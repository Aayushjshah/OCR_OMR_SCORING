#!/usr/bin/env python3
"""LightOn OCR -> OMR JSON -> evaluation pipeline.

Usage examples:
  python omr_pipeline.py --ocr-text-file ocr.txt --output parsed.json
  python omr_pipeline.py --image sheet.jpg --ocr-endpoint "$LIGHTON_OCR_ENDPOINT"
  python omr_pipeline.py --ocr-text-file ocr.txt --answer-key answers.json --weights weights.json

Answer key JSON can be either:
  {"1": "C", "2": "B"}
or:
  {"answers": {"1": "C", "2": "B"}}

Weight JSON can be either:
  {"1": 2, "2": 1}
or:
  {"weights": {"1": 2, "2": 1}}
"""

from __future__ import annotations

import argparse
import base64
import csv
import html as html_lib
import itertools
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageOps


FILLED_MARK_CHARS = {"●", "⬤", "◉", "⦿", "■", "◆", "•"}
EMPTY_MARK_CHARS = {"○", "◯", "◌", "◇", "□"}
OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_SET_OPTIONS = ["1", "2", "3", "4"]
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_DOCUMENT_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf", ".txt", ".json"}
ANSWER_GRID_TOP_FRACTION = 0.28
ANSWER_GRID_BOTTOM_FRACTION = 0.96
ANSWER_ROW_CLUSTER_FRACTION = 0.032


@dataclass(frozen=True)
class Selection:
    selected: str | list[str] | None
    status: str


def marker_value(token: str) -> bool | None:
    """Return True for filled, False for empty, None for a non-marker token."""
    cleaned = token.strip().strip("[](){}.,;:")
    if cleaned in {"O", "o"}:
        return False
    if any(char in FILLED_MARK_CHARS for char in cleaned):
        return True
    if any(char in EMPTY_MARK_CHARS for char in cleaned):
        return False
    return None


def markers_from_tokens(tokens: list[str]) -> list[bool] | None:
    values = [marker_value(token) for token in tokens]
    if not values or any(value is None for value in values):
        return None
    return [bool(value) for value in values]


def generated_options(count: int) -> list[str]:
    if count > len(OPTION_LETTERS):
        raise ValueError(f"Only up to {len(OPTION_LETTERS)} options are supported, got {count}")
    return list(OPTION_LETTERS[:count])


def parse_option_header(line: str) -> list[str] | None:
    tokens = line.split()
    if 2 <= len(tokens) <= len(OPTION_LETTERS) and all(
        len(token) == 1 and token.upper() in OPTION_LETTERS for token in tokens
    ):
        return [token.upper() for token in tokens]
    return None


def select_from_marks(options: list[str], marks: list[bool]) -> Selection:
    if len(options) != len(marks):
        options = generated_options(len(marks))

    selected = [options[index] for index, is_filled in enumerate(marks) if is_filled]
    if not selected:
        return Selection(selected=None, status="unmarked")
    if len(selected) == 1:
        return Selection(selected=selected[0], status="answered")
    return Selection(selected=selected, status="multiple")


def parse_key_value(line: str, key: str) -> str | None:
    match = re.match(rf"^\s*{re.escape(key)}\s*:\s*(.*)$", line, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def clean_ocr_value(value: str) -> str | None:
    cleaned = re.sub(r"<[^>]+>", " ", value)
    cleaned = html_lib.unescape(cleaned)
    for _ in range(3):
        updated = re.sub(
            r"\\(?:text|mathrm|mathbf|underline)\s*\{\s*([^{}]+?)\s*\}",
            r"\1",
            cleaned,
            flags=re.IGNORECASE,
        )
        if updated == cleaned:
            break
        cleaned = updated
    cleaned = re.sub(r"[$\\{}]+", " ", cleaned)
    cleaned = re.sub(r"[*_`#]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" :-–—")
    return cleaned or None


def html_table_cells(text: str) -> list[str]:
    cells = re.findall(r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>", text, flags=re.IGNORECASE | re.DOTALL)
    return [clean_ocr_value(cell) or "" for cell in cells]


def normalized_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def is_identity_label(value: str) -> bool:
    label = normalized_label(value)
    return label in {
        "name",
        "email",
        "email id",
        "email address",
        "roll no",
        "roll number",
        "set no",
        "set number",
    }


def extract_labeled_html_value(text: str, labels: set[str]) -> str | None:
    cells = html_table_cells(text)
    wanted = {normalized_label(label) for label in labels}
    for index, cell in enumerate(cells[:-1]):
        if normalized_label(cell) not in wanted:
            continue
        for value in cells[index + 1 :]:
            if value and not is_identity_label(value):
                return value
    return None


def normalize_roll_no(value: Any) -> str | None:
    if isinstance(value, dict):
        raw = clean_ocr_value(str(value.get("raw") or ""))
        if raw:
            return re.sub(r"\s+", "", raw).upper()

        selected = value.get("selected_by_column") or []
        if isinstance(selected, list):
            parts: list[str] = []
            for item in selected:
                if item in (None, "") or isinstance(item, list):
                    return None
                parts.append(str(item))
            return "".join(parts) or None

    cleaned = clean_ocr_value(str(value)) if value not in (None, "") else None
    if not cleaned:
        return None
    compact = re.sub(r"\s+", "", cleaned).upper()
    return compact if re.search(r"[A-Z0-9]", compact) else None


def normalize_ocr_set_name(value: str) -> str | None:
    cleaned = clean_ocr_value(value)
    if not cleaned:
        return None
    compact = re.sub(r"[\s_:\-–—]+", "", cleaned)
    compact = re.sub(r"(?i)^set", "", compact)
    if not compact:
        return None
    return f"set{compact}".lower()


def normalize_marked_set_selection(value: str | list[str] | None) -> str | list[str] | None:
    if isinstance(value, list):
        normalized = [normalize_marked_set_selection(item) for item in value]
        return [item for item in normalized if isinstance(item, str)] or None
    if value in (None, ""):
        return None
    if re.fullmatch(r"[A-Z]", str(value).strip(), flags=re.IGNORECASE):
        index = OPTION_LETTERS.index(str(value).strip().upper())
        if index < len(DEFAULT_SET_OPTIONS):
            return normalize_ocr_set_name(DEFAULT_SET_OPTIONS[index])
        return None
    return normalize_ocr_set_name(str(value))


def normalize_set_search_line(line: str) -> str:
    normalized = html_lib.unescape(line)
    normalized = re.sub(r"\\text\s*\{\s*([^{}]+?)\s*\}", r"\1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\\(?:long)?(?:right)?arrow|\\to", " -> ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[$*_`{}]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_name(text: str) -> str | None:
    html_value = extract_labeled_html_value(text, {"name"})
    if html_value:
        return clean_ocr_value(html_value)

    for line in text.splitlines():
        match = re.search(
            r"\b(?:candidate\s*)?(?:full\s*)?name\b\s*(?:id)?\s*[:\-–—]\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            return clean_ocr_value(match.group(1))
    return None


def extract_email(text: str) -> str | None:
    html_value = extract_labeled_html_value(text, {"email", "email id", "email address"})
    if html_value:
        compact = re.sub(r"\s+", "", html_value)
        email_match = re.search(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", compact, flags=re.IGNORECASE)
        if email_match:
            return email_match.group(0).strip()

    label_pattern = re.compile(
        r"\b(?:e[-\s]*)?mail(?:\s*(?:id|address))?\b\s*[:\-–—]\s*(.+)$",
        flags=re.IGNORECASE,
    )
    labeled_pattern = re.compile(
        r"\b(?:e[-\s]*)?mail(?:\s*(?:id|address))?\b\s*[:\-–—]\s*"
        r"([A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,})",
        flags=re.IGNORECASE,
    )
    for line in text.splitlines():
        label_match = label_pattern.search(line)
        if label_match:
            cleaned = clean_ocr_value(label_match.group(1))
            if cleaned:
                compact = re.sub(r"\s+", "", cleaned)
                email_match = re.search(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", compact, flags=re.IGNORECASE)
                if email_match:
                    return email_match.group(0).strip()

        match = labeled_pattern.search(line)
        if match:
            return match.group(1).strip()

    match = re.search(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", text, flags=re.IGNORECASE)
    return match.group(0).strip() if match else None


def extract_roll_no(text: str) -> str | None:
    html_value = extract_labeled_html_value(text, {"roll no", "roll number"})
    if html_value:
        return normalize_roll_no(html_value)

    for line in text.splitlines():
        match = re.search(r"\broll\s*(?:no|number)\.?\s*[:#=\-–—]?\s*(.+)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        value = clean_ocr_value(match.group(1))
        if not value:
            continue
        value = re.split(
            r"\b(?:instructions|candidate\s+signature|for\s+office\s+use)\b",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        return normalize_roll_no(value)
    return None


def extract_set_name(text: str) -> str | None:
    set_pattern = re.compile(
        r"\b(?:exam|paper|question\s*paper)?\s*set\s*"
        r"(?:no\.?|number|name)?\s*(?:[:#=\-–—>→⇒➜➔↦]+\s*|\s+)"
        r"((?:set\s*[-_:]?\s*)?[A-Z0-9]+)\b",
        flags=re.IGNORECASE,
    )
    for line in text.splitlines():
        normalized = normalize_set_search_line(line)
        match = set_pattern.search(normalized)
        if match:
            candidate = match.group(1)
            if candidate.lower() in {"no", "number", "name"}:
                continue
            return normalize_ocr_set_name(candidate)
    return None


def parse_set_option_header(line: str) -> list[str] | None:
    tokens = line.split()
    if 2 <= len(tokens) <= 10 and all(re.fullmatch(r"[A-Z0-9]+", token, flags=re.IGNORECASE) for token in tokens):
        return [token.upper() for token in tokens]
    return None


def marked_options_from_text(text: str, default_options: list[str] | None = None) -> list[tuple[str, bool]]:
    marker_chars = "".join(re.escape(char) for char in FILLED_MARK_CHARS | EMPTY_MARK_CHARS)
    option_marks: list[tuple[str, bool]] = []

    for match in re.finditer(rf"\b([A-Z0-9]+)\b\s*([{marker_chars}])", text, flags=re.IGNORECASE):
        option_marks.append((match.group(1).upper(), bool(marker_value(match.group(2)))))

    if option_marks:
        return option_marks

    markers = re.findall(rf"[{marker_chars}]", text)
    if not markers:
        return []

    options = default_options or generated_options(len(markers))
    if len(options) != len(markers):
        options = generated_options(len(markers))
    return [(options[index], bool(marker_value(marker))) for index, marker in enumerate(markers)]


def extract_marked_set_name(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    label_pattern = re.compile(r"\b(?:exam\s*)?set(?:\s*(?:no\.?|number|name))?\b", flags=re.IGNORECASE)

    for index, line in enumerate(lines):
        label = label_pattern.search(normalize_set_search_line(line))
        if not label:
            continue

        candidates = [line[label.end() :]]
        if index + 1 < len(lines):
            candidates.append(lines[index + 1])
        if index + 2 < len(lines):
            candidates.append(f"{lines[index + 1]} {lines[index + 2]}")

        for candidate in candidates:
            option_marks = marked_options_from_text(candidate, DEFAULT_SET_OPTIONS)
            if not option_marks:
                continue
            selection = selection_from_option_marks(option_marks)
            normalized = normalize_marked_set_selection(selection.selected)
            return normalized if isinstance(normalized, str) else None

    return None


def normalize_section(line: str) -> str | None:
    match = re.match(r"^section\s*(\d+)$", line.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return f"Section{match.group(1)}"


def is_question_row(line: str) -> tuple[str, list[bool]] | None:
    tokens = line.split()
    if len(tokens) < 2 or not tokens[0].isdigit():
        return None
    marks = markers_from_tokens(tokens[1:])
    if marks is None:
        return None
    return tokens[0], marks


def is_marker_only_row(line: str) -> list[bool] | None:
    return markers_from_tokens(line.split())


def parse_roll_no_block(lines: list[str], start_index: int) -> tuple[dict[str, Any], int]:
    index = start_index + 1
    raw = ""

    if index < len(lines) and is_question_row(lines[index]) is None:
        raw = lines[index]
        index += 1

    rows: list[tuple[str, list[bool]]] = []
    while index < len(lines):
        parsed = is_question_row(lines[index])
        if parsed is None:
            break
        label, marks = parsed
        rows.append((label, marks))
        index += 1

    selected_by_column: list[str | list[str] | None] = []
    statuses: list[str] = []
    if rows:
        width = max(len(marks) for _, marks in rows)
        for column_index in range(width):
            selected = [
                row_label
                for row_label, marks in rows
                if column_index < len(marks) and marks[column_index]
            ]
            if not selected:
                selected_by_column.append(None)
                statuses.append("unmarked")
            elif len(selected) == 1:
                selected_by_column.append(selected[0])
                statuses.append("answered")
            else:
                selected_by_column.append(selected)
                statuses.append("multiple")

    return {
        "raw": raw,
        "selected_by_column": selected_by_column,
        "statuses": statuses,
    }, index


def parse_omr_text(ocr_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]
    candidate: dict[str, Any] = {
        "name": None,
        "exam": None,
        "date": None,
        "exam_set": None,
        "roll_no": None,
    }
    answers: list[dict[str, Any]] = []

    current_subject: str | None = None
    current_section: str | None = None
    current_options: list[str] | None = None
    pending_exam_set = False
    index = 0

    while index < len(lines):
        line = lines[index]

        name = parse_key_value(line, "NAME")
        if name is not None:
            candidate["name"] = name or None
            index += 1
            continue

        exam = parse_key_value(line, "EXAM")
        if exam is not None:
            candidate["exam"] = exam or None
            index += 1
            continue

        date = parse_key_value(line, "DATE")
        if date is not None:
            candidate["date"] = date or None
            index += 1
            continue

        if re.fullmatch(r"(?:exam\s*)?set(?:\s*(?:no\.?|number|name))?", line, flags=re.IGNORECASE):
            pending_exam_set = True
            current_subject = None
            current_section = None
            index += 1
            continue

        roll_value = parse_key_value(line, "Roll No.") or parse_key_value(line, "Roll No") or parse_key_value(line, "Roll Number")
        if roll_value is not None:
            candidate["roll_no"] = normalize_roll_no(roll_value)
            index += 1
            continue

        if re.fullmatch(r"roll\s*(?:no|number)\.?", line, flags=re.IGNORECASE):
            candidate["roll_no"], index = parse_roll_no_block(lines, index)
            continue

        if pending_exam_set:
            set_option_header = parse_set_option_header(line)
            if set_option_header is not None:
                current_options = set_option_header
                index += 1
                continue

        option_header = parse_option_header(line)
        if option_header is not None:
            current_options = option_header
            index += 1
            continue

        marker_row = is_marker_only_row(line)
        if marker_row is not None and pending_exam_set:
            options = current_options or DEFAULT_SET_OPTIONS[: len(marker_row)]
            selection = select_from_marks(options, marker_row)
            candidate["exam_set"] = {
                "options": options if len(options) == len(marker_row) else DEFAULT_SET_OPTIONS[: len(marker_row)],
                "selected": normalize_marked_set_selection(selection.selected),
                "status": selection.status,
            }
            pending_exam_set = False
            index += 1
            continue

        section = normalize_section(line)
        if section is not None:
            current_section = section
            index += 1
            continue

        question_row = is_question_row(line)
        if question_row is not None:
            question_id, marks = question_row
            options = current_options or generated_options(len(marks))
            if len(options) != len(marks):
                options = generated_options(len(marks))
            selection = select_from_marks(options, marks)
            answers.append(
                {
                    "question_id": question_id,
                    "subject": current_subject,
                    "section": current_section,
                    "options": options,
                    "selected": selection.selected,
                    "status": selection.status,
                }
            )
            index += 1
            continue

        current_subject = line
        current_section = None
        index += 1

    return {
        "candidate": candidate,
        "answers": answers,
    }


def coerce_ocr_text(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith(("{", "[")):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return raw_text
        extracted = find_chat_message_content(payload)
        return extracted or raw_text
    return raw_text


def parse_option_cell(cell: str) -> tuple[str, bool] | None:
    match = re.match(r"^\s*([A-D])\b\s*(.*)$", cell, flags=re.IGNORECASE)
    if not match:
        return None
    option = match.group(1).upper()
    marker = marker_value(match.group(2))
    return option, bool(marker)


def parse_table_option_cell(cell: str, option_index: int) -> tuple[str, bool] | None:
    explicit = parse_option_cell(cell)
    if explicit is not None:
        return explicit

    marker = marker_value(cell)
    if marker is None:
        return None
    options = generated_options(option_index + 1)
    return options[option_index], bool(marker)


def selection_from_option_marks(option_marks: list[tuple[str, bool]]) -> Selection:
    if not option_marks:
        return Selection(selected=None, status="missing")
    selected = [option for option, is_filled in option_marks if is_filled]
    if not selected:
        return Selection(selected=None, status="unmarked")
    if len(selected) == 1:
        return Selection(selected=selected[0], status="answered")
    return Selection(selected=selected, status="multiple")


def append_marked_answer(
    answers_by_id: dict[str, dict[str, Any]],
    question_id: str | None,
    option_marks: list[tuple[str, bool]],
) -> None:
    if not question_id:
        return
    selection = selection_from_option_marks(option_marks)
    answers_by_id[question_id] = {
        "question_id": question_id,
        "options": [option for option, _ in option_marks],
        "selected": selection.selected,
        "status": selection.status,
    }


def clean_table_cell(cell: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", cell)
    cleaned = html_lib.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_html_table_answers(text: str) -> dict[str, dict[str, Any]]:
    cells = [clean_table_cell(cell) for cell in re.findall(r"<td\b[^>]*>(.*?)</td>", text, flags=re.IGNORECASE | re.DOTALL)]
    answers_by_id: dict[str, dict[str, Any]] = {}
    current_question: str | None = None
    option_marks: list[tuple[str, bool]] = []

    for cell in cells:
        question_match = re.fullmatch(r"(?:q(?:uestion)?\s*)?(\d{1,3})", cell, flags=re.IGNORECASE)
        option_mark = parse_table_option_cell(cell, len(option_marks)) if current_question else None
        if question_match:
            append_marked_answer(answers_by_id, current_question, option_marks)
            current_question = question_match.group(1)
            option_marks = []
        elif current_question and option_mark:
            option_marks.append(option_mark)

    append_marked_answer(answers_by_id, current_question, option_marks)
    return answers_by_id


def extract_inline_marked_answers(text: str) -> dict[str, dict[str, Any]]:
    marker_chars = "".join(re.escape(char) for char in FILLED_MARK_CHARS | EMPTY_MARK_CHARS)
    answer_pattern = re.compile(
        rf"(?<!\w)(\d{{1,3}})\b\s*((?:[A-D]\s*[{marker_chars}]\s*){{1,4}})",
        flags=re.IGNORECASE,
    )
    option_pattern = re.compile(rf"\b([A-D])\s*([{marker_chars}])", flags=re.IGNORECASE)
    answers_by_id: dict[str, dict[str, Any]] = {}

    for match in answer_pattern.finditer(text):
        option_marks = [
            (option.upper(), bool(marker_value(marker)))
            for option, marker in option_pattern.findall(match.group(2))
        ]
        append_marked_answer(answers_by_id, match.group(1), option_marks)

    return answers_by_id


def extract_simple_labeled_answers(text: str) -> dict[str, dict[str, Any]]:
    answers_by_id: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r"^(?:q(?:uestion)?\s*)?(\d+)\s*[:.)\-\s]\s*([A-D])\b", flags=re.IGNORECASE)
    for raw_line in text.splitlines():
        match = pattern.match(raw_line.strip())
        if match:
            question_id = match.group(1)
            answers_by_id[question_id] = {
                "question_id": question_id,
                "selected": match.group(2).upper(),
                "status": "answered",
            }
    return answers_by_id


def extract_submission_answers(text: str) -> list[dict[str, Any]]:
    answers_by_id: dict[str, dict[str, Any]] = {}
    answers_by_id.update(extract_simple_labeled_answers(text))
    answers_by_id.update(extract_inline_marked_answers(text))
    answers_by_id.update(extract_html_table_answers(text))
    return [
        answers_by_id[question_id]
        for question_id in sorted(answers_by_id, key=lambda value: int(value) if value.isdigit() else value)
    ]


def parse_simple_submission_text(ocr_text: str) -> dict[str, Any]:
    """Parse the interim OCR contract: Name, Email, Set, and question answers."""
    ocr_text = coerce_ocr_text(ocr_text)
    candidate: dict[str, Any] = {
        "name": extract_name(ocr_text),
        "email": extract_email(ocr_text),
        "roll_no": extract_roll_no(ocr_text),
        "exam_set": extract_marked_set_name(ocr_text) or extract_set_name(ocr_text),
    }

    for raw_line in ocr_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        for field, key in (("name", "name"), ("email", "email")):
            value = parse_key_value(line, field)
            if value is not None:
                candidate[key] = clean_ocr_value(value) or None
                break

    return {
        "candidate": candidate,
        "answers": extract_submission_answers(ocr_text),
    }


def parse_submission_text(ocr_text: str) -> dict[str, Any]:
    ocr_text = coerce_ocr_text(ocr_text)
    parsed = parse_simple_submission_text(ocr_text)
    if any(value not in (None, "") for value in parsed["candidate"].values()) or parsed["answers"]:
        return parsed

    parsed = parse_omr_text(ocr_text)
    candidate = parsed["candidate"]
    if candidate.get("exam_set") and isinstance(candidate["exam_set"], dict):
        candidate["exam_set"] = candidate["exam_set"].get("selected")
    return parsed


def normalize_choice_label(value: Any) -> str | list[str]:
    text = str(value).strip().upper()
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return text


def is_choice_label(value: Any) -> bool:
    normalized = normalize_choice_label(value)
    if isinstance(normalized, list):
        return bool(normalized) and all(part in {"A", "B", "C", "D"} for part in normalized)
    return normalized in {"A", "B", "C", "D"}


def parse_marks(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    normalized = {key.strip().lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = normalized.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def answer_key_from_csv(csv_path: str | Path, set_name: str) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    weights: dict[str, float] = {}
    questions: list[dict[str, Any]] = []

    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=2):
            question_id = _first_present(row, ("Question Number", "Q#", "Question #", "question_id", "question"))
            choice = _first_present(row, ("Answer Choice Label", "Answer Label", "Correct Option", "Answer Choice"))
            marks = _first_present(row, ("Marks", "Mark", "Score", "Weight"))
            if question_id is None or choice is None or marks is None:
                if question_id is None and choice is None:
                    continue
                raise ValueError(f"Missing question, answer label, or marks in CSV row {row_number}")
            if not is_choice_label(choice) and is_choice_label(row.get("A")):
                choice = row.get("A")
                marks = row.get("Time Given") or marks

            parsed_marks = parse_marks(marks)
            if parsed_marks is None:
                extras = row.get(None) or []
                candidates = [row.get("Time Given"), *extras]
                parsed_marks = next((value for value in (parse_marks(item) for item in candidates) if value is not None), None)
            if parsed_marks is None:
                raise ValueError(f"Invalid marks value in CSV row {row_number}: {marks!r}")

            question_key = str(question_id).strip()
            answers[question_key] = normalize_choice_label(choice)
            weights[question_key] = parsed_marks
            questions.append(
                {
                    "question_id": question_key,
                    "answer": answers[question_key],
                    "marks": weights[question_key],
                }
            )

    if not answers:
        raise ValueError("Answer CSV did not contain any questions")

    return {
        "set": set_name,
        "answers": answers,
        "weights": weights,
        "questions": questions,
    }


def save_answer_key_json(csv_path: str | Path, set_name: str, output_dir: str | Path) -> Path:
    payload = answer_key_from_csv(csv_path, set_name)
    safe_set_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", set_name.strip()).strip("_")
    if not safe_set_name:
        raise ValueError("Set name is required")
    output_path = Path(output_dir) / f"{safe_set_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_answer_key_for_set(set_name: str, key_dir: str | Path) -> dict[str, Any]:
    safe_set_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", set_name.strip()).strip("_")
    if not safe_set_name:
        raise ValueError("Set is missing from OCR output")
    path = Path(key_dir) / f"{safe_set_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No answer key JSON found for set {set_name!r} at {path}")
    return load_json_file(path)


def candidate_set_name(parsed: dict[str, Any]) -> str | None:
    value = parsed.get("candidate", {}).get("exam_set")
    if isinstance(value, dict):
        value = value.get("selected")
    return str(value).strip() if value not in (None, "") else None


def candidate_roll_no(parsed: dict[str, Any]) -> str | None:
    return normalize_roll_no(parsed.get("candidate", {}).get("roll_no"))


def identity_warnings(parsed: dict[str, Any]) -> list[str]:
    return []


def score_submission(parsed: dict[str, Any], answer_key_payload: dict[str, Any]) -> dict[str, Any]:
    evaluation = evaluate_answers(
        parsed,
        answer_key=normalize_answer_map(answer_key_payload, "answers"),
        weights=normalize_weight_map(answer_key_payload.get("weights", {})),
    )
    candidate = parsed.get("candidate", {})
    total_questions = len(evaluation["details"])
    answered_questions = sum(1 for detail in evaluation["details"] if detail.get("status") == "answered")
    return {
        "name": candidate.get("name"),
        "email": candidate.get("email"),
        "roll_no": candidate_roll_no(parsed),
        "set": candidate_set_name(parsed),
        "answered_questions": answered_questions,
        "total_questions": total_questions,
        "unanswered_questions": total_questions - answered_questions,
        "score": evaluation["score"],
        "max_score": evaluation["max_score"],
        "identity_warnings": identity_warnings(parsed),
        "evaluation": evaluation,
    }


def normalize_answer_map(payload: Any, top_level_key: str) -> dict[str, Any]:
    if isinstance(payload, dict) and top_level_key in payload:
        payload = payload[top_level_key]

    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items()}

    if isinstance(payload, list):
        normalized: dict[str, Any] = {}
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid {top_level_key} item: {item!r}")
            question_id = item.get("question_id") or item.get("id") or item.get("question")
            answer = item.get("answer") or item.get("correct") or item.get("correct_answer")
            if question_id is None or answer is None:
                raise ValueError(f"Invalid {top_level_key} item: {item!r}")
            normalized[str(question_id)] = answer
        return normalized

    raise ValueError(f"{top_level_key} JSON must be an object or a list")


def normalize_weight_map(payload: Any) -> dict[str, float]:
    if isinstance(payload, dict) and "weights" in payload:
        payload = payload["weights"]

    if isinstance(payload, list):
        raw_weights: dict[str, Any] = {}
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid weights item: {item!r}")
            question_id = item.get("question_id") or item.get("id") or item.get("question")
            weight = item.get("weight", item.get("marks", item.get("score")))
            if question_id is None or weight is None:
                raise ValueError(f"Invalid weights item: {item!r}")
            raw_weights[str(question_id)] = weight
    elif isinstance(payload, dict):
        raw_weights = {str(key): value for key, value in payload.items()}
    else:
        raise ValueError("weights JSON must be an object or a list")

    weights: dict[str, float] = {}
    for question_id, value in raw_weights.items():
        if isinstance(value, dict):
            value = value.get("weight", value.get("marks", value.get("score")))
        if value is None:
            raise ValueError(f"Missing weight for question {question_id}")
        weights[question_id] = float(value)
    return weights


def answer_matches(selected: Any, correct: Any) -> bool:
    if selected is None:
        return False
    if isinstance(selected, list):
        return selected == correct
    if isinstance(correct, list):
        return selected in correct
    return str(selected).strip().upper() == str(correct).strip().upper()


def evaluate_answers(
    parsed: dict[str, Any],
    answer_key: dict[str, Any],
    weights: dict[str, float],
    default_weight: float = 1.0,
) -> dict[str, Any]:
    answer_by_id = {answer["question_id"]: answer for answer in parsed.get("answers", [])}
    details: list[dict[str, Any]] = []
    total_score = 0.0
    max_score = 0.0

    for question_id in sorted(answer_key, key=lambda value: int(value) if value.isdigit() else value):
        correct = answer_key[question_id]
        weight = weights.get(question_id, default_weight)
        max_score += weight

        answer = answer_by_id.get(question_id)
        selected = answer.get("selected") if answer else None
        status = answer.get("status") if answer else "missing"
        is_correct = answer_matches(selected, correct)
        score = weight if is_correct else 0.0
        total_score += score

        details.append(
            {
                "question_id": question_id,
                "selected": selected,
                "correct": correct,
                "status": status,
                "weight": weight,
                "is_correct": is_correct,
                "score": score,
            }
        )

    return {
        "score": total_score,
        "max_score": max_score,
        "details": details,
    }


def load_json_file(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def build_multipart_body(field_name: str, image_path: Path) -> tuple[bytes, str]:
    boundary = f"----omr-pipeline-{uuid.uuid4().hex}"
    filename = image_path.name
    content_type = "application/octet-stream"
    file_bytes = image_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    body += file_bytes
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    return body, f"multipart/form-data; boundary={boundary}"


def find_text_value(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "ocr_text", "raw_text", "markdown", "content", "output", "result"):
            value = payload.get(key)
            text = find_text_value(value)
            if text:
                return text
        for value in payload.values():
            text = find_text_value(value)
            if text:
                return text
    if isinstance(payload, list):
        parts = [find_text_value(item) for item in payload]
        parts = [part for part in parts if part]
        if parts:
            return "\n".join(parts)
    return None


def find_chat_message_content(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return find_text_value(payload)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            return find_text_value(message.get("content"))
    return find_text_value(payload)


def load_omr_layout_image(image_path: str | Path) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    image = normalize_omr_page_perspective(image)
    return normalize_omr_layout_orientation(image)


def normalize_omr_page_perspective(image: Image.Image) -> Image.Image:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return image

    rgb = np.array(image)
    quad = _detect_page_marker_quad(rgb)
    if quad is None:
        return image

    tl, tr, br, bl = quad
    quad_area = float(cv2.contourArea(quad))
    coverage = quad_area / float(rgb.shape[0] * rgb.shape[1])
    top_width = float(np.linalg.norm(tr - tl))
    bottom_width = float(np.linalg.norm(br - bl))
    left_height = float(np.linalg.norm(bl - tl))
    right_height = float(np.linalg.norm(br - tr))
    width_delta = abs(top_width - bottom_width) / max(1.0, top_width, bottom_width)
    height_delta = abs(left_height - right_height) / max(1.0, left_height, right_height)
    if coverage > 0.70 or width_delta > 0.35 or height_delta > 0.22:
        return image

    min_x, min_y = quad.min(axis=0)
    max_x, max_y = quad.max(axis=0)
    if min_y > rgb.shape[0] * 0.25 or max_y < rgb.shape[0] * 0.72:
        return image
    if min_x > rgb.shape[1] * 0.28 or max_x < rgb.shape[1] * 0.75:
        return image

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 400 or height < 600:
        return image

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    warped = cv2.warpPerspective(rgb, transform, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def _detect_page_marker_quad(rgb: Any) -> Any | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    height, width = rgb.shape[:2]
    image_area = height * width
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates: list[dict[str, Any]] = []
    for component_index in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_index]
        if area < max(400, image_area * 0.00008) or area > image_area * 0.02:
            continue
        aspect_ratio = component_width / float(component_height)
        if not (0.45 <= aspect_ratio <= 2.20):
            continue
        fill_ratio = area / float(component_width * component_height)
        if fill_ratio < 0.35:
            continue
        center_x, center_y = centroids[component_index]
        candidates.append(
            {
                "center": (float(center_x), float(center_y)),
                "area": float(area),
                "box": (int(x), int(y), int(component_width), int(component_height)),
            }
        )

    if len(candidates) < 4:
        return None

    candidates = sorted(candidates, key=lambda item: float(item["area"]), reverse=True)[:40]

    def order_points(points: list[tuple[float, float]]) -> Any:
        pts = np.array(points, dtype=np.float32)
        sums = pts.sum(axis=1)
        diffs = np.diff(pts, axis=1)[:, 0]
        return np.array(
            [pts[np.argmin(sums)], pts[np.argmin(diffs)], pts[np.argmax(sums)], pts[np.argmax(diffs)]],
            dtype=np.float32,
        )

    def score_quad(combo: tuple[dict[str, Any], ...]) -> tuple[float, Any]:
        quad = order_points([candidate["center"] for candidate in combo])
        area = float(cv2.contourArea(quad))
        if area < image_area * 0.25:
            return -float("inf"), quad

        tl, tr, br, bl = quad
        page_width = max(float(np.linalg.norm(br - bl)), float(np.linalg.norm(tr - tl)))
        page_height = max(float(np.linalg.norm(tr - br)), float(np.linalg.norm(tl - bl)))
        if page_width <= 0 or page_height <= 0:
            return -float("inf"), quad

        aspect_ratio = page_height / page_width
        if not (1.15 <= aspect_ratio <= 1.85):
            return -float("inf"), quad

        corner_distance = (
            float(np.linalg.norm(tl - np.array([0, 0])))
            + float(np.linalg.norm(tr - np.array([width, 0])))
            + float(np.linalg.norm(br - np.array([width, height])))
            + float(np.linalg.norm(bl - np.array([0, height])))
        )
        marker_area = sum(float(candidate["area"]) for candidate in combo)
        aspect_penalty = abs(aspect_ratio - 1.42) * image_area * 0.05
        return area + (marker_area * 20.0) - (corner_distance * 300.0) - aspect_penalty, quad

    best_score = -float("inf")
    best_quad = None
    for combo in itertools.combinations(candidates, 4):
        score, quad = score_quad(combo)
        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad


def normalize_omr_layout_orientation(image: Image.Image) -> Image.Image:
    variants = [(angle, image.rotate(angle, expand=True) if angle else image) for angle in (0, 90, 180, 270)]
    scored_variants = [(layout_orientation_score(variant), angle, variant) for angle, variant in variants]
    best_score, _, best_image = max(scored_variants, key=lambda item: (item[0], -item[1]))
    original_score = scored_variants[0][0]
    if best_score <= 0 or best_score < original_score + 8.0:
        return image
    return best_image


def layout_orientation_score(image: Image.Image) -> float:
    try:
        import numpy as np
    except ImportError:
        return 0.0

    width, height = image.size
    rgb = np.array(image)
    answers = _detect_answers_from_rgb(rgb)
    answered_count = sum(1 for answer in answers if answer.get("selected") is not None)
    set_bonus = 8 if _detect_marked_set_from_rgb(rgb, allow_filled_position=False) else 0
    portrait_bonus = 2 if height > width else 0
    header_bonus = _layout_upright_header_score(rgb)
    return float((len(answers) * 10) + answered_count + set_bonus + portrait_bonus + (header_bonus * 1.5))


def _layout_upright_header_score(rgb: Any) -> float:
    """Favor orientations with the title/identity header at the top of the page."""
    marked = _marked_pixel_mask(rgb)
    if marked is None or marked.size == 0:
        return 0.0

    height, width = marked.shape[:2]

    def fraction(x0: float, y0: float, x1: float, y1: float) -> float:
        region = marked[int(height * y0) : int(height * y1), int(width * x0) : int(width * x1)]
        return float(region.mean()) if region.size else 0.0

    top_title_density = fraction(0.20, 0.03, 0.82, 0.10)
    top_identity_density = fraction(0.06, 0.10, 0.96, 0.22)
    top_set_density = fraction(0.62, 0.09, 0.94, 0.18)
    top_instruction_density = fraction(0.04, 0.22, 0.78, 0.34)
    bottom_title_density = fraction(0.18, 0.90, 0.82, 0.98)
    bottom_identity_density = fraction(0.04, 0.78, 0.96, 0.91)
    bottom_set_density = fraction(0.04, 0.80, 0.40, 0.91)
    bottom_footer_density = fraction(0.08, 0.86, 0.92, 0.98)

    return (
        (top_title_density * 420.0)
        + (top_identity_density * 90.0)
        + (top_set_density * 70.0)
        + (top_instruction_density * 35.0)
        - (bottom_title_density * 520.0)
        - (bottom_identity_density * 55.0)
        - (bottom_set_density * 45.0)
        - (bottom_footer_density * 25.0)
    )


def _marked_pixel_mask(rgb: Any) -> Any | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue_or_purple_ink = (hue > 90) & (hue < 145) & (saturation > 35) & (value < 215)
    dark_threshold = 85.0
    if float(np.percentile(value, 1)) > dark_threshold:
        otsu_threshold, _ = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_threshold = min(185.0, max(dark_threshold, float(otsu_threshold)))
    dark_ink = value < dark_threshold
    return blue_or_purple_ink | dark_ink


def preprocess_image_for_ocr(image_path: str | Path, output_path: str | Path) -> Path:
    image = load_omr_layout_image(image_path)

    try:
        import cv2
        import numpy as np

        rgb = np.array(ImageEnhance.Color(image).enhance(1.45))
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        blue_ink = (
            (blue > 55)
            & (blue > red + 18)
            & (blue > green + 8)
            & (red < 170)
            & (green < 185)
        )
        mask = (blue_ink.astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        rgb[mask > 0] = (0, 0, 0)
        image = Image.fromarray(rgb)
    except ImportError:
        image = ImageEnhance.Color(image).enhance(1.45)

    image = ImageEnhance.Contrast(image).enhance(1.35)
    image = ImageEnhance.Sharpness(image).enhance(1.15)
    try:
        import numpy as np

        rgb = np.array(image, dtype=np.uint16)
        rgb = ((rgb + 4) // 8) * 8
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    except ImportError:
        pass

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    return output


def crop_identity_header_for_ocr(image_path: str | Path, output_path: str | Path) -> Path:
    image = load_omr_layout_image(image_path)
    width, height = image.size
    crop_box = (
        int(width * 0.10),
        int(height * 0.12),
        int(width * 0.90),
        int(height * 0.26),
    )
    crop = image.crop(crop_box)
    crop = crop.resize((crop.width * 3, crop.height * 3))
    crop = ImageEnhance.Contrast(crop).enhance(1.6)
    crop = ImageEnhance.Sharpness(crop).enhance(1.4)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output, format="PNG")
    return output


def detect_marked_set_from_image(image_path: str | Path) -> str | None:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    allow_filled_position = image.width >= 2000 and image.height >= 2800
    image = normalize_omr_layout_orientation(image)
    try:
        import numpy as np
    except ImportError:
        return None

    detected_set = _detect_marked_set_from_rgb(np.array(image), allow_filled_position=allow_filled_position)
    if detected_set:
        return detected_set

    image = load_omr_layout_image(image_path)
    return _detect_marked_set_from_rgb(np.array(image), allow_filled_position=False)


def _detect_marked_set_from_rgb(rgb: Any, *, allow_filled_position: bool = True) -> str | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    height, width = rgb.shape[:2]
    crop = rgb[int(height * 0.12) : int(height * 0.26), int(width * 0.58) : int(width * 0.91)]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    min_radius = max(6, int(width * 0.008))
    max_radius = max(12, int(width * 0.025))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, crop.shape[1] // 12),
        param1=80,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    marked = _marked_pixel_mask(crop)
    if marked is None:
        return None

    candidates: list[dict[str, float | str]] = []

    def add_candidate(x: float, y: float, radius: float, source: str) -> None:
        if not (crop.shape[1] * 0.20 < x < crop.shape[1] * 0.95):
            return
        for existing in candidates:
            distance = ((float(existing["x"]) - x) ** 2 + (float(existing["y"]) - y) ** 2) ** 0.5
            if distance <= max(9.0, (float(existing["radius"]) + radius) * 0.45):
                if source == "filled":
                    existing["x"] = x
                    existing["y"] = y
                else:
                    existing["x"] = (float(existing["x"]) + x) / 2
                    existing["y"] = (float(existing["y"]) + y) / 2
                existing["radius"] = max(float(existing["radius"]), radius)
                existing["source"] = f"{existing['source']},{source}"
                return
        candidates.append({"x": x, "y": y, "radius": radius, "source": source})

    if circles is not None:
        for x, y, radius in np.round(circles[0]).astype(int).tolist():
            add_candidate(float(x), float(y), float(radius), "outline")

    component_mask = (marked.astype(np.uint8) * 255)
    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(component_mask, 8)
    min_area = math.pi * (min_radius * 0.45) ** 2
    max_area = math.pi * (max_radius * 1.15) ** 2
    filled_components: list[dict[str, float]] = []
    for component_index in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_index]
        center_x, center_y = centroids[component_index]
        if not (min_area <= area <= max_area):
            continue
        if component_width < min_radius * 0.8 or component_height < min_radius * 0.8:
            continue
        if component_width > max_radius * 2.4 or component_height > max_radius * 2.4:
            continue
        aspect_ratio = component_width / float(component_height)
        if not (0.55 <= aspect_ratio <= 1.85):
            continue
        filled_components.append(
            {
                "x": float(center_x),
                "y": float(center_y),
                "radius": float(max(component_width, component_height) / 2),
                "area": float(area),
                "aspect_ratio": float(aspect_ratio),
            }
        )
        add_candidate(float(center_x), float(center_y), float(max(component_width, component_height) / 2), "filled")

    if len(candidates) < len(DEFAULT_SET_OPTIONS):
        return None

    def fill_score(candidate: dict[str, float | str]) -> float:
        x = int(round(float(candidate["x"])))
        y = int(round(float(candidate["y"])))
        radius = float(candidate["radius"])
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), max(3, int(radius * 0.60)), 255, -1)
        pixels = marked[mask > 0]
        return float(pixels.mean()) if pixels.size else 0.0

    def fill_scores(group: tuple[dict[str, float | str], ...]) -> list[float]:
        scores: list[float] = []
        for candidate in sorted(group, key=lambda item: float(item["x"])):
            scores.append(fill_score(candidate))
        return scores

    def detect_from_filled_position() -> str | None:
        plausible_components: list[tuple[float, dict[str, float]]] = []
        for component in filled_components:
            x_ratio = component["x"] / float(crop.shape[1])
            y_ratio = component["y"] / float(crop.shape[0])
            if not (0.10 <= x_ratio <= 0.84 and 0.40 <= y_ratio <= 0.84):
                continue
            if not (0.72 <= component["aspect_ratio"] <= 1.45):
                continue
            set_score = component["area"] - (abs(y_ratio - 0.62) * 900.0)
            plausible_components.append((set_score, component))

        if not plausible_components:
            return None

        _, component = max(plausible_components, key=lambda item: item[0])
        x_ratio = component["x"] / float(crop.shape[1])
        if x_ratio < 0.36:
            selected_index = 0
        elif x_ratio < 0.43:
            selected_index = 1
        elif x_ratio < 0.525:
            selected_index = 2
        else:
            selected_index = 3
        return normalize_ocr_set_name(DEFAULT_SET_OPTIONS[selected_index])

    if allow_filled_position:
        positioned_set = detect_from_filled_position()
        if positioned_set:
            return positioned_set

    def detect_from_filled_anchor() -> str | None:
        expected_gap = width * 0.042
        anchor_candidates: list[tuple[float, dict[str, float | str]]] = []
        for candidate in candidates:
            source = str(candidate["source"])
            if "filled" not in source:
                continue
            x = float(candidate["x"])
            y = float(candidate["y"])
            if not (crop.shape[1] * 0.20 < x < crop.shape[1] * 0.84):
                continue
            if not (crop.shape[0] * 0.42 < y < crop.shape[0] * 0.84):
                continue
            score = fill_score(candidate)
            if score > 0.25:
                anchor_candidates.append((score, candidate))

        best_index: int | None = None
        best_score = -float("inf")
        for anchor_score, anchor in anchor_candidates:
            anchor_x = float(anchor["x"])
            anchor_y = float(anchor["y"])
            row_candidates = [
                candidate
                for candidate in candidates
                if crop.shape[1] * 0.12 < float(candidate["x"]) < crop.shape[1] * 0.92
                and abs(float(candidate["y"]) - anchor_y) <= crop.shape[0] * 0.18
            ]
            for group in itertools.combinations(row_candidates, len(DEFAULT_SET_OPTIONS)):
                ordered = tuple(sorted(group, key=lambda item: float(item["x"])))
                xs = [float(candidate["x"]) for candidate in ordered]
                ys = [float(candidate["y"]) for candidate in ordered]
                gaps = [xs[position + 1] - xs[position] for position in range(len(xs) - 1)]
                mean_gap = float(np.mean(gaps))
                if not (crop.shape[1] * 0.24 <= xs[0] <= crop.shape[1] * 0.38):
                    continue
                if not (width * 0.025 <= mean_gap <= width * 0.058):
                    continue
                if min(gaps) < mean_gap * 0.50 or max(gaps) > mean_gap * 1.55:
                    continue
                closest_index = min(range(len(xs)), key=lambda index: abs(xs[index] - anchor_x))
                alignment = abs(xs[closest_index] - anchor_x)
                if alignment > max(18.0, mean_gap * 0.45):
                    continue
                scores = fill_scores(ordered)
                if scores[closest_index] < 0.25 or sum(score > 0.25 for score in scores) != 1:
                    continue
                y_spread = float(np.std(ys))
                gap_spread = float(np.std(gaps))
                score = (
                    (anchor_score * 120.0)
                    - (abs(mean_gap - expected_gap) * 0.45)
                    - (gap_spread * 1.2)
                    - y_spread
                    - (alignment * 0.25)
                )
                if score > best_score:
                    best_score = score
                    best_index = closest_index

        if best_index is None:
            return None
        return normalize_ocr_set_name(DEFAULT_SET_OPTIONS[best_index])

    if allow_filled_position:
        anchored_set = detect_from_filled_anchor()
        if anchored_set:
            return anchored_set

    best_group: tuple[dict[str, float | str], ...] | None = None
    best_group_score = -float("inf")
    best_scores: list[float] = []
    for group in itertools.combinations(candidates, len(DEFAULT_SET_OPTIONS)):
        ordered = tuple(sorted(group, key=lambda item: float(item["x"])))
        xs = [float(candidate["x"]) for candidate in ordered]
        ys = [float(candidate["y"]) for candidate in ordered]
        gaps = [xs[position + 1] - xs[position] for position in range(len(xs) - 1)]
        mean_gap = float(np.mean(gaps))
        if not (width * 0.022 <= mean_gap <= width * 0.075):
            continue
        if min(gaps) < mean_gap * 0.35 or max(gaps) > mean_gap * 1.90:
            continue
        y_spread = float(np.std(ys))
        if y_spread > crop.shape[0] * 0.18:
            continue
        gap_spread = float(np.std(gaps))
        scores = fill_scores(ordered)
        group_score = (max(scores) * 60) - y_spread - gap_spread
        if sum(score > 0.25 for score in scores) == 1:
            group_score += 10
        if group_score > best_group_score:
            best_group_score = group_score
            best_group = ordered
            best_scores = scores

    if best_group is None or not best_scores:
        return None

    selected_index = max(range(len(best_scores)), key=best_scores.__getitem__)
    if best_scores[selected_index] < 0.25:
        return None
    return normalize_ocr_set_name(DEFAULT_SET_OPTIONS[selected_index])


def detect_answers_from_image(image_path: str | Path) -> list[dict[str, Any]]:
    image = load_omr_layout_image(image_path)
    try:
        import numpy as np
    except ImportError:
        return []

    return _detect_answers_from_rgb(np.array(image))


def _select_answer_grid_rows(rows: list[list[list[int]]], np: Any) -> list[list[list[int]]]:
    if len(rows) <= 8:
        return rows[:8]

    def median_y(row: list[list[int]]) -> float:
        return float(np.median([circle[1] for circle in row]))

    def window_score(window: list[list[list[int]]]) -> float:
        ys = [median_y(row) for row in window]
        gaps = [ys[index + 1] - ys[index] for index in range(len(ys) - 1)]
        median_gap = max(1.0, float(np.median(gaps)))
        gap_score = sum(abs(gap - median_gap) / median_gap for gap in gaps)
        extra_circle_penalty = sum(max(0, len(row) - 8) for row in window) * 0.08
        return gap_score + extra_circle_penalty

    windows = [rows[index : index + 8] for index in range(len(rows) - 7)]
    return min(windows, key=window_score)


def _answer_from_scores(question_id: str, option_scores: list[float]) -> dict[str, Any]:
    selected_options = [
        OPTION_LETTERS[index]
        for index, score in enumerate(option_scores)
        if score > 0.25
    ]
    if not selected_options:
        selected: str | list[str] | None = None
        status = "unmarked"
    elif len(selected_options) == 1:
        selected = selected_options[0]
        status = "answered"
    else:
        selected = selected_options
        status = "multiple"

    return {
        "question_id": question_id,
        "options": list(OPTION_LETTERS[:4]),
        "selected": selected,
        "status": status,
        "source": "image",
    }


def _detect_answers_from_registered_grid(rgb: Any) -> list[dict[str, Any]]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(25, int(width * 0.04)),
        param1=80,
        param2=24,
        minRadius=max(8, int(width * 0.008)),
        maxRadius=max(20, int(width * 0.025)),
    )
    if circles is None:
        return []

    detected = np.round(circles[0]).astype(int).tolist()
    detected = [
        circle
        for circle in detected
        if height * 0.25 < circle[1] < height * ANSWER_GRID_BOTTOM_FRACTION
        and width * 0.03 < circle[0] < width * 0.90
    ]
    detected = sorted(detected, key=lambda circle: (circle[1], circle[0]))

    raw_rows: list[list[list[int]]] = []
    for circle in detected:
        if not raw_rows or abs(float(np.median([item[1] for item in raw_rows[-1]])) - circle[1]) > height * ANSWER_ROW_CLUSTER_FRACTION:
            raw_rows.append([circle])
        else:
            raw_rows[-1].append(circle)

    candidate_rows: list[list[list[int]]] = []
    for row in raw_rows:
        if len(row) > 8:
            median_y = float(np.median([circle[1] for circle in row]))
            row = [circle for circle in row if abs(circle[1] - median_y) <= height * 0.02]
        if len(row) >= 4:
            candidate_rows.append(row)

    if len(candidate_rows) < 8:
        return []

    def median_y(row: list[list[int]]) -> float:
        return float(np.median([circle[1] for circle in row]))

    def row_window_score(window: list[list[list[int]]]) -> float:
        ys = [median_y(row) for row in window]
        gaps = [ys[index + 1] - ys[index] for index in range(len(ys) - 1)]
        median_gap = max(1.0, float(np.median(gaps)))
        gap_score = sum(abs(gap - median_gap) / median_gap for gap in gaps)
        missing_circle_penalty = sum(max(0, 8 - len(row)) for row in window) * 0.10
        extra_circle_penalty = sum(max(0, len(row) - 9) for row in window) * 0.05
        return gap_score + missing_circle_penalty + extra_circle_penalty

    windows = [candidate_rows[index : index + 8] for index in range(len(candidate_rows) - 7)]
    selected_rows = min(windows, key=row_window_score)
    observed_row_ys = [median_y(row) for row in selected_rows]
    gaps = [observed_row_ys[index + 1] - observed_row_ys[index] for index in range(len(observed_row_ys) - 1)]
    median_gap = max(1.0, float(np.median(gaps)))
    if any(gap < median_gap * 0.50 or gap > median_gap * 1.65 for gap in gaps):
        return []

    column_observations: list[list[tuple[float, float, float]]] = [[] for _ in range(8)]
    radii: list[int] = []
    for row_index, row in enumerate(selected_rows):
        if not (8 <= len(row) <= 9):
            continue
        row = sorted(row, key=lambda circle: circle[0])
        xs = [circle[0] for circle in row]
        valid_splits = [
            (xs[index + 1] - xs[index], index + 1)
            for index in range(len(xs) - 1)
            if index + 1 >= 4 and len(xs) - (index + 1) >= 4
        ]
        if not valid_splits:
            continue
        split_index = max(valid_splits, key=lambda item: item[0])[1]
        left = sorted(row[:split_index], key=lambda circle: circle[0])[-4:]
        right = sorted(row[split_index:], key=lambda circle: circle[0])[-4:]
        if len(left) != 4 or len(right) != 4:
            continue
        for index, circle in enumerate(left + right):
            column_observations[index].append((float(row_index), float(circle[0]), float(circle[1])))
            radii.append(int(circle[2]))

    if any(len(observations) < 3 for observations in column_observations):
        return []

    column_models: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for observations in column_observations:
        indexes = np.array([item[0] for item in observations])
        xs = np.array([item[1] for item in observations])
        ys = np.array([item[2] for item in observations])
        x_slope, x_intercept = np.polyfit(indexes, xs, 1)
        y_slope, y_intercept = np.polyfit(indexes, ys, 1)
        column_models.append(((float(x_slope), float(x_intercept)), (float(y_slope), float(y_intercept))))

    marked = _marked_pixel_mask(rgb)
    if marked is None:
        return []

    sample_radius = max(3, int(float(np.median(radii)) * 0.52)) if radii else max(3, int(width * 0.008))
    answers_by_id: dict[str, dict[str, Any]] = {}
    for row_index in range(8):
        for question_id, start_column in ((str(row_index + 1), 0), (str(row_index + 9), 4)):
            option_scores: list[float] = []
            for option_index in range(4):
                x_model, y_model = column_models[start_column + option_index]
                x = int(round((x_model[0] * row_index) + x_model[1]))
                center_y = int(round((y_model[0] * row_index) + y_model[1]))
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, center_y), sample_radius, 255, -1)
                pixels = marked[mask > 0]
                option_scores.append(float(pixels.mean()) if pixels.size else 0.0)
            answers_by_id[question_id] = _answer_from_scores(question_id, option_scores)

    if len(answers_by_id) != 16:
        return []
    return [
        answers_by_id[question_id]
        for question_id in sorted(answers_by_id, key=lambda value: int(value) if value.isdigit() else value)
    ]


def _detect_answers_from_rgb(rgb: Any) -> list[dict[str, Any]]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    registered_answers = _detect_answers_from_registered_grid(rgb)
    if len(registered_answers) == 16:
        return registered_answers

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(25, int(width * 0.04)),
        param1=80,
        param2=24,
        minRadius=max(8, int(width * 0.008)),
        maxRadius=max(20, int(width * 0.025)),
    )
    if circles is None:
        return []

    detected = np.round(circles[0]).astype(int).tolist()
    detected = [
        circle
        for circle in detected
        if height * ANSWER_GRID_TOP_FRACTION < circle[1] < height * ANSWER_GRID_BOTTOM_FRACTION
        and width * 0.04 < circle[0] < width * 0.86
    ]
    detected = sorted(detected, key=lambda circle: (circle[1], circle[0]))

    rows: list[list[list[int]]] = []
    for circle in detected:
        if not rows or abs(float(np.median([item[1] for item in rows[-1]])) - circle[1]) > height * ANSWER_ROW_CLUSTER_FRACTION:
            rows.append([circle])
        else:
            rows[-1].append(circle)

    cleaned_rows: list[list[list[int]]] = []
    for row in rows:
        if len(row) > 8:
            median_y = float(np.median([circle[1] for circle in row]))
            row = [circle for circle in row if abs(circle[1] - median_y) <= height * 0.02]
        if len(row) >= 8:
            cleaned_rows.append(row)
    rows = _select_answer_grid_rows(cleaned_rows, np)
    if len(rows) < 6:
        return []

    grouped_rows: list[tuple[list[list[int]], list[list[int]]]] = []
    right_refs: list[list[int]] = []
    for row in rows:
        row = sorted(row, key=lambda circle: circle[0])
        xs = [circle[0] for circle in row]
        valid_splits = [
            (xs[index + 1] - xs[index], index + 1)
            for index in range(len(xs) - 1)
            if index + 1 >= 4 and len(xs) - (index + 1) >= 4
        ]
        if not valid_splits:
            continue
        split_index = max(valid_splits, key=lambda item: item[0])[1]
        left = row[:split_index]
        right = row[split_index:]
        if len(right) == 4:
            right_refs.append([circle[0] for circle in right])
        grouped_rows.append((left, right))

    right_ref = np.median(np.array(right_refs), axis=0) if right_refs else None

    def pick_four(candidates: list[list[int]], side: str) -> list[list[int]]:
        candidates = sorted(candidates, key=lambda circle: circle[0])
        if len(candidates) < 4:
            return []
        if len(candidates) == 4:
            return candidates
        if side == "left":
            return candidates[-4:]
        if right_ref is not None:
            best_group: list[list[int]] = []
            best_score = float("inf")
            for index in range(len(candidates) - 3):
                group = candidates[index : index + 4]
                score = sum(abs(group[position][0] - right_ref[position]) for position in range(4))
                if score < best_score:
                    best_score = score
                    best_group = group
            return best_group
        return candidates[:4]

    marked = _marked_pixel_mask(rgb)
    if marked is None:
        return []

    answers_by_id: dict[str, dict[str, Any]] = {}
    for row_index, (left_candidates, right_candidates) in enumerate(grouped_rows[:8]):
        for question_id, candidates in (
            (str(row_index + 1), pick_four(left_candidates, "left")),
            (str(row_index + 9), pick_four(right_candidates, "right")),
        ):
            if len(candidates) != 4:
                continue
            option_scores: list[float] = []
            for x, y, radius in sorted(candidates, key=lambda circle: circle[0]):
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), max(3, int(radius * 0.52)), 255, -1)
                pixels = marked[mask > 0]
                option_scores.append(float(pixels.mean()) if pixels.size else 0.0)

            selected_options = [
                OPTION_LETTERS[index]
                for index, score in enumerate(option_scores)
                if score > 0.25
            ]
            if not selected_options:
                selected: str | list[str] | None = None
                status = "unmarked"
            elif len(selected_options) == 1:
                selected = selected_options[0]
                status = "answered"
            else:
                selected = selected_options
                status = "multiple"

            answers_by_id[question_id] = {
                "question_id": question_id,
                "options": list(OPTION_LETTERS[:4]),
                "selected": selected,
                "status": status,
                "source": "image",
            }

    return [
        answers_by_id[question_id]
        for question_id in sorted(answers_by_id, key=lambda value: int(value) if value.isdigit() else value)
    ]


def call_lighton_identity_ocr_image(
    image_path: str | Path,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout_seconds: int = 120,
) -> str:
    with tempfile.TemporaryDirectory(prefix="omr_identity_") as temp_dir:
        crop_path = crop_identity_header_for_ocr(image_path, Path(temp_dir) / "identity-header.png")
        return call_lighton_chat_ocr_image(
            crop_path,
            base_url,
            model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_tokens=1024,
        )


def call_lighton_chat_ocr_image(
    image_path: str | Path,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout_seconds: int = 120,
    max_tokens: int = 4096,
) -> str:
    path = Path(image_path)
    with tempfile.TemporaryDirectory(prefix="omr_preprocess_") as temp_dir:
        ocr_image_path = preprocess_image_for_ocr(path, Path(temp_dir) / "ocr-input.png")
        mime_type = mimetypes.guess_type(ocr_image_path.name)[0] or "image/png"
        image_data = base64.b64encode(ocr_image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                    }
                ],
            }
        ],
    }
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        error_text = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OCR endpoint returned HTTP {error.code}: {error_text}") from error

    decoded = json.loads(response_text)
    text = find_chat_message_content(decoded)
    if not text:
        raise RuntimeError("OCR endpoint response did not contain message content")
    return text


def render_pdf_pages(pdf_path: str | Path, output_dir: str | Path, dpi: int = 300) -> list[Path]:
    prefix = Path(output_dir) / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return sorted(Path(output_dir).glob("page-*.png"), key=lambda path: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)])


def call_lighton_chat_ocr_file(
    file_path: str | Path,
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout_seconds: int = 120,
    pdf_dpi: int = 300,
) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return coerce_ocr_text(path.read_text(encoding="utf-8"))
    if suffix == ".json":
        return coerce_ocr_text(path.read_text(encoding="utf-8"))
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return call_lighton_chat_ocr_image(path, base_url, model, api_key=api_key, timeout_seconds=timeout_seconds)
    if suffix == ".pdf":
        with tempfile.TemporaryDirectory(prefix="omr_pdf_") as temp_dir:
            pages = render_pdf_pages(path, temp_dir, dpi=pdf_dpi)
            if not pages:
                raise RuntimeError(f"No pages rendered from {path}")
            page_text = [
                f"<!-- ===== {page.name} ===== -->\n"
                + call_lighton_chat_ocr_image(page, base_url, model, api_key=api_key, timeout_seconds=timeout_seconds)
                for page in pages
            ]
            return "\n\n".join(page_text)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def call_lighton_ocr(
    image_path: str | Path,
    endpoint_url: str,
    api_key: str | None = None,
    field_name: str = "file",
    timeout_seconds: int = 60,
) -> str:
    path = Path(image_path)
    body, content_type = build_multipart_body(field_name, path)
    headers = {
        "Content-Type": content_type,
        "Accept": "application/json, text/plain",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(endpoint_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read()
            response_text = response_body.decode("utf-8")
    except urllib.error.HTTPError as error:
        error_text = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OCR endpoint returned HTTP {error.code}: {error_text}") from error

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return response_text

    text = find_text_value(payload)
    if not text:
        raise RuntimeError("OCR endpoint response did not contain a text field")
    return text


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    if args.ocr_text_file:
        ocr_text = Path(args.ocr_text_file).read_text(encoding="utf-8")
    elif args.image:
        endpoint = args.ocr_endpoint or os.environ.get("LIGHTON_OCR_ENDPOINT")
        if not endpoint:
            raise ValueError("--ocr-endpoint or LIGHTON_OCR_ENDPOINT is required with --image")
        api_key = args.api_key or os.environ.get("LIGHTON_OCR_API_KEY")
        ocr_text = call_lighton_ocr(
            args.image,
            endpoint,
            api_key=api_key,
            field_name=args.image_field,
            timeout_seconds=args.timeout,
        )
    else:
        ocr_text = sys.stdin.read()

    parsed = parse_submission_text(ocr_text)
    result: dict[str, Any] = {
        "source": {
            "ocr_text_file": args.ocr_text_file,
            "image": args.image,
        },
        **parsed,
    }

    if args.answer_key:
        answer_key_payload = load_json_file(args.answer_key)
        answer_key = normalize_answer_map(answer_key_payload, "answers")
        if args.weights:
            weights = normalize_weight_map(load_json_file(args.weights))
        elif isinstance(answer_key_payload, dict) and "weights" in answer_key_payload:
            weights = normalize_weight_map(answer_key_payload["weights"])
        else:
            weights = {}
        result["evaluation"] = evaluate_answers(
            parsed,
            answer_key=answer_key,
            weights=weights,
            default_weight=args.default_weight,
        )

    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LightOn OCR OMR text into standard JSON.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--ocr-text-file", help="Path to a text file containing OCR output.")
    input_group.add_argument("--image", help="Path to an OMR image to send to the OCR endpoint.")
    parser.add_argument("--ocr-endpoint", help="LightOn OCR endpoint URL. Can also use LIGHTON_OCR_ENDPOINT.")
    parser.add_argument("--api-key", help="OCR API key. Can also use LIGHTON_OCR_API_KEY.")
    parser.add_argument("--image-field", default="file", help="Multipart form field name for the image.")
    parser.add_argument("--timeout", type=int, default=60, help="OCR request timeout in seconds.")
    parser.add_argument("--answer-key", help="JSON file containing correct answers.")
    parser.add_argument("--weights", help="JSON file containing weightage per question.")
    parser.add_argument("--default-weight", type=float, default=1.0, help="Weight used if a question is missing.")
    parser.add_argument("--output", help="Write JSON output to this file instead of stdout.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        result = build_result(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
