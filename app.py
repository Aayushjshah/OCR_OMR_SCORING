#!/usr/bin/env python3
"""FastAPI service for OMR answer-key import and candidate scoring."""

from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pypdf import PdfReader

from omr_pipeline import (
    SUPPORTED_DOCUMENT_SUFFIXES,
    SUPPORTED_IMAGE_SUFFIXES,
    call_lighton_chat_ocr_image,
    call_lighton_chat_ocr_file,
    call_lighton_identity_ocr_image,
    candidate_set_name,
    detect_answers_from_image,
    detect_marked_set_from_image,
    load_answer_key_for_set,
    parse_submission_text,
    render_pdf_pages,
    save_answer_key_json,
    score_submission,
)


BASE_DIR = Path(__file__).resolve().parent
KEY_DIR = Path(os.environ.get("OMR_KEY_DIR", BASE_DIR / "answer_keys"))
OUTPUT_DIR = Path(os.environ.get("OMR_OUTPUT_DIR", BASE_DIR / "outputs"))
UPLOAD_DIR = Path(os.environ.get("OMR_UPLOAD_DIR", BASE_DIR / "uploads"))

OCR_BASE_URL = os.environ.get("LIGHTON_OCR_BASE_URL", "http://100.111.195.29:8000")
OCR_MODEL = os.environ.get("LIGHTON_OCR_MODEL", "lightonai/LightOnOCR-2-1B")
OCR_API_KEY = os.environ.get("LIGHTON_OCR_API_KEY")
OCR_TIMEOUT = int(os.environ.get("OMR_OCR_TIMEOUT", "120"))
PDF_DPI = int(os.environ.get("OMR_PDF_DPI", "300"))
MAX_COMBINED_PDF_BYTES = int(os.environ.get("OMR_MAX_COMBINED_PDF_MB", "40")) * 1024 * 1024
MAX_COMBINED_PDF_PAGES = int(os.environ.get("OMR_MAX_COMBINED_PDF_PAGES", "250"))
MAX_FOLDER_UPLOAD_BYTES = int(os.environ.get("OMR_MAX_FOLDER_UPLOAD_MB", "40")) * 1024 * 1024
MAX_FOLDER_UPLOAD_FILES = int(os.environ.get("OMR_MAX_FOLDER_UPLOAD_FILES", "250"))
ESTIMATED_SECONDS_PER_PAGE = int(os.environ.get("OMR_ESTIMATED_SECONDS_PER_PAGE", "15"))

app = FastAPI(title="OMR Evaluation Service")


def ensure_dirs() -> None:
    for directory in (KEY_DIR, OUTPUT_DIR, UPLOAD_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    basename = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", basename).strip("_")
    return cleaned or f"upload_{int(time.time())}"


def row_from_result(source: str, result: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    return {
        "source": source,
        "name": result.get("name") if result else "",
        "email": result.get("email") if result else "",
        "roll_no": result.get("roll_no") if result else "",
        "set": result.get("set") if result else "",
        "answered_questions": result.get("answered_questions") if result else "",
        "total_questions": result.get("total_questions") if result else "",
        "unanswered_questions": result.get("unanswered_questions") if result else "",
        "score": result.get("score") if result else "",
        "max_score": result.get("max_score") if result else "",
        "warnings": "; ".join(result.get("identity_warnings", [])) if result else "",
        "error": error or "",
    }


async def save_upload(upload: UploadFile, subdir: str) -> Path:
    ensure_dirs()
    target_dir = UPLOAD_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_filename(upload.filename or "upload")
    with target.open("wb") as file:
        shutil.copyfileobj(upload.file, file)
    return target


def pdf_page_count(path: Path) -> int:
    reader = PdfReader(str(path))
    return len(reader.pages)


def format_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds} sec"
    minutes, remainder = divmod(seconds, 60)
    if remainder == 0:
        return f"{minutes} min"
    return f"{minutes} min {remainder} sec"


def estimated_combined_pdf_seconds(page_count: int) -> int:
    return page_count * ESTIMATED_SECONDS_PER_PAGE


def validate_folder_upload_limits(file_count: int, total_bytes: int) -> None:
    if file_count > MAX_FOLDER_UPLOAD_FILES:
        raise ValueError(f"Folder upload has {file_count} files; maximum allowed is {MAX_FOLDER_UPLOAD_FILES}")
    if total_bytes > MAX_FOLDER_UPLOAD_BYTES:
        limit_mb = MAX_FOLDER_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(f"Folder upload exceeds {limit_mb} MB limit")


def upload_file_size(upload: UploadFile) -> int:
    size = getattr(upload, "size", None)
    if isinstance(size, int):
        return size

    current = upload.file.tell()
    upload.file.seek(0, 2)
    total = upload.file.tell()
    upload.file.seek(current)
    return total


def validate_upload_batch(files: list[UploadFile]) -> int:
    total_bytes = sum(upload_file_size(upload) for upload in files)
    validate_folder_upload_limits(len(files), total_bytes)
    return total_bytes


def validate_combined_pdf(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        raise ValueError("Combined upload must be a PDF")
    if path.stat().st_size > MAX_COMBINED_PDF_BYTES:
        limit_mb = MAX_COMBINED_PDF_BYTES // (1024 * 1024)
        raise ValueError(f"Combined PDF exceeds {limit_mb} MB limit")
    pages = pdf_page_count(path)
    if pages > MAX_COMBINED_PDF_PAGES:
        raise ValueError(f"Combined PDF has {pages} pages; maximum allowed is {MAX_COMBINED_PDF_PAGES}")
    return pages


def merge_identity_candidate(parsed: dict[str, Any], identity_ocr_text: str | None = None, image_path: Path | None = None) -> None:
    candidate = parsed.setdefault("candidate", {})
    if identity_ocr_text:
        identity = parse_submission_text(identity_ocr_text)
        identity_candidate = identity.get("candidate", {})
        for field in ("name", "email", "roll_no"):
            value = identity_candidate.get(field)
            if value not in (None, ""):
                candidate[field] = value

    if image_path is not None and image_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
        detected_set = detect_marked_set_from_image(image_path)
        if detected_set:
            candidate["exam_set"] = detected_set
        detected_answers = detect_answers_from_image(image_path)
        if len(detected_answers) >= 12:
            parsed["answers"] = detected_answers


def score_ocr_text(
    ocr_text: str,
    identity_ocr_text: str | None = None,
    image_path: Path | None = None,
) -> dict[str, Any]:
    parsed = parse_submission_text(ocr_text)
    merge_identity_candidate(parsed, identity_ocr_text=identity_ocr_text, image_path=image_path)
    set_name = candidate_set_name(parsed)
    if not set_name:
        raise ValueError("Set is missing from OCR output")
    answer_key = load_answer_key_for_set(set_name, KEY_DIR)
    return score_submission(parsed, answer_key)


def score_path(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    identity_ocr_text = None
    identity_image_path: Path | None = path if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES else None

    if path.suffix.lower() == ".pdf":
        with tempfile.TemporaryDirectory(prefix="omr_pdf_score_") as temp_dir:
            pages = render_pdf_pages(path, temp_dir, dpi=PDF_DPI)
            if not pages:
                raise RuntimeError(f"No pages rendered from {path}")
            identity_image_path = pages[0]
            page_text = [
                f"<!-- ===== {page.name} ===== -->\n"
                + call_lighton_chat_ocr_image(
                    page,
                    OCR_BASE_URL,
                    OCR_MODEL,
                    api_key=OCR_API_KEY,
                    timeout_seconds=OCR_TIMEOUT,
                )
                for page in pages
            ]
            ocr_text = "\n\n".join(page_text)
            identity_ocr_text = call_lighton_identity_ocr_image(
                identity_image_path,
                OCR_BASE_URL,
                OCR_MODEL,
                api_key=OCR_API_KEY,
                timeout_seconds=OCR_TIMEOUT,
            )
            result = score_ocr_text(ocr_text, identity_ocr_text=identity_ocr_text, image_path=identity_image_path)
    else:
        ocr_text = call_lighton_chat_ocr_file(
            path,
            OCR_BASE_URL,
            OCR_MODEL,
            api_key=OCR_API_KEY,
            timeout_seconds=OCR_TIMEOUT,
            pdf_dpi=PDF_DPI,
        )

        if identity_image_path is not None:
            identity_ocr_text = call_lighton_identity_ocr_image(
                identity_image_path,
                OCR_BASE_URL,
                OCR_MODEL,
                api_key=OCR_API_KEY,
                timeout_seconds=OCR_TIMEOUT,
            )
        result = score_ocr_text(ocr_text, identity_ocr_text=identity_ocr_text, image_path=identity_image_path)

    stamp = int(time.time() * 1000)
    base = safe_filename(path.stem)
    saved_ocr_text = ocr_text
    if identity_ocr_text:
        saved_ocr_text = f"<!-- ===== identity-header ===== -->\n{identity_ocr_text}\n\n<!-- ===== full-page ===== -->\n{ocr_text}"
    (OUTPUT_DIR / f"{base}_{stamp}.ocr.txt").write_text(saved_ocr_text + "\n", encoding="utf-8")
    (OUTPUT_DIR / f"{base}_{stamp}.result.json").write_text(
        json.dumps({"source": str(path), **result}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def write_score_csv(rows: list[dict[str, Any]], prefix: str = "scores") -> Path:
    ensure_dirs()
    output_path = OUTPUT_DIR / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source",
                "name",
                "email",
                "roll_no",
                "set",
                "answered_questions",
                "total_questions",
                "unanswered_questions",
                "score",
                "max_score",
                "warnings",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    ensure_dirs()
    key_files = sorted(path.name for path in KEY_DIR.glob("*.json"))
    key_list = "".join(f"<li>{html.escape(name)}</li>" for name in key_files) or "<li>No answer keys imported yet</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OMR Evaluation</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18202a;
      --muted: #5f6975;
      --line: #d7dce2;
      --panel: #f7f8fa;
      --accent: #176b5b;
      --accent-strong: #0f4f43;
      --danger: #9b2c2c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 18px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    header h1 {{ font-size: 20px; margin: 0; font-weight: 700; letter-spacing: 0; }}
    header span {{ color: var(--muted); font-size: 13px; }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 28px;
      align-items: start;
    }}
    section {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 18px;
    }}
    section h2 {{ margin: 0 0 14px; font-size: 16px; letter-spacing: 0; }}
    label {{ display: block; margin: 12px 0 6px; font-size: 13px; color: var(--muted); }}
    .hint {{ margin: 6px 0 0; color: var(--muted); font-size: 12px; line-height: 1.4; }}
    input[type="text"], input[type="file"] {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 10px;
      font: inherit;
    }}
    button {{
      margin-top: 14px;
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-strong); }}
    .stack {{ display: grid; gap: 18px; }}
    .result {{
      min-height: 220px;
      background: #101820;
      color: #eef6f4;
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
    }}
    .section-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 14px;
    }}
    .section-head h2 {{ margin: 0; }}
    .copy-button {{
      margin: 0;
      padding: 7px 10px;
      background: #2f5f7c;
      font-size: 13px;
    }}
    .copy-button:hover {{ background: #244b63; }}
    ul {{ margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .error {{ color: var(--danger); }}
    @media (max-width: 860px) {{
      main, .two {{ grid-template-columns: 1fr; }}
      header {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>OMR Evaluation</h1>
    <span>OCR endpoint: {html.escape(OCR_BASE_URL)} · model: {html.escape(OCR_MODEL)}</span>
  </header>
  <main>
    <aside class="stack">
      <section>
        <h2>Import Answer Key</h2>
        <form id="key-form">
          <label for="set-name">Set name</label>
          <input id="set-name" name="set_name" type="text" placeholder="set1" required>
          <label for="answer-csv">CSV</label>
          <input id="answer-csv" name="file" type="file" accept=".csv" required>
          <button type="submit">Import CSV</button>
        </form>
      </section>
      <section>
        <h2>Available Sets</h2>
        <ul>{key_list}</ul>
      </section>
    </aside>
    <div class="stack">
      <div class="two">
        <section>
          <h2>Score One File</h2>
          <form id="file-form">
            <label for="sheet-file">Image, PDF, or OCR text</label>
            <input id="sheet-file" name="file" type="file" accept=".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,.pdf,.txt,.json" required>
            <button type="submit">Score File</button>
          </form>
        </section>
        <section>
          <h2>Combined PDF</h2>
          <form id="combined-pdf-form">
            <label for="combined-pdf-file">One PDF, one OMR sheet per page</label>
            <input id="combined-pdf-file" name="file" type="file" accept=".pdf,application/pdf" required>
            <p class="hint">Maximum {MAX_COMBINED_PDF_PAGES} pages and {MAX_COMBINED_PDF_BYTES // (1024 * 1024)} MB. Approx {ESTIMATED_SECONDS_PER_PAGE} seconds per page.</p>
            <button type="submit">Process Combined PDF</button>
          </form>
        </section>
        <section>
          <h2>Score Folder</h2>
          <form id="folder-form">
            <label for="folder-files">Browser folder upload</label>
            <input id="folder-files" name="files" type="file" webkitdirectory directory multiple required>
            <p class="hint">Maximum {MAX_FOLDER_UPLOAD_FILES} files and {MAX_FOLDER_UPLOAD_BYTES // (1024 * 1024)} MB total.</p>
            <button type="submit">Process Folder</button>
          </form>
          <form action="/api/score-folder-path" method="post">
            <label for="folder-path">Server folder path</label>
            <input id="folder-path" name="folder_path" type="text" placeholder="/path/to/folder">
            <p class="hint">Use a folder path on this machine, for example /Users/aayush.shah/Downloads/omr_submissions.</p>
            <button type="submit">Download CSV</button>
          </form>
        </section>
      </div>
      <section>
        <div class="section-head">
          <h2>Result</h2>
          <button id="copy-result" class="copy-button" type="button">Copy</button>
        </div>
        <pre id="result" class="result">Ready.</pre>
      </section>
    </div>
  </main>
  <script>
    const result = document.getElementById('result');
    const copyResult = document.getElementById('copy-result');
    const maxCombinedPdfBytes = {MAX_COMBINED_PDF_BYTES};
    const maxFolderUploadBytes = {MAX_FOLDER_UPLOAD_BYTES};
    const maxFolderUploadFiles = {MAX_FOLDER_UPLOAD_FILES};
    async function copyText(text) {{
      if (navigator.clipboard && window.isSecureContext) {{
        await navigator.clipboard.writeText(text);
        return;
      }}
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.left = '-9999px';
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      document.execCommand('copy');
      textarea.remove();
    }}
    copyResult.addEventListener('click', async () => {{
      await copyText(result.textContent || '');
      const previous = copyResult.textContent;
      copyResult.textContent = 'Copied';
      window.setTimeout(() => {{ copyResult.textContent = previous; }}, 1200);
    }});
    async function postForm(form, url) {{
      result.textContent = 'Working...';
      const response = await fetch(url, {{ method: 'POST', body: new FormData(form) }});
      const text = await response.text();
      let payload;
      try {{ payload = JSON.parse(text); }} catch {{ payload = text; }}
      if (!response.ok) {{
        result.textContent = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2);
        result.classList.add('error');
        return;
      }}
      result.classList.remove('error');
      result.textContent = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2);
    }}
    async function postFolderForm(form) {{
      const data = new FormData(form);
      const files = data.getAll('files');
      const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
      result.classList.remove('error');
      if (files.length > maxFolderUploadFiles) {{
        result.classList.add('error');
        result.textContent = `Folder upload has ${{files.length}} files; maximum allowed is ${{maxFolderUploadFiles}}.`;
        return;
      }}
      if (totalBytes > maxFolderUploadBytes) {{
        result.classList.add('error');
        result.textContent = `Folder upload exceeds ${{Math.round(maxFolderUploadBytes / 1024 / 1024)}} MB limit.`;
        return;
      }}
      result.textContent = `Preparing upload for ${{files.length}} file(s)...`;
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/score-upload-batch-json');
      xhr.upload.addEventListener('progress', event => {{
        if (event.lengthComputable) {{
          const percent = Math.round((event.loaded / event.total) * 100);
          result.textContent = `Uploading folder... ${{percent}}%\\n${{files.length}} file(s) queued for scoring.`;
        }} else {{
          result.textContent = `Uploading folder...\\n${{files.length}} file(s) queued for scoring.`;
        }}
      }});
      xhr.upload.addEventListener('load', () => {{
        result.textContent = `Upload complete. Processing ${{files.length}} file(s) on server...`;
      }});
      xhr.onload = () => {{
        let payload;
        try {{ payload = JSON.parse(xhr.responseText); }} catch {{ payload = xhr.responseText; }}
        if (xhr.status < 200 || xhr.status >= 300) {{
          result.classList.add('error');
          result.textContent = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2);
          return;
        }}
        result.classList.remove('error');
        const summary = [
          `Done. Processed ${{payload.processed_files}} / ${{payload.total_files}} file(s).`,
          `Successful: ${{payload.successful_files}}`,
          `Failed: ${{payload.failed_files}}`,
          '',
          `CSV: ${{window.location.origin + payload.download_url}}`,
          '',
          JSON.stringify(payload.rows, null, 2)
        ].join('\\n');
        result.textContent = summary;
      }};
      xhr.onerror = () => {{
        result.classList.add('error');
        result.textContent = 'Upload failed before the server could process the folder.';
      }};
      result.textContent = `Uploading folder...\\n${{files.length}} file(s) queued for scoring.`;
      xhr.send(data);
    }}
    async function postCombinedPdfForm(form) {{
      const file = form.querySelector('input[type="file"]').files[0];
      if (!file) {{
        result.classList.add('error');
        result.textContent = 'Select a combined PDF first.';
        return;
      }}
      if (!file.name.toLowerCase().endsWith('.pdf')) {{
        result.classList.add('error');
        result.textContent = 'Combined upload must be a PDF.';
        return;
      }}
      if (file.size > maxCombinedPdfBytes) {{
        result.classList.add('error');
        result.textContent = `Combined PDF exceeds ${{Math.round(maxCombinedPdfBytes / 1024 / 1024)}} MB limit.`;
        return;
      }}

      const data = new FormData(form);
      result.classList.remove('error');
      result.textContent = `Preparing upload for ${{file.name}}...`;
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/score-combined-pdf');
      xhr.upload.addEventListener('progress', event => {{
        if (event.lengthComputable) {{
          const percent = Math.round((event.loaded / event.total) * 100);
          result.textContent = `Uploading combined PDF... ${{percent}}%`;
        }} else {{
          result.textContent = 'Uploading combined PDF...';
        }}
      }});
      xhr.upload.addEventListener('load', () => {{
        result.textContent = 'Upload complete. Rendering pages and processing OMR sheets...';
      }});
      xhr.onload = () => {{
        let payload;
        try {{ payload = JSON.parse(xhr.responseText); }} catch {{ payload = xhr.responseText; }}
        if (xhr.status < 200 || xhr.status >= 300) {{
          result.classList.add('error');
          result.textContent = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2);
          return;
        }}
        result.classList.remove('error');
        const summary = [
          `Done. Processed ${{payload.processed_pages}} / ${{payload.total_pages}} page(s).`,
          `Approx time: ${{payload.estimated_processing_time}} (${{payload.estimated_processing_seconds}} sec)`,
          `Successful: ${{payload.successful_pages}}`,
          `Failed: ${{payload.failed_pages}}`,
          '',
          `CSV: ${{window.location.origin + payload.download_url}}`,
          '',
          JSON.stringify(payload.rows, null, 2)
        ].join('\\n');
        result.textContent = summary;
      }};
      xhr.onerror = () => {{
        result.classList.add('error');
        result.textContent = 'Upload failed before the server could process the combined PDF.';
      }};
      xhr.send(data);
    }}
    document.getElementById('key-form').addEventListener('submit', event => {{
      event.preventDefault();
      postForm(event.currentTarget, '/api/answer-keys');
    }});
    document.getElementById('file-form').addEventListener('submit', event => {{
      event.preventDefault();
      postForm(event.currentTarget, '/api/score-file');
    }});
    document.getElementById('folder-form').addEventListener('submit', event => {{
      event.preventDefault();
      postFolderForm(event.currentTarget);
    }});
    document.getElementById('combined-pdf-form').addEventListener('submit', event => {{
      event.preventDefault();
      postCombinedPdfForm(event.currentTarget);
    }});
  </script>
</body>
</html>"""


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/api/answer-keys")
async def import_answer_key(set_name: str = Form(...), file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV answer key")
    saved = await save_upload(file, "answer_keys")
    try:
        output_path = save_answer_key_json(saved, set_name, KEY_DIR)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return JSONResponse({"answer_key_path": str(output_path), **payload})


@app.post("/api/score-file")
async def score_file(file: UploadFile = File(...)) -> JSONResponse:
    saved = await save_upload(file, "submissions")
    try:
        result = score_path(saved)
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return JSONResponse({"source": str(saved), **result})


@app.post("/api/score-upload-batch")
async def score_upload_batch(files: list[UploadFile] = File(...)) -> FileResponse:
    try:
        output_path, _ = await process_upload_batch(files)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return FileResponse(output_path, media_type="text/csv", filename=output_path.name)


async def process_upload_batch(files: list[UploadFile]) -> tuple[Path, list[dict[str, Any]]]:
    validate_upload_batch(files)
    rows: list[dict[str, Any]] = []
    batch = f"batch_{int(time.time())}"
    for upload in files:
        saved = await save_upload(upload, batch)
        try:
            result = score_path(saved)
            rows.append(row_from_result(upload.filename or saved.name, result))
        except Exception as error:
            rows.append(row_from_result(upload.filename or saved.name, {}, str(error)))
    output_path = write_score_csv(rows, prefix="upload_scores")
    return output_path, rows


def process_combined_pdf(path: Path, original_name: str) -> tuple[Path, list[dict[str, Any]], int]:
    page_count = validate_combined_pdf(path)
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="omr_combined_pdf_") as temp_dir:
        pages = render_pdf_pages(path, temp_dir, dpi=PDF_DPI)
        if len(pages) != page_count:
            page_count = len(pages)
        for index, page in enumerate(pages, start=1):
            source = f"{original_name} page {index}"
            try:
                ocr_text = call_lighton_chat_ocr_image(
                    page,
                    OCR_BASE_URL,
                    OCR_MODEL,
                    api_key=OCR_API_KEY,
                    timeout_seconds=OCR_TIMEOUT,
                )
                identity_ocr_text = call_lighton_identity_ocr_image(
                    page,
                    OCR_BASE_URL,
                    OCR_MODEL,
                    api_key=OCR_API_KEY,
                    timeout_seconds=OCR_TIMEOUT,
                )
                result = score_ocr_text(ocr_text, identity_ocr_text=identity_ocr_text, image_path=page)
                rows.append(row_from_result(source, result))
            except Exception as error:
                rows.append(row_from_result(source, {}, str(error)))

    output_path = write_score_csv(rows, prefix="combined_pdf_scores")
    return output_path, rows, page_count


@app.post("/api/score-upload-batch-json")
async def score_upload_batch_json(files: list[UploadFile] = File(...)) -> JSONResponse:
    try:
        output_path, rows = await process_upload_batch(files)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    failed_files = sum(1 for row in rows if row.get("error"))
    return JSONResponse(
        {
            "total_files": len(files),
            "processed_files": len(rows),
            "successful_files": len(rows) - failed_files,
            "failed_files": failed_files,
            "csv_path": str(output_path),
            "download_url": f"/api/download/{output_path.name}",
            "rows": rows,
        }
    )


@app.post("/api/score-combined-pdf")
async def score_combined_pdf(file: UploadFile = File(...)) -> JSONResponse:
    saved = await save_upload(file, "combined_pdfs")
    try:
        output_path, rows, page_count = process_combined_pdf(saved, file.filename or saved.name)
    except Exception as error:
        saved.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error)) from error

    failed_pages = sum(1 for row in rows if row.get("error"))
    estimated_seconds = estimated_combined_pdf_seconds(page_count)
    return JSONResponse(
        {
            "total_pages": page_count,
            "processed_pages": len(rows),
            "estimated_processing_seconds": estimated_seconds,
            "estimated_processing_time": format_duration(estimated_seconds),
            "successful_pages": len(rows) - failed_pages,
            "failed_pages": failed_pages,
            "csv_path": str(output_path),
            "download_url": f"/api/download/{output_path.name}",
            "rows": rows,
        }
    )


@app.get("/api/download/{filename}")
def download_output(filename: str) -> FileResponse:
    safe_name = safe_filename(filename)
    path = OUTPUT_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.post("/api/score-folder-path")
def score_folder_path(folder_path: str = Form(...)) -> FileResponse:
    folder = Path(folder_path).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder does not exist: {folder}")

    rows: list[dict[str, Any]] = []
    candidates = [
        path
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
    ]
    if not candidates:
        raise HTTPException(status_code=400, detail=f"No supported files found in {folder}")

    for path in candidates:
        try:
            result = score_path(path)
            rows.append(row_from_result(str(path), result))
        except Exception as error:
            rows.append(row_from_result(str(path), {}, str(error)))
    output_path = write_score_csv(rows, prefix="folder_scores")
    return FileResponse(output_path, media_type="text/csv", filename=output_path.name)
