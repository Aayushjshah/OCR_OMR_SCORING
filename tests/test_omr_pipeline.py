import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from app import (
    MAX_FOLDER_UPLOAD_BYTES,
    MAX_FOLDER_UPLOAD_FILES,
    create_batch_job,
    estimated_combined_pdf_seconds,
    format_duration,
    get_batch_job,
    normalize_manual_set,
    process_combined_pdf_job,
    process_saved_upload_batch_job,
    row_from_result,
    score_ocr_text,
    validate_combined_pdf,
    validate_folder_upload_limits,
)
from omr_pipeline import (
    OPTION_LETTERS,
    answer_key_from_csv,
    detect_answers_from_image,
    detect_marked_set_from_image,
    evaluate_answers,
    normalize_weight_map,
    parse_omr_text,
    parse_submission_text,
    preprocess_image_for_ocr,
    save_answer_key_json,
    score_submission,
)


SAMPLE_OCR_TEXT = """NAME: Lalitha

EXAM:

DATE: 18/06/25

Exam Set

A B C D
○ ● ○ ○

Roll No

3 P G 0 3 T T 0 0 9

0 ● ○ ○ ○ ○ ○ ○ ○ ○
1 ○ ○ ○ ● ○ ○ ○ ○ ○
2 ○ ○ ○ ○ ○ ● ○ ○ ○
3 ○ ○ ● ○ ○ ○ ○ ○ ○
4 ○ ○ ○ ● ○ ○ ○ ○ ○
5 ○ ○ ○ ○ ● ○ ○ ○ ○
6 ○ ○ ○ ○ ● ○ ○ ○ ○
7 ○ ○ ○ ○ ○ ● ○ ○ ○
8 ○ ○ ● ○ ○ ○ ○ ○ ○
9 ○ ○ ○ ○ ● ○ ○ ○ ○

maths

Section1

A B C D
1 ○ ○ ● ○

Section2

A B C D
2 ○ ● ○ ○

phy

Section1

A B C D
3 ○ ● ○ ○

Section2

A B C D
4 ○ ○ ○ ○

system

Section1

A B C D
5 ○ ○ ○ ●
6 ○ ● ○ ○
7 ● ○ ○ ○
8 ○ ○ ● ○
9 ○ ○ ● ○

A B C D
10 ○ ● ○ ○
11 ○ ○ ● ○

CS

Section1

A B C D
12 ○ ● ○ ○
13 ○ ○ ● ○
14 ○ ● ○ ○
15 ○ ○ ● ○
16 ○ ● ○ ○
"""


class OmrPipelineTest(unittest.TestCase):
    def test_parse_sample_ocr_text(self):
        parsed = parse_omr_text(SAMPLE_OCR_TEXT)

        self.assertEqual(parsed["candidate"]["name"], "Lalitha")
        self.assertEqual(parsed["candidate"]["date"], "18/06/25")
        self.assertEqual(parsed["candidate"]["exam_set"]["selected"], "set2")
        self.assertEqual(parsed["candidate"]["roll_no"]["raw"], "3 P G 0 3 T T 0 0 9")

        answers = {answer["question_id"]: answer for answer in parsed["answers"]}
        self.assertEqual(len(answers), 16)
        self.assertEqual(answers["1"]["selected"], "C")
        self.assertEqual(answers["2"]["selected"], "B")
        self.assertEqual(answers["4"]["selected"], None)
        self.assertEqual(answers["4"]["status"], "unmarked")
        self.assertEqual(answers["16"]["selected"], "B")

    def test_evaluate_answers_with_weights(self):
        parsed = parse_omr_text(SAMPLE_OCR_TEXT)
        answer_key = {"1": "C", "2": "A", "4": "D"}
        weights = normalize_weight_map({"weights": {"1": 2, "2": 3, "4": 5}})

        evaluation = evaluate_answers(parsed, answer_key, weights)

        self.assertEqual(evaluation["score"], 2)
        self.assertEqual(evaluation["max_score"], 10)
        self.assertEqual(evaluation["details"][0]["is_correct"], True)
        self.assertEqual(evaluation["details"][1]["is_correct"], False)
        self.assertEqual(evaluation["details"][2]["status"], "unmarked")

    def test_weight_list_format(self):
        weights = normalize_weight_map([{"question_id": "1", "weight": 2}])

        self.assertEqual(weights, {"1": 2.0})

    def test_answer_key_from_csv_uses_choice_label_and_marks(self):
        payload = answer_key_from_csv("answers/set1.csv", "set1")

        self.assertEqual(payload["set"], "set1")
        self.assertEqual(payload["answers"]["1"], "D")
        self.assertEqual(payload["weights"]["1"], 3.0)

    def test_parse_simple_submission_text_and_score(self):
        parsed = parse_submission_text(
            """Name: Ada Lovelace
Email: ada@example.com
Roll No.: 23EE0446
Set: set1
1: D
2: B
3: A
"""
        )
        key = {
            "answers": {"1": "D", "2": "A", "3": "A"},
            "weights": {"1": 3, "2": 3, "3": 2},
        }

        result = score_submission(parsed, key)

        self.assertEqual(result["name"], "Ada Lovelace")
        self.assertEqual(result["email"], "ada@example.com")
        self.assertEqual(result["roll_no"], "23EE0446")
        self.assertEqual(result["set"], "set1")
        self.assertEqual(result["answered_questions"], 3)
        self.assertEqual(result["total_questions"], 3)
        self.assertEqual(result["unanswered_questions"], 0)
        self.assertEqual(result["score"], 5.0)
        self.assertEqual(result["max_score"], 8.0)

    def test_score_ocr_text_without_override_uses_detected_set(self):
        result = score_ocr_text(
            """Name: Manual Candidate
Email: manual@example.com
Roll No.: 23EE0446
Set: set1
1: A
"""
        )

        self.assertEqual(result["set"], "set1")
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["evaluation"]["details"][0]["is_correct"])

    def test_score_ocr_text_override_uses_selected_set(self):
        result = score_ocr_text(
            """Name: Manual Candidate
Email: manual@example.com
Roll No.: 23EE0446
Set: set1
1: A
""",
            set_override=normalize_manual_set("2"),
        )

        self.assertEqual(result["set"], "set2")
        self.assertEqual(result["score"], 3.0)
        self.assertTrue(result["evaluation"]["details"][0]["is_correct"])

    def test_normalize_manual_set_rejects_invalid_values(self):
        self.assertEqual(normalize_manual_set("1"), "set1")
        self.assertEqual(normalize_manual_set(" 4 "), "set4")

        for value in ("", "0", "5", "set5"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_manual_set(value)

    def test_parse_final_layout_marked_set_and_roll_no(self):
        parsed = parse_submission_text(
            """OMR ANSWER SHEET
Name: Saravana Prabhu T
Email ID: saravanaprabhu.t.eee2023@citchennai.net
Set No.: 1 ○ 2 ● 3 ○ 4 ○
Roll No.: 23EE0446

1 A ○ B ○ C ● D ○
2 A ○ B ● C ○ D ○
"""
        )
        key = {
            "answers": {"1": "C", "2": "A"},
            "weights": {"1": 1, "2": 1},
        }

        result = score_submission(parsed, key)
        row = row_from_result("filled.jpg", result)

        self.assertEqual(parsed["candidate"]["exam_set"], "set2")
        self.assertEqual(parsed["candidate"]["roll_no"], "23EE0446")
        self.assertEqual(result["set"], "set2")
        self.assertEqual(result["roll_no"], "23EE0446")
        self.assertEqual(row["roll_no"], "23EE0446")
        self.assertEqual(result["score"], 1.0)

    def test_parse_latex_wrapped_identity_values(self):
        parsed = parse_submission_text(
            """Name: $\\text{NITHIN S}$
Email ID: $\\text{ns2470@symist.edu.in}$
Set No.: 4
Roll No.: $\\TEXT{RA231104701014}$
"""
        )

        self.assertEqual(parsed["candidate"]["name"], "NITHIN S")
        self.assertEqual(parsed["candidate"]["email"], "ns2470@symist.edu.in")
        self.assertEqual(parsed["candidate"]["exam_set"], "set4")
        self.assertEqual(parsed["candidate"]["roll_no"], "RA231104701014")

    def test_parse_identity_from_cropped_ocr_table(self):
        parsed = parse_submission_text(
            """<table>
  <tr><td>Name:</td><td>SARAVANA PRABHU . I</td><td>Set No.:</td><td>○ 1 ● ○ 2 ○ 4</td></tr>
  <tr><td>Email ID:</td><td>SARAVANAPRABHU I. EEE 2023@CITCHENNAI.NET</td><td>Roll No.:</td><td>23E5046</td></tr>
</table>"""
        )

        self.assertEqual(parsed["candidate"]["name"], "SARAVANA PRABHU . I")
        self.assertEqual(parsed["candidate"]["email"], "SARAVANAPRABHUI.EEE2023@CITCHENNAI.NET")
        self.assertEqual(parsed["candidate"]["roll_no"], "23E5046")

    def test_parse_identity_table_without_set_preserves_roll_no(self):
        parsed = parse_submission_text(
            """<table>
  <tr><td>Name:</td><td>Sabaresh C</td><td>Sat No.:</td><td>1</td><td>2</td></tr>
  <tr><td>Email ID:</td><td>SABARESHC.CSE2023@CITCHENNAI.NET</td><td>Roll No.:</td><td>2100010773</td></tr>
</table>"""
        )

        self.assertEqual(parsed["candidate"]["name"], "Sabaresh C")
        self.assertEqual(parsed["candidate"]["email"], "SABARESHC.CSE2023@CITCHENNAI.NET")
        self.assertEqual(parsed["candidate"]["roll_no"], "2100010773")

    def test_detect_marked_set_from_image(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "set.png"
            image = Image.new("RGB", (1200, 1600), "white")
            centers = [(835, 277), (871, 278), (914, 279), (952, 282)]
            for index, (x, y) in enumerate(centers):
                for dx in range(-13, 14):
                    for dy in range(-13, 14):
                        distance = (dx * dx + dy * dy) ** 0.5
                        if 11 <= distance <= 13:
                            image.putpixel((x + dx, y + dy), (0, 0, 0))
                        elif index == 1 and distance <= 9:
                            image.putpixel((x + dx, y + dy), (30, 35, 120))
            image.save(path)

            self.assertEqual(detect_marked_set_from_image(path), "set2")

    def test_detect_answers_from_image(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "answers.png"
            image = Image.new("RGB", (1200, 1600), "white")
            left_xs = [260, 320, 380, 440]
            right_xs = [700, 760, 820, 880]
            row_ys = [515, 628, 745, 865, 990, 1121, 1257, 1396]
            selections = {
                "1": "C",
                "2": "B",
                "3": "A",
                "4": "B",
                "5": "C",
                "6": "B",
                "7": "C",
                "8": "B",
                "9": "B",
                "10": "B",
                "11": "B",
                "12": "C",
                "13": "A",
                "14": "C",
                "15": "B",
                "16": "C",
            }
            for row_index, y in enumerate(row_ys):
                for question_id, xs in ((str(row_index + 1), left_xs), (str(row_index + 9), right_xs)):
                    selected = selections[question_id]
                    for option_index, x in enumerate(xs):
                        for dx in range(-18, 19):
                            for dy in range(-18, 19):
                                distance = (dx * dx + dy * dy) ** 0.5
                                if 16 <= distance <= 18:
                                    image.putpixel((x + dx, y + dy), (0, 0, 0))
                                elif OPTION_LETTERS[option_index] == selected and distance <= 13:
                                    image.putpixel((x + dx, y + dy), (30, 35, 120))
            image.save(path)

            answers = {answer["question_id"]: answer["selected"] for answer in detect_answers_from_image(path)}

            self.assertEqual(answers, selections)

    def test_detect_answers_includes_high_first_row(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "answers_high_first_row.png"
            image = Image.new("RGB", (1200, 1600), "white")
            left_xs = [260, 320, 380, 440]
            right_xs = [700, 760, 820, 880]
            row_ys = [455, 585, 715, 845, 975, 1105, 1235, 1365]
            selections = {
                "1": "D",
                "2": "A",
                "3": "B",
                "4": "C",
                "5": "D",
                "6": "A",
                "7": "B",
                "8": "C",
                "9": "A",
                "10": "B",
                "11": "C",
                "12": "D",
                "13": "A",
                "14": "B",
                "15": "C",
                "16": "D",
            }
            for row_index, y in enumerate(row_ys):
                for question_id, xs in ((str(row_index + 1), left_xs), (str(row_index + 9), right_xs)):
                    selected = selections[question_id]
                    for option_index, x in enumerate(xs):
                        for dx in range(-18, 19):
                            for dy in range(-18, 19):
                                distance = (dx * dx + dy * dy) ** 0.5
                                if 16 <= distance <= 18:
                                    image.putpixel((x + dx, y + dy), (0, 0, 0))
                                elif OPTION_LETTERS[option_index] == selected and distance <= 13:
                                    image.putpixel((x + dx, y + dy), (30, 35, 120))
            image.save(path)

            answers = {answer["question_id"]: answer["selected"] for answer in detect_answers_from_image(path)}

            self.assertEqual(answers, selections)

    def test_lalitha_scan_88_orientation_and_rows_when_available(self):
        path = Path(
            "/Users/aayush.shah/Downloads/OMR images/Archive 2/lalitha 2 folder/"
            "Adobe Scan 24 Jun 2026 (88)_1.jpg"
        )
        if not path.exists():
            self.skipTest("local Lalitha scan fixture is not available")

        answers = {answer["question_id"]: answer["selected"] for answer in detect_answers_from_image(path)}

        self.assertEqual(detect_marked_set_from_image(path), "set3")
        self.assertEqual(len(answers), 16)
        self.assertEqual(answers["1"], "C")
        self.assertEqual(answers["8"], "C")
        self.assertEqual(answers["16"], "B")

    def test_detect_answers_ignores_instruction_circle_row(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "answers_with_instruction_circles.png"
            image = Image.new("RGB", (1200, 1600), "white")
            left_xs = [260, 320, 380, 440]
            right_xs = [700, 760, 820, 880]
            row_ys = [570, 705, 840, 975, 1110, 1245, 1380, 1520]
            selections = {
                "1": "B",
                "2": "D",
                "3": "A",
                "4": "C",
                "5": "B",
                "6": "D",
                "7": "A",
                "8": "C",
                "9": "A",
                "10": "D",
                "11": "B",
                "12": "C",
                "13": "A",
                "14": "D",
                "15": "B",
                "16": "C",
            }
            for x in [190, 260, 350, 430, 510, 590, 670, 760, 850]:
                for dx in range(-17, 18):
                    for dy in range(-17, 18):
                        distance = (dx * dx + dy * dy) ** 0.5
                        if 15 <= distance <= 17:
                            image.putpixel((x + dx, 500 + dy), (0, 0, 0))
            for row_index, y in enumerate(row_ys):
                for question_id, xs in ((str(row_index + 1), left_xs), (str(row_index + 9), right_xs)):
                    selected = selections[question_id]
                    for option_index, x in enumerate(xs):
                        for dx in range(-18, 19):
                            for dy in range(-18, 19):
                                distance = (dx * dx + dy * dy) ** 0.5
                                if 16 <= distance <= 18:
                                    image.putpixel((x + dx, y + dy), (0, 0, 0))
                                elif OPTION_LETTERS[option_index] == selected and distance <= 13:
                                    image.putpixel((x + dx, y + dy), (30, 35, 120))
            image.save(path)

            answers = {answer["question_id"]: answer["selected"] for answer in detect_answers_from_image(path)}

            self.assertEqual(answers, selections)

    def test_new_folder_scan_90_top_and_bottom_rows_when_available(self):
        path = Path(
            "/Users/aayush.shah/Downloads/OMR images/Archive 2/New Folder With Items 2/"
            "Adobe Scan 24 Jun 2026 (90)_1.jpg"
        )
        if not path.exists():
            self.skipTest("local New Folder scan fixture is not available")

        answers = {answer["question_id"]: answer["selected"] for answer in detect_answers_from_image(path)}

        self.assertEqual(detect_marked_set_from_image(path), "set4")
        self.assertEqual(len(answers), 16)
        self.assertEqual(answers["1"], "B")
        self.assertEqual(answers["8"], "C")
        self.assertEqual(answers["9"], "A")
        self.assertEqual(answers["10"], "D")
        self.assertEqual(answers["16"], "B")

    def test_phone_photo_perspective_registration_when_available(self):
        paths = [
            Path("/Users/aayush.shah/Downloads/IMG_8188.JPG.jpeg"),
            Path("/Users/aayush.shah/Downloads/Hadn written/IMG_8188.JPG.jpeg"),
        ]
        path = next((candidate for candidate in paths if candidate.exists()), None)
        if path is None:
            self.skipTest("local phone photo fixture is not available")

        answers = {answer["question_id"]: answer["selected"] for answer in detect_answers_from_image(path)}

        self.assertEqual(detect_marked_set_from_image(path), "set1")
        self.assertEqual(
            answers,
            {
                "1": "D",
                "2": "B",
                "3": "B",
                "4": "B",
                "5": "D",
                "6": "C",
                "7": "D",
                "8": "A",
                "9": "A",
                "10": "A",
                "11": "C",
                "12": "D",
                "13": "B",
                "14": "D",
                "15": "A",
                "16": "A",
            },
        )

    def test_phone_photo_skewed_bottom_rows_when_available(self):
        path = Path("/Users/aayush.shah/Downloads/Hadn written/IMG_8190.JPG.jpeg")
        if not path.exists():
            self.skipTest("local skewed phone photo fixture is not available")

        answers = {answer["question_id"]: answer["selected"] for answer in detect_answers_from_image(path)}

        self.assertEqual(detect_marked_set_from_image(path), "set4")
        self.assertEqual(
            answers,
            {
                "1": "D",
                "2": "C",
                "3": "A",
                "4": "C",
                "5": "A",
                "6": "B",
                "7": "D",
                "8": "C",
                "9": "D",
                "10": "C",
                "11": "B",
                "12": "D",
                "13": "A",
                "14": "A",
                "15": "D",
                "16": "A",
            },
        )

    def test_phone_photo_set_markers_when_available(self):
        cases = {
            "IMG_8192.JPG.jpeg": "set2",
            "IMG_8195.JPG.jpeg": "set3",
            "IMG_8197.JPG.jpeg": "set1",
            "IMG_8199.JPG.jpeg": "set2",
        }
        folder = Path("/Users/aayush.shah/Downloads/Hadn written")
        paths = {name: folder / name for name in cases}
        if not any(path.exists() for path in paths.values()):
            self.skipTest("local phone photo set marker fixtures are not available")

        for name, expected_set in cases.items():
            path = paths[name]
            if path.exists():
                self.assertEqual(detect_marked_set_from_image(path), expected_set)

    def test_detect_answers_and_set_from_rotated_image(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rotated_answers.png"
            image = Image.new("RGB", (1200, 1600), "white")
            set_centers = [(835, 277), (871, 278), (914, 279), (952, 282)]
            left_xs = [260, 320, 380, 440]
            right_xs = [700, 760, 820, 880]
            row_ys = [515, 628, 745, 865, 990, 1121, 1257, 1396]
            selections = {
                "1": "C",
                "2": "B",
                "3": "A",
                "4": "B",
                "5": "C",
                "6": "B",
                "7": "C",
                "8": "B",
                "9": "B",
                "10": "B",
                "11": "B",
                "12": "C",
                "13": "A",
                "14": "C",
                "15": "B",
                "16": "C",
            }
            for index, (x, y) in enumerate(set_centers):
                for dx in range(-13, 14):
                    for dy in range(-13, 14):
                        distance = (dx * dx + dy * dy) ** 0.5
                        if 11 <= distance <= 13:
                            image.putpixel((x + dx, y + dy), (0, 0, 0))
                        elif index == 2 and distance <= 9:
                            image.putpixel((x + dx, y + dy), (5, 5, 5))
            for row_index, y in enumerate(row_ys):
                for question_id, xs in ((str(row_index + 1), left_xs), (str(row_index + 9), right_xs)):
                    selected = selections[question_id]
                    for option_index, x in enumerate(xs):
                        for dx in range(-18, 19):
                            for dy in range(-18, 19):
                                distance = (dx * dx + dy * dy) ** 0.5
                                if 16 <= distance <= 18:
                                    image.putpixel((x + dx, y + dy), (0, 0, 0))
                                elif OPTION_LETTERS[option_index] == selected and distance <= 13:
                                    image.putpixel((x + dx, y + dy), (5, 5, 5))
            image.rotate(90, expand=True).save(path)

            answers = {answer["question_id"]: answer["selected"] for answer in detect_answers_from_image(path)}

            self.assertEqual(detect_marked_set_from_image(path), "set3")
            self.assertEqual(answers, selections)

    def test_set_marker_row_defaults_to_numbered_sets(self):
        parsed = parse_omr_text(
            """Set No.
○ ● ○ ○
Roll No.: 23EE0446
"""
        )

        self.assertEqual(parsed["candidate"]["exam_set"]["selected"], "set2")

    def test_parse_lighton_json_table_output(self):
        sample = """{
  "choices": [
    {
      "message": {
        "content": "SET-1\\n\\n# OMR ANSWER SHEET\\n\\nName: Test Candidate\\n\\nEmail ID: candidate@example.com\\n\\n<table>\\n  <tr><td>1</td><td>A ●</td><td>B ○</td><td>C ○</td><td>D ○</td><td>9</td><td>A ○</td><td>B ○</td><td>C ○</td><td>D ●</td></tr>\\n  <tr><td>2</td><td>A ○</td><td>B ●</td><td>C ○</td><td>D ○</td><td>10</td><td>A ○</td><td>B ○</td><td>C ○</td><td>D ○</td></tr>\\n  <tr><td>3</td><td>B ●</td><td>C ○</td><td>D ○</td><td>11</td><td>A ○</td><td>B ○</td><td>C ○</td><td>D ○</td></tr>\\n  <tr><td>4</td><td>A ○</td><td>B ○</td><td>C ○</td><td>D ●</td><td>12</td><td>B ●</td><td>C ○</td><td>D ○</td></tr>\\n  <tr><td>5</td><td>A ○</td><td>B ○</td><td>D ●</td><td>13</td><td>A ○</td><td>B ○</td><td>C ○</td><td>D ●</td></tr>\\n  <tr><td>6</td><td>A ○</td><td>B ●</td><td>C ○</td><td>D ○</td><td>14</td><td>B ●</td><td>C ○</td><td>D ○</td></tr>\\n  <tr><td>7</td><td>A ○</td><td>B ○</td><td>C ○</td><td>D ●</td><td>15</td><td>A ○</td><td>B ○</td><td>C ○</td><td>D ●</td></tr>\\n  <tr><td>8</td><td>B ●</td><td>C ○</td><td>D ○</td><td>16</td><td>A ○</td><td>B ○</td><td>C ○</td><td>D ○</td></tr>\\n</table>"
      }
    }
  ]
}"""

        parsed = parse_submission_text(sample)
        answers = {answer["question_id"]: answer for answer in parsed["answers"]}

        self.assertEqual(parsed["candidate"]["name"], "Test Candidate")
        self.assertEqual(parsed["candidate"]["email"], "candidate@example.com")
        self.assertEqual(parsed["candidate"]["exam_set"], "set1")
        self.assertEqual(len(answers), 16)
        self.assertEqual(answers["1"]["selected"], "A")
        self.assertEqual(answers["3"]["selected"], "B")
        self.assertEqual(answers["9"]["selected"], "D")
        self.assertEqual(answers["10"]["selected"], None)
        self.assertEqual(answers["10"]["status"], "unmarked")
        self.assertEqual(answers["16"]["selected"], None)

    def test_parse_marker_only_pdf_table_output(self):
        sample = """SET-1

Name: *Test Candidate*
Email ID: *candidate 57@example.com*

<table>
  <tr><td>1</td><td>●</td><td>B</td><td>C</td><td>D</td><td>9</td><td>A</td><td>B</td><td>C</td><td>●</td></tr>
  <tr><td>2</td><td>A</td><td>●</td><td>C</td><td>D</td><td>10</td><td>A</td><td>B</td><td>C</td><td>D</td></tr>
</table>
"""

        parsed = parse_submission_text(sample)
        answers = {answer["question_id"]: answer for answer in parsed["answers"]}

        self.assertEqual(parsed["candidate"]["email"], "candidate57@example.com")
        self.assertEqual(parsed["candidate"]["exam_set"], "set1")
        self.assertEqual(answers["1"]["selected"], "A")
        self.assertEqual(answers["2"]["selected"], "B")
        self.assertEqual(answers["9"]["selected"], "D")
        self.assertEqual(answers["10"]["selected"], None)
        self.assertEqual(answers["10"]["status"], "unmarked")

    def test_set_regex_variants(self):
        cases = {
            "SET-1\nName: A": "set1",
            "Exam Set: set2\nName: A": "set2",
            "paper set - A\nName: A": "seta",
            "$\\text{SET} \\rightarrow 3$\nName: A": "set3",
            "SET -> 4\nName: A": "set4",
            "Name: A\nEmail: a@example.com": None,
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                parsed = parse_submission_text(text)
                self.assertEqual(parsed["candidate"]["exam_set"], expected)

    def test_preprocess_darkens_blue_ink(self):
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "blue.png"
            output = Path(temp_dir) / "processed.png"
            image = Image.new("RGB", (40, 40), "white")
            for x in range(12, 28):
                for y in range(12, 28):
                    image.putpixel((x, y), (35, 45, 140))
            image.save(source)

            preprocess_image_for_ocr(source, output)
            processed = Image.open(output).convert("RGB")

            self.assertLess(sum(processed.getpixel((20, 20))), 30)
            self.assertTrue(all(channel == 255 or channel % 8 == 0 for channel in processed.getpixel((5, 5))))

    def test_validate_combined_pdf_counts_pages(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "combined.pdf"
            pages = [
                Image.new("RGB", (80, 80), "white"),
                Image.new("RGB", (80, 80), "white"),
            ]
            pages[0].save(path, save_all=True, append_images=pages[1:])

            self.assertEqual(validate_combined_pdf(path), 2)

    def test_combined_pdf_estimated_time(self):
        self.assertEqual(estimated_combined_pdf_seconds(4), 60)
        self.assertEqual(format_duration(45), "45 sec")
        self.assertEqual(format_duration(75), "1 min 15 sec")

    def test_folder_upload_limits(self):
        validate_folder_upload_limits(MAX_FOLDER_UPLOAD_FILES, MAX_FOLDER_UPLOAD_BYTES)

        with self.assertRaises(ValueError):
            validate_folder_upload_limits(MAX_FOLDER_UPLOAD_FILES + 1, 1)

        with self.assertRaises(ValueError):
            validate_folder_upload_limits(1, MAX_FOLDER_UPLOAD_BYTES + 1)

    def test_background_batch_job_writes_csv(self):
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sheet.jpg"
            source.write_bytes(b"fake image")
            result = {
                "name": "Ada",
                "email": "ada@example.com",
                "roll_no": "23EE0446",
                "set": "set1",
                "answered_questions": 16,
                "total_questions": 16,
                "unanswered_questions": 0,
                "score": 10,
                "max_score": 16,
                "identity_warnings": [],
            }
            job_id = create_batch_job(1, "batch_test")

            with patch("app.OUTPUT_DIR", Path(temp_dir)), patch("app.score_path", return_value=result):
                process_saved_upload_batch_job(job_id, [("sheet.jpg", source)])

            job = get_batch_job(job_id)
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["processed_files"], 1)
            self.assertEqual(job["successful_files"], 1)
            self.assertEqual(job["failed_files"], 0)
            self.assertEqual(job["rows"][0]["roll_no"], "23EE0446")
            self.assertTrue(Path(job["csv_path"]).exists())
            self.assertTrue(job["download_url"].startswith("/api/download/upload_scores_"))

    def test_background_batch_jobs_are_serialized(self):
        with TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.jpg"
            second = Path(temp_dir) / "second.jpg"
            first.write_bytes(b"fake first image")
            second.write_bytes(b"fake second image")
            result = {
                "name": "Ada",
                "email": "ada@example.com",
                "roll_no": "23EE0446",
                "set": "set1",
                "answered_questions": 16,
                "total_questions": 16,
                "unanswered_questions": 0,
                "score": 10,
                "max_score": 16,
                "identity_warnings": [],
            }
            first_started = threading.Event()
            release_first = threading.Event()
            second_started = threading.Event()
            call_order: list[str] = []

            def score_path_once_at_a_time(path: Path) -> dict:
                call_order.append(path.name)
                if path == first:
                    first_started.set()
                    self.assertFalse(second_started.is_set())
                    self.assertTrue(release_first.wait(timeout=5))
                else:
                    second_started.set()
                return result

            first_job_id = create_batch_job(1, "first_batch")
            second_job_id = create_batch_job(1, "second_batch")

            with patch("app.OUTPUT_DIR", Path(temp_dir)), patch("app.score_path", side_effect=score_path_once_at_a_time):
                first_thread = threading.Thread(
                    target=process_saved_upload_batch_job,
                    args=(first_job_id, [("first.jpg", first)]),
                )
                second_thread = threading.Thread(
                    target=process_saved_upload_batch_job,
                    args=(second_job_id, [("second.jpg", second)]),
                )
                first_thread.start()
                self.assertTrue(first_started.wait(timeout=5))
                second_thread.start()

                second_job = None
                for _ in range(100):
                    second_job = get_batch_job(second_job_id)
                    if second_job and second_job["message"]:
                        break
                    time.sleep(0.01)

                self.assertIsNotNone(second_job)
                assert second_job is not None
                self.assertEqual(second_job["status"], "queued")
                self.assertEqual(second_job["processed_files"], 0)
                self.assertEqual(second_job["message"], "Waiting for another OCR batch to finish")
                self.assertFalse(second_started.is_set())

                release_first.set()
                first_thread.join(timeout=5)
                second_thread.join(timeout=5)

            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertEqual(call_order, ["first.jpg", "second.jpg"])
            first_job = get_batch_job(first_job_id)
            second_job = get_batch_job(second_job_id)
            self.assertIsNotNone(first_job)
            self.assertIsNotNone(second_job)
            assert first_job is not None
            assert second_job is not None
            self.assertEqual(first_job["status"], "completed")
            self.assertEqual(second_job["status"], "completed")
            self.assertEqual(second_job["processed_files"], 1)

    def test_combined_pdf_background_job_writes_csv(self):
        with TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "combined.pdf"
            source.write_bytes(b"fake pdf")
            pages = [Path(temp_dir) / "page-1.png", Path(temp_dir) / "page-2.png"]
            for page in pages:
                page.write_bytes(b"fake page")
            result = {
                "name": "Ada",
                "email": "ada@example.com",
                "roll_no": "23EE0446",
                "set": "set1",
                "answered_questions": 16,
                "total_questions": 16,
                "unanswered_questions": 0,
                "score": 10,
                "max_score": 16,
                "identity_warnings": [],
            }
            job_id = create_batch_job(2, "combined_batch", job_type="combined_pdf", unit_label="page(s)")

            with (
                patch("app.OUTPUT_DIR", Path(temp_dir)),
                patch("app.render_pdf_pages", return_value=pages),
                patch("app.call_lighton_chat_ocr_image", return_value="ocr text"),
                patch("app.call_lighton_identity_ocr_image", return_value="identity text"),
                patch("app.score_ocr_text", return_value=result),
            ):
                process_combined_pdf_job(job_id, source, "combined.pdf", 2)

            job = get_batch_job(job_id)
            self.assertIsNotNone(job)
            assert job is not None
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["job_type"], "combined_pdf")
            self.assertEqual(job["unit_label"], "page(s)")
            self.assertEqual(job["processed_files"], 2)
            self.assertEqual(job["successful_files"], 2)
            self.assertEqual(job["failed_files"], 0)
            self.assertEqual(job["rows"][0]["source"], "combined.pdf page 1")
            self.assertTrue(Path(job["csv_path"]).exists())
            self.assertTrue(job["download_url"].startswith("/api/download/combined_pdf_scores_"))

    def test_folder_and_combined_pdf_jobs_share_large_job_queue(self):
        with TemporaryDirectory() as temp_dir:
            folder_source = Path(temp_dir) / "folder-sheet.jpg"
            pdf_source = Path(temp_dir) / "combined.pdf"
            page = Path(temp_dir) / "page-1.png"
            folder_source.write_bytes(b"fake image")
            pdf_source.write_bytes(b"fake pdf")
            page.write_bytes(b"fake page")
            result = {
                "name": "Ada",
                "email": "ada@example.com",
                "roll_no": "23EE0446",
                "set": "set1",
                "answered_questions": 16,
                "total_questions": 16,
                "unanswered_questions": 0,
                "score": 10,
                "max_score": 16,
                "identity_warnings": [],
            }
            folder_started = threading.Event()
            release_folder = threading.Event()
            pdf_render_started = threading.Event()
            call_order: list[str] = []

            def score_folder(path: Path) -> dict:
                call_order.append(path.name)
                folder_started.set()
                self.assertFalse(pdf_render_started.is_set())
                self.assertTrue(release_folder.wait(timeout=5))
                return result

            def render_pdf(*args, **kwargs) -> list[Path]:
                call_order.append("render-pdf")
                pdf_render_started.set()
                return [page]

            folder_job_id = create_batch_job(1, "folder_batch")
            pdf_job_id = create_batch_job(1, "combined_batch", job_type="combined_pdf", unit_label="page(s)")

            with (
                patch("app.OUTPUT_DIR", Path(temp_dir)),
                patch("app.score_path", side_effect=score_folder),
                patch("app.render_pdf_pages", side_effect=render_pdf),
                patch("app.call_lighton_chat_ocr_image", return_value="ocr text"),
                patch("app.call_lighton_identity_ocr_image", return_value="identity text"),
                patch("app.score_ocr_text", return_value=result),
            ):
                folder_thread = threading.Thread(
                    target=process_saved_upload_batch_job,
                    args=(folder_job_id, [("folder-sheet.jpg", folder_source)]),
                )
                pdf_thread = threading.Thread(
                    target=process_combined_pdf_job,
                    args=(pdf_job_id, pdf_source, "combined.pdf", 1),
                )
                folder_thread.start()
                self.assertTrue(folder_started.wait(timeout=5))
                pdf_thread.start()

                pdf_job = None
                for _ in range(100):
                    pdf_job = get_batch_job(pdf_job_id)
                    if pdf_job and pdf_job["message"]:
                        break
                    time.sleep(0.01)

                self.assertIsNotNone(pdf_job)
                assert pdf_job is not None
                self.assertEqual(pdf_job["status"], "queued")
                self.assertEqual(pdf_job["processed_files"], 0)
                self.assertEqual(pdf_job["message"], "Waiting for another OCR batch to finish")
                self.assertFalse(pdf_render_started.is_set())

                release_folder.set()
                folder_thread.join(timeout=5)
                pdf_thread.join(timeout=5)

            self.assertFalse(folder_thread.is_alive())
            self.assertFalse(pdf_thread.is_alive())
            self.assertEqual(call_order, ["folder-sheet.jpg", "render-pdf"])
            folder_job = get_batch_job(folder_job_id)
            pdf_job = get_batch_job(pdf_job_id)
            self.assertIsNotNone(folder_job)
            self.assertIsNotNone(pdf_job)
            assert folder_job is not None
            assert pdf_job is not None
            self.assertEqual(folder_job["status"], "completed")
            self.assertEqual(pdf_job["status"], "completed")
            self.assertEqual(pdf_job["processed_files"], 1)

    def test_save_answer_key_json(self):
        with TemporaryDirectory() as temp_dir:
            path = save_answer_key_json("answers/set1.csv", "Set 1", temp_dir)

            self.assertEqual(path, Path(temp_dir) / "Set_1.json")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
