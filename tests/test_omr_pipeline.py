import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from app import (
    MAX_FOLDER_UPLOAD_BYTES,
    MAX_FOLDER_UPLOAD_FILES,
    estimated_combined_pdf_seconds,
    format_duration,
    row_from_result,
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

    def test_save_answer_key_json(self):
        with TemporaryDirectory() as temp_dir:
            path = save_answer_key_json("answers/set1.csv", "Set 1", temp_dir)

            self.assertEqual(path, Path(temp_dir) / "Set_1.json")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
