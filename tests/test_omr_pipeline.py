import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from omr_pipeline import (
    answer_key_from_csv,
    evaluate_answers,
    normalize_weight_map,
    parse_omr_text,
    parse_submission_text,
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
        self.assertEqual(parsed["candidate"]["exam_set"]["selected"], "B")
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
        self.assertEqual(result["set"], "set1")
        self.assertEqual(result["answered_questions"], 3)
        self.assertEqual(result["total_questions"], 3)
        self.assertEqual(result["unanswered_questions"], 0)
        self.assertEqual(result["score"], 5.0)
        self.assertEqual(result["max_score"], 8.0)

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

    def test_save_answer_key_json(self):
        with TemporaryDirectory() as temp_dir:
            path = save_answer_key_json("answers/set1.csv", "Set 1", temp_dir)

            self.assertEqual(path, Path(temp_dir) / "Set_1.json")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
