"""Unit tests for Qualitative Error Taxonomy & Case Study Extractor."""

import json
import os
import tempfile
import unittest

from scripts.extract_case_studies import (
    analyze_and_extract_case_studies,
    classify_trajectory,
    generate_latex_case_study_block,
)


class TestExtractCaseStudies(unittest.TestCase):
    def test_semantic_recovery_classification(self):
        rec = {
            "problem_id": "HE_10",
            "turn_history": [
                {"turn": 0, "status": "WA", "code": "def solve(): return False", "feedback": "AssertionError: expected True"},
                {"turn": 1, "status": "AC", "code": "def solve(): return True", "feedback": "All test cases PASSED successfully."},
            ],
        }
        self.assertEqual(classify_trajectory(rec), "SEMANTIC_RECOVERY")

    def test_runtime_recovery_classification(self):
        rec_re = {
            "problem_id": "HE_20",
            "turn_history": [
                {"turn": 0, "status": "RE", "code": "def solve(xs): return xs[0]", "feedback": "IndexError: list index out of range"},
                {"turn": 1, "status": "AC", "code": "def solve(xs): return xs[0] if xs else None", "feedback": "All test cases PASSED successfully."},
            ],
        }
        self.assertEqual(classify_trajectory(rec_re), "RUNTIME_RECOVERY")

        rec_ce = {
            "problem_id": "HE_21",
            "turn_history": [
                {"turn": 0, "status": "CE", "code": "def solve()", "feedback": "SyntaxError: invalid syntax"},
                {"turn": 1, "status": "AC", "code": "def solve(): return 1", "feedback": "All test cases PASSED successfully."},
            ],
        }
        self.assertEqual(classify_trajectory(rec_ce), "RUNTIME_RECOVERY")

    def test_regression_classification(self):
        # Passed 4 tests at turn 0, but only passed 1 test at turn 1
        rec = {
            "problem_id": "HE_30",
            "turn_history": [
                {"turn": 0, "status": "WA", "passed": 4, "code": "def solve(x): return x * 2", "feedback": "Passed 4/5 tests"},
                {"turn": 1, "status": "WA", "passed": 1, "code": "def solve(x): return x + 2", "feedback": "Passed 1/5 tests"},
            ],
        }
        self.assertEqual(classify_trajectory(rec), "REGRESSION")

        # WA devolving into Syntax Error
        rec_ce = {
            "problem_id": "HE_31",
            "turn_history": [
                {"turn": 0, "status": "WA", "code": "def solve(): return 1", "feedback": "WA"},
                {"turn": 1, "status": "CE", "code": "def solve(:", "feedback": "SyntaxError"},
            ],
        }
        self.assertEqual(classify_trajectory(rec_ce), "REGRESSION")

    def test_feedback_misinterpretation_classification(self):
        rec = {
            "problem_id": "HE_40",
            "turn_history": [
                {"turn": 0, "status": "WA", "code": "def compute_factorial(n):\n    res = 1\n    for i in range(1, n+1): res *= i\n    return res",
                 "feedback": "Traceback: line 3, calculation mismatch on n=5"},
                {"turn": 1, "status": "WA", "code": "def f(n): pass", # code drastically wiped
                 "feedback": "AssertionError"},
            ],
        }
        self.assertEqual(classify_trajectory(rec), "FEEDBACK_MISINTERPRETATION")

    def test_resistant_plateau_classification(self):
        rec = {
            "problem_id": "HE_50",
            "turn_history": [
                {"turn": 0, "status": "TLE", "code": "while True: pass", "feedback": "Time Limit Exceeded"},
                {"turn": 1, "status": "TLE", "code": "while 1: pass", "feedback": "Time Limit Exceeded"},
                {"turn": 2, "status": "TLE", "code": "while True: pass", "feedback": "Time Limit Exceeded"},
            ],
        }
        self.assertEqual(classify_trajectory(rec), "RESISTANT_PLATEAU")

    def test_end_to_end_extraction_and_latex(self):
        mock_traces = [
            # Semantic recovery
            {"problem_id": "HE_1", "turn_history": [{"status": "WA", "code": "x=1"}, {"status": "AC", "code": "x=2"}]},
            # Runtime recovery
            {"problem_id": "HE_2", "turn_history": [{"status": "RE", "code": "xs[0]"}, {"status": "AC", "code": "xs[0] if xs else 0"}]},
            # Regression
            {"problem_id": "HE_3", "turn_history": [{"status": "WA", "passed": 3, "code": "a=1"}, {"status": "WA", "passed": 0, "code": "a=0"}]},
            # Resistant
            {"problem_id": "HE_4", "turn_history": [{"status": "TLE", "code": "pass"}, {"status": "TLE", "code": "pass"}]},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_jsonl = os.path.join(tmp_dir, "mock_failures.jsonl")
            with open(input_jsonl, "w", encoding="utf-8") as f:
                for r in mock_traces:
                    f.write(json.dumps(r) + "\n")

            out_dir = os.path.join(tmp_dir, "out")
            res = analyze_and_extract_case_studies(
                input_source=input_jsonl,
                output_dir=out_dir,
                export_latex=True,
            )

            self.assertEqual(res["total_records"], 4)
            self.assertTrue(os.path.isfile(res["csv_path"]))
            self.assertTrue(os.path.isfile(res["json_path"]))
            self.assertTrue(os.path.isfile(res["latex_path"]))

            # Verify LaTeX content
            with open(res["latex_path"], "r", encoding="utf-8") as f:
                tex_content = f.read()

            self.assertIn("\\begin{listing}", tex_content)
            self.assertIn("\\begin{lstlisting}", tex_content)
            self.assertIn("HE_1", tex_content)
            self.assertIn("HE_2", tex_content)


if __name__ == "__main__":
    unittest.main()
