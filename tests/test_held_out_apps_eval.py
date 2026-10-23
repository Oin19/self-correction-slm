"""Unit tests for held-out APPS in-domain evaluation runner (RQ6)."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.run_held_out_apps_eval import run_apps_evaluation
from src.evaluation.metrics import _format_problem_prompt
from src.utils.test_cases import normalize_tests


class TestHeldOutAppsEval(unittest.TestCase):
    def test_apps_starter_code_formatting(self):
        # Case 1: Function-based problem with starter_code
        ex_fn = {
            "question": "Write a function solve(n) that computes the square of n.",
            "starter_code": "def solve(n):\n",
            "input_output": json.dumps({"inputs": [[2], [3]], "outputs": [4, 9], "fn_name": "solve"}),
        }
        prompt = _format_problem_prompt(ex_fn)
        self.assertIn("Write a function solve(n)", prompt)
        self.assertIn("def solve(n):", prompt)

        # Ensure normalize_tests extracts call assertions
        tests = normalize_tests(ex_fn)
        self.assertEqual(len(tests), 2)
        self.assertIn("assert solve(2) == 4", tests[0]["assertion"])
        self.assertIn("assert solve(3) == 9", tests[1]["assertion"])

        # Case 2: Stdin/stdout-based problem without starter_code
        ex_io = {
            "question": "Read an integer n and print n * 2.",
            "input_output": json.dumps({"inputs": ["5\n", "10\n"], "outputs": ["10\n", "20\n"]}),
        }
        prompt_io = _format_problem_prompt(ex_io)
        self.assertIn("Read an integer n and print n * 2.", prompt_io)
        tests_io = normalize_tests(ex_io)
        self.assertEqual(len(tests_io), 2)
        self.assertEqual(tests_io[0]["input"], "5\n")
        self.assertEqual(tests_io[0]["output"], "10\n")

    def test_run_apps_evaluation_end_to_end(self):
        # Mock dataset with 3 APPS-like problems
        mock_dataset = [
            {
                "problem_id": 1501,
                "question": "Solve problem 1",
                "starter_code": "def fn1():\n",
                "input_output": json.dumps({"inputs": [[]], "outputs": [1], "fn_name": "fn1"}),
            },
            {
                "problem_id": 1502,
                "question": "Solve problem 2 (fail then repair)",
                "starter_code": "def fn2():\n",
                "input_output": json.dumps({"inputs": [[]], "outputs": [2], "fn_name": "fn2"}),
            },
            {
                "problem_id": 1503,
                "question": "Solve problem 3 (always fail)",
                "starter_code": "def fn3():\n",
                "input_output": json.dumps({"inputs": [[]], "outputs": [3], "fn_name": "fn3"}),
            },
        ]

        def mock_debug_loop(model, tokenizer, problem, test_cases=None, K=3, **kwargs):
            if "problem 1" in problem:
                return [
                    {"turn": 1, "prompt": "p", "code": "c", "status": "AC", "result": {"status": "AC", "passed_tests": 1, "total_tests": 1}, "tokens": 100, "latency": 0.2}
                ]
            elif "problem 2" in problem:
                return [
                    {"turn": 1, "prompt": "p", "code": "c", "status": "WA", "result": {"status": "WA", "passed_tests": 0, "total_tests": 1}, "tokens": 120, "latency": 0.2},
                    {"turn": 2, "prompt": "p", "code": "c", "status": "AC", "result": {"status": "AC", "passed_tests": 1, "total_tests": 1}, "tokens": 130, "latency": 0.2},
                ]
            else:
                return [
                    {"turn": 1, "prompt": "p", "code": "c", "status": "WA", "result": {"status": "WA", "passed_tests": 0, "total_tests": 1}, "tokens": 120, "latency": 0.2},
                    {"turn": 2, "prompt": "p", "code": "c", "status": "WA", "result": {"status": "WA", "passed_tests": 0, "total_tests": 1}, "tokens": 130, "latency": 0.2},
                    {"turn": 3, "prompt": "p", "code": "c", "status": "WA", "result": {"status": "WA", "passed_tests": 0, "total_tests": 1}, "tokens": 140, "latency": 0.2},
                ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("scripts.run_held_out_apps_eval.load_model_and_tokenizer") as mock_load:
                with patch("scripts.run_held_out_apps_eval.load_apps_held_out_dataset") as mock_ds:
                    with patch("src.evaluation.metrics.agentic_debug_loop", side_effect=mock_debug_loop):
                        mock_load.return_value = (MagicMock(), MagicMock())
                        mock_ds.return_value = mock_dataset

                        res = run_apps_evaluation(
                            model_name="mock_model",
                            arm_label="TestArm",
                            k_values=[3],
                            feedback_mode="F2",
                            output_dir=tmp_dir,
                            seed=42,
                        )

                        # Verify generated summary and manifest files
                        self.assertTrue(os.path.isfile(res["summary_csv"]))
                        self.assertTrue(os.path.isfile(res["summary_json"]))
                        self.assertTrue(os.path.isfile(res["manifest"]))

                        # Verify metrics:
                        # 3 problems:
                        # Prob 1: pass@1 = True
                        # Prob 2: pass@1 = False, solved at turn 2 (cum@3=True, cond_repair=True)
                        # Prob 3: pass@1 = False, cum@3=False
                        rows = res["rows"]
                        self.assertEqual(len(rows), 1)
                        r = rows[0]
                        self.assertEqual(r["N_Evaluated"], 3)
                        self.assertAlmostEqual(r["Raw_Pass@1"], 1 / 3, places=2)
                        self.assertAlmostEqual(r["Raw_CumSuccess@K"], 2 / 3, places=2)
                        self.assertAlmostEqual(r["Raw_CondRepair@K"], 1 / 2, places=2)
                        self.assertEqual(r["Eligible_Repairs"], 2)
                        self.assertEqual(r["Successful_Repairs"], 1)

                        # Verify JSONL streaming output file
                        jsonl_file = r["JSONL_Path"]
                        self.assertTrue(os.path.isfile(jsonl_file))
                        with open(jsonl_file, "r", encoding="utf-8") as f:
                            lines = [json.loads(line) for line in f]
                        self.assertEqual(len(lines), 3)
                        self.assertEqual(lines[0]["problem_id"], 1501)
                        self.assertTrue(lines[0]["pass_at_1"])
                        self.assertEqual(lines[1]["problem_id"], 1502)
                        self.assertFalse(lines[1]["pass_at_1"])
                        self.assertTrue(lines[1]["cumulative_success_at_3"])
                        self.assertTrue(lines[1]["conditional_repair_at_3"])


if __name__ == "__main__":
    unittest.main()
