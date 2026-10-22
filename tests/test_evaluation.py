"""Tests for evaluation metrics (Pass@1, Fix@K, error distribution, recovery rate)."""

import json
import os
import tempfile
import unittest

from src.debugging.debug_loop import format_feedback
from src.evaluation.metrics import (
    calculate_conditional_repair,
    calculate_cumulative_success,
    calculate_error_distribution,
    calculate_fix_at_k,
    calculate_mean_repair_turn,
    calculate_pass_at_1,
    calculate_recovery_rate,
    compute_bootstrap_ci,
    evaluate,
    mcnemar_test,
    summarize_eval_metrics,
)
from src.execution.executor import ExecutionResult


class TestEvaluation(unittest.TestCase):
    def test_pass_at_1(self):
        results = [
            {"status": "AC"},
            {"status": "WA"},
            {"status": "AC"},
            {"status": "CE"},
        ]
        self.assertEqual(calculate_pass_at_1(results), 0.5)

    def test_fix_at_k(self):
        sessions = [
            {"solved": True, "solved_turn": 1, "history": [{"status": "AC"}]},
            {"solved": True, "solved_turn": 3, "history": [{"status": "WA"}, {"status": "RE"}, {"status": "AC"}]},
            {"solved": False, "solved_turn": None, "history": [{"status": "WA"}, {"status": "WA"}, {"status": "WA"}]},
        ]

        self.assertAlmostEqual(calculate_fix_at_k(sessions, k=1), 1 / 3)
        self.assertAlmostEqual(calculate_fix_at_k(sessions, k=3), 2 / 3)
        self.assertAlmostEqual(calculate_fix_at_k(sessions, k=5), 2 / 3)

    def test_error_distribution(self):
        results = [
            {"status": "AC"},
            {"status": "WA"},
            {"status": "WA"},
            {"status": "TLE"},
        ]
        dist = calculate_error_distribution(results)
        self.assertEqual(dist["AC"], 0.25)
        self.assertEqual(dist["WA"], 0.50)
        self.assertEqual(dist["TLE"], 0.25)

    def test_recovery_rate(self):
        sessions = [
            {"solved": True, "history": [{"status": "WA"}, {"status": "AC"}]},
            {"solved": False, "history": [{"status": "RE"}, {"status": "RE"}]},
        ]
        self.assertEqual(calculate_recovery_rate(sessions), 0.5)

    def test_feedback_modes_formatting(self):
        res = ExecutionResult(
            status="WA",
            passed_tests=2,
            total_tests=5,
            traceback="AssertionError: Output mismatch",
        )
        # F0: No feedback
        self.assertEqual(format_feedback(res, mode="F0"), "")
        # F1: Status only
        f1 = format_feedback(res, mode="F1")
        self.assertIn("Execution Status: WA", f1)
        self.assertNotIn("Traceback", f1)
        # F2: Sanitized traceback
        f2 = format_feedback(res, mode="F2")
        self.assertIn("Execution Status: WA", f2)
        self.assertIn("Traceback / Error Details:", f2)
        # F3: Structured summary
        f3 = format_feedback(res, mode="F3")
        self.assertIn("Execution Status: WA", f3)
        self.assertIn("Passed Tests: 2 / 5 (40.0%)", f3)

    def test_evaluate_streaming_jsonl_and_metrics(self):
        # Mock debug loop function
        def mock_debug_loop(model, tokenizer, problem, test_cases=None, K=3, **kwargs):
            if "fail" in problem:
                return [
                    {"turn": 1, "prompt": "p1", "code": "c1", "status": "WA", "result": {"status": "WA", "passed_tests": 0, "total_tests": 2}, "tokens": 20, "latency": 0.1},
                    {"turn": 2, "prompt": "p2", "code": "c2", "status": "AC", "result": {"status": "AC", "passed_tests": 2, "total_tests": 2}, "tokens": 25, "latency": 0.1},
                ]
            else:
                return [
                    {"turn": 1, "prompt": "p1", "code": "c1", "status": "AC", "result": {"status": "AC", "passed_tests": 2, "total_tests": 2}, "tokens": 15, "latency": 0.05},
                ]

        mock_dataset = [
            {"task_id": "prob_1", "prompt": "always solve", "test_list": ["assert True", "assert True"]},
            {"task_id": "prob_2", "prompt": "fail then repair", "test_list": ["assert True", "assert True"]},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = os.path.join(tmp_dir, "eval_out.jsonl")
            summary = evaluate(
                model=None,
                tokenizer=None,
                dataset=mock_dataset,
                K=3,
                label="test_eval",
                feedback_mode="F2",
                output_jsonl=jsonl_path,
                debug_loop_fn=mock_debug_loop,
            )

            # Check aggregate summary
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["pass_at_1"], 0.5)
            self.assertEqual(summary["cumulative_success_at_3"], 1.0)
            self.assertEqual(summary["conditional_repair_at_3"], 1.0)
            self.assertEqual(summary["conditional_repair_eligible"], 1)
            self.assertEqual(summary["conditional_repair_repaired"], 1)

            # Check JSONL output file content
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["problem_id"], "prob_1")
            self.assertTrue(lines[0]["pass_at_1"])
            self.assertEqual(lines[1]["problem_id"], "prob_2")
            self.assertFalse(lines[1]["pass_at_1"])
            self.assertTrue(lines[1]["cumulative_success_at_3"])
            self.assertTrue(lines[1]["conditional_repair_at_3"])

    def test_bootstrap_ci(self):
        data = [1.0] * 50 + [0.0] * 50
        mean, lower, upper = compute_bootstrap_ci(data, num_samples=500, ci=0.95)
        self.assertAlmostEqual(mean, 0.5)
        self.assertTrue(0.35 < lower < 0.5)
        self.assertTrue(0.5 < upper < 0.65)

    def test_mcnemar_test(self):
        base = [True] * 50 + [False] * 50
        treatment = [True] * 80 + [False] * 20
        res = mcnemar_test(base, treatment)
        self.assertIn("chi2", res)
        self.assertIn("p_value", res)
        self.assertTrue(res["p_value"] < 0.05)


if __name__ == "__main__":
    unittest.main()
