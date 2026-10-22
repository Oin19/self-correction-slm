"""Tests for evaluation metrics (Pass@1, Fix@K, ConditionalRepair@K, feedback modes, HumanEval normalization)."""

import json
import os
import tempfile
import unittest
from src.debugging.debug_loop import format_feedback_by_mode, sanitize_feedback_text, _execute
from src.evaluation.metrics import (
    calculate_pass_at_1,
    calculate_fix_at_k,
    calculate_error_distribution,
    calculate_recovery_rate,
    evaluate,
    summarize_eval_metrics,
)
from src.execution.executor import ExecutionResult
from src.models.generation import extract_code_block
from src.utils.test_cases import normalize_tests


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

    def test_summarize_metrics(self):
        sessions = [
            {"solved": True, "solved_turn": 1, "status": "AC", "history": [{"status": "AC"}]},
            {"solved": True, "solved_turn": 2, "status": "AC", "history": [{"status": "WA"}, {"status": "AC"}]},
        ]
        summary = summarize_eval_metrics(sessions)
        self.assertEqual(summary["pass_at_1"], 1.0)
        self.assertEqual(summary["fix_at_1"], 0.5)
        self.assertEqual(summary["fix_at_5"], 1.0)
        self.assertEqual(summary["recovery_rate"], 1.0)

    def test_humaneval_normalize_appends_entry_point_check(self):
        ex = {
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        }
        tests = normalize_tests(ex)
        self.assertEqual(len(tests), 1)
        self.assertIn("check(add)", tests[0])
        res_bad = _execute("def add(a, b):\n    return a - b\n", tests)
        self.assertNotEqual(res_bad.status, "AC")
        res_good = _execute("def add(a, b):\n    return a + b\n", tests)
        self.assertEqual(res_good.status, "AC")

    def test_extract_code_block_closing_fence_only(self):
        raw_ppo_resp = "def add(a, b):\n    return a + b\n```\nSome explanation"
        self.assertEqual(extract_code_block(raw_ppo_resp), "def add(a, b):\n    return a + b")

    def test_feedback_modes_and_oracle_sanitization(self):
        res = ExecutionResult(
            status="WA",
            passed_tests=1,
            total_tests=3,
            traceback="AssertionError: expected=42 actual=0",
        )
        self.assertEqual(format_feedback_by_mode(res, "F0"), "")
        self.assertEqual(format_feedback_by_mode(res, "F1"), "Execution Status: WA")
        f2 = format_feedback_by_mode(res, "F2")
        self.assertNotIn("42", f2)
        self.assertIn("expected=<REDACTED>", f2)
        f3 = format_feedback_by_mode(res, "F3")
        self.assertIn("Passed 1 / 3 feedback checks.", f3)
        self.assertNotIn("42", f3)

    def test_evaluate_conditional_repair_and_jsonl(self):
        dataset = [
            {"task_id": "p0", "prompt": "q0", "test_list": ["assert True"]},
            {"task_id": "p1", "prompt": "q1", "test_list": ["assert True"]},
            {"task_id": "p2", "prompt": "q2", "test_list": ["assert True"]},
        ]
        trajectories = {
            "q0": [{"result": {"status": "AC"}}],
            "q1": [{"result": {"status": "WA"}}, {"result": {"status": "AC"}}],
            "q2": [{"result": {"status": "RE"}}, {"result": {"status": "WA"}}, {"result": {"status": "WA"}}],
        }

        def fake_loop(model, tokenizer, problem, tests, K=3):
            return trajectories[problem][:K]

        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "eval.jsonl")
            out = evaluate(None, None, dataset, K=3, label="DPO", debug_loop_fn=fake_loop, output_jsonl=jsonl_path)
            self.assertAlmostEqual(out["pass_at_1"], 1 / 3)
            self.assertAlmostEqual(out["fix_at_3"], 2 / 3)
            self.assertAlmostEqual(out["conditional_repair_at_3"], 0.5)
            with open(jsonl_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[1]["solved_turn"], 2)


if __name__ == "__main__":
    unittest.main()
