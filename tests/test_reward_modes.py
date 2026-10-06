"""Tests for dense vs binary rollout reward modes and DPO test-harness alignment."""

import unittest

from src.execution.executor import ExecutionResult, PythonSandbox
from src.execution.status import ExecutionStatus
from src.rewards.execution_reward import score_rollout_reward
from src.training.dpo import parse_apps_test_cases
from src.utils.repro import set_global_seed
from src.utils.test_cases import normalize_tests


class TestRewardModes(unittest.TestCase):
    def test_dense_partial_credit(self):
        partial = ExecutionResult(status=ExecutionStatus.WA, passed_tests=3, total_tests=5)
        ce = ExecutionResult(status=ExecutionStatus.CE, passed_tests=0, total_tests=5)
        ac = ExecutionResult(status=ExecutionStatus.AC, passed_tests=5, total_tests=5)
        self.assertAlmostEqual(score_rollout_reward(partial, "dense"), 0.6)
        self.assertAlmostEqual(score_rollout_reward(ce, "dense"), -0.2)
        self.assertAlmostEqual(score_rollout_reward(ac, "dense"), 1.0)

    def test_binary_ac_only(self):
        partial = {"status": "WA", "passed_tests": 3, "total_tests": 5}
        ce = {"status": "CE", "passed_tests": 0, "total_tests": 5}
        ac = {"status": "AC", "passed_tests": 5, "total_tests": 5}
        self.assertEqual(score_rollout_reward(partial, "binary"), 0.0)
        self.assertEqual(score_rollout_reward(ce, "binary"), 0.0)
        self.assertEqual(score_rollout_reward(ac, "binary"), 1.0)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            score_rollout_reward({"status": "AC"}, "ternary")

    def test_sandbox_binary_vs_dense_on_partial_suite(self):
        sandbox = PythonSandbox(default_timeout=2.0, max_memory_mb=1024.0)
        tests = [{"assertion": "assert add(1,2) == 3"}] * 5 + [
            {"assertion": "assert add(2,3) == 5"}
        ] * 5
        code = "def add(a,b): return a+b if a==1 else 0"
        result = sandbox.run_tests(code, tests)
        dense = score_rollout_reward(result.to_dict(), "dense")
        binary = score_rollout_reward(result.to_dict(), "binary")
        self.assertAlmostEqual(dense, 0.5)
        self.assertEqual(binary, 0.0)


class TestDpoHarnessAlignment(unittest.TestCase):
    def test_set_global_seed_is_deterministic(self):
        import random
        set_global_seed(123)
        a = random.random()
        set_global_seed(123)
        b = random.random()
        self.assertEqual(a, b)

    def test_parse_apps_uses_normalize_tests(self):
        ex = {
            "test_list": ["assert f(1) == 1", "assert f(2) == 4", "assert f(3) == 9"],
            "test": "assert True",
            "input_output": {
                "inputs": [["1"]],
                "outputs": [["1"]],
                "fn_name": "f",
            },
        }
        self.assertEqual(parse_apps_test_cases(ex), normalize_tests(ex))

    def test_parse_apps_keeps_packed_tests(self):
        ex = {
            "input_output": {
                "inputs": ["3\n1\n2\n3\n"],
                "outputs": ["1\n2\n3\n"],
            }
        }
        cases = parse_apps_test_cases(ex)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].get("packed_tests"), 3)


if __name__ == "__main__":
    unittest.main()
