"""Tests for synthetic error injection engine."""

import unittest
from src.error_injection import BugInjector, BugType
from src.execution.executor import PythonSandbox
from src.execution.status import ExecutionStatus


class TestErrorInjection(unittest.TestCase):
    def test_syntax_bug_injection(self):
        injector = BugInjector(seed=42)
        code = "def is_even(n):\n    if n % 2 == 0:\n        return True\n    return False\n"
        result = injector.inject_bug(code, category=BugType.SYNTAX)
        self.assertEqual(result["bug_type"], "syntax")
        sandbox = PythonSandbox()
        res = sandbox.run_single(result["buggy_code"])
        self.assertEqual(res.status, ExecutionStatus.CE)

    def test_logic_bug_injection(self):
        injector = BugInjector(seed=42)
        code = "def add(a, b):\n    return a + b\n"
        result = injector.inject_bug(code, category=BugType.LOGIC)
        self.assertEqual(result["bug_type"], "logic")
        self.assertTrue("-" in result["buggy_code"] or "return 0" in result["buggy_code"])

    def test_runtime_bug_injection(self):
        injector = BugInjector(seed=42)
        code = "def get_item(items, idx):\n    return items[idx]\n"
        result = injector.inject_bug(code, category=BugType.RUNTIME)
        self.assertEqual(result["bug_type"], "runtime")
        sandbox = PythonSandbox()
        test_cases = [{"fn_name": "get_item", "input": [[1, 2], 0], "expected": 1}]
        res = sandbox.run_tests(result["buggy_code"], test_cases)
        self.assertIn(res.status, (ExecutionStatus.RE, ExecutionStatus.CE))

    def test_infinite_loop_injection(self):
        injector = BugInjector(seed=42)
        code = "def compute_sum(n):\n    total = 0\n    while n > 0:\n        total += n\n        n -= 1\n    return total\n"
        result = injector.inject_bug(code, category=BugType.INFINITE_LOOP)
        self.assertEqual(result["bug_type"], "infinite_loop")
        sandbox = PythonSandbox(default_timeout=1.0)
        test_cases = [{"fn_name": "compute_sum", "input": [5], "expected": 15}]
        res = sandbox.run_tests(result["buggy_code"], test_cases)
        self.assertEqual(res.status, ExecutionStatus.TLE)

    def test_dataset_augmentation(self):
        injector = BugInjector(seed=42)
        dataset = [
            {"id": 1, "solution": "def f(): return 1"},
            {"id": 2, "solution": "def g(x): return x * 2"},
        ]
        augmented = injector.create_buggy_dataset(dataset)
        self.assertEqual(len(augmented), 2)
        self.assertIn("buggy_code", augmented[0])
        self.assertIn("bug_type", augmented[0])


if __name__ == "__main__":
    unittest.main()
