"""Tests for sandboxed Python executor and status classification."""

import unittest
from src.execution.executor import PythonSandbox
from src.execution.status import ExecutionStatus, STATUSES


class TestExecutor(unittest.TestCase):
    def test_status_definitions(self):
        self.assertIn("AC", STATUSES)
        self.assertIn("WA", STATUSES)
        self.assertIn("TLE", STATUSES)
        self.assertIn("CE", STATUSES)
        self.assertIn("RE", STATUSES)

    def test_executor_syntax_error(self):
        sandbox = PythonSandbox()
        invalid_code = "def foo():\n    return 42 missing_colon"
        res = sandbox.run_single(invalid_code)
        self.assertEqual(res.status, ExecutionStatus.CE)
        self.assertEqual(res.passed_tests, 0)
        self.assertIn("SyntaxError", res.traceback)

    def test_executor_accepted_code(self):
        sandbox = PythonSandbox()
        code = "def add(a, b):\n    return a + b\n"
        test_cases = [
            {"fn_name": "add", "input": [2, 3], "expected": 5},
            {"fn_name": "add", "input": [-1, 1], "expected": 0},
        ]
        res = sandbox.run_tests(code, test_cases)
        self.assertEqual(res.status, ExecutionStatus.AC)
        self.assertEqual(res.passed_tests, 2)
        self.assertEqual(res.total_tests, 2)

    def test_executor_wrong_answer(self):
        sandbox = PythonSandbox()
        code = "def add(a, b):\n    return a - b\n"
        test_cases = [
            {"fn_name": "add", "input": [2, 3], "expected": 5},
        ]
        res = sandbox.run_tests(code, test_cases)
        self.assertEqual(res.status, ExecutionStatus.WA)
        self.assertEqual(res.passed_tests, 0)

    def test_executor_runtime_error(self):
        sandbox = PythonSandbox()
        code = "def divide(a, b):\n    return a / b\n"
        test_cases = [
            {"fn_name": "divide", "input": [10, 0], "expected": 5},
        ]
        res = sandbox.run_tests(code, test_cases)
        self.assertEqual(res.status, ExecutionStatus.RE)
        self.assertEqual(res.passed_tests, 0)

    def test_executor_timeout(self):
        sandbox = PythonSandbox(default_timeout=1.0)
        code = "def infinite_loop():\n    while True:\n        pass\ninfinite_loop()\n"
        res = sandbox.run_single(code)
        self.assertEqual(res.status, ExecutionStatus.TLE)
        self.assertEqual(res.passed_tests, 0)

    def test_io_canonical_ac(self):
        sandbox = PythonSandbox()
        code = "import sys\ndata = sys.stdin.read().split()\nif data:\n    print(int(data[0]) * 2)\n"
        test_cases = [{"input": "5\n", "output": "10\n"}]
        res = sandbox.run_tests(code, test_cases)
        self.assertEqual(res.status, ExecutionStatus.AC)
        self.assertEqual(res.passed_tests, 1)

    def test_io_solve_autocall_ac(self):
        sandbox = PythonSandbox()
        code = "def solve():\n    import sys\n    data = sys.stdin.read().split()\n    if data:\n        print(int(data[0]) + 10)\n"
        test_cases = [{"input": "5\n", "output": "15\n"}]
        res = sandbox.run_tests(code, test_cases)
        self.assertEqual(res.status, ExecutionStatus.AC)
        self.assertEqual(res.passed_tests, 1)

    def test_io_wrong_answer_wa(self):
        sandbox = PythonSandbox()
        code = "import sys\ndata = sys.stdin.read().split()\nif data:\n    print(int(data[0]) * 999)\n"
        test_cases = [{"input": "5\n", "output": "10\n"}]
        res = sandbox.run_tests(code, test_cases)
        self.assertEqual(res.status, ExecutionStatus.WA)
        self.assertEqual(res.passed_tests, 0)

    def test_io_compilation_and_runtime_errors(self):
        sandbox = PythonSandbox()
        ce_code = "def solve():\n    return 42 invalid_syntax"
        re_code = "import sys\nx = 1 / 0"
        test_cases = [{"input": "5\n", "output": "10\n"}]
        res_ce = sandbox.run_tests(ce_code, test_cases)
        res_re = sandbox.run_tests(re_code, test_cases)
        self.assertEqual(res_ce.status, ExecutionStatus.CE)
        self.assertEqual(res_re.status, ExecutionStatus.RE)


if __name__ == "__main__":
    unittest.main()

