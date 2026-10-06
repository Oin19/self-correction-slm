import unittest

from src.execution.executor import PythonSandbox
from src.rewards.execution_reward import compute_partial_reward


class ExecutionRewardRegressionTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = PythonSandbox(default_timeout=2.0, max_memory_mb=1024.0)
        self.tests = [{"assertion": "assert add(1,2) == 3"}] * 5 + [
            {"assertion": "assert add(2,3) == 5"}
        ] * 5

    def test_dense_reward_full_partial_and_compile_failure(self):
        cases = {
            "full": "def add(a,b): return a+b",
            "partial": "def add(a,b): return a+b if a==1 else 0",
            "broken": "def add(a,b): return a+",
        }
        expected = {"full": 1.0, "partial": 0.5, "broken": -0.2}
        for name, code in cases.items():
            result = self.sandbox.run_tests(code, self.tests)
            reward = compute_partial_reward(result)
            self.assertAlmostEqual(reward, expected[name], places=6, msg=name)

    def test_zero_pass_nonfatal_result_is_zero(self):
        result = {"status": "WA", "passed_tests": 0, "total_tests": 10}
        self.assertEqual(compute_partial_reward(result), 0.0)

    def test_empty_suite_is_not_success(self):
        result = self.sandbox.run_tests("print('hello')", [])
        self.assertNotEqual(result.status, "AC")
        self.assertEqual(compute_partial_reward(result), -0.2)

    def test_distinct_tests_not_duplicated_by_normalizer(self):
        from src.utils.test_cases import normalize_tests

        ex = {
            "input_output": {
                "inputs": [["1"], ["2"]],
                "outputs": [["1"], ["2"]],
                "fn_name": "solve",
            }
        }
        cases = normalize_tests(ex)
        self.assertEqual(len(cases), 2)
        self.assertNotEqual(cases[0], cases[1])

    def test_single_input_output_case_stays_single(self):
        from src.utils.test_cases import normalize_tests

        ex = {
            "input_output": {
                "inputs": [["1"]],
                "outputs": [["1"]],
                "fn_name": "solve",
            }
        }
        cases = normalize_tests(ex)
        self.assertEqual(len(cases), 1)

    def test_prefers_multi_case_over_monolithic_test_script(self):
        from src.utils.test_cases import normalize_tests

        ex = {
            "test": "assert solve(1) == 1\nassert solve(2) == 2",
            "input_output": {
                "inputs": [["1"], ["2"], ["3"]],
                "outputs": [["1"], ["2"], ["3"]],
                "fn_name": "solve",
            },
        }
        cases = normalize_tests(ex)
        self.assertEqual(len(cases), 3)

    def test_prefers_longer_test_list_over_single_io_case(self):
        from src.utils.test_cases import normalize_tests

        ex = {
            "test": "assert True",
            "test_list": ["assert f(1) == 1", "assert f(2) == 4", "assert f(3) == 9"],
            "input_output": {
                "inputs": [["1"]],
                "outputs": [["1"]],
                "fn_name": "f",
            },
        }
        cases = normalize_tests(ex)
        self.assertEqual(len(cases), 3)
        self.assertIn("assertion", cases[0])


if __name__ == "__main__":
    unittest.main()
