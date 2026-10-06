import unittest

from src.execution.executor import PythonSandbox
from src.rewards.execution_reward import compute_partial_reward
from src.utils.test_cases import normalize_tests, packed_test_count, score_io_output


class PackedSuitePartialCreditTests(unittest.TestCase):
    def test_packed_test_count(self):
        self.assertEqual(packed_test_count("4\n1 2\n3 4\n"), 4)
        self.assertEqual(packed_test_count("1\n5\n"), 1)
        self.assertEqual(packed_test_count("abc\n"), 1)
        self.assertEqual(packed_test_count(""), 1)

    def test_score_full_token_match_uses_packed_total(self):
        passed, total = score_io_output("1 2 3", "1 2 3", packed_tests=4)
        self.assertEqual((passed, total), (4, 4))

    def test_score_line_partial_when_t_matches_line_count(self):
        expected = "1\n2\n3\n4"
        actual = "1\n9\n3\n4"
        passed, total = score_io_output(actual, expected, packed_tests=4)
        self.assertEqual((passed, total), (3, 4))

    def test_score_block_partial_when_blank_lines_split_suite(self):
        expected = "yes\n\nyes\n\nno\n\nyes"
        actual = "yes\n\nno\n\nno\n\nyes"
        passed, total = score_io_output(actual, expected, packed_tests=4)
        self.assertEqual((passed, total), (3, 4))

    def test_score_multi_line_fallback_without_matching_t(self):
        expected = "a\nb\nc"
        actual = "a\nx\nc"
        passed, total = score_io_output(actual, expected, packed_tests=1)
        self.assertEqual((passed, total), (2, 3))

    def test_score_single_line_mismatch(self):
        self.assertEqual(score_io_output("999", "10", packed_tests=1), (0, 1))

    def test_normalize_tags_packed_stdin(self):
        ex = {
            "input_output": {
                "inputs": ["3\n1 2\n3 4\n5 6\n"],
                "outputs": ["1\n2\n3\n"],
                "fn_name": None,
            }
        }
        cases = normalize_tests(ex)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].get("packed_tests"), 3)

    def test_normalize_does_not_tag_fn_cases(self):
        ex = {
            "input_output": {
                "inputs": [[1, 2]],
                "outputs": [3],
                "fn_name": "add",
            }
        }
        cases = normalize_tests(ex)
        self.assertNotIn("packed_tests", cases[0])

    def test_sandbox_packed_partial_reward(self):
        sandbox = PythonSandbox(default_timeout=3.0)
        code = (
            "import sys\n"
            "data = sys.stdin.read().split()\n"
            "t = int(data[0])\n"
            "vals = list(map(int, data[1:]))\n"
            "for i in range(0, len(vals), 2):\n"
            "    a, b = vals[i], vals[i + 1]\n"
            "    # wrong on the second pair only\n"
            "    print(a + b if i == 0 else a - b)\n"
        )
        tests = [
            {
                "input": "3\n1 2\n3 4\n5 6\n",
                "output": "3\n7\n11\n",
                "packed_tests": 3,
            }
        ]
        result = sandbox.run_tests(code, tests)
        self.assertEqual(result.passed_tests, 1)
        self.assertEqual(result.total_tests, 3)
        self.assertNotEqual(result.status, "AC")
        self.assertAlmostEqual(compute_partial_reward(result), 1.0 / 3.0)

    def test_sandbox_packed_full_pass(self):
        sandbox = PythonSandbox(default_timeout=3.0)
        code = (
            "import sys\n"
            "data = sys.stdin.read().split()\n"
            "t = int(data[0])\n"
            "vals = list(map(int, data[1:]))\n"
            "for i in range(0, len(vals), 2):\n"
            "    print(vals[i] + vals[i + 1])\n"
        )
        tests = [
            {
                "input": "3\n1 2\n3 4\n5 6\n",
                "output": "3\n7\n11\n",
                "packed_tests": 3,
            }
        ]
        result = sandbox.run_tests(code, tests)
        self.assertEqual(result.status, "AC")
        self.assertEqual(result.passed_tests, 3)
        self.assertEqual(result.total_tests, 3)
        self.assertEqual(compute_partial_reward(result), 1.0)


if __name__ == "__main__":
    unittest.main()
