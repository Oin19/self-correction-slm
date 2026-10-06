import unittest
from src.execution.executor import run_code
from src.rewards.execution_reward import compute_reward, compute_reward_binary


class TestSandbox(unittest.TestCase):
    def test_sandbox_ac(self):
        result = run_code("print(2 + 2)")
        self.assertEqual(result["status"], "AC")
        self.assertIn("4", result["output"])

    def test_sandbox_ce(self):
        result = run_code("print(2 +")
        self.assertEqual(result["status"], "CE")
        self.assertIn("SyntaxError", result["traceback"])

    def test_sandbox_re(self):
        result = run_code("print(1 / 0)")
        self.assertEqual(result["status"], "RE")
        self.assertIn("ZeroDivisionError", result["traceback"])

    def test_sandbox_tle(self):
        result = run_code("import time; time.sleep(5)", timeout=1)
        self.assertEqual(result["status"], "TLE")
        self.assertIn("timed out", result["traceback"])

    def test_compute_reward(self):
        self.assertEqual(compute_reward("AC", 5, 5), 1.0)
        self.assertEqual(compute_reward("WA", 3, 5), 0.6)
        self.assertEqual(compute_reward("CE", 0, 5), -0.2)
        self.assertEqual(compute_reward_binary("AC", 5, 5), 1.0)
        self.assertEqual(compute_reward_binary("WA", 3, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
