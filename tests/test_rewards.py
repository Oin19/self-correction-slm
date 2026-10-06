"""Tests for execution reward functions."""

import unittest
from src.execution.executor import ExecutionResult
from src.execution.status import ExecutionStatus
from src.rewards.execution_reward import (
    compute_binary_reward,
    compute_partial_reward,
    compute_status_aware_reward,
)


class TestRewards(unittest.TestCase):
    def test_binary_reward(self):
        ac_res = ExecutionResult(status=ExecutionStatus.AC, passed_tests=5, total_tests=5)
        wa_res = ExecutionResult(status=ExecutionStatus.WA, passed_tests=2, total_tests=5)

        self.assertEqual(compute_binary_reward(ac_res), 1.0)
        self.assertEqual(compute_binary_reward(wa_res), -0.2)

    def test_partial_reward(self):
        ac_res = ExecutionResult(status=ExecutionStatus.AC, passed_tests=5, total_tests=5)
        partial_res = ExecutionResult(status=ExecutionStatus.WA, passed_tests=3, total_tests=5)
        ce_res = ExecutionResult(status=ExecutionStatus.CE, passed_tests=0, total_tests=5)

        self.assertEqual(compute_partial_reward(ac_res), 1.0)
        self.assertAlmostEqual(compute_partial_reward(partial_res), 0.6)
        self.assertEqual(compute_partial_reward(ce_res), -0.2)

    def test_partial_reward_tle_mle_re_keep_partial_credit(self):
        tle_partial = ExecutionResult(status=ExecutionStatus.TLE, passed_tests=3, total_tests=5)
        mle_partial = ExecutionResult(status=ExecutionStatus.MLE, passed_tests=2, total_tests=4)
        re_partial = ExecutionResult(status=ExecutionStatus.RE, passed_tests=1, total_tests=5)

        self.assertAlmostEqual(compute_partial_reward(tle_partial), 0.6)
        self.assertAlmostEqual(compute_partial_reward(mle_partial), 0.5)
        self.assertAlmostEqual(compute_partial_reward(re_partial), 0.2)

    def test_partial_reward_zero_pass_penalty_split(self):
        wa_zero = ExecutionResult(status=ExecutionStatus.WA, passed_tests=0, total_tests=5)
        tle_zero = ExecutionResult(status=ExecutionStatus.TLE, passed_tests=0, total_tests=5)
        re_zero = ExecutionResult(status=ExecutionStatus.RE, passed_tests=0, total_tests=5)

        self.assertEqual(compute_partial_reward(wa_zero), 0.0)
        self.assertEqual(compute_partial_reward(tle_zero), -0.2)
        self.assertEqual(compute_partial_reward(re_zero), -0.2)

    def test_status_aware_reward(self):
        self.assertEqual(compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.AC)), 1.0)
        self.assertEqual(compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.PE)), 0.5)
        self.assertEqual(compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.WA)), 0.0)
        self.assertEqual(compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.TLE)), -0.1)
        self.assertEqual(compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.CE)), -0.2)
        self.assertEqual(compute_status_aware_reward(ExecutionResult(status=ExecutionStatus.RE)), -0.2)


if __name__ == "__main__":
    unittest.main()
