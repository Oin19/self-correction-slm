"""Tests for evaluation metrics (Pass@1, Fix@K, error distribution, recovery rate)."""

import unittest
from src.evaluation.metrics import (
    calculate_pass_at_1,
    calculate_fix_at_k,
    calculate_error_distribution,
    calculate_recovery_rate,
    summarize_eval_metrics,
)


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


if __name__ == "__main__":
    unittest.main()
