import unittest
from src.execution.status import STATUSES, is_valid_status


class TestStatus(unittest.TestCase):
    def test_execution_statuses(self):
        self.assertIn("AC", STATUSES)
        self.assertIn("WA", STATUSES)
        self.assertIn("TLE", STATUSES)
        self.assertIn("RE", STATUSES)
        self.assertTrue(is_valid_status("AC"))
        self.assertFalse(is_valid_status("INVALID"))


if __name__ == "__main__":
    unittest.main()
