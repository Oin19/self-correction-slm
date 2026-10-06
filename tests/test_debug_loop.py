"""Tests for multi-turn debug loop controller."""

import unittest
from src.debugging.debug_loop import DebugLoop
from src.execution.executor import PythonSandbox
from src.execution.status import ExecutionStatus


class TestDebugLoop(unittest.TestCase):
    def test_debug_loop_success(self):
        sandbox = PythonSandbox()
        problem = "Write a function add(a, b) that adds two numbers."
        test_cases = [{"fn_name": "add", "input": [2, 3], "expected": 5}]

        def mock_model(prompt: str) -> str:
            if "previous solution failed" in prompt.lower():
                return "```python\ndef add(a, b):\n    return a + b\n```"
            return "```python\ndef add(a, b):\n    return a - b\n```"

        loop = DebugLoop(sandbox=sandbox, max_turns=3)
        session = loop.run_session(problem, test_cases, model_fn=mock_model)

        self.assertTrue(session["solved"])
        self.assertEqual(session["solved_turn"], 2)
        self.assertEqual(session["total_turns"], 2)
        self.assertEqual(session["final_status"], ExecutionStatus.AC)


if __name__ == "__main__":
    unittest.main()
