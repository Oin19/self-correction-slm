"""Tests for multi-turn debug loop controller."""

import json
import os
import tempfile
import unittest

from src.debugging.debug_loop import DebugLoop
from src.execution.executor import PythonSandbox
from src.execution.status import ExecutionStatus
from src.models.generation import extract_code_block
from src.utils.repro import write_run_manifest


class TestDebugLoop(unittest.TestCase):
    def test_debug_loop_success(self):
        sandbox = PythonSandbox()
        problem = "Write a function add(a, b) that adds two numbers."
        test_cases = [{"fn_name": "add", "input": [2, 3], "expected": 5}]

        def mock_model(prompt: str) -> str:
            if "execution feedback" in prompt.lower() or "previous solution" in prompt.lower():
                return "```python\ndef add(a, b):\n    return a + b\n```"
            return "```python\ndef add(a, b):\n    return a - b\n```"

        loop = DebugLoop(sandbox=sandbox, max_turns=3)
        session = loop.run_session(problem, test_cases, model_fn=mock_model)

        self.assertTrue(session["solved"])
        self.assertEqual(session["solved_turn"], 2)
        self.assertEqual(session["total_turns"], 2)
        self.assertEqual(session["final_status"], ExecutionStatus.AC)

    def test_extract_code_block_ppo_closing_fence(self):
        # PPO response with code ending in closing fence ``` (opening fence was in prompt)
        ppo_output = "def add(a, b):\n    return a + b\n```\nHere is some explanation."
        extracted = extract_code_block(ppo_output)
        self.assertEqual(extracted, "def add(a, b):\n    return a + b")

    def test_extract_code_block_standard_fenced(self):
        fenced = "```python\ndef foo():\n    return 42\n```"
        self.assertEqual(extract_code_block(fenced), "def foo():\n    return 42")

    def test_run_manifest_12_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = write_run_manifest(
                output_dir=tmp_dir,
                run_id="test_run_1",
                model_name_or_path="test-model",
                stage="sft",
                seed=42,
                hyperparameters={"lr": 2e-5, "epochs": 3},
                dataset_split="train[:100]",
                turn_budget_k=3,
                reward_mode_or_loss="cross_entropy",
                parent_checkpoint="none",
            )
            # Verify 12 core provenance fields
            required_fields = [
                "run_id", "timestamp", "git_commit_sha", "model_name_or_path",
                "parent_checkpoint", "stage", "seed", "dataset_split",
                "turn_budget_k", "reward_mode_or_loss", "hyperparameters", "environment"
            ]
            for field in required_fields:
                self.assertIn(field, manifest, f"Missing required manifest field: {field}")

            # Verify saved file on disk
            with open(os.path.join(tmp_dir, "run_manifest.json"), encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved["run_id"], "test_run_1")


if __name__ == "__main__":
    unittest.main()
