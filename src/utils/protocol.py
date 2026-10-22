"""Phase 1 Gate G0: Shared Experimental Protocol Freeze.

Defines canonical model identifiers, split definitions, prompt templates,
seeds, turn budgets, and frozen problem ID registries.
"""

from typing import List

# Frozen Model Identifiers
PRIMARY_MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
SECONDARY_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Experimental Budgets & Replications
DEFAULT_SEED = 42
REPLICATION_SEEDS: List[int] = [42, 43, 44]
TURN_BUDGETS: List[int] = [1, 3, 5]

# Dataset Splits & Subsets
APPS_TRAIN_SPLIT = "train[:1500]"
APPS_HELD_OUT_SPLIT = "train[1500:2000]"

# Benchmark Problem Registries
HUMANEVAL_TOTAL_PROBLEMS = 164
FROZEN_HUMANEVAL_IDS: List[str] = [f"HumanEval/{i}" for i in range(HUMANEVAL_TOTAL_PROBLEMS)]

# Feedback Modes (RQ3 Ablation)
FEEDBACK_MODES = {
    "F0": "no_feedback",          # Pure repeated sampling control (B1)
    "F1": "status_only",          # Status classification only (AC, WA, TLE, RE, CE)
    "F2": "sanitized_traceback",  # Traceback with oracle outputs redacted
    "F3": "structured_summary",   # Passed tests count + status
}

# Unified Prompt Templates
PROMPT_TEMPLATE = "### Problem:\n{problem}\n\n### Solution:\n```python\n"
REPAIR_TEMPLATE = (
    "### Problem:\n{problem}\n\n"
    "### Previous Solution:\n```python\n{prev_code}\n```\n\n"
    "### Execution Feedback:\n{feedback}\n\n"
    "### Fixed Solution:\n```python\n"
)


def get_frozen_contract() -> dict:
    """Return dictionary representation of the frozen experimental protocol."""
    return {
        "primary_model": PRIMARY_MODEL_ID,
        "secondary_model": SECONDARY_MODEL_ID,
        "seeds": REPLICATION_SEEDS,
        "turn_budgets": TURN_BUDGETS,
        "apps_train_split": APPS_TRAIN_SPLIT,
        "apps_held_out_split": APPS_HELD_OUT_SPLIT,
        "humaneval_total": HUMANEVAL_TOTAL_PROBLEMS,
        "feedback_modes": FEEDBACK_MODES,
    }
