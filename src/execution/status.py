"""Execution status definitions for code sandbox execution."""

from enum import Enum
from typing import Dict, Tuple

STATUSES: Tuple[str, ...] = ("AC", "PE", "WA", "TLE", "MLE", "CE", "RE")


class ExecutionStatus(str, Enum):
    AC = "AC"    # Accepted / All Correct
    PE = "PE"    # Presentation Error / Partial Execution
    WA = "WA"    # Wrong Answer
    TLE = "TLE"  # Time Limit Exceeded
    MLE = "MLE"  # Memory Limit Exceeded
    CE = "CE"    # Compilation / Syntax Error
    RE = "RE"    # Runtime Error


STATUS_DESCRIPTIONS: Dict[str, str] = {
    "AC": "Accepted - Code passed all test cases.",
    "PE": "Partial Execution - Output format or partial test pass mismatch.",
    "WA": "Wrong Answer - Code produced incorrect output on test cases.",
    "TLE": "Time Limit Exceeded - Code exceeded execution time limit.",
    "MLE": "Memory Limit Exceeded - Code exceeded memory limit.",
    "CE": "Compilation Error - Syntax or parse error in Python code.",
    "RE": "Runtime Error - Exception raised during execution.",
}

# Status hierarchy (lower rank number = better outcome)
STATUS_RANK: Dict[str, int] = {
    "AC": 1,
    "PE": 2,
    "WA": 3,
    "TLE": 4,
    "MLE": 5,
    "CE": 6,
    "RE": 7,
}


def is_valid_status(status: str) -> bool:
    """Check if status code is valid."""
    return status in STATUSES

