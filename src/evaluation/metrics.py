"""Evaluation metrics and benchmark runner."""

from typing import Any, Dict, List
from src.debugging.debug_loop import agentic_debug_loop
from src.utils.test_cases import normalize_tests as extract_eval_test_cases


def evaluate(model, tokenizer, dataset, K: int = 3, label: str = "", debug_loop_fn=None) -> dict:
    """Compute Pass@1 and Fix@K using explicit executable tests."""
    debug_loop_fn = debug_loop_fn or agentic_debug_loop
    total = len(dataset)
    pass1 = fixk = 0
    for i, example in enumerate(dataset):
        tests = extract_eval_test_cases(example)
        if not tests:
            raise ValueError(f"No executable tests for evaluation example {i}; refusing to score it.")
        problem = example.get("prompt", example.get("question", example.get("text", "")))
        history = debug_loop_fn(model, tokenizer, problem, tests, K=K)
        statuses = [h["result"]["status"] for h in history]
        pass1 += int(bool(statuses) and statuses[0] == "AC")
        fixk += int("AC" in statuses)
        if (i + 1) % 5 == 0 or i + 1 == total:
            print(f"{label} [{i+1}/{total}] Pass@1={pass1/(i+1):.2%} Fix@{K}={fixk/(i+1):.2%}")
    return {"label": label, "pass_at_1": pass1/total if total else 0.0, f"fix_at_{K}": fixk/total if total else 0.0, "total": total}


def calculate_pass_at_1(results: List[Dict[str, Any]]) -> float:
    if not results: return 0.0
    return sum(1 for r in results if r.get("status") == "AC" or r.get("solved", False)) / len(results)


def calculate_fix_at_k(sessions: List[Dict[str, Any]], k: int) -> float:
    if not sessions: return 0.0
    return sum(1 for s in sessions if (s.get("solved") and s.get("solved_turn", 10**9) <= k) or any(t.get("status") == "AC" for t in s.get("history", [])[:k])) / len(sessions)


def calculate_error_distribution(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results: return {}
    counts: Dict[str, int] = {}
    for r in results:
        st = r.get("status") or r.get("final_status", "UNKNOWN")
        counts[st] = counts.get(st, 0) + 1
    return {st: n / len(results) for st, n in counts.items()}


def calculate_recovery_rate(sessions: List[Dict[str, Any]]) -> float:
    buggy = [s for s in sessions if s.get("history") and s["history"][0].get("status") != "AC"]
    return sum(1 for s in buggy if s.get("solved", False)) / len(buggy) if buggy else 0.0


def summarize_eval_metrics(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"pass_at_1": calculate_pass_at_1(sessions), "fix_at_1": calculate_fix_at_k(sessions, 1), "fix_at_3": calculate_fix_at_k(sessions, 3), "fix_at_5": calculate_fix_at_k(sessions, 5), "recovery_rate": calculate_recovery_rate(sessions), "error_distribution": calculate_error_distribution(sessions), "total_eval_samples": len(sessions)}
