"""Evaluation metrics and benchmark runner with per-problem JSONL persistence."""

import json
import math
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.debugging.debug_loop import agentic_debug_loop
from src.utils.test_cases import normalize_tests as extract_eval_test_cases
from src.utils.test_cases import split_feedback_and_scoring_tests


def evaluate(
    model,
    tokenizer,
    dataset,
    K: int = 3,
    label: str = "",
    feedback_mode: str = "F2",
    output_jsonl: Optional[str] = None,
    split_feedback_scoring: bool = True,
    debug_loop_fn: Optional[Callable] = None,
    max_samples: Optional[int] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Compute Pass@1, Fix@K, ConditionalRepair@K, and stream per-problem JSONL records.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        dataset: Iterable of evaluation problems (e.g. HumanEval, MBPP, APPS).
        K: Maximum debugging turns budget (K in {1, 3, 5}).
        label: Descriptive label for the evaluation run.
        feedback_mode: One of F0 (no feedback), F1 (status), F2 (sanitized traceback), F3 (structured).
        output_jsonl: Destination file path for streaming per-problem JSONL records.
        split_feedback_scoring: If True, partitions cases into feedback_tests and held-out scoring_tests.
        debug_loop_fn: Custom debug loop function (defaults to agentic_debug_loop).
        max_samples: Optional cap on evaluated problem count.
        temperature: Sampling temperature for initial generation (0.0 for greedy B0).
    """
    debug_loop_fn = debug_loop_fn or agentic_debug_loop
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    pass1_count = 0
    cum_success_k_count = 0
    cond_repair_eligible = 0
    cond_repair_success = 0
    repair_turns: List[int] = []
    all_tokens: List[int] = []
    all_latencies: List[float] = []
    final_statuses: Dict[str, int] = {}
    records: List[Dict[str, Any]] = []

    if output_jsonl:
        out_dir = os.path.dirname(output_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    for i, example in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break

        all_tests = extract_eval_test_cases(example)
        if not all_tests:
            raise ValueError(f"No executable tests for evaluation example {i}; refusing to score it.")

        if split_feedback_scoring and len(all_tests) > 1:
            fb_tests, sc_tests = split_feedback_and_scoring_tests(all_tests)
        else:
            fb_tests, sc_tests = all_tests, all_tests

        problem = example.get("prompt", example.get("question", example.get("text", "")))
        prob_id = example.get("task_id", example.get("problem_id", f"problem_{i}"))

        prob_start = time.time()
        history = debug_loop_fn(
            model=model,
            tokenizer=tokenizer,
            problem=problem,
            test_cases=all_tests,
            K=K,
            feedback_mode=feedback_mode,
            feedback_tests=fb_tests,
            scoring_tests=sc_tests,
            temperature=temperature,
        )
        prob_latency = time.time() - prob_start

        statuses = [h.get("status") or h.get("result", {}).get("status", "UNKNOWN") for h in history]
        turn_tokens = [h.get("tokens", 0) for h in history]
        prob_tokens = sum(turn_tokens)

        # Pass@1: Turn 1 is AC
        pass1 = bool(statuses and statuses[0] == "AC")
        # CumulativeSuccess@K: Any turn <= K is AC
        cum_success_k = ("AC" in statuses)

        # ConditionalRepair@K: Given turn 1 failed, did it recover by turn K?
        cond_repair_k: Optional[bool] = None
        solved_turn: Optional[int] = None
        if not pass1:
            cond_repair_eligible += 1
            cond_repair_k = cum_success_k
            if cum_success_k:
                cond_repair_success += 1

        if cum_success_k:
            solved_turn = statuses.index("AC") + 1
            repair_turns.append(solved_turn)

        final_st = statuses[-1] if statuses else "UNKNOWN"
        final_statuses[final_st] = final_statuses.get(final_st, 0) + 1

        pass1_count += int(pass1)
        cum_success_k_count += int(cum_success_k)
        all_tokens.append(prob_tokens)
        all_latencies.append(prob_latency)

        # Structure per-problem JSONL record
        record = {
            "problem_id": prob_id,
            "label": label,
            "feedback_mode": feedback_mode,
            "K": K,
            "pass_at_1": pass1,
            f"cumulative_success_at_{K}": cum_success_k,
            f"conditional_repair_at_{K}": cond_repair_k,
            "solved_turn": solved_turn,
            "total_turns": len(history),
            "final_status": final_st,
            "total_tokens": prob_tokens,
            "latency_seconds": prob_latency,
            "turn_history": [
                {
                    "turn": h["turn"],
                    "status": h.get("status"),
                    "passed_tests": h.get("result", {}).get("passed_tests", 0),
                    "total_tests": h.get("result", {}).get("total_tests", 0),
                    "tokens": h.get("tokens", 0),
                    "latency": h.get("latency", 0.0),
                }
                for h in history
            ],
        }
        records.append(record)

        # Stream immediately to JSONL
        if output_jsonl:
            with open(output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if (i + 1) % 5 == 0 or (i + 1) == total:
            cur_p1 = pass1_count / (i + 1)
            cur_cum = cum_success_k_count / (i + 1)
            cur_rep = (cond_repair_success / cond_repair_eligible) if cond_repair_eligible else 0.0
            print(
                f"{label} [{i+1}/{total}] Pass@1={cur_p1:.2%} CumSucc@{K}={cur_cum:.2%} "
                f"CondRepair@{K}={cur_rep:.2%} ({cond_repair_success}/{cond_repair_eligible})",
                flush=True,
            )

    pass_at_1_rate = pass1_count / total if total else 0.0
    cum_success_rate = cum_success_k_count / total if total else 0.0
    cond_repair_rate = (cond_repair_success / cond_repair_eligible) if cond_repair_eligible else 0.0
    mean_repair_turn = (sum(repair_turns) / len(repair_turns)) if repair_turns else None

    summary = {
        "label": label,
        "feedback_mode": feedback_mode,
        "K": K,
        "total": total,
        "pass_at_1": pass_at_1_rate,
        f"fix_at_{K}": cum_success_rate,  # backward compatibility alias
        f"cumulative_success_at_{K}": cum_success_rate,
        f"conditional_repair_at_{K}": cond_repair_rate,
        "conditional_repair_eligible": cond_repair_eligible,
        "conditional_repair_repaired": cond_repair_success,
        "mean_repair_turn": mean_repair_turn,
        "mean_tokens_per_problem": (sum(all_tokens) / total) if total else 0.0,
        "mean_latency_per_problem": (sum(all_latencies) / total) if total else 0.0,
        "error_distribution": {k: v / total for k, v in final_statuses.items()},
        "output_jsonl": output_jsonl,
    }
    return summary


def calculate_pass_at_1(results: List[Dict[str, Any]]) -> float:
    """Calculate Pass@1 proportion across problem results."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("status") == "AC" or r.get("pass_at_1", False) or r.get("solved", False)) / len(results)


def calculate_fix_at_k(sessions: List[Dict[str, Any]], k: int) -> float:
    """Calculate CumulativeSuccess@K proportion across sessions."""
    if not sessions:
        return 0.0
    key = f"cumulative_success_at_{k}"
    return sum(
        1 for s in sessions
        if s.get(key, False)
        or (s.get("solved") and s.get("solved_turn", 10**9) <= k)
        or any(t.get("status") == "AC" for t in s.get("history", s.get("turn_history", []))[:k])
    ) / len(sessions)


def calculate_cumulative_success(sessions: List[Dict[str, Any]], k: int) -> float:
    """Explicit alias for calculate_fix_at_k."""
    return calculate_fix_at_k(sessions, k)


def calculate_conditional_repair(sessions: List[Dict[str, Any]], k: int) -> float:
    """Calculate ConditionalRepair@K: fraction of initially buggy problems repaired by turn K."""
    key = f"conditional_repair_at_{k}"
    eligible = [s for s in sessions if (s.get("pass_at_1") is False) or (s.get("history") and s["history"][0].get("status") != "AC")]
    if not eligible:
        return 0.0
    repaired = sum(
        1 for s in eligible
        if s.get(key) is True
        or (s.get("solved") and s.get("solved_turn", 10**9) <= k)
        or any(t.get("status") == "AC" for t in s.get("history", s.get("turn_history", []))[:k])
    )
    return repaired / len(eligible)


def calculate_recovery_rate(sessions: List[Dict[str, Any]]) -> float:
    """Backward compatibility alias for conditional repair rate."""
    return calculate_conditional_repair(sessions, k=5)


def calculate_mean_repair_turn(sessions: List[Dict[str, Any]]) -> Optional[float]:
    """Calculate average turn index among successfully repaired problems."""
    turns = [
        s.get("solved_turn")
        for s in sessions
        if s.get("solved") and s.get("solved_turn") is not None and s.get("solved_turn") > 1
    ]
    return (sum(turns) / len(turns)) if turns else None


def calculate_error_distribution(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate status classification percentage distribution."""
    if not results:
        return {}
    counts: Dict[str, int] = {}
    for r in results:
        st = r.get("status") or r.get("final_status", "UNKNOWN")
        counts[st] = counts.get(st, 0) + 1
    return {st: n / len(results) for st, n in counts.items()}


def summarize_eval_metrics(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compile comprehensive dictionary of evaluation metrics across multi-turn sessions."""
    return {
        "pass_at_1": calculate_pass_at_1(sessions),
        "fix_at_1": calculate_fix_at_k(sessions, 1),
        "fix_at_3": calculate_fix_at_k(sessions, 3),
        "fix_at_5": calculate_fix_at_k(sessions, 5),
        "cumulative_success_at_1": calculate_cumulative_success(sessions, 1),
        "cumulative_success_at_3": calculate_cumulative_success(sessions, 3),
        "cumulative_success_at_5": calculate_cumulative_success(sessions, 5),
        "conditional_repair_at_3": calculate_conditional_repair(sessions, 3),
        "conditional_repair_at_5": calculate_conditional_repair(sessions, 5),
        "recovery_rate": calculate_recovery_rate(sessions),
        "error_distribution": calculate_error_distribution(sessions),
        "total_eval_samples": len(sessions),
    }


def compute_bootstrap_ci(
    data: List[Union[int, float, bool]],
    metric_fn: Optional[Callable[[List[Any]], float]] = None,
    num_samples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence interval.

    Returns:
        (mean, ci_lower, ci_upper)
    """
    if not data:
        return (0.0, 0.0, 0.0)

    numeric_data = [float(x) for x in data]
    estimator = metric_fn or (lambda xs: sum(xs) / len(xs))
    point_estimate = estimator(numeric_data)

    rng = random.Random(seed)
    n = len(numeric_data)
    boot_estimates = []
    for _ in range(num_samples):
        sample = [numeric_data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_estimates.append(estimator(sample))

    boot_estimates.sort()
    lower_idx = int(((1.0 - ci) / 2.0) * num_samples)
    upper_idx = int(((1.0 + ci) / 2.0) * num_samples)
    ci_lower = boot_estimates[max(0, lower_idx)]
    ci_upper = boot_estimates[min(num_samples - 1, upper_idx)]

    return (point_estimate, ci_lower, ci_upper)


def mcnemar_test(
    baseline_successes: List[bool],
    treatment_successes: List[bool],
) -> Dict[str, float]:
    """Paired McNemar test with continuity correction for comparing repair treatments.

    Returns:
        Dictionary containing contingency counts, chi2 statistic, and two-tailed p-value.
    """
    if len(baseline_successes) != len(treatment_successes):
        raise ValueError("Baseline and treatment lists must have identical sample length")

    # b: baseline pass, treatment fail
    # c: baseline fail, treatment pass
    b = sum(1 for base, treat in zip(baseline_successes, treatment_successes) if base and not treat)
    c = sum(1 for base, treat in zip(baseline_successes, treatment_successes) if not base and treat)

    if (b + c) == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}

    # McNemar with continuity correction
    chi2 = ((abs(b - c) - 1.0) ** 2) / (b + c)
    # 1 degree of freedom: survival function approximation via erfc
    p_value = math.erfc(math.sqrt(chi2) / math.sqrt(2))
    return {
        "b": b,
        "c": c,
        "chi2": float(chi2),
        "p_value": float(max(0.0, min(1.0, p_value))),
    }
