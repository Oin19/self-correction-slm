"""Evaluation metrics and benchmark runner."""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from src.debugging.debug_loop import (
    agentic_debug_loop,
    agentic_loop_no_feedback,
    agentic_loop_repeated_sampling,
    batch_agentic_debug_loop,
)
from src.utils.test_cases import normalize_tests as extract_eval_test_cases


def _format_problem_prompt(example: dict) -> str:
    """Extract problem statement and include required function signature for MBPP-style items."""
    problem = example.get("prompt", example.get("question", example.get("text", "")))
    if "prompt" not in example and "text" in example:
        code_ref = example.get("code", "")
        test_list = example.get("test_list") or []
        target_fn = None
        if test_list and isinstance(test_list[0], str):
            m = re.search(r"assert\s+.*?([A-Za-z_][A-Za-z0-9_]*)\s*\(", test_list[0])
            if m:
                cand = m.group(1)
                if cand not in {"set", "list", "tuple", "dict", "len", "abs", "round", "sorted", "math"}:
                    target_fn = cand
        sig_line = None
        for line in code_ref.splitlines():
            s = line.strip()
            if s.startswith("def ") and s.endswith(":"):
                if target_fn is None or f"def {target_fn}(" in s:
                    sig_line = s
                    break
                sig_line = s
        if sig_line and sig_line not in problem:
            problem = f"{problem.rstrip()}\n\nRequired function signature: `{sig_line}`"
    return problem


def evaluate(
    model,
    tokenizer,
    dataset,
    K: int = 3,
    label: str = "",
    debug_loop_fn=None,
    output_jsonl: Optional[str] = None,
    batch_size: int = 16,
    max_new_tokens: int = 320,
    feedback_mode: str = "F2",
    resume: bool = False,
) -> dict:
    """Compute Pass@1, Fix@K, and ConditionalRepair@K using batched GPU rollouts and executable tests."""
    total = len(dataset)
    pass1 = fixk = 0
    fix_at = {1: 0, 3: 0, 5: 0, 7: 0, K: 0}
    initial_failures = 0
    repaired_by = {1: 0, 3: 0, 5: 0, 7: 0, K: 0}
    records: List[Dict[str, Any]] = []
    completed_ids: Dict[str, Dict[str, Any]] = {}

    if output_jsonl:
        out_dir = os.path.dirname(os.path.abspath(output_jsonl))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if resume and os.path.exists(output_jsonl):
            with open(output_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        completed_ids[str(rec["problem_id"])] = rec
                    except Exception:
                        continue
            if completed_ids:
                print(f"{label}: Resumed {len(completed_ids)}/{total} already-evaluated problems from {output_jsonl}", flush=True)
        elif os.path.exists(output_jsonl):
            os.remove(output_jsonl)

    def _accumulate_record(rec: Dict[str, Any]) -> None:
        nonlocal pass1, fixk, initial_failures
        statuses = rec.get("statuses") or [h["result"]["status"] for h in rec.get("history", [])]
        first_ac = bool(statuses) and statuses[0] == "AC"
        any_ac = "AC" in statuses
        pass1 += int(first_ac)
        fixk += int(any_ac)
        for k_val in list(fix_at.keys()):
            fix_at[k_val] += int("AC" in statuses[:k_val])
        if not first_ac:
            initial_failures += 1
            for k_val in list(repaired_by.keys()):
                repaired_by[k_val] += int("AC" in statuses[:k_val])
        records.append(rec)

    # Determine whether we can run fast batched GPU evaluation
    use_batched = debug_loop_fn in (
        None,
        agentic_debug_loop,
        agentic_loop_repeated_sampling,
        agentic_loop_no_feedback,
    )
    eff_fb_mode = "F0" if debug_loop_fn is agentic_loop_repeated_sampling else feedback_mode
    eff_fake_fb = debug_loop_fn is agentic_loop_no_feedback
    eff_batch_size = max(1, int(batch_size)) if use_batched else 1

    pending_items: List[Dict[str, Any]] = []
    for i, example in enumerate(dataset):
        tests = extract_eval_test_cases(example)
        if not tests:
            raise ValueError(f"No executable tests for evaluation example {i}; refusing to score it.")
        problem = _format_problem_prompt(example)
        problem_id = example.get("task_id", example.get("problem_id", i))
        pid_key = str(problem_id)
        if pid_key in completed_ids:
            _accumulate_record(completed_ids[pid_key])
        else:
            pending_items.append({"idx": i, "problem_id": problem_id, "problem": problem, "tests": tests})

    for start_idx in range(0, len(pending_items), eff_batch_size):
        chunk = pending_items[start_idx : start_idx + eff_batch_size]
        t0 = time.time()
        if use_batched:
            histories = batch_agentic_debug_loop(
                model=model,
                tokenizer=tokenizer,
                problems=[it["problem"] for it in chunk],
                test_cases_list=[it["tests"] for it in chunk],
                K=K,
                max_new_tokens=max_new_tokens,
                feedback_mode=eff_fb_mode,
                fake_feedback=eff_fake_fb,
            )
        else:
            histories = [
                debug_loop_fn(model, tokenizer, it["problem"], it["tests"], K=K)
                for it in chunk
            ]
        batch_elapsed = time.time() - t0
        per_item_latency = round(batch_elapsed / max(1, len(chunk)), 4)

        new_recs: List[Dict[str, Any]] = []
        for it, history in zip(chunk, histories):
            statuses = [h["result"]["status"] for h in history]
            first_ac = bool(statuses) and statuses[0] == "AC"
            any_ac = "AC" in statuses
            rec = {
                "problem_id": it["problem_id"],
                "label": label,
                "K": K,
                "pass_at_1": first_ac,
                "solved": any_ac,
                "solved_turn": (statuses.index("AC") + 1) if any_ac else None,
                "statuses": statuses,
                "final_status": statuses[-1] if statuses else "CE",
                "latency_sec": per_item_latency,
                "history": history,
            }
            _accumulate_record(rec)
            new_recs.append(rec)

        if output_jsonl and new_recs:
            with open(output_jsonl, "a", encoding="utf-8") as f:
                for rec in new_recs:
                    f.write(json.dumps(rec) + "\n")

        done_n = len(records)
        cond_rep = (repaired_by[K] / initial_failures) if initial_failures else 0.0
        print(
            f"{label} [{done_n}/{total}] Pass@1={pass1/done_n:.2%} "
            f"Fix@{K}={fixk/done_n:.2%} CondRepair@{K}={cond_rep:.2%} "
            f"({batch_elapsed:.1f}s/{len(chunk)} probs)",
            flush=True,
        )

    return {
        "label": label,
        "pass_at_1": pass1 / total if total else 0.0,
        f"fix_at_{K}": fixk / total if total else 0.0,
        "fix_at_1": fix_at[1] / total if total else 0.0,
        "fix_at_3": fix_at[3] / total if total else 0.0,
        "fix_at_5": fix_at[5] / total if total else 0.0,
        "fix_at_7": fix_at[7] / total if total else 0.0,
        f"conditional_repair_at_{K}": (repaired_by[K] / initial_failures) if initial_failures else 0.0,
        "conditional_repair_at_3": (repaired_by[3] / initial_failures) if initial_failures else 0.0,
        "conditional_repair_at_5": (repaired_by[5] / initial_failures) if initial_failures else 0.0,
        "conditional_repair_at_7": (repaired_by[7] / initial_failures) if initial_failures else 0.0,
        "initial_failures": initial_failures,
        "total": total,
        "records": records,
    }


def calculate_pass_at_1(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("status") == "AC" or r.get("solved", False)) / len(results)


def calculate_fix_at_k(sessions: List[Dict[str, Any]], k: int) -> float:
    if not sessions:
        return 0.0
    return sum(
        1
        for s in sessions
        if (s.get("solved") and s.get("solved_turn", 10**9) <= k)
        or any(t.get("status") == "AC" for t in s.get("history", [])[:k])
    ) / len(sessions)


def calculate_error_distribution(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not results:
        return {}
    counts: Dict[str, int] = {}
    for r in results:
        st = r.get("status") or r.get("final_status", "UNKNOWN")
        counts[st] = counts.get(st, 0) + 1
    return {st: n / len(results) for st, n in counts.items()}


def calculate_recovery_rate(sessions: List[Dict[str, Any]]) -> float:
    buggy = [s for s in sessions if s.get("history") and s["history"][0].get("status") != "AC"]
    return sum(1 for s in buggy if s.get("solved", False)) / len(buggy) if buggy else 0.0


def summarize_eval_metrics(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "pass_at_1": calculate_pass_at_1(sessions),
        "fix_at_1": calculate_fix_at_k(sessions, 1),
        "fix_at_3": calculate_fix_at_k(sessions, 3),
        "fix_at_5": calculate_fix_at_k(sessions, 5),
        "recovery_rate": calculate_recovery_rate(sessions),
        "error_distribution": calculate_error_distribution(sessions),
        "total_eval_samples": len(sessions),
    }
