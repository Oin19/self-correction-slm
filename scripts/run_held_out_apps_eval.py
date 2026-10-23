"""Held-Out APPS In-Domain Evaluation Runner (Gate G0 / RQ6).

Executes multi-turn self-correction evaluation on the frozen held-out APPS split:
    APPS_HELD_OUT_SPLIT = "train[1500:2000]" (500 problems)

Answers Research Question 6 (RQ6):
- In-domain (APPS held-out) vs. Out-of-domain (HumanEval / MBPP) repair transfer.
- Evaluates B0 (greedy K=1), B1 (repeated sampling F0), and B2 (execution feedback F2)
  or fine-tuned/RL checkpoints (SFT, PPO-Dense, PPO-Binary, DPO).
- Streams per-problem records to ./results/apps/ with anti-leakage test splitting.
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from src.evaluation.metrics import (
    compute_bootstrap_ci,
    evaluate,
    mcnemar_test,
)
from src.models.loader import load_model_and_tokenizer
from src.utils.protocol import (
    APPS_HELD_OUT_SPLIT,
    PRIMARY_MODEL_ID,
    TURN_BUDGETS,
)
from src.utils.repro import git_commit_sha, set_global_seed


def load_apps_held_out_dataset():
    """Safely load the held-out APPS partition with dual-URI fallback."""
    print(f"Loading held-out APPS partition: {APPS_HELD_OUT_SPLIT}...", flush=True)
    try:
        return load_dataset("codeparrot/apps", split=APPS_HELD_OUT_SPLIT)
    except Exception as e:
        print(f"codeparrot/apps load failed ({e}), attempting fallback 'apps'...", flush=True)
        return load_dataset("apps", split=APPS_HELD_OUT_SPLIT)


def run_apps_evaluation(
    model_name: str = PRIMARY_MODEL_ID,
    checkpoint_path: Optional[str] = None,
    arm_label: str = "Base",
    k_values: Optional[List[int]] = None,
    feedback_mode: str = "F2",
    output_dir: str = "./results/apps",
    max_samples: Optional[int] = None,
    load_in_4bit: bool = False,
    run_all_baselines: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute APPS held-out evaluation across specified turn budgets and feedback modes.

    Args:
        model_name: Base model identifier or HF repo.
        checkpoint_path: Optional local path to LoRA or fine-tuned checkpoint.
        arm_label: Model arm description (e.g. "B2 (Base + Feedback)", "P1 (PPO-Dense)").
        k_values: Turn budgets (defaults to [1, 3, 5]).
        feedback_mode: Feedback mode for repair turns (F0, F1, F2, F3).
        output_dir: Destination folder for JSONL traces and summary tables.
        max_samples: Optional cap on problem count (e.g. 50 for quick check, None for 500).
        load_in_4bit: Whether to load base weights in NF4 4-bit precision.
        run_all_baselines: If True, evaluates B0 (K=1 greedy), B1 (K in {1,3,5} F0), and B2 (K in {1,3,5} F2).
        seed: Random seed for reproducibility.
    """
    set_global_seed(seed)
    k_values = k_values or TURN_BUDGETS
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================", flush=True)
    print(f"  AAAI-27 Workshop APPS Held-Out In-Domain Evaluation  ", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Model ID         : {model_name}")
    print(f"Checkpoint       : {checkpoint_path or 'None (Base Model)'}")
    print(f"Arm Label        : {arm_label}")
    print(f"APPS Split       : {APPS_HELD_OUT_SPLIT}")
    print(f"Turn Budgets K   : {k_values}")
    print(f"Feedback Mode    : {feedback_mode}")
    print(f"Output Directory : {output_dir}")
    print(f"Seed             : {seed}\n", flush=True)

    # 1. Load Model & Tokenizer
    if checkpoint_path and os.path.isdir(checkpoint_path):
        from src.models.loader import load_canonical_checkpoint
        print(f"Loading trained canonical checkpoint from {checkpoint_path}...", flush=True)
        model, tokenizer = load_canonical_checkpoint(
            checkpoint_path=checkpoint_path,
            base_model_name=model_name,
            load_in_4bit=load_in_4bit,
            is_trainable=False,
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            attach_lora=False,
            is_trainable=False,
        )
    model.eval()

    # 2. Load Held-Out Dataset
    dataset = load_apps_held_out_dataset()
    total_avail = len(dataset)
    num_to_eval = min(total_avail, max_samples) if max_samples else total_avail
    print(f"Dataset successfully loaded. Total available: {total_avail}, Evaluating: {num_to_eval}", flush=True)

    summary_rows = []
    safe_arm = arm_label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").lower()

    if run_all_baselines:
        eval_configs = []
        # B0: Base Greedy (K=1)
        eval_configs.append({"arm": "B0 (Base Greedy)", "k": 1, "fb": "F0", "temp": 0.0, "code": "b0"})
        # B1: Repeated Sampling (K in k_values)
        for k in k_values:
            eval_configs.append({"arm": f"B1 (Repeated Sampling K={k})", "k": k, "fb": "F0", "temp": 1.0, "code": f"b1_k{k}"})
        # B2: Base + Feedback (K in k_values)
        for k in k_values:
            eval_configs.append({"arm": f"B2 (Base + Feedback K={k})", "k": k, "fb": "F2", "temp": 0.2, "code": f"b2_k{k}"})
    else:
        eval_configs = [
            {"arm": f"{arm_label} (K={k})", "k": k, "fb": feedback_mode, "temp": 0.2, "code": f"{safe_arm}_k{k}"}
            for k in k_values
        ]

    for cfg in eval_configs:
        k = cfg["k"]
        fb = cfg["fb"]
        temp = cfg["temp"]
        arm_title = cfg["arm"]
        tag = cfg["code"]

        out_jsonl = os.path.join(output_dir, f"{tag}_apps.jsonl")
        print(f"\n--- Running Evaluation: {arm_title} [K={k}, Mode={fb}, Temp={temp}] ---", flush=True)

        res = evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            K=k,
            label=f"APPS-{tag}",
            feedback_mode=fb,
            output_jsonl=out_jsonl,
            max_samples=max_samples,
            temperature=temp,
            split_feedback_scoring=True,
            skip_empty_tests=True,
        )

        # Compute 95% Bootstrap Confidence Intervals
        rec_list = res.get("records") or []
        if not rec_list and os.path.isfile(out_jsonl):
            with open(out_jsonl, "r", encoding="utf-8") as jf:
                rec_list = [json.loads(line) for line in jf if line.strip()]

        pass1_list = [
            1.0 if (r.get("pass_at_1") or (r.get("turn_history") and r["turn_history"][0].get("status") == "AC")) else 0.0
            for r in rec_list
        ]
        cumk_list = [
            1.0 if (r.get(f"cumulative_success_at_{k}") or (isinstance(r.get("solved_turn"), (int, float)) and r["solved_turn"] <= k) or (r.get("turn_history") and any(h.get("status") == "AC" for h in r["turn_history"][:k]))) else 0.0
            for r in rec_list
        ]
        init_fails = [
            r for r in rec_list
            if not (r.get("pass_at_1") or (r.get("turn_history") and r["turn_history"][0].get("status") == "AC"))
        ]
        condk_list = [
            1.0 if (r.get(f"conditional_repair_at_{k}") or (isinstance(r.get("solved_turn"), (int, float)) and r["solved_turn"] <= k) or (r.get("turn_history") and any(h.get("status") == "AC" for h in r["turn_history"][:k]))) else 0.0
            for r in init_fails
        ]

        p1_m, p1_l, p1_u = compute_bootstrap_ci(pass1_list, ci=0.95, seed=seed)
        cumk_m, cumk_l, cumk_u = compute_bootstrap_ci(cumk_list, ci=0.95, seed=seed)
        condk_m, condk_l, condk_u = compute_bootstrap_ci(condk_list, ci=0.95, seed=seed) if condk_list else (0.0, 0.0, 0.0)

        row = {
            "Model_Arm": arm_title,
            "Benchmark": "APPS (Held-Out train[1500:2000])",
            "K": k,
            "Feedback_Mode": fb,
            "N_Evaluated": res["total"],
            "Pass@1": f"{p1_m:.1%} [{p1_l:.1%}, {p1_u:.1%}]",
            "CumSuccess@K": f"{cumk_m:.1%} [{cumk_l:.1%}, {cumk_u:.1%}]",
            "CondRepair@K": f"{condk_m:.1%} [{condk_l:.1%}, {condk_u:.1%}]" if init_fails else "N/A",
            "Eligible_Repairs": len(init_fails),
            "Successful_Repairs": res["conditional_repair_repaired"],
            "Mean_Repair_Turn": f"{res['mean_repair_turn']:.2f}" if res['mean_repair_turn'] else "N/A",
            "Mean_Tokens": f"{res['mean_tokens_per_problem']:.1f}",
            "Mean_Latency_s": f"{res['mean_latency_per_problem']:.2f}",
            "Raw_Pass@1": p1_m,
            "Raw_CumSuccess@K": cumk_m,
            "Raw_CondRepair@K": condk_m,
            "JSONL_Path": out_jsonl,
        }
        summary_rows.append(row)

    # 3. Export Summary CSV & JSON
    summary_csv = os.path.join(output_dir, "apps_evaluation_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Model_Arm", "Benchmark", "K", "Feedback_Mode", "N_Evaluated",
            "Pass@1", "CumSuccess@K", "CondRepair@K", "Eligible_Repairs",
            "Successful_Repairs", "Mean_Repair_Turn", "Mean_Tokens", "Mean_Latency_s", "JSONL_Path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nAPPS evaluation summary saved to: {summary_csv}")

    summary_json = os.path.join(output_dir, "apps_evaluation_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    # 4. Provenance Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit_sha(),
        "model_id": model_name,
        "checkpoint_path": checkpoint_path,
        "apps_split": APPS_HELD_OUT_SPLIT,
        "k_values": k_values,
        "summary_results": summary_rows,
    }
    manifest_path = os.path.join(output_dir, "apps_eval_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"APPS evaluation manifest locked: {manifest_path}")

    # 5. Display Pretty Table
    print("\n================== APPS IN-DOMAIN EVALUATION SUMMARY ==================", flush=True)
    try:
        import pandas as pd
        df = pd.DataFrame(summary_rows)[fieldnames[:11]]
        print(df.to_string(index=False), flush=True)
    except Exception:
        for r in summary_rows:
            print(f"{r['Model_Arm']} | N={r['N_Evaluated']} | Pass@1={r['Pass@1']} | CumSucc={r['CumSuccess@K']} | CondRepair={r['CondRepair@K']}")

    return {"summary_csv": summary_csv, "summary_json": summary_json, "manifest": manifest_path, "rows": summary_rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-Out APPS In-Domain Evaluation Runner")
    parser.add_argument("--model", type=str, default=PRIMARY_MODEL_ID)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned or RL adapter")
    parser.add_argument("--arm_label", type=str, default="Base")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--feedback_mode", type=str, default="F2")
    parser.add_argument("--output_dir", type=str, default="./results/apps")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--run_all_baselines", action="store_true", help="Run B0, B1, and B2 grids")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_apps_evaluation(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        arm_label=args.arm_label,
        k_values=args.k_values,
        feedback_mode=args.feedback_mode,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        load_in_4bit=args.load_in_4bit,
        run_all_baselines=args.run_all_baselines,
        seed=args.seed,
    )
