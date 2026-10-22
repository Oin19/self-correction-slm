"""Phase 2: Baseline Evaluation Runner (B0, B1, B2).

Executes the three canonical baseline arms across benchmarks:
- B0: Base model greedy decoding (K=1, temperature=0.0)
- B1: Base model with F0 (repeated-sampling control, K in {1, 3, 5})
- B2: Base model with F2 (sanitized execution feedback, K in {1, 3, 5})

Saves per-problem JSONL traces to results/baselines/ for statistical auditing.
"""

import argparse
import os
import sys

from datasets import load_dataset

from src.evaluation.metrics import evaluate
from src.models.loader import load_model_and_tokenizer
from src.utils.protocol import (
    APPS_HELD_OUT_SPLIT,
    PRIMARY_MODEL_ID,
    TURN_BUDGETS,
)


def run_all_baselines(
    model_name: str = PRIMARY_MODEL_ID,
    benchmarks: list = None,
    k_values: list = None,
    output_dir: str = "./results/baselines",
    max_samples: int = None,
    load_in_4bit: bool = False,
):
    benchmarks = benchmarks or ["humaneval", "mbpp"]
    k_values = k_values or TURN_BUDGETS
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Base Model {model_name} for Phase 2 Baselines...", flush=True)
    model, tokenizer = load_model_and_tokenizer(model_name=model_name, load_in_4bit=load_in_4bit, attach_lora=False)
    model.eval()

    datasets_map = {}
    if "humaneval" in benchmarks:
        print("Loading HumanEval test split...", flush=True)
        datasets_map["humaneval"] = load_dataset("openai_humaneval", split="test")
    if "mbpp" in benchmarks:
        print("Loading MBPP test split...", flush=True)
        datasets_map["mbpp"] = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    if "apps" in benchmarks:
        print(f"Loading held-out APPS split {APPS_HELD_OUT_SPLIT}...", flush=True)
        datasets_map["apps"] = load_dataset("codeparrot/apps", split=APPS_HELD_OUT_SPLIT)

    summary_grid = []

    for b_name, dset in datasets_map.items():
        print(f"\n================ BENCHMARK: {b_name.upper()} ================", flush=True)

        # 1. B0: Base Greedy (K=1)
        b0_out = os.path.join(output_dir, f"B0_{b_name}_K1.jsonl")
        print(f"\n--- Running B0 (Base Greedy K=1) on {b_name} ---", flush=True)
        b0_res = evaluate(
            model=model,
            tokenizer=tokenizer,
            dataset=dset,
            K=1,
            label=f"B0-Greedy-{b_name}",
            feedback_mode="F0",
            temperature=0.0,
            output_jsonl=b0_out,
            max_samples=max_samples,
        )
        summary_grid.append({
            "Baseline": "B0 (Greedy)",
            "Benchmark": b_name,
            "K": 1,
            "Feedback_Mode": "F0",
            "Pass@1": b0_res["pass_at_1"],
            "CumulativeSuccess@K": b0_res["cumulative_success_at_1"],
            "ConditionalRepair@K": b0_res["conditional_repair_at_1"],
        })

        for k in k_values:
            # 2. B1: Base Repeated-Sampling Control (F0)
            b1_out = os.path.join(output_dir, f"B1_{b_name}_K{k}.jsonl")
            print(f"\n--- Running B1 (F0 No-Feedback Repeated Sampling K={k}) on {b_name} ---", flush=True)
            b1_res = evaluate(
                model=model,
                tokenizer=tokenizer,
                dataset=dset,
                K=k,
                label=f"B1-RepeatedSampling-{b_name}-K{k}",
                feedback_mode="F0",
                temperature=1.0,
                output_jsonl=b1_out,
                max_samples=max_samples,
            )
            summary_grid.append({
                "Baseline": "B1 (Repeated Sampling)",
                "Benchmark": b_name,
                "K": k,
                "Feedback_Mode": "F0",
                "Pass@1": b1_res["pass_at_1"],
                "CumulativeSuccess@K": b1_res[f"cumulative_success_at_{k}"],
                "ConditionalRepair@K": b1_res[f"conditional_repair_at_{k}"],
            })

            # 3. B2: Base with Execution Feedback (F2)
            b2_out = os.path.join(output_dir, f"B2_{b_name}_K{k}.jsonl")
            print(f"\n--- Running B2 (F2 Sanitized Feedback K={k}) on {b_name} ---", flush=True)
            b2_res = evaluate(
                model=model,
                tokenizer=tokenizer,
                dataset=dset,
                K=k,
                label=f"B2-ExecutionFeedback-{b_name}-K{k}",
                feedback_mode="F2",
                temperature=0.2,
                output_jsonl=b2_out,
                max_samples=max_samples,
            )
            summary_grid.append({
                "Baseline": "B2 (Base + Feedback)",
                "Benchmark": b_name,
                "K": k,
                "Feedback_Mode": "F2",
                "Pass@1": b2_res["pass_at_1"],
                "CumulativeSuccess@K": b2_res[f"cumulative_success_at_{k}"],
                "ConditionalRepair@K": b2_res[f"conditional_repair_at_{k}"],
            })

    print("\n================ PHASE 2 BASELINE RESULTS GRID ================", flush=True)
    import pandas as pd
    df = pd.DataFrame(summary_grid)
    print(df.to_string(index=False), flush=True)
    return summary_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Baseline Evaluation Runner")
    parser.add_argument("--model", type=str, default=PRIMARY_MODEL_ID)
    parser.add_argument("--benchmarks", nargs="+", default=["humaneval", "mbpp"])
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--output_dir", type=str, default="./results/baselines")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    run_all_baselines(
        model_name=args.model,
        benchmarks=args.benchmarks,
        k_values=args.k_values,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        load_in_4bit=args.load_in_4bit,
    )
