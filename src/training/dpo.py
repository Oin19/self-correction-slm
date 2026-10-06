"""DPO training entry points."""

import json
import os
from typing import TYPE_CHECKING, List, Optional

from src.debugging.debug_loop import agentic_debug_loop
from src.utils.repro import git_commit_sha, set_global_seed
from src.utils.test_cases import normalize_tests

if TYPE_CHECKING:
    from datasets import Dataset


def parse_apps_test_cases(prob: dict) -> list:
    """Extract executor-ready tests via the same normalizer PPO/eval use."""
    return normalize_tests(prob)


def make_preference_pairs(problems, model, tokenizer, K: int = 3, seed: int = 42,
                          checkpoint_path: Optional[str] = None, max_gen_time: Optional[float] = None):
    """Collect (chosen, rejected) preference pairs from execution debug loop rollouts.

    When ``checkpoint_path`` is set, each problem's outcome (pair or ``None``) is
    appended to a JSONL file immediately after it is processed, so an interrupted
    or frozen run resumes without recomputing finished problems.
    """
    from datasets import Dataset

    set_global_seed(seed)
    pairs: List[dict] = []
    processed: dict = {}

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # interrupted mid-write: that problem is reprocessed
                processed[entry["idx"]] = entry.get("pair")
        pairs = [p for p in processed.values() if p]
        print(f"Resuming from {checkpoint_path}: {len(processed)} problems already "
              f"processed, {len(pairs)} pairs recovered.", flush=True)

    total = len(problems)
    print(f"Starting DPO preference pair generation across {total} APPS problems (seed={seed})...", flush=True)

    ck = open(checkpoint_path, "a", encoding="utf-8") if checkpoint_path else None
    if ck and os.path.getsize(checkpoint_path) > 0:
        with open(checkpoint_path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                ck.write("\n")  # interrupted write left no newline: don't merge records
    try:
        for idx, prob in enumerate(problems):
            if idx in processed:
                continue
            question = prob.get("question", prob.get("prompt", ""))
            test_cases = parse_apps_test_cases(prob)
            pair = None
            if test_cases:
                history = agentic_debug_loop(model, tokenizer, question, test_cases,
                                             K=K, max_gen_time=max_gen_time)

                ac_turns = [h for h in history if h["result"]["status"] == "AC"]
                bad_turns = [h for h in history if h["result"]["status"] in ("CE", "RE", "WA", "TLE", "MLE")]

                if ac_turns and bad_turns:
                    pair = {
                        "prompt": question,
                        "chosen": ac_turns[0]["code"],
                        "rejected": bad_turns[0]["code"],
                    }
                # No reference-solution fallback: DPO pairs must come from the model's
                # own execution-grounded rollouts so the comparison remains methodologically clean.

            processed[idx] = pair
            if pair:
                pairs.append(pair)
            if ck:
                ck.write(json.dumps({"idx": idx, "pair": pair}) + "\n")
                ck.flush()

            if (idx + 1) % 5 == 0 or (idx + 1) == total:
                print(f"   [Progress: {idx + 1}/{total}] Generated {len(pairs)} preference pairs...", flush=True)
    finally:
        if ck:
            ck.close()

    return Dataset.from_list(pairs)


def run_dpo_training(
    model,
    tokenizer,
    preference_data,
    output_dir: str = "./checkpoints/dpo",
    beta: float = 0.1,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    seed: int = 42,
    commit_sha: str = "",
    preference_split: str = "",
    debug_turns_k: int = 0,
):
    """Run Direct Preference Optimization (DPO) training."""
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        raise ImportError(f"TRL library is required for DPO training. Install with `pip install trl`: {e}")

    set_global_seed(seed)
    print(f"DPO seed={seed} commit={commit_sha or git_commit_sha() or 'unknown'}", flush=True)

    try:
        dpo_config = DPOConfig(
            beta=beta,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=output_dir,
            fp16=True,
            report_to="none",
        )
    except Exception:
        dpo_config = DPOConfig(
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            output_dir=output_dir,
            fp16=True,
            report_to="none",
        )

    # Compatible with TRL versions using processing_class or tokenizer
    try:
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=preference_data,
            processing_class=tokenizer,
        )
    except (TypeError, Exception):
        try:
            dpo_trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=dpo_config,
                train_dataset=preference_data,
                tokenizer=tokenizer,
            )
        except Exception:
            dpo_trainer = DPOTrainer(
                model=model,
                args=dpo_config,
                train_dataset=preference_data,
                processing_class=tokenizer,
            )

    dpo_trainer.train()
    final_dir = f"{output_dir}/final"
    dpo_trainer.save_model(final_dir)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(final_dir)
    with open(f"{final_dir}/dpo_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "checkpoint_type": "dpo_execution_grounded",
            "preference_source": "model_generated_execution_rollouts",
            "reference_solution_fallback": False,
            "test_harness": "normalize_tests",
            "seed": seed,
            "commit_sha": commit_sha or git_commit_sha(),
            "beta": beta,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "num_preference_pairs": len(preference_data),
            "preference_split": preference_split,
            "debug_turns_k": debug_turns_k,
        }, f, indent=2)
    return dpo_trainer
