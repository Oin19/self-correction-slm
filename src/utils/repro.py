"""Reproducibility helpers (no heavy ML deps)."""
import os
import random
import subprocess


def set_global_seed(seed: int) -> None:
    """Seed python/numpy/torch for reproducible rollouts when available."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def git_commit_sha(cwd: str = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or os.getcwd(),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def write_run_manifest(
    output_dir: str,
    run_id: str,
    model_name_or_path: str,
    stage: str,
    seed: int,
    hyperparameters: dict,
    dataset_split: str = "train",
    turn_budget_k: int = 3,
    reward_mode_or_loss: str = "dense",
    parent_checkpoint: str = None,
    additional_metadata: dict = None,
) -> dict:
    """Write standardized 12-field provenance manifest for auditability and paper reproducibility."""
    import json
    import platform
    import time

    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit_sha": git_commit_sha(),
        "model_name_or_path": model_name_or_path,
        "parent_checkpoint": parent_checkpoint or "none",
        "stage": stage,
        "seed": seed,
        "dataset_split": dataset_split,
        "turn_budget_k": turn_budget_k,
        "reward_mode_or_loss": reward_mode_or_loss,
        "hyperparameters": hyperparameters,
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "cuda_available": False,
            "device_count": 0,
        },
    }

    try:
        import torch
        manifest["environment"]["cuda_available"] = torch.cuda.is_available()
        manifest["environment"]["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        manifest["environment"]["torch_version"] = torch.__version__
    except Exception:
        pass

    if additional_metadata:
        manifest["additional_metadata"] = additional_metadata

    manifest_path = os.path.join(output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Run manifest successfully saved -> {manifest_path}", flush=True)
    return manifest
