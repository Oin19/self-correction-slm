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
