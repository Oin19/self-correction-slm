"""Auto-checkpointing and session duration safety utilities."""

import os
import threading
import time


def auto_checkpoint(trainer, save_dir: str, interval_min: int = 30):
    """Saves checkpoints on a background thread — non-blocking."""
    os.makedirs(save_dir, exist_ok=True)

    def _loop():
        while True:
            time.sleep(interval_min * 60)
            ckpt = f"{save_dir}/auto_{int(time.time())}"
            try:
                trainer.save_model(ckpt)
                print(f"Checkpoint saved -> {ckpt} at {time.strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"Failed to auto-save checkpoint: {e}")

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    print(f"Auto-checkpoint enabled every {interval_min} min.")


def guard_session_limit(trainer, save_dir: str, session_start: float, limit_hours: float = 11.5) -> bool:
    """Force-saves near the 12h session limit to prevent data loss."""
    elapsed = (time.time() - session_start) / 3600
    if elapsed >= limit_hours:
        trainer.save_model(f"{save_dir}/forced_final")
        print(f"FORCED SAVE at {elapsed:.1f}h — session near limit!")
        return True
    return False
