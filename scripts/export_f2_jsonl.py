"""Materialize results/ablation_f2_traceback.jsonl (K=3) from results/ablation_k7_trajectory.jsonl."""

import json
import os
import sys


def export_f2_from_k7(
    k7_path: str = "./results/ablation_k7_trajectory.jsonl",
    f2_path: str = "./results/ablation_f2_traceback.jsonl",
) -> int:
    if not os.path.isfile(k7_path):
        raise FileNotFoundError(f"Could not find {k7_path}. Unzip notebook07_final_results.zip into ./results first.")
    count = 0
    with open(k7_path, "r", encoding="utf-8") as fin, open(f2_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            st3 = rec.get("statuses", [])[:3]
            ac3 = "AC" in st3
            f2_rec = {
                **rec,
                "label": "F2 Sanitized Traceback Feedback",
                "K": 3,
                "solved": ac3,
                "solved_turn": (st3.index("AC") + 1) if ac3 else None,
                "statuses": st3,
                "final_status": st3[-1] if st3 else "CE",
                "history": rec.get("history", [])[:3],
            }
            fout.write(json.dumps(f2_rec) + "\n")
            count += 1
    print(f"Wrote {count} K=3 F2 per-problem records to {f2_path}")
    return count


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "./results/ablation_k7_trajectory.jsonl"
    dst = sys.argv[2] if len(sys.argv) > 2 else "./results/ablation_f2_traceback.jsonl"
    export_f2_from_k7(src, dst)
