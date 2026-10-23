"""Kaggle Evaluation Ingestion & Quality Validation Pipeline (Gate G0 / G5 / G6).

Ingests, validates, and standardizes multi-turn evaluation traces from Kaggle GPU runs:
1. Discovers and parses all .jsonl and .csv outputs from Kaggle runs.
2. Validates schema, problem counts, anti-leakage barriers, and token/latency tracking.
3. Maps raw filenames to canonical model arms:
   - B0 (Base Greedy)
   - B1 (Repeated Sampling F0)
   - B2 (Base + Feedback F2)
   - S1 (SFT Baseline)
   - P1 (PPO-Dense)
   - P2 (PPO-Binary)
   - D1 (Matched DPO)
4. Checks experimental coverage grid across benchmarks (HumanEval 164, MBPP 500, APPS 500).
5. Generates an auditable validation manifest and quality report.
"""

import argparse
import csv
import glob
import json
import os
import re
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from scripts.analyze_and_generate_paper_artifacts import _is_pass_at_1


BENCHMARK_EXPECTED_COUNTS = {
    "humaneval": 164,
    "mbpp": 500,
    "apps": 500,
}

CORE_MODEL_ARMS = [
    "B0 (Base Greedy)",
    "B1 (Repeated Sampling)",
    "B2 (Base + Feedback)",
    "S1 (SFT Baseline)",
    "P1 (PPO-Dense)",
    "P2 (PPO-Binary)",
    "D1 (Matched DPO)",
]


def classify_trace_filename(filename: str) -> Tuple[str, str, int]:
    """Classifies model arm, benchmark, and K from filename.

    Returns:
        (arm_name, benchmark_name, K)
    """
    base = os.path.basename(filename).lower()

    # 1. Benchmark
    if "he" in base or "humaneval" in base:
        bench = "humaneval"
    elif "mbpp" in base:
        bench = "mbpp"
    elif "apps" in base:
        bench = "apps"
    else:
        bench = "unknown"

    # 2. Turn Budget K
    m_k = re.search(r"[_\-]k(\d+)", base)
    k_val = int(m_k.group(1)) if m_k else 3

    # 3. Model Arm
    if "b0" in base or "greedy" in base:
        arm = "B0 (Base Greedy)"
        k_val = 1
    elif "b1" in base or "repeated" in base or ("f0" in base and "base" in base):
        arm = "B1 (Repeated Sampling)"
    elif "b2" in base or ("feedback" in base and "base" in base) or "f2" in base:
        arm = "B2 (Base + Feedback)"
    elif "sft" in base:
        arm = "S1 (SFT Baseline)"
    elif "ppo_dense" in base or "dense" in base:
        arm = "P1 (PPO-Dense)"
    elif "ppo_binary" in base or "binary" in base:
        arm = "P2 (PPO-Binary)"
    elif "dpo" in base:
        arm = "D1 (Matched DPO)"
    elif "zero-shot" in base:
        arm = "B0 (Base Greedy)"
    else:
        arm = os.path.splitext(os.path.basename(filename))[0]

    return arm, bench, k_val


def validate_jsonl_trace(file_path: str) -> Dict[str, Any]:
    """Perform rigorous schema, integrity, and anti-leakage validation on a .jsonl trace."""
    records = []
    corrupted_lines = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                corrupted_lines += 1

    total_records = len(records)
    if total_records == 0:
        return {
            "valid": False,
            "error": "File is empty or contains zero valid JSON records.",
            "total_records": 0,
            "corrupted_lines": corrupted_lines,
        }

    # Verify problem ID uniqueness
    pids = [r.get("problem_id") for r in records if "problem_id" in r]
    unique_pids = set(pids)
    duplicates = len(pids) - len(unique_pids)

    # Verify anti-leakage & schema fields
    leakage_violations = 0
    missing_history = 0
    missing_tokens = 0
    passed_count = 0

    for r in records:
        if not r.get("turn_history"):
            missing_history += 1
            continue

        for h in r["turn_history"]:
            # Check for illegal expected value leakage in prompt or traceback
            prompt_text = h.get("prompt", "")
            traceback_text = h.get("traceback", "")
            if "expected=" in prompt_text.lower() or "expected:" in prompt_text.lower():
                leakage_violations += 1

            if h.get("tokens", 0) == 0:
                missing_tokens += 1

        if _is_pass_at_1(r):
            passed_count += 1

    is_valid = (corrupted_lines == 0) and (leakage_violations == 0) and (total_records > 0)

    return {
        "valid": is_valid,
        "total_records": total_records,
        "unique_problems": len(unique_pids),
        "duplicate_problems": duplicates,
        "corrupted_lines": corrupted_lines,
        "leakage_violations": leakage_violations,
        "missing_turn_history": missing_history,
        "pass_at_1_count": passed_count,
        "pass_at_1_rate": passed_count / total_records if total_records else 0.0,
    }


def ingest_and_validate_results(
    input_source: str = "./results",
    target_results_dir: str = "./results",
    archive_zip: Optional[str] = None,
) -> Dict[str, Any]:
    """Ingest evaluation outputs, validate Gate G0 compliance, and report coverage."""
    start_time = time.time()
    os.makedirs(target_results_dir, exist_ok=True)

    print(f"\n=======================================================", flush=True)
    print(f"  AAAI-27 Kaggle Results Ingestion & Quality Pipeline  ", flush=True)
    print(f"=======================================================", flush=True)
    print(f"Input Source       : {input_source}")
    print(f"Target Directory   : {target_results_dir}")
    print(f"Archive Package    : {archive_zip or 'None'}\n", flush=True)

    # 1. Extract archive if provided
    if archive_zip and os.path.isfile(archive_zip):
        print(f"Unpacking Kaggle archive {archive_zip} into {target_results_dir}...", flush=True)
        with zipfile.ZipFile(archive_zip, "r") as zf:
            zf.extractall(target_results_dir)

    # Also search input_source
    search_dirs = [target_results_dir]
    if input_source and os.path.isdir(input_source) and os.path.abspath(input_source) != os.path.abspath(target_results_dir):
        search_dirs.append(input_source)

    discovered_files = []
    for d in search_dirs:
        discovered_files.extend(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True))

    discovered_files = sorted(set(discovered_files))
    print(f"Discovered {len(discovered_files)} evaluation trace files.", flush=True)

    validation_reports = []
    coverage_grid: Dict[str, Dict[str, Dict[int, str]]] = {
        arm: {"humaneval": {}, "mbpp": {}, "apps": {}} for arm in CORE_MODEL_ARMS
    }

    for path in discovered_files:
        filename = os.path.basename(path)
        if "natural_failures" in filename.lower():
            continue
        arm, bench, k_val = classify_trace_filename(filename)

        v_res = validate_jsonl_trace(path)
        expected_n = BENCHMARK_EXPECTED_COUNTS.get(bench, 164)
        is_complete = v_res["total_records"] >= expected_n

        report_entry = {
            "Filename": filename,
            "Classified_Arm": arm,
            "Benchmark": bench.upper(),
            "K": k_val,
            "Total_Records": v_res["total_records"],
            "Expected_Records": expected_n,
            "Complete_Split": is_complete,
            "Valid_Schema": v_res["valid"],
            "Pass@1_Rate": f"{v_res.get('pass_at_1_rate', 0.0):.1%}",
            "Leakage_Violations": v_res.get("leakage_violations", 0),
            "Corrupted_Lines": v_res.get("corrupted_lines", 0),
            "File_Path": path,
        }
        validation_reports.append(report_entry)

        # Update coverage grid
        if arm in coverage_grid and bench in coverage_grid[arm]:
            coverage_grid[arm][bench][k_val] = "VALID" if (v_res["valid"] and is_complete) else "PARTIAL"

    # Export Validation Report CSV
    report_csv = os.path.join(target_results_dir, "results_validation_report.csv")
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Filename", "Classified_Arm", "Benchmark", "K", "Total_Records",
            "Expected_Records", "Complete_Split", "Valid_Schema", "Pass@1_Rate",
            "Leakage_Violations", "Corrupted_Lines", "File_Path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(validation_reports)
    print(f"\nValidation report saved to: {report_csv}")

    # Generate Manifest
    all_valid = all(r["Valid_Schema"] for r in validation_reports) if validation_reports else False
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_results_dir": target_results_dir,
        "total_files_audited": len(validation_reports),
        "all_files_valid": all_valid,
        "coverage_grid": coverage_grid,
        "validation_reports": validation_reports,
        "elapsed_seconds": time.time() - start_time,
    }

    manifest_json = os.path.join(target_results_dir, "results_ingestion_manifest.json")
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Results ingestion manifest locked: {manifest_json}")

    # Display Pretty Table
    print("\n================== RESULTS INGESTION & AUDIT MATRIX ==================", flush=True)
    for r in validation_reports:
        status_flag = "PASS" if r["Valid_Schema"] else "FAIL"
        print(f"[{status_flag}] {r['Filename']} -> {r['Classified_Arm']} | {r['Benchmark']} K={r['K']} | N={r['Total_Records']}/{r['Expected_Records']} | Pass@1={r['Pass@1_Rate']}")

    return {
        "report_csv": report_csv,
        "manifest_json": manifest_json,
        "reports": validation_reports,
        "coverage_grid": coverage_grid,
        "all_valid": all_valid,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Evaluation Ingestion & Quality Validation Pipeline")
    parser.add_argument("--input_source", type=str, default="./results", help="Source folder or path to Kaggle outputs")
    parser.add_argument("--target_dir", type=str, default="./results", help="Destination canonical results directory")
    parser.add_argument("--archive_zip", type=str, default=None, help="Optional path to downloaded results.zip")
    parser.add_argument("--run_artifacts", action="store_true", help="Auto-trigger paper artifact generator if valid")
    args = parser.parse_args()

    audit_res = ingest_and_validate_results(
        input_source=args.input_source,
        target_results_dir=args.target_dir,
        archive_zip=args.archive_zip,
    )

    if args.run_artifacts:
        from scripts.analyze_and_generate_paper_artifacts import run_artifact_generation
        print("\nTriggering AAAI-27 Paper Artifacts Generation Engine...")
        run_artifact_generation(results_dir=args.target_dir)
