"""Qualitative Error Taxonomy and Case Study Extraction Engine.

Fulfills Section 6.8 and Section 10 of the AAAI-27 Workshop Research Plan.
Parses multi-turn self-correction traces (natural_failures_1.3b.jsonl or
evaluation JSONL trace archives) and performs:
1. Behavioral Categorization into 5 standardized failure/repair modes:
   - SEMANTIC_RECOVERY: Initial assertion failure (WA) -> Accepted (AC).
   - RUNTIME_RECOVERY: Initial syntax/runtime failure (RE/CE) -> Accepted (AC).
   - REGRESSION: Early turn achieved partial pass, but later edit broke tests.
   - FEEDBACK_MISINTERPRETATION: Diagnostic was ignored or caused unrelated edits.
   - RESISTANT_PLATEAU: Unresolved failure persisting across all turns (e.g. TLE/WA).
2. Quantified Error-Category Frequency Distribution (CSV + JSON).
3. Automated LaTeX Code Listing Generation for Appendix Qualitative Tables.
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# 1. Trajectory Categorization Engine
# ==============================================================================

def classify_trajectory(record: Dict[str, Any]) -> str:
    """Classifies a multi-turn evaluation trace into one of 5 behavioral modes.

    Returns:
        One of: 'SEMANTIC_RECOVERY', 'RUNTIME_RECOVERY', 'REGRESSION',
                'FEEDBACK_MISINTERPRETATION', 'RESISTANT_PLATEAU'.
    """
    history = record.get("turn_history", [])
    if not isinstance(history, list) or len(history) == 0:
        # Fallback if no history
        if record.get("solved_turn") and record["solved_turn"] > 1:
            return "SEMANTIC_RECOVERY"
        return "RESISTANT_PLATEAU"

    statuses = [h.get("status", "") for h in history]
    t0_status = statuses[0] if statuses else ""
    is_solved = any(s == "AC" for s in statuses)

    # 1. Recovery Cases (Failed Turn 0 -> Solved Turn >= 1)
    if is_solved and t0_status != "AC":
        if t0_status in ("RE", "CE"):
            return "RUNTIME_RECOVERY"
        elif t0_status == "WA":
            return "SEMANTIC_RECOVERY"
        else:
            return "SEMANTIC_RECOVERY"

    # 2. Regression Cases: Early turn had partial passes, later turn degraded
    pass_counts = []
    for h in history:
        # Check passed count or test pass fraction
        passed = h.get("tests_passed") or h.get("passed")
        if passed is not None and isinstance(passed, (int, float)):
            pass_counts.append(float(passed))
        elif h.get("status") == "AC":
            pass_counts.append(10.0)
        elif h.get("status") == "WA":
            pass_counts.append(1.0)
        else:
            pass_counts.append(0.0)

    for i in range(len(pass_counts) - 1):
        if pass_counts[i] > 0 and pass_counts[i+1] < pass_counts[i]:
            return "REGRESSION"
        if statuses[i] == "WA" and statuses[i+1] in ("CE", "RE", "TLE"):
            return "REGRESSION"

    # 3. Feedback Misinterpretation:
    # Model received specific error feedback but output completely broke or hallucinated
    for i in range(len(history) - 1):
        curr_fb = history[i].get("feedback", "")
        next_code = history[i+1].get("code", "")
        # If traceback mentioned a line, but code truncated or syntax error resulted
        if "Traceback" in curr_fb or "Error" in curr_fb:
            if statuses[i+1] == "CE":
                return "FEEDBACK_MISINTERPRETATION"
            # If code became drastically shorter (accidental deletion)
            curr_code = history[i].get("code", "")
            if len(curr_code) > 80 and len(next_code) < len(curr_code) * 0.3:
                return "FEEDBACK_MISINTERPRETATION"

    # 4. Resistant Plateau: Never solved, stagnant failure
    return "RESISTANT_PLATEAU"


# ==============================================================================
# 2. Case Study Formatting & LaTeX Generator
# ==============================================================================

def _latex_clean_code(code: str) -> str:
    """Cleans code snippet for LaTeX verbatim/listing environment."""
    if not code:
        return "# No code emitted"
    # Ensure lines do not exceed printable width
    lines = []
    for line in code.splitlines():
        if len(line) > 78:
            lines.append(line[:75] + "...")
        else:
            lines.append(line)
    return "\n".join(lines[:25])  # Cap at 25 lines for appendix compactness


def generate_latex_case_study_block(
    category: str,
    record: Dict[str, Any],
    label: str,
) -> str:
    """Generates a structured LaTeX code block for the qualitative appendix."""
    pid = record.get("problem_id", "Unknown_Problem")
    history = record.get("turn_history", [])
    t0 = history[0] if history else {}
    t_last = history[-1] if len(history) > 1 else t0

    t0_code = _latex_clean_code(t0.get("code", ""))
    t0_status = t0.get("status", "WA")
    feedback = t0.get("feedback", "AssertionError: test failed") or "No feedback recorded"
    feedback = feedback[:200]  # truncate feedback

    final_turn = len(history)
    final_code = _latex_clean_code(t_last.get("code", ""))
    final_status = t_last.get("status", "WA")

    category_title = category.replace("_", " ").title()

    block = f"""% --- Case Study: {category_title} ({pid}) ---
\\begin{{listing}}[t]
\\caption{{{category_title}: Problem \\texttt{{{pid}}}. Initial attempt encountered \\texttt{{{t0_status}}}; repaired at Turn {final_turn} with final status \\texttt{{{final_status}}}.}}
\\label{{lst:case_{label}}}
\\begin{{lstlisting}}[language=Python]
# --- Turn 0 Initial Code (Status: {t0_status}) ---
{t0_code}

# --- Sanitized Execution Feedback ---
# {feedback.replace(chr(10), chr(10) + '# ')}

# --- Final Turn {final_turn} Code (Status: {final_status}) ---
{final_code}
\\end{{lstlisting}}
\\end{{listing}}
"""
    return block


# ==============================================================================
# 3. Batch Analysis & Extraction Engine
# ==============================================================================

def analyze_and_extract_case_studies(
    input_source: str,
    output_dir: str = "./FINAL_RESULTS",
    export_latex: bool = True,
    latex_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Reads problem traces, categorizes trajectories, and exports deliverables."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    # Ingest from file or directory
    if os.path.isfile(input_source):
        with open(input_source, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    elif os.path.isdir(input_source):
        for fname in os.listdir(input_source):
            if fname.endswith(".jsonl"):
                p = os.path.join(input_source, fname)
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                pass

    print(f"Loaded {len(records)} problem evaluation traces from {input_source}.")

    # Categorize traces
    category_counts = {
        "SEMANTIC_RECOVERY": 0,
        "RUNTIME_RECOVERY": 0,
        "REGRESSION": 0,
        "FEEDBACK_MISINTERPRETATION": 0,
        "RESISTANT_PLATEAU": 0,
    }
    categorized_records: Dict[str, List[Dict[str, Any]]] = {k: [] for k in category_counts}

    for r in records:
        cat = classify_trajectory(r)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        categorized_records.setdefault(cat, []).append(r)

    total = sum(category_counts.values()) or 1
    summary_rows = []
    for cat, count in category_counts.items():
        summary_rows.append({
            "Failure Category": cat,
            "Count": count,
            "Percentage": f"{(count / total) * 100:.1f}%",
            "Description": {
                "SEMANTIC_RECOVERY": "Initial WA resolved to AC via execution feedback",
                "RUNTIME_RECOVERY": "Initial RE/CE resolved to AC via exception feedback",
                "REGRESSION": "Negative repair: edit broke previously passing tests",
                "FEEDBACK_MISINTERPRETATION": "Feedback misread: syntax corruption or code wipe",
                "RESISTANT_PLATEAU": "Unresolved complex failure persisting across all turns",
            }.get(cat, ""),
        })

    # Save CSV summary
    csv_path = os.path.join(output_dir, "case_studies_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Failure Category", "Count", "Percentage", "Description"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved failure category distribution: {csv_path}")

    # Save JSON summary
    json_path = os.path.join(output_dir, "case_studies_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_analyzed": len(records),
            "distribution": category_counts,
            "proportions": {k: v / total for k, v in category_counts.items()},
        }, f, indent=2)

    # Export LaTeX Case Studies
    latex_blocks = []
    if export_latex:
        tex_path = latex_output_path or os.path.join(output_dir, "case_studies_qualitative.tex")
        latex_blocks.append("% AAAI-27 Qualitative Case Studies Appendix")
        latex_blocks.append("% Automatically generated by scripts/extract_case_studies.py\n")

        for cat_name in ["SEMANTIC_RECOVERY", "RUNTIME_RECOVERY", "REGRESSION", "FEEDBACK_MISINTERPRETATION", "RESISTANT_PLATEAU"]:
            pool = categorized_records.get(cat_name, [])
            if pool:
                # Select a representative sample with non-empty code
                sample = next((r for r in pool if r.get("turn_history") and len(r["turn_history"]) > 1), pool[0])
                latex_blocks.append(generate_latex_case_study_block(cat_name, sample, cat_name.lower()))

        with open(tex_path, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_blocks))
        print(f"Exported qualitative LaTeX listings: {tex_path}")

    return {
        "total_records": len(records),
        "distribution": category_counts,
        "csv_path": csv_path,
        "json_path": json_path,
        "latex_path": latex_output_path or os.path.join(output_dir, "case_studies_qualitative.tex"),
    }


# ==============================================================================
# 4. CLI Interface
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualitative Error Taxonomy & Case Study Extractor")
    parser.add_argument("--input_source", type=str, default="./FINAL_RESULTS/natural_failures_1.3b.jsonl",
                        help="Path to JSONL file or directory containing evaluation traces")
    parser.add_argument("--output_dir", type=str, default="./FINAL_RESULTS",
                        help="Directory to save case studies and distributions")
    parser.add_argument("--latex_output_path", type=str, default=None,
                        help="Optional specific output path for case_studies_qualitative.tex")
    args = parser.parse_args()

    analyze_and_extract_case_studies(
        input_source=args.input_source,
        output_dir=args.output_dir,
        export_latex=True,
        latex_output_path=args.latex_output_path,
    )
