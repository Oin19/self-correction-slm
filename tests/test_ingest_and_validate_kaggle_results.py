"""Unit tests for Kaggle evaluation ingestion and validation pipeline (Gate G5 / G6)."""

import json
import os
import tempfile
import unittest

from scripts.ingest_and_validate_kaggle_results import (
    classify_trace_filename,
    ingest_and_validate_results,
    validate_jsonl_trace,
)


class TestIngestAndValidateKaggleResults(unittest.TestCase):
    def test_classify_trace_filename(self):
        # B0 greedy HumanEval
        arm, bench, k = classify_trace_filename("b0_humaneval_k1.jsonl")
        self.assertEqual(arm, "B0 (Base Greedy)")
        self.assertEqual(bench, "humaneval")
        self.assertEqual(k, 1)

        # B1 repeated sampling MBPP K=3
        arm, bench, k = classify_trace_filename("b1_mbpp_k3.jsonl")
        self.assertEqual(arm, "B1 (Repeated Sampling)")
        self.assertEqual(bench, "mbpp")
        self.assertEqual(k, 3)

        # B2 feedback APPS K=5
        arm, bench, k = classify_trace_filename("b2_k5_apps.jsonl")
        self.assertEqual(arm, "B2 (Base + Feedback)")
        self.assertEqual(bench, "apps")
        self.assertEqual(k, 5)

        # PPO-Dense HumanEval K=3
        arm, bench, k = classify_trace_filename("ppo_dense_humaneval_k3.jsonl")
        self.assertEqual(arm, "P1 (PPO-Dense)")
        self.assertEqual(bench, "humaneval")
        self.assertEqual(k, 3)

        # Matched DPO HumanEval K=3
        arm, bench, k = classify_trace_filename("d1_dpo_humaneval_k3.jsonl")
        self.assertEqual(arm, "D1 (Matched DPO)")
        self.assertEqual(bench, "humaneval")
        self.assertEqual(k, 3)

    def test_validate_jsonl_trace_clean(self):
        clean_records = [
            {
                "problem_id": f"HE_{i}",
                "pass_at_1": True,
                "turn_history": [
                    {"turn": 1, "status": "AC", "prompt": "def solve():\n", "tokens": 100, "latency": 0.2}
                ],
            }
            for i in range(10)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "clean_trace.jsonl")
            with open(file_path, "w", encoding="utf-8") as f:
                for r in clean_records:
                    f.write(json.dumps(r) + "\n")

            v = validate_jsonl_trace(file_path)
            self.assertTrue(v["valid"])
            self.assertEqual(v["total_records"], 10)
            self.assertEqual(v["corrupted_lines"], 0)
            self.assertEqual(v["leakage_violations"], 0)
            self.assertEqual(v["pass_at_1_rate"], 1.0)

    def test_validate_jsonl_trace_leakage_and_corruption(self):
        bad_records = [
            # Record with illegal expected= leakage in prompt
            {
                "problem_id": "HE_0",
                "pass_at_1": False,
                "turn_history": [
                    {"turn": 1, "status": "WA", "prompt": "Fix this code: expected=42", "tokens": 100}
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "bad_trace.jsonl")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(bad_records[0]) + "\n")
                f.write("{unclosed_json_line\n")  # Corrupted line

            v = validate_jsonl_trace(file_path)
            self.assertFalse(v["valid"])
            self.assertEqual(v["total_records"], 1)
            self.assertEqual(v["corrupted_lines"], 1)
            self.assertEqual(v["leakage_violations"], 1)

    def test_ingest_and_validate_results_end_to_end(self):
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dst_dir:
                # Create sample trace in src_dir
                sample_path = os.path.join(src_dir, "b2_humaneval_k3.jsonl")
                with open(sample_path, "w", encoding="utf-8") as f:
                    for i in range(5):
                        rec = {
                            "problem_id": f"HumanEval/{i}",
                            "pass_at_1": (i == 0),
                            "turn_history": [
                                {"turn": 1, "status": "AC" if i == 0 else "WA", "prompt": "clean", "tokens": 120}
                            ],
                        }
                        f.write(json.dumps(rec) + "\n")

                res = ingest_and_validate_results(input_source=src_dir, target_results_dir=dst_dir)
                self.assertTrue(os.path.isfile(res["report_csv"]))
                self.assertTrue(os.path.isfile(res["manifest_json"]))
                self.assertEqual(len(res["reports"]), 1)
                r = res["reports"][0]
                self.assertEqual(r["Classified_Arm"], "B2 (Base + Feedback)")
                self.assertEqual(r["Benchmark"], "HUMANEVAL")
                self.assertEqual(r["Total_Records"], 5)
                self.assertTrue(r["Valid_Schema"])


if __name__ == "__main__":
    unittest.main()
