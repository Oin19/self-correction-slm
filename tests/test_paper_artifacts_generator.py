"""Unit tests for AAAI-27 paper analysis and artifact generation engine."""

import json
import os
import tempfile
import unittest

from scripts.analyze_and_generate_paper_artifacts import (
    compute_bootstrap_ci,
    discover_evaluation_traces,
    extract_natural_failures,
    generate_figure_1_repair_curves,
    generate_figure_2_failure_transitions,
    generate_figure_3_efficiency_pareto,
    generate_table_1_protocol,
    generate_table_2_main_results,
    generate_table_3_feedback_ablation,
    generate_table_4_hypothesis_testing,
    holm_bonferroni_correction,
    mcnemar_test,
    run_artifact_generation,
)


class TestPaperArtifactsGenerator(unittest.TestCase):
    def test_bootstrap_ci_exact(self):
        data = [1.0] * 50 + [0.0] * 50
        mean, lower, upper = compute_bootstrap_ci(data, num_samples=500, ci=0.95, seed=42)
        self.assertAlmostEqual(mean, 0.5)
        self.assertTrue(0.35 <= lower <= 0.50)
        self.assertTrue(0.50 <= upper <= 0.65)

    def test_mcnemar_test_statistical_accuracy(self):
        # 100 samples: baseline passes 50, treatment passes 80
        # b (base pass, treat fail) = 5
        # c (base fail, treat pass) = 35
        # chi2 = (|5 - 35| - 1)^2 / (5 + 35) = 29^2 / 40 = 841 / 40 = 21.025
        base = [True] * 45 + [True] * 5 + [False] * 35 + [False] * 15
        treat = [True] * 45 + [False] * 5 + [True] * 35 + [False] * 15
        res = mcnemar_test(base, treat)
        self.assertEqual(res["b"], 5)
        self.assertEqual(res["c"], 35)
        self.assertAlmostEqual(res["chi2"], 21.025, places=2)
        self.assertTrue(res["p_value"] < 0.0001)
        self.assertAlmostEqual(res["odds_ratio"], 7.0)

    def test_holm_bonferroni_monotonicity(self):
        # 3 raw p-values: 0.005, 0.02, 0.04
        raw = [("Test1", 0.04), ("Test2", 0.005), ("Test3", 0.02)]
        adj = holm_bonferroni_correction(raw, alpha=0.05)
        self.assertEqual(len(adj), 3)
        # Test2 was rank 1 (p=0.005 * 3 = 0.015)
        t2 = next(r for r in adj if r["name"] == "Test2")
        self.assertAlmostEqual(t2["adjusted_p"], 0.015)
        self.assertTrue(t2["significant"])
        # In step-down, adjusted p-values must be non-decreasing
        p_vals = [r["adjusted_p"] for r in sorted(adj, key=lambda x: x["raw_p"])]
        self.assertTrue(p_vals[0] <= p_vals[1] <= p_vals[2])

    def test_table_1_protocol_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = generate_table_1_protocol("configs/protocol_freeze.yaml", tmp_dir)
            self.assertTrue(os.path.isfile(res["csv"]))
            self.assertTrue(os.path.isfile(res["tex"]))
            with open(res["tex"], encoding="utf-8") as f:
                content = f.read()
            self.assertIn("deepseek-ai/deepseek-coder-1.3b-instruct", content)
            self.assertIn("\\begin{table*}", content)

    def test_end_to_end_artifact_generation(self):
        # Build mock traces for B0, B1, B2, SFT, PPO
        mock_traces = {
            "B0 (Base Greedy)": {
                "humaneval": [
                    {"problem_id": f"HE_{i}", "pass_at_1": (i < 38), "turn_history": [{"turn": 1, "status": "AC" if i < 38 else "WA"}]}
                    for i in range(100)
                ]
            },
            "P1 (PPO-Dense)": {
                "humaneval": [
                    {
                        "problem_id": f"HE_{i}",
                        "pass_at_1": (i < 38),
                        "cumulative_success_at_3": (i < 58),
                        "conditional_repair_at_3": (38 <= i < 58),
                        "solved_turn": 1 if i < 38 else (2 if i < 58 else None),
                        "turn_history": [
                            {"turn": 1, "status": "AC" if i < 38 else "WA", "tokens": 150},
                            {"turn": 2, "status": "AC" if 38 <= i < 58 else "WA", "tokens": 160},
                        ],
                    }
                    for i in range(100)
                ]
            },
            "P2 (PPO-Binary)": {
                "humaneval": [
                    {
                        "problem_id": f"HE_{i}",
                        "pass_at_1": (i < 38),
                        "cumulative_success_at_3": (i < 50),
                        "conditional_repair_at_3": (38 <= i < 50),
                        "solved_turn": 1 if i < 38 else (2 if i < 50 else None),
                        "turn_history": [
                            {"turn": 1, "status": "AC" if i < 38 else "WA", "tokens": 150},
                            {"turn": 2, "status": "AC" if 38 <= i < 50 else "WA", "tokens": 160},
                        ],
                    }
                    for i in range(100)
                ]
            },
        }

        with tempfile.TemporaryDirectory() as results_tmp:
            with tempfile.TemporaryDirectory() as out_tmp:
                # Write mock JSONL files in results_tmp
                for arm, benches in mock_traces.items():
                    safe_arm = arm.split()[0].replace("(", "").replace(")", "").lower()
                    for b_name, recs in benches.items():
                        path = os.path.join(results_tmp, f"{safe_arm}_{b_name}_k3.jsonl")
                        with open(path, "w", encoding="utf-8") as f:
                            for r in recs:
                                f.write(json.dumps(r) + "\n")

                # Run complete generation engine
                audit = run_artifact_generation(
                    results_dir=results_tmp,
                    output_dir=out_tmp,
                    protocol_path="configs/protocol_freeze.yaml",
                )

                # Verify all 4 tables exist in CSV and TeX
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "table1_protocol.tex")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "table1_protocol.csv")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "table2_main_results.tex")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "table2_main_results.csv")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "table3_feedback_ablation.tex")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "table3_feedback_ablation.csv")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "table4_hypothesis_testing.tex")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "table4_hypothesis_testing.csv")))

                # Verify all 3 figures exist in SVG and CSV
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "figure1_repair_curves.svg")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "figure1_repair_curves.csv")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "figure2_failure_transitions.svg")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "figure2_failure_transitions.csv")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "figure3_efficiency_pareto.svg")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "figure3_efficiency_pareto.csv")))

                # Verify natural failures archive and provenance audit manifest
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "natural_failures_1.3b.jsonl")))
                self.assertTrue(os.path.isfile(os.path.join(out_tmp, "provenance_audit_manifest.json")))
                self.assertIn("table_2", audit["generated_artifacts"])


if __name__ == "__main__":
    unittest.main()
