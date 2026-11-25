# Research Plan

## Project
Self-Correction in Small Code Language Models via Execution-Guided Reinforcement Learning

## Primary Model & Scale
- **Primary Model**: `deepseek-ai/deepseek-coder-1.3b-instruct` (`1.346B` base parameters; `3.15M` trainable LoRA parameters at `r=16, alpha=32` on `q_proj, v_proj`).
- **Training Dataset (In-Domain)**: `codeparrot/apps` (`train` split).
- **Out-of-Distribution Evaluation Benchmarks**:
  - `openai/openai_humaneval` (`test` split, full $N = 164$ problems)
  - `google-research-datasets/mbpp` (`test` split, full $N = 500$ problems)

## Research Questions
1. **RQ1 (Learning vs. Sampling)**: Does execution feedback improve multi-turn repair over repeated sampling (`F0`) under matched turn and decoding budgets?
2. **RQ2 (Reward Density)**: How do dense partial-test rewards (`PPO-Dense`) vs. binary pass/fail rewards (`PPO-Binary`) affect `Pass@1` vs. `ConditionalRepair@K`?
3. **RQ3 (Feedback Granularity)**: Which feedback signal (`F0` none, `F1` status-only, `F2` sanitized traceback, `F3` structured summary, vs. fake error control) drives code repair?
4. **RQ4 / RQ5 (PPO vs. Execution-DPO)**: Under matched SFT initialization and execution-verified trajectories, how does offline Execution-DPO compare with online PPO in repair quality and stability?
5. **RQ6 (Turn Budget Plateau)**: How do `Fix@K` and `ConditionalRepair@K` progress across $K \in \{1, 3, 5, 7\}$?
6. **RQ7 (Out-of-Distribution Transfer)**: How well do policies trained on APPS competitive-programming problems generalize to HumanEval and MBPP?

## Completed Validation Matrix (`1.3B` Full Benchmark Run)
- **Evaluated Policies**: `Zero-Shot Base`, `SFT Baseline`, `PPO-Dense`, `PPO-Binary`, `Execution-DPO`
- **Primary Metrics**: `Pass@1`, `Fix@1`, `Fix@3`, `Fix@5`, `ConditionalRepair@3`, `ConditionalRepair@5`
- **Summary of Headline Outcomes**:
  - Multi-turn execution feedback yields consistent gains from $K=1 \rightarrow K=5$ across every evaluated arm on both HumanEval and MBPP.
  - **Execution-DPO** achieves the strongest trained performance (`68.90%` HumanEval `Fix@5`, `+3.05%` over SFT; `44.00%` MBPP `Fix@5`), outperforming `Zero-Shot Base` across every metric on full MBPP-500 (`39.20%` `Pass@1`, `42.20%` `Fix@3`, `44.00%` `Fix@5`, `7.89%` `CondRepair@5`).
  - **`PPO-Dense` vs. `PPO-Binary`** reveals a clear trade-off: dense partial rewards favor Turn-1 `Pass@1` (`62.80%` vs. `60.37%` HE), whereas binary `AC` rewards favor multi-turn `ConditionalRepair@5` (`15.38%` vs. `8.20%` HE; `8.09%` vs. `7.10%` MBPP).
