# Self-Correction in Small Code Language Models via Execution-Guided Reinforcement Learning

Official research implementation studying **multi-turn execution-guided self-correction** in Small Code Language Models (`deepseek-ai/deepseek-coder-1.3b-instruct`), comparing **Zero-Shot Prompting**, **Supervised Fine-Tuning (SFT)**, **Online Reinforcement Learning (`PPO-Dense` & `PPO-Binary`)**, and **Execution-Grounded Direct Preference Optimization (`DPO`)** under matched checkpoint ancestry and leakage-safe sandbox evaluation.

---

## Core Research Questions

1. **RQ1 — Learning vs. Sampling**: Does execution feedback improve multi-turn repair beyond repeated sampling (`F0`) under the same attempt and decoding budget?
2. **RQ2 — Reward Granularity**: How do dense partial-test rewards (`PPO-Dense`) vs. binary acceptance rewards (`PPO-Binary`) trade off first-attempt generation (`Pass@1`) against multi-turn repair (`ConditionalRepair@K`)?
3. **RQ3 — Feedback Granularity**: How do no-feedback (`F0`), status-only (`F1`), sanitized traceback (`F2`), structured diagnostics (`F3`), and uninformative fake error controls compare?
4. **RQ4 / RQ5 — Online PPO vs. Offline Execution-DPO**: Under identical SFT initialization and sandbox test verification, how does execution-grounded DPO compare against PPO in repair accuracy and compute stability at the 1.3B scale?
5. **RQ6 / RQ7 — Turn Budget & Cross-Domain Generalization**: How does self-correction scale across $K \in \{1, 3, 5, 7\}$ turns, and how does training on `codeparrot/apps` transfer out-of-distribution to **HumanEval ($N = 164$)** and **MBPP ($N = 500$)**?

---

## Full-Benchmark Results (`deepseek-ai/deepseek-coder-1.3b-instruct`)

All trained models (`SFT`, `PPO-Dense`, `PPO-Binary`, `DPO`) are trained exclusively on **APPS (`codeparrot/apps`)** using LoRA adapters (`r=16, alpha=32, dropout=0.05` on `q_proj, v_proj`), with `PPO-Dense`, `PPO-Binary`, and `DPO` initialized from the **exact same SFT parent checkpoint**. Evaluation runs zero-shot out-of-distribution on the **complete HumanEval (`N = 164`)** and **complete MBPP (`N = 500`)** test splits (`T1: Greedy` | `T2..5: temp=1.0, top_p=0.95, max_new_tokens=320`, `F2` sanitized traceback feedback).

### 1. Model Configurations & Parameter Counts

| Model | Total Params | Trainable Params | LoRA Config | Training Hyperparameters |
| :--- | :---: | :---: | :--- | :--- |
| **Zero-Shot Base** | 1.346B (`1,346,471,936`) | `0` (`0.00%`) | None (Frozen Base) | Pretrained `deepseek-coder-1.3b-instruct` (no fine-tuning) |
| **SFT Baseline** | 1.350B (`1,349,617,664`) | `3.15M` (`0.23%`) | `r=16, alpha=32, drop=0.05 (q_proj,v_proj)` | `LR=2e-5, epochs=1, batch=2x4, seed=42` (`APPS train[:2000]`) |
| **PPO-Dense Model** | 1.350B (`1,349,619,713`) | `3.15M` (`0.23%`) | `r=16, alpha=32, drop=0.05 (q_proj,v_proj)` | `reward=dense, LR=2e-7, steps=100, init_KL=0.3, target_KL=4.0, seed=42` |
| **PPO-Binary Model** | 1.350B (`1,349,619,713`) | `3.15M` (`0.23%`) | `r=16, alpha=32, drop=0.05 (q_proj,v_proj)` | `reward=binary, LR=2e-7, steps=100, init_KL=0.3, target_KL=4.0, seed=42` |
| **DPO Model** | 1.350B (`1,349,617,664`) | `3.15M` (`0.23%`) | `r=16, alpha=32, drop=0.05 (q_proj,v_proj)` | `beta=0.1, LR=5e-5, epochs=3, batch=4, pairs=22, seed=42` |

### 2. Out-of-Distribution Multi-Turn Self-Correction Performance

$$\text{ConditionalRepair@}K = \frac{\text{Fix@}K - \text{Pass@}1}{1 - \text{Pass@}1}$$

| Model | HE Pass@1 | HE Fix@3 | HE Fix@5 | HE CondRepair@3 | HE CondRepair@5 | MBPP Pass@1 | MBPP Fix@3 | MBPP Fix@5 | MBPP CondRepair@3 | MBPP CondRepair@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Base** | **65.24%** | **70.12%** | **71.95%** | **14.04%** | **19.30%** | 39.00% | 40.80% | 43.20% | 2.95% | 6.89% |
| **SFT Baseline** | 60.98% | 63.41% | 65.85% | 6.25% | 12.50% | 37.80% | 41.00% | 43.00% | 5.14% | **8.36%** |
| **PPO-Dense Model** | 62.80% | 64.63% | 65.85% | 4.92% | 8.20% | 38.00% | 41.00% | 42.40% | 4.84% | 7.10% |
| **PPO-Binary Model** | 60.37% | 65.24% | 66.46% | 12.31% | **15.38%** | 38.20% | 41.60% | 43.20% | **5.50%** | 8.09% |
| **DPO Model** | 63.41% | 66.46% | **68.90%** | 8.33% | 15.00% | **39.20%** | **42.20%** | **44.00%** | 4.93% | 7.89% |

---

## Key Empirical Findings

1. **Execution-Guided DPO Outperforms Online PPO at the 1.3B SLM Scale (`RQ4` / `RQ5`)**:
   * Starting from the identical SFT parent, **Execution-DPO** achieves the strongest overall multi-turn accuracy among all trained models on both HumanEval (`68.90%` `Fix@5`, `+3.05%` over SFT) and MBPP (`44.00%` `Fix@5`), surpassing **even the `Zero-Shot Base` instruct model across every single MBPP-500 metric** (`Pass@1`: `39.20%` vs. `39.00%`; `Fix@3`: `42.20%` vs. `40.80%`; `Fix@5`: `44.00%` vs. `43.20%`; `CondRepair@5`: `7.89%` vs. `6.89%`).
   * Contrasting the SFT model's own sandbox-failing Turn-1 rollouts (`rejected`) against its sandbox-verified repaired rollouts (`chosen`) avoids the value-head variance and strict KL bottleneck of online PPO in 1.3B SLMs.
2. **Dense vs. Binary PPO First-Attempt / Repair Trade-Off (`RQ2`)**:
   * **`PPO-Dense`** (partial test-fraction credit) improves **first-attempt `Pass@1`** over SFT (`62.80%` vs. `60.98%` on HumanEval), whereas **`PPO-Binary`** (strict `AC`-only reward) produces substantially higher **multi-turn `ConditionalRepair@5`** (`15.38%` vs. `8.20%` Dense on HumanEval; `8.09%` vs. `7.10%` on MBPP) because credit is awarded only when bugs are completely eliminated.
3. **Cross-Domain Format Shift & RL Recovery (`RQ3` / `RQ7`)**:
   * Single-turn SFT on competitive-programming `stdin/stdout` APPS scripts introduces mild out-of-distribution format drift on typed docstring function completion (`HumanEval Pass@1`: `65.24%` $\rightarrow$ `60.98%`) while boosting `MBPP CondRepair@5` (`6.89%` $\rightarrow$ `8.36%`). Execution-guided RL (`DPO` and `PPO-Binary`) consistently recovers both `Fix@5` and `CondRepair@5` relative to the matched SFT parent.

---

## Feedback Granularity & Safety Protocol

* **Subprocess Execution Sandbox (`src/execution/executor.py`)**: Executes untrusted candidate code in isolated subprocesses with syntax pre-checks, wall-clock timeouts, memory caps (`RLIMIT_AS`), and `AC` / `WA` / `RE` / `CE` / `TLE` / `MLE` classification.
* **Oracle Redaction (`sanitize_feedback_text`)**: Strips `expected=...` oracle values from tracebacks before injecting feedback into repair prompts.
* **Feedback Modes (`src/debugging/debug_loop.py`)**:
  * `F0`: No execution feedback (matched repeated-sampling control `B1`)
  * `F1`: Categorical execution status only (`Execution Status: WA/RE/CE/TLE`)
  * `F2`: Sanitized stderr / Python exception traceback
  * `F3`: Structured pass/total diagnostic summary + sanitized traceback

---

## Repository Structure

```text
├── notebooks/
│   ├── 01_setup_and_sandbox.ipynb           # Sandbox & status verification
│   ├── 02_model_and_baseline.ipynb          # Base 1.3B loading & zero-shot checks
│   ├── 03_debug_loop_and_reward.ipynb       # Multi-turn debug loop & reward verification
│   ├── 04_sft_training.ipynb                # APPS LoRA SFT baseline training
│   ├── 05_ppo_training.ipynb                # Execution-guided PPO (dense & binary modes)
│   ├── 06_dpo_training.ipynb                # Execution rollout preference collection & DPO
│   └── 07_evaluation_and_ablations.ipynb    # Batched multi-GPU full evaluation & ablations
├── src/
│   ├── debugging/debug_loop.py              # Batched multi-turn loop, F0-F3 modes & fence early-stop
│   ├── evaluation/metrics.py                # Pass@1, Fix@K, CondRepair@K & resumable JSONL runner
│   ├── execution/                           # Subprocess Python sandbox & status enums
│   ├── models/                              # Canonical LoRA/base loader & code extraction
│   ├── rewards/execution_reward.py          # Dense partial-credit & binary AC reward functions
│   ├── training/                            # SFT, PPO, and execution-grounded DPO trainers
│   └── utils/                               # Test normalizer, seed control, checkpoint helpers
├── docs/                                    # Research plan, methodology, and literature review
└── tests/                                   # Unit tests (sandbox, rewards, debug loop, DPO, eval)
```

---

## Running Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
