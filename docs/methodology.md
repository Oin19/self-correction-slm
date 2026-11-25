# Methodology

## End-to-End Execution-Guided Loop

`problem → Turn 1 candidate (greedy) → sandbox execution → sanitized feedback → Turn 2..K repair (sampled) → sandbox execution`

1. **Turn 1 (`Pass@1`)**: Given a programming problem statement (preserving top-level prompt imports for HumanEval and required function signatures for MBPP), the policy generates an initial Python solution using deterministic greedy decoding.
2. **Sandbox Verification**: Candidate code is checked via `ast.parse` and executed in an isolated Python subprocess (`src/execution/executor.py`) against normalized unit tests (`src/utils/test_cases.py`) with wall-clock timeouts and memory bounds.
3. **Sanitized Feedback Formatting**: Expected/oracle test values are redacted (`sanitize_feedback_text`), and feedback is formatted under one of four controlled granularity modes (`F0`–`F3`).
4. **Iterative Repair (`Turns 2..K`)**: Unsolved problems enter batched repair rollouts (`temperature=1.0, top_p=0.95`, early-stopped immediately at the closing ```` ``` ```` code fence via `_FenceEosLogitsProcessor`) until all unit tests pass (`AC`) or the turn budget $K$ is exhausted.

## Execution Status Taxonomy

- `AC` — Accepted (all unit tests / output segments passed)
- `WA` — Wrong Answer (one or more assertions / output checks failed)
- `CE` — Compilation / Syntax / Indentation Error
- `RE` — Runtime Exception (`IndexError`, `TypeError`, `NameError`, `ZeroDivisionError`, etc.)
- `TLE` — Time Limit Exceeded
- `MLE` — Memory Limit Exceeded

## Feedback Granularity Modes (`src/debugging/debug_loop.py`)

- **`F0` (No Feedback / Repeated-Sampling Control `B1`)**: Re-prompts the original problem without execution status or error details under the identical turn and sampling budget.
- **`F1` (Status-Only)**: Injects `Execution Status: <STATUS>` only.
- **`F2` (Sanitized Traceback — Default)**: Injects bounded, oracle-redacted Python exception / assertion traceback text.
- **`F3` (Structured Summary)**: Injects execution status, passed/total feedback check counts, and sanitized diagnostic text.

## Training Arms & Matched Checkpoint Ancestry

All trained arms use `deepseek-ai/deepseek-coder-1.3b-instruct` with LoRA (`r=16, lora_alpha=32, lora_dropout=0.05`, targets `q_proj, v_proj`, `3.15M` trainable parameters = `0.23%` of `1.350B`):

1. **`SFT Baseline` (`src/training/sft.py`)**: Supervised fine-tuning on `codeparrot/apps` (`train[:2000]`), `LR=2e-5`, `1` epoch, `seed=42`. Serves as the frozen parent checkpoint for all RL/preference arms.
2. **`PPO-Dense` (`src/training/ppo.py`, `reward_mode="dense"`)**: Initialized from `SFT`:
   - `AC` $\rightarrow +1.0$
   - Non-`AC` with `passed > 0` $\rightarrow \text{passed} / \text{total}$ (including packed APPS multi-test I/O via `score_io_output`)
   - `WA` with `0` passed $\rightarrow 0.0$
   - `CE` / `RE` / `TLE` / `MLE` with `0` passed $\rightarrow -0.2$
3. **`PPO-Binary` (`src/training/ppo.py`, `reward_mode="binary"`)**: Initialized from `SFT` with identical hyperparameters (`LR=2e-7, steps=100, init_KL=0.30, target_KL=4.0, seed=42`), rewarding `AC` $\rightarrow +1.0$ and all non-`AC` $\rightarrow 0.0$.
4. **`Execution-DPO` (`src/training/dpo.py`)**: Initialized from the same `SFT` parent; preference pairs `(chosen, rejected)` are collected strictly from the SFT model's own execution-guided rollouts on APPS (`train[:500]`, `K=3`, no reference-solution fallback), trained with `beta=0.1, LR=5e-5, epochs=3, batch=4, seed=42`.

## Primary Evaluation Metrics (`src/evaluation/metrics.py`)

- **`Pass@1`**: Fraction of benchmark problems solved (`AC`) on Turn 1.
- **`Fix@K` (`CumulativeSuccess@K`)**: Fraction of benchmark problems solved at or before Turn $K$ ($K \in \{1, 3, 5, 7\}$).
- **`ConditionalRepair@K`**: Fraction of Turn-1 failures subsequently repaired to `AC` by Turn $K$:
  $$\text{ConditionalRepair@}K = \frac{\text{Fix@}K - \text{Pass@}1}{1 - \text{Pass@}1}$$
- **Per-Problem Trace Persistence**: Every run streams resumable JSONL trajectories (`problem_id`, `pass_at_1`, `solved_turn`, `statuses`, `latency_sec`, `history`) for paired statistical tests and failure-transition analysis.
