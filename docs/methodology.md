# Methodology

## End-to-End Loop

`problem → initial code → execute → feedback → correction → execute → ...`

The model receives the programming problem and generates an initial solution. The generated code is executed in a sandbox. Execution feedback may include traceback, failed-test summary, and execution status. The model then generates a correction for the next turn.

## Execution Status

Track at minimum:
- AC — Accepted
- PE — Presentation/format issue where applicable
- WA — Wrong Answer
- TLE — Time Limit Exceeded
- MLE — Memory Limit Exceeded
- CE — Compilation/Syntax Error
- RE — Runtime Error

## Reward Variants

Selected at train time via `reward_mode` on `run_ppo_training(...)`:

- **`reward_mode="dense"`** (default; RQ3) — partial test-pass reward
  - AC → `+1.0`
  - Non-AC with `passed > 0` → `passed / total` (never discarded)
  - WA/PE with `0` passed → `0.0`
  - CE/RE/TLE/MLE (or empty suite) with `0` passed → `−0.2`
  - Packed APPS multi-test stdin (`first line = T`): segment/line partial credit via `score_io_output` / `packed_tests=T`
- **`reward_mode="binary"`** (RQ4 ablation only)
  - AC → `+1.0`
  - Any non-AC → `0.0` (no partial credit, no CE/RE penalty)
- Execution-status-aware reward (exploratory)
- Invalid/syntax output penalty (exploratory)

Both modes share the same sandbox tests (`normalize_tests`) so only the reward mapping differs between RQ3 and RQ4 arms. Checkpoints: `./checkpoints/ppo_dense/final` vs `./checkpoints/ppo_binary/final`; `ppo_metadata.json` records `reward_mode`, `seed`, and `commit_sha`.

DPO preference pairs are collected with the **same** `normalize_tests` harness (chosen = first AC turn; rejected = first CE/RE/WA/TLE/MLE turn within K=3).

## Baselines

- Zero-shot prompting
- No-feedback debugging
- True execution feedback
- Random/incorrect feedback ablation
- SFT
- PPO
- DPO

## Controlled Variables

Record model, dataset split, prompt format, sampling temperature, top-p, debugging turns, test budget, reward version, random seed, GPU, wall-clock time, and VRAM.

## Evaluation

Primary metrics are Pass@1, Fix@3, Fix@5, execution success, wall-clock time, VRAM usage, token usage where available, and reward stability.
