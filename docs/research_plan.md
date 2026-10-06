# Research Plan

## Project
Self-Correction in LLMs via Execution-Guided Reinforcement Learning

## Core Question
Can small Code LLMs learn to iteratively detect and correct programming errors using execution feedback?

## Research Questions
1. Can SLMs perform agentic debugging?
2. How does feedback quality/type affect self-correction?
3. Does RL outperform prompting and SFT?
4. Do SLMs benefit from dense execution rewards?
5. Is DPO viable compared with PPO?
6. How many debugging turns are useful?
7. Does training generalize across benchmarks?

## Minimum Validation Grid
- Models: Qwen2.5-Coder 1.5B and 7B where compute permits
- Data: APPS subset + HumanEval-Fix
- Methods: Zero-shot / SFT / PPO / DPO
- Turns: K=1,3,5
- Metrics: Pass@1 / Fix@3 / Fix@5 / wall-clock / VRAM

## Hypothesis
Execution-guided RL is hypothesized to improve agentic code generation and debugging in small language models compared with zero-shot prompting and SFT, particularly under fixed sample budgets and constrained compute.

## Novelty Positioning
The project does not claim that execution-feedback RL itself is new. The study focuses on small Code LLMs, a dedicated debugging formulation, PPO vs DPO, reward density, feedback quality, multi-turn efficiency, and transfer across benchmarks.
