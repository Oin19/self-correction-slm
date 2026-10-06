# Literature Review

## Closest Prior Work

### RLEF — Grounding Code LLMs in Execution Feedback with Reinforcement Learning

RLEF is a key prior work because it uses iterative code generation with execution feedback and PPO. Therefore, this project must not claim that execution-guided RL for code self-correction is entirely new.

### RePair — Automated Program Repair with Process-based Feedback

RePair is another closely related direction using process-based feedback for program repair and iterative optimization.

## Planned Differentiation

Focus the contribution on a systematic study of self-correction in smaller Code LLMs, including:

- model-size scaling;
- dedicated buggy-code/debugging formulation;
- PPO versus DPO;
- binary versus partial/error-status rewards;
- feedback-quality ablations;
- multi-turn Fix@K efficiency;
- cross-benchmark transfer under constrained compute.

Claims should be updated if later literature reveals closer or newer overlap.
