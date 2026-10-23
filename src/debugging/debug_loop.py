import time
import torch
from typing import Any, Callable, Dict, List, Optional

from src.execution.executor import ExecutionResult, PythonSandbox, run_code, sanitize_traceback
from src.execution.status import ExecutionStatus
from src.models.generation import (
    build_prompt as _canonical_build_prompt,
    build_repair_prompt,
    extract_code_block,
)
from src.rewards.execution_reward import compute_partial_reward
from src.utils.test_cases import split_feedback_and_scoring_tests


def _prepare_executable_code(code: str, problem: str = "") -> str:
    """Ensure HumanEval prompt imports or function headers are preserved when executing extracted code."""
    if not code or not problem:
        return code
    prob_lines = problem.splitlines()
    has_prob_def = any(line.startswith("def ") for line in prob_lines)
    has_code_def = any(line.lstrip().startswith("def ") for line in code.splitlines())
    if has_prob_def and not has_code_def:
        body_lines = code.splitlines()
        if body_lines and not body_lines[0].startswith((" ", "\t")):
            body_lines = ["    " + ln if ln.strip() else ln for ln in body_lines]
        return problem.rstrip() + "\n" + "\n".join(body_lines)
    import_lines = [
        ln for ln in prob_lines
        if ln.startswith(("import ", "from ")) and ln not in code
    ]
    if import_lines:
        return "\n".join(import_lines) + "\n" + code
    return code


def format_feedback(
    result: ExecutionResult,
    mode: str = "F2",
    max_traceback_len: int = 512,
) -> str:
    """Format execution feedback according to feedback mode F0, F1, F2, F3.

    Modes:
      F0 (No feedback): Empty string (pure repeated-sampling control B1).
      F1 (Status only): Status indicator only (e.g. WA, RE, TLE, CE, MLE).
      F2 (Sanitized traceback): Status + sanitized traceback with oracle/expected values redacted.
      F3 (Structured summary): Status + test fraction passed + sanitized error details.
    """
    if result.status == ExecutionStatus.AC:
        return "All test cases PASSED successfully."

    if mode == "F0":
        return ""

    if mode == "F1":
        return f"Execution Status: {result.status}"

    if mode == "F2":
        feedback = [f"Execution Status: {result.status}"]
        raw_error = result.traceback or result.stderr or result.stdout or ""
        clean_error = sanitize_traceback(raw_error)[:max_traceback_len].strip()
        if clean_error:
            feedback.append(f"Traceback / Error Details:\n{clean_error}")
        return "\n".join(feedback)

    if mode == "F3":
        feedback = [f"Execution Status: {result.status}"]
        if result.total_tests > 0:
            feedback.append(f"Passed Tests: {result.passed_tests} / {result.total_tests} ({result.pass_rate:.1%})")
        raw_error = result.traceback or result.stderr or result.stdout or ""
        clean_error = sanitize_traceback(raw_error)[:max_traceback_len].strip()
        if clean_error:
            feedback.append(f"Error Details:\n{clean_error}")
        return "\n".join(feedback)

    # Fallback to F2
    return format_feedback(result, mode="F2", max_traceback_len=max_traceback_len)


def format_execution_feedback(result: ExecutionResult) -> str:
    """Backward-compatible default feedback formatter (uses F2 sanitized traceback)."""
    return format_feedback(result, mode="F2")


def build_prompt(problem: str, prev_code: str = None, traceback: str = None) -> str:
    """Construct prompt for turn 0 or multi-turn repair."""
    if prev_code is None:
        return _canonical_build_prompt(problem)
    return build_repair_prompt(problem, prev_code, traceback or "")


def build_mode_prompt(
    problem: str,
    prev_code: Optional[str] = None,
    result: Optional[ExecutionResult] = None,
    mode: str = "F2",
) -> str:
    """Construct multi-turn prompt according to experimental feedback mode F0-F3."""
    if prev_code is None or mode == "F0" or result is None:
        return _canonical_build_prompt(problem)
    feedback_str = format_feedback(result, mode=mode)
    return build_repair_prompt(problem, prev_code, feedback_str)


def _execute(code: str, test_cases: list, sandbox: Optional[PythonSandbox] = None, problem: str = "") -> ExecutionResult:
    """Execute against explicit benchmark tests; empty tests are never AC."""
    sandbox = sandbox or PythonSandbox()
    if not test_cases:
        return ExecutionResult("CE", 0, 0, traceback="No executable test cases supplied.")
    exec_code = _prepare_executable_code(code, problem)
    if all(isinstance(t, str) for t in test_cases):
        r = run_code(exec_code + "\n\n" + "\n\n".join(test_cases))
        return ExecutionResult(r["status"], 1 if r["status"] == "AC" else 0, 1,
                               stdout=r.get("output", ""), traceback=r.get("traceback", ""))
    return sandbox.run_tests(exec_code, test_cases)


def agentic_debug_loop(
    model,
    tokenizer,
    problem: str,
    test_cases: list = None,
    K: int = 3,
    initial_code: str = None,
    max_gen_time: Optional[float] = None,
    feedback_mode: str = "F2",
    feedback_tests: Optional[list] = None,
    scoring_tests: Optional[list] = None,
    temperature: float = 0.2,
) -> list:
    """Run up to K turns with configurable feedback modes (F0-F3) and anti-leakage protection.

    When feedback_tests and scoring_tests are provided, feedback injected into the prompt
    is derived exclusively from feedback_tests, while scoring and benchmark pass rate
    are evaluated on held-out scoring_tests.
    """
    history = []
    code = initial_code
    last_fb_result: Optional[ExecutionResult] = None
    device = next(model.parameters()).device
    sandbox = PythonSandbox()

    # Determine feedback and scoring tests
    all_tests = test_cases or []
    if feedback_tests is not None and scoring_tests is not None:
        fb_cases = feedback_tests
        sc_cases = scoring_tests
    elif len(all_tests) > 1:
        fb_cases, sc_cases = split_feedback_and_scoring_tests(all_tests)
    else:
        fb_cases, sc_cases = all_tests, all_tests

    for turn in range(K):
        turn_start = time.time()
        tokens_generated = 0

        if turn == 0 and code is not None:
            prompt = _canonical_build_prompt(problem)
        else:
            prompt = build_mode_prompt(problem, code, last_fb_result, mode=feedback_mode)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            gen_kwargs = {
                "max_new_tokens": 512,
                "do_sample": turn > 0 or temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if turn > 0:
                gen_kwargs.update({"temperature": 1.0, "top_p": 0.95})
            elif temperature > 0:
                gen_kwargs.update({"temperature": temperature, "top_p": 0.95})

            if max_gen_time:
                gen_kwargs["max_time"] = max_gen_time

            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)

            # Count generated tokens
            in_len = inputs["input_ids"].shape[1]
            out_len = output.shape[1]
            tokens_generated = max(0, out_len - in_len)

            full = tokenizer.decode(output[0], skip_special_tokens=True)
            code = extract_code_block(full, prompt)

        # Run feedback tests (for prompt injection)
        fb_res = _execute(code, fb_cases, sandbox, problem=problem)
        last_fb_result = fb_res

        # Run scoring tests (for benchmark metric evaluation)
        sc_res = _execute(code, sc_cases, sandbox, problem=problem) if sc_cases != fb_cases else fb_res

        elapsed = time.time() - turn_start
        history.append({
            "turn": turn + 1,
            "prompt": prompt,
            "code": code,
            "result": sc_res.to_dict(),
            "feedback_result": fb_res.to_dict(),
            "status": sc_res.status,
            "tokens": tokens_generated,
            "latency": elapsed,
        })

        if sc_res.status == ExecutionStatus.AC:
            break

    return history


def agentic_loop_no_feedback(model, tokenizer, problem: str, test_cases: list = None, K: int = 3) -> list:
    """RQ2 / B1 control: genuine repeated sampling without feedback prompt injection."""
    return agentic_debug_loop(
        model=model,
        tokenizer=tokenizer,
        problem=problem,
        test_cases=test_cases,
        K=K,
        feedback_mode="F0",
    )


class DebugLoop:
    """Controls multi-turn debugging sessions with configurable feedback mode."""

    def __init__(
        self,
        sandbox: Optional[PythonSandbox] = None,
        max_turns: int = 5,
        traceback_token_cap: int = 500,
        feedback_mode: str = "F2",
    ):
        self.sandbox = sandbox or PythonSandbox()
        self.max_turns = max_turns
        self.traceback_token_cap = traceback_token_cap
        self.feedback_mode = feedback_mode

    def build_initial_prompt(self, problem_description: str) -> str:
        return _canonical_build_prompt(problem_description)

    def build_turn_prompt(self, problem_description: str, previous_code: str, execution_result: ExecutionResult) -> str:
        return build_repair_prompt(
            problem_description,
            previous_code,
            format_feedback(execution_result, mode=self.feedback_mode, max_traceback_len=self.traceback_token_cap),
        )

    def run_session(
        self,
        problem_description: str,
        test_cases: List[Dict[str, Any]],
        model_fn: Callable[[str], str],
        initial_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        history = []
        current_code = initial_code
        last_exec: Optional[ExecutionResult] = None
        solved, solved_turn = False, None

        for turn_idx in range(1, self.max_turns + 1):
            if turn_idx == 1 and current_code is not None:
                prompt = self.build_initial_prompt(problem_description)
            elif turn_idx == 1:
                prompt = self.build_initial_prompt(problem_description)
                current_code = extract_code_block(model_fn(prompt))
            else:
                prompt = self.build_turn_prompt(problem_description, current_code, last_exec)
                current_code = extract_code_block(model_fn(prompt))

            exec_code = _prepare_executable_code(current_code, problem_description)
            exec_res = self.sandbox.run_tests(exec_code, test_cases)
            last_exec = exec_res
            history.append({
                "turn": turn_idx,
                "prompt": prompt,
                "code": current_code,
                "execution_result": exec_res.to_dict(),
                "reward": compute_partial_reward(exec_res),
                "status": exec_res.status,
            })
            if exec_res.status == ExecutionStatus.AC:
                solved, solved_turn = True, turn_idx
                break

        return {
            "solved": solved,
            "solved_turn": solved_turn,
            "total_turns": len(history),
            "max_turns": self.max_turns,
            "history": history,
            "final_status": history[-1]["status"] if history else "CE",
            "final_reward": history[-1]["reward"] if history else -0.2,
        }
