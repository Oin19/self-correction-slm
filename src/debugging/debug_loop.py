"""Iterative execution-feedback debugging loop."""

import random
import torch
from typing import Any, Callable, Dict, List, Optional

from src.execution.executor import ExecutionResult, PythonSandbox, run_code
from src.execution.status import ExecutionStatus
from src.models.generation import extract_code_block
from src.rewards.execution_reward import compute_partial_reward


def build_prompt(problem: str, prev_code: str = None, traceback: str = None) -> str:
    if prev_code is None:
        return f"### Problem:\n{problem}\n\n### Write a Python solution:\n```python"
    return (f"### Problem:\n{problem}\n\n### Your previous code:\n```python\n{prev_code}\n```\n\n"
            f"### Error you received:\n{(traceback or '')[:512]}\n\n### Fixed version:\n```python")


def _execute(code: str, test_cases: list, sandbox: Optional[PythonSandbox] = None) -> ExecutionResult:
    """Execute against explicit benchmark tests; empty tests are never AC."""
    sandbox = sandbox or PythonSandbox()
    if not test_cases:
        return ExecutionResult("CE", 0, 0, traceback="No executable test cases supplied.")
    if all(isinstance(t, str) for t in test_cases):
        r = run_code(code + "\n\n" + "\n\n".join(test_cases))
        return ExecutionResult(r["status"], 1 if r["status"] == "AC" else 0, 1,
                               stdout=r.get("output", ""), traceback=r.get("traceback", ""))
    return sandbox.run_tests(code, test_cases)


def agentic_debug_loop(model, tokenizer, problem: str, test_cases: list = None, K: int = 3,
                       initial_code: str = None, max_gen_time: Optional[float] = None) -> list:
    """Run up to K turns; every retry receives real execution feedback.

    ``max_gen_time`` (seconds) caps each ``model.generate`` call (wall clock)
    so a wedged GPU/driver cannot hang the loop indefinitely.
    """
    history, code, error = [], initial_code, ""
    device = next(model.parameters()).device
    tests, sandbox = test_cases or [], PythonSandbox()
    for turn in range(K):
        if turn == 0 and code is not None:
            prompt = build_prompt(problem)
        else:
            prompt = build_prompt(problem, code, error)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            gen_kwargs = {
                "max_new_tokens": 512,
                "do_sample": turn > 0,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if turn > 0:
                gen_kwargs.update({"temperature": 1.0, "top_p": 0.95})
            if max_gen_time:
                gen_kwargs["max_time"] = max_gen_time
            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)
            full = tokenizer.decode(output[0], skip_special_tokens=True)
            code = extract_code_block(full, prompt)
        result = _execute(code, tests, sandbox)
        history.append({"turn": turn + 1, "prompt": prompt, "code": code, "result": result.to_dict()})
        if result.status == ExecutionStatus.AC:
            break
        error = (result.traceback or result.stderr or result.stdout or "")[:512]
    return history


def agentic_loop_no_feedback(model, tokenizer, problem: str, test_cases: list = None, K: int = 3) -> list:
    """RQ2 control: random fake error context, but correctness is checked on real tests."""
    history, code = [], None
    fake_errors = ['NameError: name "x" is not defined', "IndexError: list index out of range", "TypeError: unsupported operand type(s)"]
    device = next(model.parameters()).device
    tests, sandbox = test_cases or [], PythonSandbox()
    for turn in range(K):
        prompt = build_prompt(problem, code, random.choice(fake_errors))
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": turn > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if turn > 0:
            gen_kwargs.update({"temperature": 1.0, "top_p": 0.95})
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        full = tokenizer.decode(output[0], skip_special_tokens=True)
        code = extract_code_block(full, prompt)
        result = _execute(code, tests, sandbox)
        history.append({"turn": turn + 1, "code": code, "result": result.to_dict()})
        if result.status == ExecutionStatus.AC:
            break
    return history


def format_execution_feedback(result: ExecutionResult) -> str:
    if result.status == ExecutionStatus.AC:
        return "All test cases PASSED successfully."
    feedback = [f"Execution Status: {result.status}"]
    if result.traceback: feedback.append(f"Traceback / Error Details:\n{result.traceback}")
    elif result.stderr: feedback.append(f"Error Output (stderr):\n{result.stderr}")
    elif result.stdout: feedback.append(f"Output (stdout):\n{result.stdout}")
    if result.total_tests > 0: feedback.append(f"Passed {result.passed_tests} / {result.total_tests} test cases.")
    return "\n".join(feedback)


class DebugLoop:
    """Controls multi-turn debugging sessions."""
    def __init__(self, sandbox: Optional[PythonSandbox] = None, max_turns: int = 5, traceback_token_cap: int = 500):
        self.sandbox, self.max_turns, self.traceback_token_cap = sandbox or PythonSandbox(), max_turns, traceback_token_cap

    def build_initial_prompt(self, problem_description: str) -> str:
        return f"Solve the following programming problem in Python.\nProblem Statement:\n{problem_description}\n\nWrite clean, self-contained Python code wrapped in ```python ... ```."

    def build_turn_prompt(self, problem_description: str, previous_code: str, execution_result: ExecutionResult) -> str:
        return (f"Problem Statement:\n{problem_description}\n\nPrevious solution failed execution.\n"
                f"Previous Code:\n```python\n{previous_code}\n```\n\nExecution Feedback:\n{format_execution_feedback(execution_result)}\n\n"
                "Identify the bug, correct the code, and provide the fixed Python solution wrapped in ```python ... ```." )

    def run_session(self, problem_description: str, test_cases: List[Dict[str, Any]], model_fn: Callable[[str], str], initial_code: Optional[str] = None) -> Dict[str, Any]:
        history, current_code, last_exec = [], initial_code, None
        solved, solved_turn = False, None
        for turn_idx in range(1, self.max_turns + 1):
            if current_code is None:
                prompt = self.build_initial_prompt(problem_description)
                current_code = extract_code_block(model_fn(prompt))
            else:
                prompt = self.build_turn_prompt(problem_description, current_code, last_exec)
            exec_res = self.sandbox.run_tests(current_code, test_cases)
            last_exec = exec_res
            history.append({"turn": turn_idx, "prompt": prompt, "code": current_code,
                            "execution_result": exec_res.to_dict(), "reward": compute_partial_reward(exec_res),
                            "status": exec_res.status})
            if exec_res.status == ExecutionStatus.AC:
                solved, solved_turn = True, turn_idx
                break
            if turn_idx < self.max_turns:
                current_code = extract_code_block(model_fn(self.build_turn_prompt(problem_description, current_code, exec_res)))
        return {"solved": solved, "solved_turn": solved_turn, "total_turns": len(history),
                "max_turns": self.max_turns, "history": history,
                "final_status": history[-1]["status"] if history else "CE",
                "final_reward": history[-1]["reward"] if history else -0.2}
