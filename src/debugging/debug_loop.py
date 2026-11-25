"""Iterative execution-feedback debugging loop."""

import ast
from concurrent.futures import ThreadPoolExecutor
import os
import random
import re
import subprocess
import sys
import time
import torch
from typing import Any, Callable, Dict, List, Optional
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from src.execution.executor import ExecutionResult, PythonSandbox, run_code
from src.execution.status import ExecutionStatus
from src.models.generation import extract_code_block
from src.rewards.execution_reward import compute_partial_reward

_BACKTICK_IDS_CACHE: Dict[int, set] = {}


def _get_backtick_token_ids(tokenizer) -> set:
    key = id(tokenizer)
    if key not in _BACKTICK_IDS_CACHE:
        try:
            vocab = tokenizer.get_vocab()
            _BACKTICK_IDS_CACHE[key] = {tid for tok, tid in vocab.items() if "`" in str(tok)}
        except Exception:
            _BACKTICK_IDS_CACHE[key] = set()
    return _BACKTICK_IDS_CACHE[key]


def _has_closed_code_fence(gen_text: str) -> bool:
    """Return True once generated text has closed its first markdown code block."""
    stripped = gen_text.lstrip()
    if not stripped:
        return False
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl == -1:
            return False
        return "```" in stripped[first_nl + 1 :]
    return "```" in stripped


class _FenceEosLogitsProcessor(LogitsProcessor):
    """Force EOS per batch row immediately after its closing ``` code fence is emitted."""

    def __init__(self, tokenizer, prompt_len: int, batch_size: int, eos_token_id: Optional[int]):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.eos_token_id = eos_token_id
        self.backtick_ids = _get_backtick_token_ids(tokenizer)
        self.done = [False] * batch_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] <= self.prompt_len or self.eos_token_id is None:
            return scores
        for b in range(input_ids.shape[0]):
            if self.done[b]:
                scores[b, :] = -float("inf")
                scores[b, self.eos_token_id] = 0.0
                continue
            last_tok = int(input_ids[b, -1])
            if last_tok == self.eos_token_id:
                self.done[b] = True
                continue
            if not self.backtick_ids or last_tok in self.backtick_ids:
                gen_text = self.tokenizer.decode(input_ids[b, self.prompt_len :], skip_special_tokens=True)
                if _has_closed_code_fence(gen_text):
                    self.done[b] = True
                    scores[b, :] = -float("inf")
                    scores[b, self.eos_token_id] = 0.0
        return scores


class _AllRowsDoneCriteria(StoppingCriteria):
    """Stop batched generation as soon as every sequence has hit EOS or closed its code fence."""

    def __init__(self, processor: _FenceEosLogitsProcessor):
        self.processor = processor

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return all(self.processor.done)


def sanitize_feedback_text(text: str, max_chars: int = 512) -> str:
    """Sanitize traceback/error text so hidden oracle expected values are not leaked."""
    if not text:
        return ""
    sanitized = re.sub(r"expected=.*?(?=\s+actual=|$)", "expected=<REDACTED>", text)
    sanitized = re.sub(r"Expected\s+.*?(?=,\s+got\s+)", "Expected <REDACTED>", sanitized)
    return sanitized[:max_chars]


def format_feedback_by_mode(result: ExecutionResult, feedback_mode: str = "F2", max_chars: int = 512) -> str:
    """Format execution feedback according to F0/F1/F2/F3 granularity modes."""
    mode = (feedback_mode or "F2").upper()
    if mode == "F0":
        return ""
    if mode == "F1":
        return f"Execution Status: {result.status}"
    raw_err = result.traceback or result.stderr or result.stdout or ""
    sanitized_err = sanitize_feedback_text(raw_err, max_chars=max_chars)
    if mode == "F2":
        return sanitized_err or f"Execution Status: {result.status}"
    if mode == "F3":
        parts = [f"Execution Status: {result.status}"]
        if result.total_tests > 0:
            parts.append(f"Passed {result.passed_tests} / {result.total_tests} feedback checks.")
        if sanitized_err:
            parts.append(f"Diagnostic:\n{sanitized_err}")
        return "\n".join(parts)[:max_chars]
    return sanitized_err


def build_prompt(problem: str, prev_code: str = None, traceback: str = None, feedback_mode: str = "F2") -> str:
    if prev_code is None or (feedback_mode and feedback_mode.upper() == "F0"):
        return f"### Problem:\n{problem}\n\n### Write a Python solution:\n```python"
    return (
        f"### Problem:\n{problem}\n\n### Your previous code:\n```python\n{prev_code}\n```\n\n"
        f"### Error you received:\n{(traceback or '')[:512]}\n\n### Fixed version:\n```python"
    )


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


def _run_assertion_suite_fast(code: str, assertions: List[str], timeout: float = 2.5) -> ExecutionResult:
    """Run all assertion test cases inside a single Python subprocess with fast syntax pre-check."""
    total = len(assertions)
    try:
        ast.parse(code)
    except SyntaxError as e:
        return ExecutionResult(
            status=ExecutionStatus.CE,
            passed_tests=0,
            total_tests=total,
            stderr=str(e),
            traceback=f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}\n{e.text or ''}",
        )
    except Exception as e:
        return ExecutionResult(status=ExecutionStatus.CE, passed_tests=0, total_tests=total, stderr=str(e), traceback=str(e))

    runner = (
        "import sys, traceback\n"
        f"_code = {code!r}\n"
        f"_assertions = {assertions!r}\n"
        "_ns = {'__name__': '__main__'}\n"
        "try:\n"
        "    exec(compile(_code, '<solution>', 'exec'), _ns)\n"
        "except SystemExit:\n"
        "    pass\n"
        "except Exception:\n"
        "    sys.stderr.write('###FAST_RES###0|' + str(len(_assertions)) + '|RE\\n')\n"
        "    traceback.print_exc(file=sys.stderr)\n"
        "    sys.exit(0)\n"
        "if 'Solution' in _ns and isinstance(_ns['Solution'], type):\n"
        "    try:\n"
        "        _inst = _ns['Solution']()\n"
        "        for _a in dir(_inst):\n"
        "            if not _a.startswith('_') and callable(getattr(_inst, _a)) and _a not in _ns:\n"
        "                _ns[_a] = getattr(_inst, _a)\n"
        "    except Exception:\n"
        "        pass\n"
        "_passed = 0\n"
        "_status = 'AC'\n"
        "_first_tb = ''\n"
        "for _stmt in _assertions:\n"
        "    try:\n"
        "        exec(compile(_stmt, '<test>', 'exec'), _ns)\n"
        "        _passed += 1\n"
        "    except AssertionError:\n"
        "        if _status == 'AC':\n"
        "            _status = 'WA'\n"
        "        if not _first_tb:\n"
        "            _first_tb = traceback.format_exc()\n"
        "    except MemoryError:\n"
        "        _status = 'MLE'\n"
        "        if not _first_tb:\n"
        "            _first_tb = traceback.format_exc()\n"
        "    except Exception:\n"
        "        if _status not in ('MLE',):\n"
        "            _status = 'RE'\n"
        "        if not _first_tb:\n"
        "            _first_tb = traceback.format_exc()\n"
        "if _passed == len(_assertions) and len(_assertions) > 0:\n"
        "    _status = 'AC'\n"
        "sys.stderr.write(f'###FAST_RES###{_passed}|{len(_assertions)}|{_status}\\n' + _first_tb)\n"
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable or "python", "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        err = proc.stderr or ""
        marker = "###FAST_RES###"
        if marker in err:
            after = err.split(marker, 1)[1]
            header, _, tb_rest = after.partition("\n")
            parts = header.strip().split("|")
            if len(parts) == 3:
                passed = int(parts[0])
                tot = int(parts[1])
                st = parts[2]
                return ExecutionResult(
                    status=st,
                    passed_tests=passed,
                    total_tests=tot,
                    stdout=proc.stdout,
                    stderr=tb_rest.strip(),
                    traceback=tb_rest.strip(),
                    execution_time=elapsed,
                )
        if "MemoryError" in err:
            st = ExecutionStatus.MLE
        elif "SyntaxError" in err or "IndentationError" in err:
            st = ExecutionStatus.CE
        else:
            st = ExecutionStatus.RE
        return ExecutionResult(st, 0, total, stdout=proc.stdout, stderr=err.strip(), traceback=err.strip(), execution_time=elapsed)
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            status=ExecutionStatus.TLE,
            passed_tests=0,
            total_tests=total,
            stderr=f"Execution timed out after {timeout} seconds.",
            traceback=f"TimeoutError: Code execution exceeded {timeout}s.",
            execution_time=timeout,
        )
    except Exception as e:
        return ExecutionResult(status=ExecutionStatus.RE, passed_tests=0, total_tests=total, stderr=str(e), traceback=str(e))


def _execute(
    code: str,
    test_cases: list,
    sandbox: Optional[PythonSandbox] = None,
    problem: str = "",
    timeout: float = 2.5,
) -> ExecutionResult:
    """Execute against explicit benchmark tests; empty tests are never AC."""
    sandbox = sandbox or PythonSandbox(default_timeout=timeout)
    if not test_cases:
        return ExecutionResult("CE", 0, 0, traceback="No executable test cases supplied.")
    exec_code = _prepare_executable_code(code, problem)
    if all(isinstance(t, str) for t in test_cases):
        syntax_err = sandbox.check_syntax(exec_code)
        if syntax_err:
            syntax_err.total_tests = 1
            return syntax_err
        r = run_code(exec_code + "\n\n" + "\n\n".join(test_cases), timeout=max(1, int(round(timeout))))
        return ExecutionResult(
            r["status"],
            1 if r["status"] == "AC" else 0,
            1,
            stdout=r.get("output", ""),
            traceback=r.get("traceback", ""),
        )
    if all(isinstance(t, dict) and "assertion" in t and "input" not in t for t in test_cases):
        return _run_assertion_suite_fast(exec_code, [t["assertion"] for t in test_cases], timeout=timeout)
    return sandbox.run_tests(exec_code, test_cases, timeout_per_test=timeout)


def _resolve_eos_token_id(tokenizer, model) -> Optional[int]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, list):
        eos_id = eos_id[0] if eos_id else None
    if eos_id is None and hasattr(model, "config"):
        cfg_eos = getattr(model.config, "eos_token_id", None)
        eos_id = cfg_eos[0] if isinstance(cfg_eos, list) and cfg_eos else cfg_eos
    return eos_id


def _generate_code_batch(
    model,
    tokenizer,
    prompts: List[str],
    do_sample: bool,
    max_new_tokens: int = 320,
    max_gen_time: Optional[float] = None,
) -> List[str]:
    """Generate and extract Python code for a batch of prompts with early fence stopping and OOM backoff."""
    if not prompts:
        return []
    device = next(model.parameters()).device
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = _resolve_eos_token_id(tokenizer, model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        fence_proc = _FenceEosLogitsProcessor(tokenizer, prompt_len=prompt_len, batch_size=len(prompts), eos_token_id=eos_id)
        stop_crit = _AllRowsDoneCriteria(fence_proc)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": pad_id,
            "logits_processor": LogitsProcessorList([fence_proc]),
            "stopping_criteria": StoppingCriteriaList([stop_crit]),
        }
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        if do_sample:
            gen_kwargs.update({"temperature": 1.0, "top_p": 0.95})
        if max_gen_time:
            gen_kwargs["max_time"] = max_gen_time

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        codes: List[str] = []
        for b, prompt in enumerate(prompts):
            gen_text = tokenizer.decode(outputs[b, prompt_len:], skip_special_tokens=True)
            codes.append(extract_code_block(prompt + gen_text, prompt))
        return codes
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and len(prompts) > 1:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(prompts) // 2
            left = _generate_code_batch(model, tokenizer, prompts[:mid], do_sample, max_new_tokens, max_gen_time)
            right = _generate_code_batch(model, tokenizer, prompts[mid:], do_sample, max_new_tokens, max_gen_time)
            return left + right
        raise
    finally:
        tokenizer.padding_side = orig_padding_side


def batch_agentic_debug_loop(
    model,
    tokenizer,
    problems: List[str],
    test_cases_list: List[list],
    K: int = 3,
    max_new_tokens: int = 320,
    max_gen_time: Optional[float] = 45.0,
    feedback_mode: str = "F2",
    fake_feedback: bool = False,
    exec_workers: int = 0,
    exec_timeout: float = 2.5,
) -> List[list]:
    """Run multi-turn execution-guided debugging across a batch of problems in lockstep."""
    n = len(problems)
    if n == 0:
        return []
    histories: List[list] = [[] for _ in range(n)]
    codes: List[Optional[str]] = [None] * n
    errors: List[str] = [""] * n
    active: List[int] = list(range(n))
    sandbox = PythonSandbox(default_timeout=exec_timeout)
    workers = exec_workers if exec_workers > 0 else min(8, max(2, os.cpu_count() or 4))
    fake_errors = [
        'NameError: name "x" is not defined',
        "IndexError: list index out of range",
        "TypeError: unsupported operand type(s)",
    ]

    for turn in range(K):
        if not active:
            break
        prompts = [
            build_prompt(problems[idx], codes[idx], random.choice(fake_errors))
            if fake_feedback
            else build_prompt(problems[idx], codes[idx], errors[idx], feedback_mode=feedback_mode)
            for idx in active
        ]
        batch_codes = _generate_code_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            do_sample=(turn > 0),
            max_new_tokens=max_new_tokens,
            max_gen_time=max_gen_time,
        )

        def _exec_item(pair):
            prob_idx, cand_code = pair
            return _execute(cand_code, test_cases_list[prob_idx], sandbox, problem=problems[prob_idx], timeout=exec_timeout)

        if len(active) > 1 and workers > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(active))) as pool:
                results = list(pool.map(_exec_item, zip(active, batch_codes)))
        else:
            results = [_exec_item(pair) for pair in zip(active, batch_codes)]

        next_active: List[int] = []
        for idx, prompt, cand_code, res in zip(active, prompts, batch_codes, results):
            codes[idx] = cand_code
            entry = {
                "turn": turn + 1,
                "prompt": prompt,
                "code": cand_code,
                "feedback_mode": "FAKE" if fake_feedback else feedback_mode,
                "result": res.to_dict(),
            }
            histories[idx].append(entry)
            if res.status != ExecutionStatus.AC:
                errors[idx] = format_feedback_by_mode(res, feedback_mode=feedback_mode, max_chars=512)
                next_active.append(idx)
        active = next_active

    return histories


def agentic_debug_loop(
    model,
    tokenizer,
    problem: str,
    test_cases: list = None,
    K: int = 3,
    initial_code: str = None,
    max_gen_time: Optional[float] = None,
    feedback_mode: str = "F2",
    max_new_tokens: int = 320,
) -> list:
    """Run up to K turns; every retry receives execution feedback under ``feedback_mode``.

    ``max_gen_time`` (seconds) caps each ``model.generate`` call (wall clock)
    so a wedged GPU/driver cannot hang the loop indefinitely.
    """
    if initial_code is None:
        return batch_agentic_debug_loop(
            model=model,
            tokenizer=tokenizer,
            problems=[problem],
            test_cases_list=[test_cases or []],
            K=K,
            max_new_tokens=max_new_tokens,
            max_gen_time=max_gen_time,
            feedback_mode=feedback_mode,
            exec_workers=1,
        )[0]

    history, code, error = [], initial_code, ""
    tests, sandbox = test_cases or [], PythonSandbox()
    for turn in range(K):
        if turn == 0 and code is not None:
            prompt = build_prompt(problem)
        else:
            prompt = build_prompt(problem, code, error, feedback_mode=feedback_mode)
            code = _generate_code_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=[prompt],
                do_sample=(turn > 0),
                max_new_tokens=max_new_tokens,
                max_gen_time=max_gen_time,
            )[0]
        result = _execute(code, tests, sandbox, problem=problem)
        history.append(
            {
                "turn": turn + 1,
                "prompt": prompt,
                "code": code,
                "feedback_mode": feedback_mode,
                "result": result.to_dict(),
            }
        )
        if result.status == ExecutionStatus.AC:
            break
        error = format_feedback_by_mode(result, feedback_mode=feedback_mode, max_chars=512)
    return history


def agentic_loop_repeated_sampling(model, tokenizer, problem: str, test_cases: list = None, K: int = 3) -> list:
    """B1 / F0 control: repeated sampling with no execution feedback under the same K & decoding budget."""
    return agentic_debug_loop(model, tokenizer, problem, test_cases=test_cases, K=K, feedback_mode="F0")


def agentic_loop_no_feedback(model, tokenizer, problem: str, test_cases: list = None, K: int = 3) -> list:
    """RQ2 control: random fake error context, but correctness is checked on real tests."""
    return batch_agentic_debug_loop(
        model=model,
        tokenizer=tokenizer,
        problems=[problem],
        test_cases_list=[test_cases or []],
        K=K,
        fake_feedback=True,
        exec_workers=1,
    )[0]


def format_execution_feedback(result: ExecutionResult) -> str:
    if result.status == ExecutionStatus.AC:
        return "All test cases PASSED successfully."
    feedback = [f"Execution Status: {result.status}"]
    if result.traceback:
        feedback.append(f"Traceback / Error Details:\n{result.traceback}")
    elif result.stderr:
        feedback.append(f"Error Output (stderr):\n{result.stderr}")
    elif result.stdout:
        feedback.append(f"Output (stdout):\n{result.stdout}")
    if result.total_tests > 0:
        feedback.append(f"Passed {result.passed_tests} / {result.total_tests} test cases.")
    return "\n".join(feedback)


class DebugLoop:
    """Controls multi-turn debugging sessions."""

    def __init__(self, sandbox: Optional[PythonSandbox] = None, max_turns: int = 5, traceback_token_cap: int = 500):
        self.sandbox, self.max_turns, self.traceback_token_cap = (
            sandbox or PythonSandbox(),
            max_turns,
            traceback_token_cap,
        )

    def build_initial_prompt(self, problem_description: str) -> str:
        return (
            "Solve the following programming problem in Python.\nProblem Statement:\n"
            f"{problem_description}\n\nWrite clean, self-contained Python code wrapped in ```python ... ```."
        )

    def build_turn_prompt(
        self, problem_description: str, previous_code: str, execution_result: ExecutionResult
    ) -> str:
        return (
            f"Problem Statement:\n{problem_description}\n\nPrevious solution failed execution.\n"
            f"Previous Code:\n```python\n{previous_code}\n```\n\nExecution Feedback:\n"
            f"{format_execution_feedback(execution_result)}\n\n"
            "Identify the bug, correct the code, and provide the fixed Python solution wrapped in ```python ... ```."
        )

    def run_session(
        self,
        problem_description: str,
        test_cases: List[Dict[str, Any]],
        model_fn: Callable[[str], str],
        initial_code: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            history.append(
                {
                    "turn": turn_idx,
                    "prompt": prompt,
                    "code": current_code,
                    "execution_result": exec_res.to_dict(),
                    "reward": compute_partial_reward(exec_res),
                    "status": exec_res.status,
                }
            )
            if exec_res.status == ExecutionStatus.AC:
                solved, solved_turn = True, turn_idx
                break
            if turn_idx < self.max_turns:
                current_code = extract_code_block(
                    model_fn(self.build_turn_prompt(problem_description, current_code, exec_res))
                )
        return {
            "solved": solved,
            "solved_turn": solved_turn,
            "total_turns": len(history),
            "max_turns": self.max_turns,
            "history": history,
            "final_status": history[-1]["status"] if history else "CE",
            "final_reward": history[-1]["reward"] if history else -0.2,
        }
