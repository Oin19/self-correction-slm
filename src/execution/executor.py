"""Sandboxed execution interface for running Python code safely."""

import ast
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.execution.status import ExecutionStatus
from src.utils.test_cases import score_io_output

try:
    import resource
except ImportError:
    resource = None


def run_code(code: str, timeout: int = 3) -> dict:
    """Run a Python program and classify process-level failures.

    This function is intentionally process-level only. Benchmark correctness must
    use ``PythonSandbox.run_tests`` with explicit test cases.
    """
    python_cmd = sys.executable or ("python3" if os.name != "nt" else "python")

    def preexec_limit():
        if resource is not None and os.name != "nt":
            try:
                resource.setrlimit(resource.RLIMIT_AS, (1_000_000_000, 1_000_000_000))
            except Exception:
                pass

    try:
        result = subprocess.run(
            [python_cmd, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=preexec_limit if (resource is not None and os.name != "nt") else None,
        )
        if result.returncode == 0:
            return {"status": "AC", "output": result.stdout, "traceback": ""}
        tb = result.stderr
        if "SyntaxError" in tb or "IndentationError" in tb:
            status = "CE"
        elif "MemoryError" in tb:
            status = "MLE"
        else:
            status = "RE"
        return {"status": status, "output": result.stdout, "traceback": tb}
    except subprocess.TimeoutExpired:
        return {"status": "TLE", "output": "", "traceback": "Execution timed out"}
    except MemoryError:
        return {"status": "MLE", "output": "", "traceback": "Memory limit exceeded"}
    except Exception as e:
        return {"status": "RE", "output": "", "traceback": str(e)}


@dataclass
class ExecutionResult:
    status: str
    passed_tests: int = 0
    total_tests: int = 0
    stdout: str = ""
    stderr: str = ""
    traceback: str = ""
    execution_time: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "pass_rate": self.pass_rate,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "traceback": self.traceback,
            "execution_time": self.execution_time,
            "details": self.details,
        }


class PythonSandbox:
    """Execute Python code against explicit assertion or I/O test cases."""

    def __init__(self, default_timeout: float = 5.0, max_memory_mb: float = 1024.0):
        self.default_timeout = default_timeout
        self.max_memory_mb = max_memory_mb

    def check_syntax(self, code: str) -> Optional[ExecutionResult]:
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return ExecutionResult(
                status=ExecutionStatus.CE,
                stderr=str(e),
                traceback=f"SyntaxError: {e.msg} at line {e.lineno}, column {e.offset}\n{e.text or ''}",
            )
        except Exception as e:
            return ExecutionResult(status=ExecutionStatus.CE, stderr=str(e), traceback=traceback.format_exc())

    def run_single(self, code: str, test_case: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> ExecutionResult:
        timeout = timeout or self.default_timeout
        syntax_err = self.check_syntax(code)
        if syntax_err:
            syntax_err.total_tests = 1
            return syntax_err

        harness_code = self._build_runner_script(code, test_case)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(harness_code)
            tmp_path = tmp_file.name

        start = time.time()
        try:
            def preexec_limit():
                if resource is not None and os.name != "nt":
                    try:
                        limit = int(self.max_memory_mb * 1024 * 1024)
                        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                    except Exception:
                        pass

            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=preexec_limit if (resource is not None and os.name != "nt") else None,
            )
            elapsed = time.time() - start
            stdout, stderr = proc.stdout, proc.stderr

            if test_case and "input" in test_case and "output" in test_case and isinstance(test_case.get("input"), str):
                actual = self._extract_actual_output(stderr)
                if actual is None:
                    actual = stdout
                expected = str(test_case.get("output", "")).strip()
                packed = int(test_case.get("packed_tests", 1) or 1)
                passed, total = score_io_output(actual, expected, packed)
                tb = ""
                if passed < total:
                    tb = f"AssertionError: expected {total} matching segment(s)/line(s), got {passed}. expected={expected!r} actual={actual.strip()!r}"
                if proc.returncode == 0:
                    status = ExecutionStatus.AC if passed >= total and total > 0 else ExecutionStatus.WA
                    return ExecutionResult(status, passed, total, actual, stderr, tb, execution_time=elapsed)
                if proc.returncode == 2:
                    return ExecutionResult(ExecutionStatus.WA, passed, total, actual, stderr, tb or stderr.strip(), execution_time=elapsed)
                if "MemoryError" in stderr:
                    status = ExecutionStatus.MLE
                elif "SyntaxError" in stderr or "IndentationError" in stderr:
                    status = ExecutionStatus.CE
                else:
                    status = ExecutionStatus.RE
                return ExecutionResult(status, passed, total, actual, stderr, tb or stderr.strip(), execution_time=elapsed)

            if proc.returncode == 0 and "PASSED_TEST_MARKER" in (stdout + stderr):
                return ExecutionResult("AC", 1, 1, stdout.replace("PASSED_TEST_MARKER", "").strip(), stderr, execution_time=elapsed)
            if proc.returncode == 0:
                # No explicit test marker means the harness did not verify correctness.
                return ExecutionResult("WA", 0, 1, stdout, stderr, "No test assertion/output check was executed.", execution_time=elapsed)
            if proc.returncode == 2:
                return ExecutionResult("WA", 0, 1, stdout, stderr, stderr.strip(), execution_time=elapsed)
            if "MemoryError" in stderr:
                status = "MLE"
            elif "SyntaxError" in stderr or "IndentationError" in stderr:
                status = "CE"
            else:
                status = "RE"
            return ExecutionResult(status, 0, 1, stdout, stderr, stderr.strip(), execution_time=elapsed)
        except subprocess.TimeoutExpired:
            if test_case and "input" in test_case and "output" in test_case:
                packed = int(test_case.get("packed_tests", 1) or 1)
                return ExecutionResult(
                    ExecutionStatus.TLE,
                    0,
                    packed,
                    stderr=f"Execution timed out after {timeout} seconds.",
                    traceback=f"TimeoutError: Code execution exceeded {timeout}s.",
                )
            return ExecutionResult("TLE", 0, 1, stderr=f"Execution timed out after {timeout} seconds.", traceback=f"TimeoutError: Code execution exceeded {timeout}s.")
        except Exception as e:
            return ExecutionResult("RE", 0, 1, stderr=str(e), traceback=traceback.format_exc())
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def run_tests(self, code: str, test_cases: List[Dict[str, Any]], timeout_per_test: Optional[float] = None) -> ExecutionResult:
        """Run code against every supplied test case; never treat an empty suite as AC."""
        if not test_cases:
            return ExecutionResult("CE", 0, 0, traceback="No executable test cases supplied.")

        syntax_err = self.check_syntax(code)
        if syntax_err:
            syntax_err.total_tests = len(test_cases)
            return syntax_err

        passed = 0
        details = []
        first_error = ""
        overall_status = ExecutionStatus.AC
        total = 0
        start = time.time()

        for i, test in enumerate(test_cases):
            if isinstance(test, str):
                test = {"assertion": test}
            res = self.run_single(code, test_case=test, timeout=timeout_per_test)
            case_total = res.total_tests if res.total_tests > 0 else 1
            case_passed = max(0, min(res.passed_tests, case_total))
            passed += case_passed
            total += case_total
            details.append({
                "test_index": i,
                "status": res.status,
                "passed": case_passed,
                "total": case_total,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "traceback": res.traceback,
            })
            if res.status != ExecutionStatus.AC and not first_error:
                first_error = res.traceback or res.stderr
            if res.status == ExecutionStatus.TLE:
                overall_status = ExecutionStatus.TLE
            elif res.status == ExecutionStatus.MLE and overall_status != ExecutionStatus.TLE:
                overall_status = ExecutionStatus.MLE
            elif res.status == ExecutionStatus.RE and overall_status not in (ExecutionStatus.TLE, ExecutionStatus.MLE):
                overall_status = ExecutionStatus.RE
            elif res.status in (ExecutionStatus.CE, ExecutionStatus.WA) and overall_status not in (ExecutionStatus.TLE, ExecutionStatus.MLE, ExecutionStatus.RE):
                overall_status = res.status

        if total > 0 and passed == total:
            overall_status = ExecutionStatus.AC
        return ExecutionResult(
            status=overall_status,
            passed_tests=passed,
            total_tests=total,
            traceback=first_error,
            execution_time=time.time() - start,
            details=details,
        )

    @staticmethod
    def _extract_actual_output(stderr: str) -> Optional[str]:
        start = "###ACTUAL###"
        end = "###END###"
        if start not in stderr:
            return None
        body = stderr.split(start, 1)[1]
        if end not in body:
            return None
        return body.split(end, 1)[0]

    def _build_runner_script(self, code: str, test_case: Optional[Dict[str, Any]]) -> str:
        """Build a harness that executes the solution only after test I/O is installed."""
        test_case = test_case or {}

        if "input" in test_case and "output" in test_case and isinstance(test_case["input"], str):
            inp = repr(test_case["input"])
            return (
                "import io, sys, traceback\n"
                f"_solution = {code!r}\n"
                f"sys.stdin = io.StringIO({inp})\n"
                "out_buf = io.StringIO()\n"
                "sys.stdout = out_buf\n"
                "def _emit(actual):\n"
                "    sys.stderr.write('###ACTUAL###' + actual + '###END###\\n')\n"
                "try:\n"
                "    exec(_solution, globals())\n"
                "    actual = out_buf.getvalue().strip()\n"
                "    if not actual:\n"
                "        for fn_name in ['solve', 'main', 'solution', 'run']:\n"
                "            if fn_name in globals() and callable(globals()[fn_name]):\n"
                "                try:\n"
                "                    res = globals()[fn_name]()\n"
                "                    if res is not None and not out_buf.getvalue().strip():\n"
                "                        print(res)\n"
                "                    actual = out_buf.getvalue().strip()\n"
                "                    if actual:\n"
                "                        break\n"
                "                except Exception:\n"
                "                    pass\n"
                "    _emit(actual)\n"
                "    sys.exit(0)\n"
                "except SystemExit as e:\n"
                "    _emit(out_buf.getvalue().strip())\n"
                "    sys.exit(0 if e.code in (0, None) else 2)\n"
                "except Exception:\n"
                "    _emit(out_buf.getvalue().strip())\n"
                "    traceback.print_exc(file=sys.stderr)\n"
                "    sys.exit(1)\n"
            )

        script = f"{code}\n\n"
        if "assertion" in test_case:
            script += (
                "import sys, traceback\n"
                "if 'Solution' in globals() and isinstance(globals()['Solution'], type):\n"
                "    try:\n"
                "        _sol_inst = globals()['Solution']()\n"
                "        for _attr in dir(_sol_inst):\n"
                "            if not _attr.startswith('_') and callable(getattr(_sol_inst, _attr)) and _attr not in globals():\n"
                "                globals()[_attr] = getattr(_sol_inst, _attr)\n"
                "    except Exception:\n"
                "        pass\n"
                "try:\n"
                f"    {test_case['assertion']}\n"
                "    print('PASSED_TEST_MARKER')\n"
                "except AssertionError:\n"
                "    traceback.print_exc()\n"
                "    sys.exit(2)\n"
            )
        elif "fn_name" in test_case and "input" in test_case and "expected" in test_case:
            fn_name = test_case["fn_name"]
            inputs = test_case["input"]
            expected = repr(test_case["expected"])
            if isinstance(inputs, list):
                args_str = ", ".join(repr(x) for x in inputs)
            elif isinstance(inputs, dict):
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in inputs.items())
            else:
                args_str = repr(inputs)
            script += (
                "import sys, traceback\ntry:\n"
                f"    actual = {fn_name}({args_str})\n"
                f"    expected = {expected}\n"
                "    if actual == expected:\n"
                "        print('PASSED_TEST_MARKER')\n"
                "    else:\n"
                "        print(f'AssertionError: Expected {expected!r}, got {actual!r}', file=sys.stderr)\n"
                "        sys.exit(2)\n"
                "except SystemExit as e:\n    sys.exit(e.code)\n"
                "except Exception:\n    traceback.print_exc(); sys.exit(1)\n"
            )
        else:
            script += "import sys\nsys.exit(2)\n"
        return script
