"""Benchmark test-case normalization (no heavy ML deps)."""
import ast
import json


def packed_test_count(stdin_text: str) -> int:
    """Return T when stdin looks like a packed multi-test suite (first line = T)."""
    if not isinstance(stdin_text, str) or not stdin_text:
        return 1
    first = stdin_text.split("\n", 1)[0].strip()
    if not first.isdigit():
        return 1
    t = int(first)
    if 1 < t <= 1000:
        return t
    return 1


def score_io_output(actual: str, expected: str, packed_tests: int = 1) -> tuple:
    """Partial credit for I/O tests, including packed APPS multi-test stdin.

    Returns (passed, total):
    - Full match (string or token): (max(t,1), max(t,1))
    - Expected non-empty lines == T: score line i as test i
    - Blank-line blocks == T: score block i as test i
    - Else multi-line expected: line-level partial (total = # non-empty lines)
    - Single-line mismatch: (0, max(t, 1))
    """
    exp = (expected or "").strip()
    act = (actual or "").strip()
    t = max(int(packed_tests or 1), 1)

    if not exp:
        return (0, t)
    if act == exp or (act.split() and act.split() == exp.split()):
        return (t, t)

    exp_lines = [ln.strip() for ln in exp.splitlines() if ln.strip()]
    act_lines = [ln.strip() for ln in act.splitlines() if ln.strip()]
    exp_blocks = [b.strip() for b in exp.split("\n\n") if b.strip()]
    act_blocks = [b.strip() for b in act.split("\n\n") if b.strip()]

    if t > 1 and len(exp_lines) == t:
        passed = sum(
            1 for i in range(t) if i < len(act_lines) and act_lines[i] == exp_lines[i]
        )
        return (passed, t)

    if t > 1 and len(exp_blocks) == t:
        passed = sum(
            1 for i in range(t) if i < len(act_blocks) and act_blocks[i] == exp_blocks[i]
        )
        return (passed, t)

    if len(exp_lines) > 1:
        n = len(exp_lines)
        m = len(act_lines)
        passed = sum(1 for i in range(min(n, m)) if act_lines[i] == exp_lines[i])
        return (passed, n)

    return (0, t)


def _cases_from_input_output(raw) -> list:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []
    ins, outs, fn = raw.get("inputs", []), raw.get("outputs", []), raw.get("fn_name")
    cases = []
    for inp, out in zip(ins, outs):
        if fn:
            if isinstance(inp, list):
                args = ", ".join(repr(x) for x in inp)
            elif isinstance(inp, dict):
                args = ", ".join(f"{k}={v!r}" for k, v in inp.items())
            else:
                args = repr(inp)
            cases.append({"assertion": f"assert {fn}({args}) == {out!r}"})
        else:
            inp_s = "\n".join(inp) if isinstance(inp, list) else str(inp)
            out_s = "\n".join(out) if isinstance(out, list) else str(out)
            case = {"input": inp_s, "output": out_s}
            t = packed_test_count(inp_s)
            if t > 1:
                case["packed_tests"] = t
            cases.append(case)
    return cases


def _extract_humaneval_assertions(test_script: str, entry_point: str) -> list:
    """Extract individual assertion cases from HumanEval check(candidate) function.

    Replaces `candidate` identifier with `entry_point` so each assertion can run
    independently, allowing partial credit and feedback/scoring splitting.
    """
    if not test_script or not entry_point:
        return []
    try:
        tree = ast.parse(test_script)
        preamble_nodes = []
        assertions = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "check":
                for stmt in node.body:
                    if isinstance(stmt, ast.Assert):
                        # Replace 'candidate' with entry_point
                        class CandidateReplacer(ast.NodeTransformer):
                            def visit_Name(self, n):
                                if n.id == "candidate":
                                    return ast.copy_location(ast.Name(id=entry_point, ctx=n.ctx), n)
                                return n
                        new_stmt = CandidateReplacer().visit(stmt)
                        ast.fix_missing_locations(new_stmt)
                        if hasattr(ast, "unparse"):
                            assertions.append(ast.unparse(new_stmt))
            elif not isinstance(node, ast.FunctionDef):
                preamble_nodes.append(node)

        preamble = ""
        if preamble_nodes and hasattr(ast, "unparse"):
            preamble = "\n".join(ast.unparse(n) for n in preamble_nodes).strip()
            if preamble:
                preamble += "\n"

        if assertions:
            return [{"assertion": f"{preamble}{a}".strip()} for a in assertions]
    except Exception:
        pass
    return []


def normalize_tests(ex: dict) -> list:
    """Normalize APPS/HumanEval/MBPP test fields into executor-ready cases.

    Prefers multi-case sources (input_output / test_list / granular assertions)
    over a monolithic ``test`` script so passed/total can be > 1 and partial reward works.
    Packed APPS stdin suites keep ``packed_tests=T`` so the executor can
    award segment/line partial credit instead of a single 0/1.
    Never duplicates cases to pad the count.
    Ensures HumanEval check(entry_point) is always appended and executed.
    """
    sources = []

    raw = ex.get("input_output", ex.get("test_cases", []))
    io_cases = _cases_from_input_output(raw)
    if io_cases:
        sources.append(io_cases)

    if isinstance(ex.get("test_list"), list) and ex["test_list"]:
        tl = [{"assertion": x} for x in ex["test_list"] if isinstance(x, str) and x.strip()]
        if tl:
            sources.append(tl)

    test_str = ex.get("test")
    entry_point = ex.get("entry_point")
    if isinstance(test_str, str) and test_str.strip():
        test_script = test_str.strip()
        # Attempt fine-grained assertion extraction for HumanEval
        if entry_point:
            granular = _extract_humaneval_assertions(test_script, entry_point)
            if granular:
                sources.append(granular)

        # Ensure monolithic check(entry_point) invocation is present
        if entry_point and f"check({entry_point})" not in test_script:
            test_script = f"{test_script}\ncheck({entry_point})\n"
        sources.append([{"assertion": test_script}])

    if not sources:
        return []
    # Most cases first so partial credit has room; ties keep preference order above.
    return max(sources, key=len)


def split_feedback_and_scoring_tests(
    test_cases: list,
    feedback_ratio: float = 0.5,
    min_feedback: int = 1,
    min_scoring: int = 1,
) -> tuple:
    """Split test cases into feedback_tests (for debug loop) and scoring_tests (held-out for evaluation).

    If there is only 1 test case, it is mirrored into both feedback and scoring.
    For >= 2 test cases, cases are partitioned deterministically to guarantee
    disjoint sets where scoring_tests are never leaked into feedback.
    """
    if not test_cases:
        return ([], [])
    if len(test_cases) == 1:
        return (list(test_cases), list(test_cases))

    n = len(test_cases)
    n_feedback = int(round(n * feedback_ratio))
    n_feedback = max(min_feedback, min(n - min_scoring, n_feedback))

    feedback_tests = list(test_cases[:n_feedback])
    scoring_tests = list(test_cases[n_feedback:])
    return (feedback_tests, scoring_tests)


def get_problem_test_split(ex: dict, feedback_ratio: float = 0.5) -> dict:
    """Extract normalized tests and return partitioned feedback and scoring test sets."""
    all_tests = normalize_tests(ex)
    fb, sc = split_feedback_and_scoring_tests(all_tests, feedback_ratio=feedback_ratio)
    return {
        "all_tests": all_tests,
        "feedback_tests": fb,
        "scoring_tests": sc,
    }
