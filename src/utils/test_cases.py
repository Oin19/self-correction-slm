"""Benchmark test-case normalization (no heavy ML deps)."""
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


def normalize_tests(ex: dict) -> list:
    """Normalize APPS/HumanEval/MBPP test fields into executor-ready cases.

    Prefers multi-case sources (input_output / test_list) over a monolithic
    ``test`` script so passed/total can be > 1 and partial reward works.
    Packed APPS stdin suites keep ``packed_tests=T`` so the executor can
    award segment/line partial credit instead of a single 0/1.
    Never duplicates cases to pad the count.
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

    if isinstance(ex.get("test"), str) and ex["test"].strip():
        sources.append([ex["test"]])

    if not sources:
        return []
    # Most cases first so partial credit has room; ties keep preference order above.
    return max(sources, key=len)
