from src.execution.executor import PythonSandbox


def test_empty_suite_is_not_correct():
    result = PythonSandbox().run_tests("print('hello')", [])
    assert result.status != "AC"


def test_assertion_suite_checks_correctness():
    code = "def add(a, b):\n    return a + b\n"
    result = PythonSandbox().run_tests(code, [{"assertion": "assert add(2, 3) == 5"}])
    assert result.status == "AC"
    assert result.passed_tests == 1


def test_assertion_suite_detects_wrong_answer():
    code = "def add(a, b):\n    return a - b\n"
    result = PythonSandbox().run_tests(code, [{"assertion": "assert add(2, 3) == 5"}])
    assert result.status == "WA"
    assert result.passed_tests == 0


def test_stdio_suite_sets_input_before_solution_runs():
    code = "x = int(input())\nprint(x * 2)\n"
    result = PythonSandbox().run_tests(code, [{"input": "4\n", "output": "8"}])
    assert result.status == "AC"
