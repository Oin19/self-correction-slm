"""Synthetic Error Injection Engine for Code LLM Self-Correction.
Generates controlled buggy variants of correct Python solutions across 4 categories:
1. Syntax Errors (missing colons, keyword typos, unclosed parentheses)
2. Logic Errors (operator mutation, off-by-one, inverted boolean conditions)
3. Runtime Errors (division by zero, index out of bounds, type mismatches)
4. Infinite Loops / Timeout Errors (unbounded loops, missing base case)

Implementation owned by Junior B.
"""

import ast
import random
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class BugType(str, Enum):
    SYNTAX = "syntax"
    LOGIC = "logic"
    RUNTIME = "runtime"
    INFINITE_LOOP = "infinite_loop"


class BugInjector:
    """Injects controlled synthetic programming errors into valid Python code."""

    def __init__(self, seed: Optional[int] = 42):
        if seed is not None:
            random.seed(seed)

    def inject_syntax_error(self, code: str) -> Tuple[str, str]:
        """Injects syntax errors (missing colons, keyword typos, bad parens)."""
        lines = code.splitlines()
        colon_lines = [i for i, line in enumerate(lines) if line.rstrip().endswith(":")]
        def_lines = [i for i, line in enumerate(lines) if "def " in line]
        return_lines = [i for i, line in enumerate(lines) if "return " in line]

        choice = random.choice(["missing_colon", "keyword_typo", "unmatched_paren"])

        if choice == "missing_colon" and colon_lines:
            target_idx = random.choice(colon_lines)
            lines[target_idx] = lines[target_idx].rstrip()[:-1]
            return "\n".join(lines), f"Syntax Error: Removed colon on line {target_idx + 1}"

        elif choice == "keyword_typo" and (def_lines or return_lines):
            if def_lines and random.random() < 0.5:
                idx = random.choice(def_lines)
                lines[idx] = lines[idx].replace("def ", "deff ", 1)
                return "\n".join(lines), f"Syntax Error: Typo 'deff' on line {idx + 1}"
            elif return_lines:
                idx = random.choice(return_lines)
                lines[idx] = lines[idx].replace("return ", "retun ", 1)
                return "\n".join(lines), f"Syntax Error: Typo 'retun' on line {idx + 1}"

        # Fallback: remove random colon or bracket
        if ":" in code:
            buggy = code.replace(":", "", 1)
            return buggy, "Syntax Error: Removed first colon in file"

        buggy = code + "\n    ("
        return buggy, "Syntax Error: Added unmatched parenthesis"

    def inject_logic_error(self, code: str) -> Tuple[str, str]:
        """Injects logic errors (operator mutation, off-by-one, condition inversion)."""
        op_map = {
            " + ": " - ",
            " - ": " + ",
            " == ": " != ",
            " != ": " == ",
            " > ": " < ",
            " < ": " > ",
            " >= ": " <= ",
            " <= ": " >= ",
            " and ": " or ",
            " or ": " and ",
        }

        # Strategy 1: Operator mutation
        for op, repl in op_map.items():
            if op in code:
                buggy = code.replace(op, repl, 1)
                return buggy, f"Logic Error: Replaced '{op.strip()}' with '{repl.strip()}'"

        # Strategy 2: Off-by-one error in range() or indexing
        if "range(" in code:
            buggy = re.sub(r"range\(([^)]+)\)", r"range(\1 - 1)", code, count=1)
            return buggy, "Logic Error: Off-by-one error injected into range()"

        # Strategy 3: Invert boolean if statement
        if "if " in code:
            buggy = re.sub(r"if\s+([^:]+):", r"if not (\1):", code, count=1)
            return buggy, "Logic Error: Inverted condition in if-statement"

        # Fallback mutation
        buggy = code.replace("return ", "return 0 # ", 1) if "return " in code else code + "\n    pass"
        return buggy, "Logic Error: Overrode return value"

    def inject_runtime_error(self, code: str) -> Tuple[str, str]:
        """Injects runtime errors (zero division, invalid index, type mismatch)."""
        choice = random.choice(["zero_division", "index_out_of_bounds", "type_error"])
        lines = code.splitlines()

        if choice == "zero_division":
            # Insert a division by zero before return or main logic
            for i, line in enumerate(lines):
                if line.strip().startswith("return "):
                    indent = line[: len(line) - len(line.lstrip())]
                    lines.insert(i, f"{indent}_dummy_div = 1 / 0")
                    return "\n".join(lines), "Runtime Error: Injected division by zero (ZeroDivisionError)"

        elif choice == "index_out_of_bounds":
            for i, line in enumerate(lines):
                if "return " in line or line.strip().startswith("def "):
                    indent = line[: len(line) - len(line.lstrip())]
                    lines.insert(i, f"{indent}_dummy_arr = []; _val = _dummy_arr[999]")
                    return "\n".join(lines), "Runtime Error: Injected index out of bounds (IndexError)"

        # Fallback: type error
        for i, line in enumerate(lines):
            if "def " in line:
                indent = line[: len(line) - len(line.lstrip())] + "    "
                lines.insert(i + 1, f"{indent}_dummy_type = 'str' + 42")
                return "\n".join(lines), "Runtime Error: Injected type mismatch (TypeError)"

        return code + "\n_dummy = 1 / 0", "Runtime Error: Division by zero appended"

    def inject_infinite_loop(self, code: str) -> Tuple[str, str]:
        """Injects infinite loop / time limit exceeded behavior."""
        lines = code.splitlines()

        # Strategy 1: Replace existing loop condition with while True: without break
        for i, line in enumerate(lines):
            if line.strip().startswith("while "):
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = f"{indent}while True:"
                return "\n".join(lines), "Infinite Loop: Converted while condition to infinite loop"

        # Strategy 2: Inject infinite while loop at start of function body
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                indent = line[: len(line) - len(line.lstrip())] + "    "
                lines.insert(i + 1, f"{indent}while True: pass")
                return "\n".join(lines), "Infinite Loop: Injected `while True: pass` into function"

        return code + "\nwhile True: pass", "Infinite Loop: Appended infinite while loop"

    def inject_bug(
        self, code: str, category: Optional[BugType] = None
    ) -> Dict[str, Any]:
        """Injects a specified or randomly chosen bug category into Python code."""
        if category is None:
            category = random.choice(list(BugType))

        if category == BugType.SYNTAX:
            buggy_code, desc = self.inject_syntax_error(code)
        elif category == BugType.LOGIC:
            buggy_code, desc = self.inject_logic_error(code)
        elif category == BugType.RUNTIME:
            buggy_code, desc = self.inject_runtime_error(code)
        elif category == BugType.INFINITE_LOOP:
            buggy_code, desc = self.inject_infinite_loop(code)
        else:
            buggy_code, desc = self.inject_logic_error(code)

        return {
            "original_code": code,
            "buggy_code": buggy_code,
            "bug_type": category.value if isinstance(category, BugType) else str(category),
            "description": desc,
        }

    def create_buggy_dataset(
        self, dataset: List[Dict[str, Any]], solution_key: str = "solution"
    ) -> List[Dict[str, Any]]:
        """Processes a list of problem instances and creates buggy solution variants."""
        augmented = []
        categories = list(BugType)

        for i, item in enumerate(dataset):
            code = item.get(solution_key, "")
            if not code:
                continue
            cat = categories[i % len(categories)]
            result = self.inject_bug(code, category=cat)

            item_copy = dict(item)
            item_copy["buggy_code"] = result["buggy_code"]
            item_copy["bug_type"] = result["bug_type"]
            item_copy["bug_description"] = result["description"]
            augmented.append(item_copy)

        return augmented
