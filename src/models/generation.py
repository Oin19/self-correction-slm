"""Code generation, prompt formatting, and extraction utilities."""

import re
import torch

PROMPT_TEMPLATE = "### Problem:\n{problem}\n\n### Solution:\n```python\n"
REPAIR_TEMPLATE = (
    "### Problem:\n{problem}\n\n"
    "### Previous Solution:\n```python\n{prev_code}\n```\n\n"
    "### Execution Feedback:\n{feedback}\n\n"
    "### Fixed Solution:\n```python\n"
)


def build_prompt(problem: str) -> str:
    """Canonical unified problem prompt across SFT, PPO, DPO, and evaluation."""
    return PROMPT_TEMPLATE.format(problem=problem.strip())


def build_repair_prompt(problem: str, prev_code: str, feedback: str) -> str:
    """Canonical unified multi-turn repair prompt."""
    return REPAIR_TEMPLATE.format(
        problem=problem.strip(),
        prev_code=prev_code.strip(),
        feedback=(feedback or "Execution failed.").strip(),
    )


def generate_code(model, tokenizer, problem: str, temperature: float = 0.2, max_new_tokens: int = 512) -> str:
    """Generate one Python code solution for a given problem statement."""
    prompt = build_prompt(problem)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01) if temperature > 0 else 1.0,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
        )

    full = tokenizer.decode(output[0], skip_special_tokens=True)
    return extract_code_block(full, prompt)


def extract_code_block(text: str, prompt: str = "") -> str:
    """Extract code block from raw generated LLM text.

    Handles:
    1. Generations where prompt ends with ```python and model output completes directly before closing ```
    2. Model responses containing full ```python ... ``` fences
    3. Multi-block responses and unfenced plain Python code
    Never returns <EMPTY> when executable code is present.
    """
    if not text:
        return ""

    # Strip prompt prefix if present
    content = text
    prompt_stripped = False
    if prompt and content.startswith(prompt):
        content = content[len(prompt):]
        prompt_stripped = True
    else:
        for marker in (
            "### Fixed Solution:\n```python",
            "### Solution:\n```python",
            "### Fixed version:\n```python",
            "### Write a Python solution:\n```python",
            "### Python Solution:\n```python",
        ):
            if marker in content:
                content = content.split(marker, 1)[-1]
                prompt_stripped = True
                break

    # Case A: Prompt already opened ```python (either stripped or prompt ended with ```python)
    prompt_had_opening_fence = prompt_stripped or (prompt and prompt.strip().endswith("```python"))
    if prompt_had_opening_fence and not content.lstrip().startswith("```"):
        if "```" in content:
            first_block = content.split("```", 1)[0].strip()
            if first_block:
                return first_block

    # Case B: If text contains ```python ... ```
    if "```python" in content:
        after = content.split("```python", 1)[1]
        code = after.split("```", 1)[0].strip()
        if code:
            return code

    # Case C: If text contains general ``` ... ``` fences
    fence_count = content.count("```")
    if fence_count >= 2:
        after = content.split("```", 1)[1]
        code = after.split("```", 1)[0].strip()
        lines = code.splitlines()
        if lines and lines[0].strip().lower() in ("python", "py", "python3"):
            code = "\n".join(lines[1:]).strip()
        if code:
            return code
    elif fence_count == 1:
        prefix = content.split("```", 1)[0].strip()
        suffix = content.split("```", 1)[1].strip()
        if len(prefix) >= len(suffix) and prefix:
            return prefix
        elif suffix:
            lines = suffix.splitlines()
            if lines and lines[0].strip().lower() in ("python", "py", "python3"):
                suffix = "\n".join(lines[1:]).strip()
            return suffix

    # Fallback to full text search if prompt stripping was imprecise
    if "```python" in text:
        after = text.split("```python")[-1]
        code = after.split("```", 1)[0].strip()
        if code:
            return code

    if "```" in text and text.count("```") >= 2:
        chunks = text.split("```")
        if len(chunks) >= 3:
            code = chunks[-2].strip()
            lines = code.splitlines()
            if lines and lines[0].strip().lower() in ("python", "py", "python3"):
                code = "\n".join(lines[1:]).strip()
            if code:
                return code

    return content.strip()
