"""Code generation and prompt extraction utilities."""

import torch


def generate_code(model, tokenizer, problem: str, temperature: float = 0.2, max_new_tokens: int = 512) -> str:
    """Generate one Python code solution for a given problem statement."""
    prompt = f"### Problem:\n{problem}\n\n### Python Solution:\n```python"
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
    """Extract code block from raw generated LLM text."""
    if "```python" in text:
        code = text.split("```python")[-1].split("```")[0].strip()
    elif "```" in text:
        code = text.split("```")[-1].split("```")[0].strip()
    elif prompt and text.startswith(prompt):
        code = text[len(prompt) :].strip()
    else:
        code = text.strip()
    return code
