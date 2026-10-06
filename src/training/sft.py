"""Supervised fine-tuning entry points."""

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


def format_for_sft(example, tokenizer=None):
    """Format problem + solution into a single training string."""
    problem = example.get("question", example.get("prompt", ""))
    solutions = example.get("solutions", [])
    if isinstance(solutions, list) and len(solutions) > 0:
        solution = solutions[0]
    else:
        solution = example.get("solution", example.get("canonical_solution", ""))

    text = f"### Problem:\n{problem}\n\n### Solution:\n```python\n{solution}\n```"

    if tokenizer is not None:
        return tokenizer(text, truncation=True, max_length=1024, padding="max_length")
    return {"text": text}


def run_sft_training(
    model,
    tokenizer,
    dataset,
    output_dir: str = "./checkpoints/sft",
    num_epochs: int = 3,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
):
    """Run supervised fine-tuning on APPS problem-solution pairs."""
    tokenized_dataset = dataset.map(
        lambda x: format_for_sft(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        save_steps=200,
        logging_steps=50,
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    return trainer
