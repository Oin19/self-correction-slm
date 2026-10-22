"""Supervised fine-tuning entry points."""

import os
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.models.generation import build_prompt, build_repair_prompt
from src.utils.repro import set_global_seed, write_run_manifest


def format_for_sft(example, tokenizer=None):
    """Format problem + solution (or multi-turn repair trajectory) into a single training string."""
    problem = example.get("question", example.get("prompt", ""))
    
    # Check if this is a repair trajectory
    if "prev_code" in example and "feedback" in example:
        solution = example.get("solution", example.get("fixed_code", ""))
        text = f"{build_repair_prompt(problem, example['prev_code'], example['feedback'])}{solution.strip()}\n```"
    else:
        solutions = example.get("solutions", [])
        if isinstance(solutions, list) and len(solutions) > 0:
            solution = solutions[0]
        else:
            solution = example.get("solution", example.get("canonical_solution", ""))
        text = f"{build_prompt(problem)}{solution.strip()}\n```"

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
    seed: int = 42,
    dataset_split: str = "train[:1500]",
):
    """Run supervised fine-tuning on APPS problem-solution pairs with provenance logging."""
    set_global_seed(seed)
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

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
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    base_name = getattr(getattr(model, "config", None), "_name_or_path", "deepseek-ai/deepseek-coder-1.3b-instruct")
    write_run_manifest(
        output_dir=final_dir,
        run_id=f"sft_seed_{seed}",
        model_name_or_path=base_name,
        stage="sft",
        seed=seed,
        hyperparameters={
            "num_epochs": num_epochs,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
        },
        dataset_split=dataset_split,
        turn_budget_k=1,
        reward_mode_or_loss="cross_entropy",
        parent_checkpoint="none",
    )
    return trainer
