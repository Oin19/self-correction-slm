"""Model loading utilities with quantization and LoRA configuration."""

import torch

# Safely handle Kaggle incompatible torchao version check in peft
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    attach_lora: bool = True,
):
    """Loads tokenizer and CausalLM model with quantization, multi-GPU distribution, and optional LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import bitsandbytes
        has_bnb = True
    except (ImportError, Exception):
        has_bnb = False

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        print(f"Multi-GPU detected! ({num_gpus} GPUs available). Distributing model across GPU 0 & GPU 1 using device_map='auto'", flush=True)
        device_map = "auto"
    elif num_gpus == 1:
        print("Single GPU detected! Using cuda:0", flush=True)
        device_map = {"": 0}
    else:
        device_map = None

    if load_in_4bit and has_bnb and torch.cuda.is_available():
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception:
            bnb_config = None
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )

    if attach_lora:
        try:
            import peft.import_utils
            peft.import_utils.is_torchao_available = lambda: False
        except Exception:
            pass
        try:
            import peft.tuners.lora.torchao
            peft.tuners.lora.torchao.is_torchao_available = lambda: False
        except Exception:
            pass

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer
