"""Model loading utilities with quantization and LoRA configuration."""

import json
import os
import torch

# Safely handle Kaggle incompatible torchao version check in peft
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

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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
    """Loads tokenizer and CausalLM model with quantization, multi-GPU distribution, and LoRA adapters.

    If ``model_name`` points to a directory containing ``adapter_config.json`` (e.g. a saved
    SFT, PPO, or DPO checkpoint), loads the base model specified in ``adapter_config.json``
    and restores the trained LoRA adapter via ``PeftModel.from_pretrained`` instead of
    initializing a fresh random LoRA adapter.
    """
    adapter_cfg_path = (
        os.path.join(model_name, "adapter_config.json")
        if isinstance(model_name, str) and os.path.isdir(model_name)
        else None
    )
    is_saved_adapter = bool(adapter_cfg_path and os.path.isfile(adapter_cfg_path))

    base_model_name = model_name
    if is_saved_adapter:
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path") or DEFAULT_MODEL_NAME

    tok_source = (
        model_name
        if (is_saved_adapter and os.path.isfile(os.path.join(model_name, "tokenizer_config.json")))
        else base_model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import bitsandbytes
        has_bnb = True
    except (ImportError, Exception):
        has_bnb = False

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        print(
            f"Multi-GPU detected! ({num_gpus} GPUs available). Distributing model across GPU 0 & GPU 1 using device_map='auto'",
            flush=True,
        )
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
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )

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

    if is_saved_adapter:
        print(
            f"Restoring trained LoRA adapter from {model_name} (base={base_model_name}, trainable={attach_lora})",
            flush=True,
        )
        model = PeftModel.from_pretrained(model, model_name, is_trainable=attach_lora)
    elif attach_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer
