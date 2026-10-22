import json
import os
import torch

# Safely handle Kaggle incompatible torchao version check in peft
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"


def load_canonical_checkpoint(
    checkpoint_path: str,
    base_model_name: str = None,
    load_in_4bit: bool = False,
    is_trainable: bool = False,
):
    """Load a trained LoRA adapter checkpoint using PeftModel.from_pretrained.

    Extracts base_model_name_or_path from adapter_config.json and attaches
    the saved weights, preventing the accidental initialization of a blank LoRA.
    """
    adapter_cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base = base_model_name or cfg.get("base_model_name_or_path", DEFAULT_MODEL_NAME)
    else:
        base = base_model_name or checkpoint_path

    # Try loading tokenizer from adapter dir; fallback to base
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model, _ = load_model_and_tokenizer(
        model_name=base,
        load_in_4bit=load_in_4bit,
        attach_lora=False,
    )

    if os.path.isfile(adapter_cfg_path):
        print(f"Loading trained LoRA adapter from {checkpoint_path} onto base {base}...", flush=True)
        model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=is_trainable)
    else:
        model = base_model

    return model, tokenizer


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    attach_lora: bool = True,
    is_trainable: bool = True,
):
    """Loads tokenizer and CausalLM model with quantization, multi-GPU distribution, and optional LoRA adapters.

    If model_name points to a saved LoRA adapter checkpoint (has adapter_config.json),
    canonical adapter loading is used instead of creating a random blank LoRA.
    """
    if os.path.isdir(model_name) and os.path.isfile(os.path.join(model_name, "adapter_config.json")):
        return load_canonical_checkpoint(
            checkpoint_path=model_name,
            load_in_4bit=load_in_4bit,
            is_trainable=is_trainable,
        )

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
