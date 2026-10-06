"""Execution-guided PPO training."""
import gc, json, os
import torch
from peft import PeftModel
from src.execution.executor import PythonSandbox
from src.models.generation import extract_code_block
from src.rewards.execution_reward import score_rollout_reward
from src.utils.repro import git_commit_sha, set_global_seed
from src.utils.test_cases import normalize_tests
from trl import PPOTrainer, PPOConfig
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead

_git_commit_sha = git_commit_sha
REWARD_MODES = ("dense", "binary")

def load_sft_adapter(path,device_map,trainable=True):
    cfg_path=os.path.join(path,"adapter_config.json")
    if not os.path.isfile(cfg_path): raise FileNotFoundError("SFT adapter not found: %s"%path)
    with open(cfg_path,encoding="utf-8") as f: cfg=json.load(f)
    base=cfg.get("base_model_name_or_path")
    if not base: raise ValueError("SFT adapter has no base_model_name_or_path")
    wrapper=AutoModelForCausalLMWithValueHead.from_pretrained(base,torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,device_map=device_map,trust_remote_code=True)
    wrapper.pretrained_model=PeftModel.from_pretrained(wrapper.pretrained_model,path,is_trainable=trainable)
    # TRL: is_peft_model=True forces ref logprobs via disable_adapter (base, no SFT)
    # and ignores an explicitly passed ref_model. Keep False so KL uses our frozen SFT ref.
    wrapper.is_peft_model=False
    if not trainable:
        for p in wrapper.parameters(): p.requires_grad=False
    return wrapper,base

def _parameter_snapshot(parameters):
    total_sq = 0.0
    for p in parameters:
        if p.requires_grad:
            total_sq += float(p.detach().float().pow(2).sum().item())
    return total_sq ** 0.5

def _gradient_norm(parameters):
    total_sq = 0.0
    found = False
    for p in parameters:
        if p.requires_grad and p.grad is not None:
            found = True
            total_sq += float(p.grad.detach().float().pow(2).sum().item())
    return (total_sq ** 0.5) if found else 0.0

def run_ppo_training(sft_model_path,tokenizer,dataset,output_dir=None,num_epochs=1,learning_rate=2e-7,batch_size=2,mini_batch_size=1,gradient_accumulation_steps=2,init_kl_coef=0.30,target_kl=4.0,max_steps=100,reward_mode="dense",seed=42,commit_sha=None):
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}")
    if output_dir is None:
        output_dir = f"./checkpoints/ppo_{reward_mode}"
    set_global_seed(seed)
    print(f"PPO seed={seed} reward_mode={reward_mode} output_dir={output_dir} commit={commit_sha or _git_commit_sha() or 'unknown'}", flush=True)
    tokenizer.padding_side="left"
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    # Keep only examples with executable tests. Use select() (not Dataset.filter) so a
    # stale HF datasets cache cannot wipe the training set after normalize_tests changes.
    n_in=len(dataset)
    sample=dataset[0] if n_in else None
    keep=[i for i,ex in enumerate(dataset) if len(normalize_tests(ex)) > 0]
    if not keep:
        keys=sorted(sample.keys()) if isinstance(sample,dict) else []
        raise ValueError(
            f"No dataset examples contain valid executable benchmark tests "
            f"(input_len={n_in}, valid_len=0, columns={keys}). "
            f"Check normalize_tests against the dataset schema."
        )
    dataset=dataset.select(keep)
    all_benchmark_tests=[normalize_tests(ex) for ex in dataset]
    test_counts=[len(t) for t in all_benchmark_tests]
    multi=sum(1 for c in test_counts if c>1)
    packed_totals=[]
    for t in all_benchmark_tests:
        pt=1
        for c in t:
            if isinstance(c,dict) and int(c.get("packed_tests",1) or 1)>1:
                pt=max(pt,int(c["packed_tests"]))
        packed_totals.append(pt)
    n_packed=sum(1 for p in packed_totals if p>1)
    print(
        f"Benchmark tests: n={len(test_counts)} min={min(test_counts)} "
        f"max={max(test_counts)} mean={sum(test_counts)/len(test_counts):.2f} "
        f"multi_case={multi}/{len(test_counts)} "
        f"packed_multi={n_packed}/{len(packed_totals)} "
        f"packed_total_max={max(packed_totals) if packed_totals else 0}",
        flush=True,
    )
    for i,ex in enumerate(dataset):
        if i>=3: break
        raw_io=ex.get("input_output",None)
        if isinstance(raw_io,str):
            try: raw_io=json.loads(raw_io)
            except Exception: pass
        n_io=len(raw_io.get("inputs",[])) if isinstance(raw_io,dict) else (len(raw_io) if isinstance(raw_io,list) else 0)
        tl=ex.get("test_list") or []
        ts=ex.get("test") or ""
        print(
            f"  test_src[{i}]: normalize={len(all_benchmark_tests[i])} "
            f"io_inputs={n_io} test_list={len(tl) if isinstance(tl,list) else 0} "
            f"test_str_len={len(ts) if isinstance(ts,str) else 0} "
            f"packed_tests={packed_totals[i]}",
            flush=True,
        )
    def encode(ex):
        p=ex.get("question",ex.get("prompt",""))
        p_ids=tokenizer.encode(p,truncation=True,max_length=350)
        p_clean=tokenizer.decode(p_ids,skip_special_tokens=True)
        prompt=f"### Problem:\n{p_clean}\n\n### Solution:\n```python\n"
        t=tokenizer(prompt,truncation=False)
        return {"input_ids":t["input_ids"],"benchmark_tests":normalize_tests(ex)}
    dataset=dataset.map(encode)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    ng=torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map="auto" if ng>=2 else ({"":0} if ng==1 else None)
    model,base=load_sft_adapter(sft_model_path,device_map)
    if hasattr(model.pretrained_model, "gradient_checkpointing_enable"):
        model.pretrained_model.gradient_checkpointing_enable()
    if hasattr(model.pretrained_model, "config"):
        model.pretrained_model.config.use_cache = False
    if hasattr(model.pretrained_model, "enable_input_require_grads"):
        model.pretrained_model.enable_input_require_grads()
    for p in model.pretrained_model.parameters(): p.requires_grad=False
    for n,p in model.named_parameters():
        if "lora_" in n or "v_head" in n or "summary" in n: p.requires_grad=True
    trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable: raise RuntimeError("PPO has no trainable parameters")
    trainable_count=sum(p.numel() for p in trainable)
    total_count=sum(p.numel() for p in model.parameters())
    print(f"PPO trainable parameters: {trainable_count:,} / {total_count:,} ({100.0*trainable_count/total_count:.4f}%)",flush=True)
    opt=torch.optim.AdamW(trainable,lr=learning_rate)
    
    # Hook into optimizer.step() to capture TRUE gradient norm BEFORE zero_grad() clears p.grad!
    captured_grad_norms = []
    orig_opt_step = opt.step
    def custom_opt_step(*args, **kwargs):
        captured_grad_norms.append(_gradient_norm(trainable))
        return orig_opt_step(*args, **kwargs)
    opt.step = custom_opt_step

    cfg=PPOConfig(
        model_name=base,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kl_penalty="abs",
        init_kl_coef=init_kl_coef,
        target=target_kl,
        target_kl=target_kl,
        adap_kl_ctrl=True,
        horizon=100,
        remove_unused_columns=False,
    )
    # Frozen SFT ref: with is_peft_model=False, TRL uses this instead of disable_adapter(base).
    ref_model,_=load_sft_adapter(sft_model_path,device_map,trainable=False)
    collate=lambda rows:{k:[r[k] for r in rows] for k in rows[0]}
    trainer=PPOTrainer(config=cfg,model=model,ref_model=ref_model,tokenizer=tokenizer,dataset=dataset,optimizer=opt,data_collator=collate)
    sandbox=PythonSandbox(default_timeout=5.0,max_memory_mb=1024.0)
    # temperature=1.0 keeps sampling aligned with TRL logprob scoring (temperature>1 or <1
    # during generate() biases objective/kl and can drive the adaptive controller negative).
    gen_target = getattr(model, "pretrained_model", model)
    gen_cfg = getattr(gen_target, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.temperature = 1.0
        gen_cfg.do_sample = True
        gen_cfg.top_p = 0.95
    kwargs={
        "max_new_tokens":384,
        "do_sample":True,
        "temperature":1.0,
        "top_p":0.95,
        "pad_token_id":tokenizer.pad_token_id,
        "eos_token_id":tokenizer.eos_token_id
    }
    for step,batch in enumerate(trainer.dataloader,1):
        queries=[q.squeeze() if isinstance(q,torch.Tensor) and q.dim()>1 else torch.as_tensor(q,dtype=torch.long) for q in batch["input_ids"]]
        if hasattr(model.pretrained_model, "gradient_checkpointing_disable"):
            try: model.pretrained_model.gradient_checkpointing_disable()
            except Exception: pass
        if hasattr(model.pretrained_model, "config"):
            model.pretrained_model.config.use_cache = True
        with torch.no_grad(): responses=trainer.generate(queries,**kwargs)
        if hasattr(model.pretrained_model, "gradient_checkpointing_enable"):
            try: model.pretrained_model.gradient_checkpointing_enable()
            except Exception: pass
        if hasattr(model.pretrained_model, "config"):
            model.pretrained_model.config.use_cache = False
        rewards=[]
        batch_tests = batch.get("benchmark_tests")
        if not batch_tests:
            raise RuntimeError(
                "batch is missing benchmark_tests; remove_unused_columns=False is required "
                "so shuffled dataloader rows keep their own tests"
            )
        if len(batch_tests) != len(responses):
            raise RuntimeError(f"Benchmark-test/response mismatch: {len(batch_tests)} tests vs {len(responses)} responses")
        for sample_idx,(response,tests) in enumerate(zip(responses,batch_tests)):
            if not tests: raise ValueError("No executable benchmark tests; refusing process-only reward")
            raw_response=tokenizer.decode(response,skip_special_tokens=True)
            code=extract_code_block(raw_response)
            result=sandbox.run_tests(code,tests).to_dict()
            reward=score_rollout_reward(result, reward_mode=reward_mode)
            rewards.append(torch.tensor(reward,dtype=torch.float32))
            print(
                f"PPO sample step={step} idx={sample_idx}: "
                f"status={result['status']} passed={result['passed_tests']}/{result['total_tests']} "
                f"reward={reward:.3f}",
                flush=True,
            )
            print("Generated code preview:",(code[-800:] if code else "<EMPTY>").replace("\n"," "),flush=True)
        if not rewards: raise ValueError("No rewards generated for batch; benchmark tests missing or empty")
        before_norm = _parameter_snapshot(trainable)
        sample_lora_before = trainable[0].detach().clone() if trainable else None
        
        # Run PPO Step
        stats = trainer.step(queries, responses, rewards)
        
        after_norm = _parameter_snapshot(trainable)
        delta_norm = abs(after_norm - before_norm)
        sample_lora_delta = torch.norm(trainable[0].detach() - sample_lora_before).item() if sample_lora_before is not None else 0.0
        
        # Retrieve captured grad norm before zero_grad
        real_grad_norm = captured_grad_norms[-1] if captured_grad_norms else 0.0
        
        mean_reward = stats.get("ppo/mean_scores", stats.get("objective/scores", 0.0))
        kl_value = stats.get("objective/kl", stats.get("ppo/policy/approxkl_avg", 0.0))
        approx_kl = stats.get("ppo/policy/approxkl_avg", stats.get("objective/kl", 0.0))
        kl_coef = stats.get("objective/kl_coef", stats.get("ppo/policy/kl_coef", None))
        if kl_coef is None:
            kl_ctl = getattr(trainer, "kl_ctl", None)
            kl_coef = getattr(kl_ctl, "value", float(cfg.init_kl_coef))

        print(
            f"PPO step {step}: reward={float(mean_reward):.3f} kl={float(kl_value):.3f} "
            f"approxkl={float(approx_kl):.3f} kl_coef={float(kl_coef):.4f} "
            f"param_norm_delta={delta_norm:.6e} sample_lora_delta={sample_lora_delta:.6e} grad_norm={float(real_grad_norm):.6e}",
            flush=True,
        )
        if max_steps and step >= max_steps: break
    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    policy = getattr(model, "pretrained_model", model)
    if hasattr(policy, "save_pretrained"):
        policy.save_pretrained(final_dir)
    else:
        model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    reward_type = (
        "execution_dense_partial_test_fraction"
        if reward_mode == "dense"
        else "execution_binary_ac_only"
    )
    with open(os.path.join(final_dir, "ppo_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": base,
            "checkpoint_type": "ppo_policy_adapter",
            "reward_mode": reward_mode,
            "reward_type": reward_type,
            "execution_tests_required": True,
            "seed": seed,
            "commit_sha": commit_sha or _git_commit_sha(),
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "init_kl_coef": init_kl_coef,
            "target_kl": target_kl,
            "batch_size": batch_size,
            "trainable_parameters": trainable_count,
            "total_parameters": total_count,
        }, f, indent=2)
    return trainer
