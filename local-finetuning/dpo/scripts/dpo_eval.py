"""
DPO 训练前/后 对比评估脚本。

在同一批 holdout 数据（与训练 eval_ds 一致的 20 条）上，对比：
  - before：π_θ = base + 新初始化 LoRA（B=0，等价于 π_ref）
  - after ：π_θ = base + 训练好的 adapter

验证预期：训练前 accuracies = 0（reward 恒 0，0 > 0 全 False），训练后 accuracies ≈ 0.95。

用法（运行目录 scripts/）：
  python dpo_eval.py

底层参数硬编码：MODEL_NAME / DATA_PATH / ADAPTER_PATH / NUM_TRAIN=180 / NUM_EVAL=20 / BETA=0.1
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = os.path.join("outputs", "dpo_data.json")
ADAPTER_PATH = os.path.join("outputs", "dpo_qlora_r8")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
RANK = 8
ALPHA = 16
BETA = 0.1
MAX_LENGTH = 512
SEED = 42
NUM_TRAIN = 180
NUM_EVAL = 20
OUTPUT_DIR = os.path.join("outputs", "dpo_eval_tmp")


def load_base():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    )


def load_eval_dataset():
    """与训练脚本完全一致的切法：shuffle(seed) 后取 [180, 200)。"""
    ds = load_dataset("json", data_files=DATA_PATH)["train"].shuffle(seed=SEED)
    return ds.select(range(NUM_TRAIN, NUM_TRAIN + NUM_EVAL))


def evaluate(tokenizer, eval_ds, model):
    """给 π_θ（model）配一个裸 base 作 π_ref，跑 evaluate 返回指标。"""
    ref_model = load_base()
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_eval_batch_size=1,
        beta=BETA,
        max_length=MAX_LENGTH,
        bf16=True,
        report_to="none",
        seed=SEED,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=eval_ds,  # DPOTrainer 要求必传；只评估不训练，复用 eval 数据
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    metrics = trainer.evaluate()
    del ref_model, trainer
    torch.cuda.empty_cache()
    return metrics


def show(tag, metrics):
    keys = ["eval_loss", "eval_rewards/chosen", "eval_rewards/rejected",
            "eval_rewards/margins", "eval_rewards/accuracies"]
    print(f"[{tag}]")
    for k in keys:
        if k in metrics:
            print(f"  {k:26s} = {metrics[k]:.4f}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_ds = load_eval_dataset()
    print(f"\n[数据] holdout {len(eval_ds)} 条"
          f"（shuffle seed={SEED}，第 {NUM_TRAIN}~{NUM_TRAIN + NUM_EVAL} 条）\n")

    print("[before] π_θ = base + 新初始化 LoRA（B=0，等价 π_ref）...")
    model = load_base()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.05,
                             target_modules=TARGET_MODULES, bias="none",
                             task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    show("before", evaluate(tokenizer, eval_ds, model))
    del model
    torch.cuda.empty_cache()

    print("\n[after] π_θ = base + 训练好的 adapter ...")
    model = load_base()
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    show("after", evaluate(tokenizer, eval_ds, model))


if __name__ == "__main__":
    main()
