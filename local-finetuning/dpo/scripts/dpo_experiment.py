"""
DPO 偏好对齐训练脚本。

目标：用 TRL DPOTrainer 给 Qwen2.5-1.5B 做偏好对齐（QLoRA），
      数据用 build_dpo_data.py 构造的 (prompt, chosen, rejected) 三元组。

DPO 思路（区别于 SFT）：SFT 学「标准答案」，DPO 学「哪个更好」。
  - π_θ   : 待训练策略（base + 新 LoRA）
  - π_ref : 冻结参考模型（base，无 adapter）
  - loss   : -log σ( β·log π_θ(chosen)/π_ref(chosen) - β·log π_θ(rejected)/π_ref(rejected) )

用法：
  # 正式跑（900 训练 / 100 评估 / 2 epoch）
  python dpo_experiment.py
  # 小规模验证管线
  python dpo_experiment.py --num-train 100 --num-eval 50 --epochs 1

底层参数硬编码（模块内）：
  - MODEL_NAME : Qwen/Qwen2.5-1.5B-Instruct
  - DATA_PATH  : outputs/dpo_data.json（build_dpo_data.py 产物）
  - BETA       : 0.1（偏离 π_ref 的 KL 惩罚）
  - RANK/ALPHA : 8 / 16（继承 LoRA/QLoRA 结论：r8 够用）
  - LR         : 5e-5（DPO 比 SFT 更敏感，用更小学习率）
  - MAX_LENGTH : 512
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = os.path.join("outputs", "dpo_data.json")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
RANK = 8
ALPHA = 16
LR = 5e-5
BATCH_SIZE = 2
GRAD_ACCUM = 4
BETA = 0.1
MAX_LENGTH = 512
EPOCHS = 2
SEED = 42
OUTPUT_DIR = os.path.join("outputs", "dpo_qlora_r8")


def build_models():
    """QLoRA（4bit）加载 π_ref（冻结 base）和 π_θ（base + 新 LoRA）。"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # π_ref：冻结参考模型，无 adapter
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    )
    # π_θ：base + 新 LoRA（可训练）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, ref_model


def main():
    parser = argparse.ArgumentParser(description="DPO 偏好对齐实验")
    parser.add_argument("--num-train", type=int, default=900)
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[数据] 加载 {DATA_PATH} ...")
    ds = load_dataset("json", data_files=DATA_PATH)["train"].shuffle(seed=SEED)
    train_ds = ds.select(range(args.num_train))
    eval_ds = ds.select(range(args.num_train, min(args.num_train + args.num_eval, len(ds))))
    print(f"[数据] train {len(train_ds)} / eval {len(eval_ds)}")

    model, ref_model = build_models()

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,  # 评估要 materialize 完整 logits，6GB 显存需降到 1
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        bf16=True,
        beta=BETA,
        max_length=MAX_LENGTH,
        logging_steps=10,
        report_to="none",
        save_strategy="no",
        seed=SEED,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("[训练] 开始 ...")
    trainer.train()

    print(f"[保存] adapter -> {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)  # 先保存：即使评估 OOM 也不丢训练成果

    print("[评估] holdout 偏好指标 ...")
    try:
        metrics = trainer.evaluate()
        for k in ["eval_loss", "eval_rewards/chosen", "eval_rewards/rejected",
                  "eval_rewards/margins", "eval_rewards/accuracies"]:
            if k in metrics:
                print(f"  {k} = {metrics[k]:.4f}")
    except torch.cuda.OutOfMemoryError:
        print("[评估] 6GB 显存不足，跳过 holdout 评估；训练指标见上方日志（rewards/accuracies 等）")
    print("完成")


if __name__ == "__main__":
    main()
