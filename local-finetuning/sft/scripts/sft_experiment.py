"""
SFT 指令微调实验脚本。

目标：用 TRL 的 SFTTrainer 给 Qwen2.5-1.5B 做指令微调（QLoRA），
      用真实中文指令数据（Alpaca-zh），对比训练前后 loss 与生成质量。

用法：
  # 正式跑（8k 训练 / 400 评估 / 2 epoch，约 50 分钟）
  python sft_experiment.py
  # 小规模验证管线
  python sft_experiment.py --num-train 100 --num-eval 50 --epochs 1

底层参数硬编码（模块内，不暴露为命令行参数）：
  - MODEL_NAME   : Qwen/Qwen2.5-1.5B-Instruct
  - DATASET_NAME : shibing624/alpaca-zh（中文指令数据，instruction/input/output）
  - RANK/ALPHA   : 8 / 16（继承 LoRA/QLoRA 实验结论：r8 够用）
  - TARGET_MODULES : Qwen2 attention + MLP 全量 linear
  - MAX_LENGTH   : 512
  - BATCH_SIZE/GRAD_ACCUM : 2 / 4（有效 batch 8）
  - LR           : 2e-4
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "shibing624/alpaca-zh"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
RANK = 8
ALPHA = 16
LR = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
MAX_LENGTH = 512
EPOCHS = 2
SEED = 42
OUTPUT_DIR = os.path.join("outputs", "sft_qlora_r8")


def build_model():
    """QLoRA（4bit）加载 Qwen2.5-1.5B 并套 LoRA。"""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
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
    return model


def to_messages(example):
    """Alpaca-zh 三列 → messages 格式（SFTTrainer 据此只对 assistant 算 loss）。"""
    user = example["instruction"] if not example["input"] \
        else f"{example['instruction']}\n{example['input']}"
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["output"]},
    ]}


def main():
    parser = argparse.ArgumentParser(description="SFT 指令微调实验")
    parser.add_argument("--num-train", type=int, default=8000)
    parser.add_argument("--num-eval", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[数据] 加载 {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)["train"].shuffle(seed=SEED)
    train_ds = ds.select(range(args.num_train)).map(
        to_messages, remove_columns=ds.column_names)
    eval_ds = ds.select(range(args.num_train, args.num_train + args.num_eval)).map(
        to_messages, remove_columns=ds.column_names)
    print(f"[数据] train {len(train_ds)} / eval {len(eval_ds)}")

    model = build_model()

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        bf16=True,
        max_length=MAX_LENGTH,
        assistant_only_loss=True,  # 只对 assistant 回复算 loss
        logging_steps=10,
        report_to="none",
        save_strategy="no",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("[训练] 开始 ...")
    trainer.train()

    print("[评估] holdout loss ...")
    eval_metrics = trainer.evaluate()
    print(f"[评估] eval_loss={eval_metrics.get('eval_loss', float('nan')):.4f}")

    print(f"[保存] adapter -> {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    print("完成")


if __name__ == "__main__":
    main()
