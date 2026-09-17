"""
DPO 偏好对数据构造脚本。

思路：DPO 需要 (prompt, chosen, rejected) 三元组。这里用 Alpaca-zh 构造：
  - prompt  : 指令（instruction + input）
  - chosen  : Alpaca-zh 的 output（强模型参考回答）
  - rejected: 本地 base 模型 Qwen2.5-1.5B-Instruct 的贪心生成（弱回答）

「强参考 vs 弱模型」构成天然偏好信号，且只加载一次 base 模型、
不依赖 SFT adapter。三列均存 messages 格式（role/content），
DPOTrainer 据此自动套 chat template。

用法：
  # 正式跑（1000 条）
  python build_dpo_data.py
  # 小规模验证
  python build_dpo_data.py --num-samples 50

底层参数硬编码（模块内）：
  - MODEL_NAME     : Qwen/Qwen2.5-1.5B-Instruct
  - DATASET_NAME   : shibing624/alpaca-zh
  - MAX_NEW_TOKENS : 200
  - OUTPUT_PATH    : outputs/dpo_data.json（运行目录 scripts/，outputs 已 gitignore）
"""

import argparse
import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "shibing624/alpaca-zh"
MAX_NEW_TOKENS = 200
OUTPUT_PATH = os.path.join("outputs", "dpo_data.json")


def to_prompt(example):
    """Alpaca-zh instruction(+input) → 用户消息文本。"""
    return example["instruction"] if not example["input"] \
        else f"{example['instruction']}\n{example['input']}"


def generate(model, tokenizer, prompt_text):
    """base 模型贪心生成（作为弱回答 rejected）。"""
    msgs = [{"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="构造 DPO 偏好对数据")
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n[数据] 加载 {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)["train"].shuffle(seed=42).select(range(args.num_samples))

    print("[模型] 加载 base（生成 rejected 用）...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    )

    rows = []
    for i, ex in enumerate(ds):
        prompt_text = to_prompt(ex)
        rejected = generate(model, tokenizer, prompt_text)
        rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "chosen": [{"role": "assistant", "content": ex["output"]}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{args.num_samples}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_ds = Dataset.from_list(rows)
    out_ds.to_json(OUTPUT_PATH, force_ascii=False)  # 中文直接写入，不转 \uXXXX
    print(f"\n[保存] {len(rows)} 条 -> {OUTPUT_PATH}")

    print("\n[样例] 前 2 条：")
    for r in rows[:2]:
        print(f"  prompt  : {r['prompt'][0]['content'][:50]}...")
        print(f"  chosen  : {r['chosen'][0]['content'][:80]}...")
        print(f"  rejected: {r['rejected'][0]['content'][:80]}...\n")


if __name__ == "__main__":
    main()
