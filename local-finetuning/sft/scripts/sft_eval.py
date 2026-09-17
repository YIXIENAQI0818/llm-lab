"""
SFT 定性评估：base vs SFT 生成对比。

加载 QLoRA 基座，先出 base 结果，再挂 adapter 出 SFT 结果，并排对比。
重点看「行为对齐」：是否只答答案、不复述指令、答完就停。

用法：
  python sft_eval.py

底层参数硬编码：
  - MODEL_NAME   : Qwen/Qwen2.5-1.5B-Instruct（官方已 SFT 的 instruct 版本）
  - ADAPTER_PATH : outputs/sft_qlora_r8（sft_experiment.py 训出的 adapter）
  - TEST_PROMPTS : 6 条通用指令（翻译/总结/问答/创作）
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = os.path.join("outputs", "sft_qlora_r8")

TEST_PROMPTS = [
    "把下面的句子翻译成英文：知识就是力量。",
    "用一句话总结：深度学习是机器学习的一个分支，它通过多层神经网络学习数据的层次化表示。",
    "写一句关于坚持的名言。",
    "中国的首都是哪里？",
    "给一家咖啡店写一句宣传语。",
    "解释一下什么是 API。",
]


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


def generate(model, tokenizer, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = load_base()
    print("[加载] base 完成，先生成 base 结果 ...")
    base_outputs = [generate(base, tokenizer, p) for p in TEST_PROMPTS]

    print("[加载] 挂 adapter -> SFT 模型 ...")
    sft = PeftModel.from_pretrained(base, ADAPTER_PATH)
    sft_outputs = [generate(sft, tokenizer, p) for p in TEST_PROMPTS]

    for p, b, s in zip(TEST_PROMPTS, base_outputs, sft_outputs):
        print(f"\n[指令] {p}")
        print(f"[base] {b}")
        print(f"[sft ] {s}")


if __name__ == "__main__":
    main()
