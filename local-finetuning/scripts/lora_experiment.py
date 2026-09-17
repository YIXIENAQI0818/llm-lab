"""
LoRA / QLoRA 微调实验脚本。

目标：用 peft 给 Qwen2.5-1.5B 上 LoRA / QLoRA，调 rank / alpha，对比
      可训练参数量、显存占用、loss 收敛。

用法：
  # QLoRA（4bit 量化基座，6GB 显存默认方案）
  python lora_experiment.py --mode qlora --rank 8  --alpha 16 --steps 50
  # 全精度 LoRA（bf16 基座，作为显存/效果对照组）
  python lora_experiment.py --mode lora  --rank 8  --alpha 16 --steps 50
  # 只报告可训练参数量，不训练（快速扫 rank）
  python lora_experiment.py --dry-run

底层参数硬编码（模块内，不暴露为命令行参数）：
  - MODEL_NAME     : Qwen/Qwen2.5-1.5B-Instruct（备选：Qwen/Qwen2.5-0.5B-Instruct）
  - TARGET_MODULES : Qwen2 的 attention + MLP 全量 linear 层
  - MAX_LENGTH     : 512（备选 256/1024，平衡显存与长文）
  - BATCH_SIZE     : 2（备选 1/4，6GB 显存下 2 较稳）
  - GRAD_ACCUM     : 4（等效 batch = 2*4 = 8）
  - LR             : 2e-4（备选 1e-4 / 5e-4）
"""

import argparse
import math
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---- 底层参数（模块内硬编码，见文件头注释）----
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 2e-4

# ---- 内置玩具数据（验证 LoRA 机制用，正式 SFT 数据在下一项）----
# 格式：instruction / input / output（alpaca 风格）
SAMPLE_DATA = [
    ("把下面的句子改得更正式", "这事儿我们得抓紧办了", "此事我们须尽快处理。"),
    ("用一句话概括这段话", "深度学习是机器学习的一个分支，它通过多层神经网络来学习数据的层次化表示。", "深度学习通过多层神经网络学习数据的层次化表示。"),
    ("给这个产品写一句宣传语", "一款可以自动记笔记的 App", "让灵感随手记录，让思考永不流失。"),
    ("把英文翻译成中文", "Knowledge is power.", "知识就是力量。"),
    ("计算", "23 * 17 等于多少", "23 × 17 = 391。"),
    ("分类", "苹果、香蕉、土豆、西瓜，哪些是水果", "苹果、香蕉、西瓜是水果，土豆是蔬菜。"),
    ("改写得更简洁", "由于天气的原因，因此我们决定取消这次活动", "因天气原因，我们取消了这次活动。"),
    ("回答", "中国的首都是哪里", "中国的首都是北京。"),
    ("写邮件", "向老板请一天假", "尊敬的老板：您好，我因家中有事，申请明天请假一天，恳请批准。"),
    ("总结", "会议讨论了预算、人员安排和项目进度三个议题，最终确定了下季度目标。", "会议确定了预算、人员安排与下季度目标。"),
    ("扩写", "下雨了，他匆匆赶回家", "天空突然下起了雨，他加快脚步，匆匆往家赶去。"),
    ("判断正误", "水在 100 摄氏度会沸腾", "正确。标准大气压下，水在 100 摄氏度沸腾。"),
    ("命名", "一个卖咖啡的小店", "取名：慢时光咖啡。"),
    ("解释", "什么是 API", "API 是应用程序编程接口，用于不同软件之间进行数据交互。"),
    ("仿写", "春风又绿江南岸", "秋月再明塞北关。"),
    ("排序", "把'苹果 香蕉 樱桃'按首字母拼音排序", "按拼音排序：菠萝? 不对，应为：苹果(p)、香蕉(x)、樱桃(y)。"),
    ("找错别字", "他工作很认真负则", "“负则”应为“负责”。"),
    ("续写", "从前有座山，山里有座庙，", "庙里有个老和尚在给小和尚讲故事。"),
    ("问天气", "今天适合穿什么", "建议根据实际气温选择，降温请添衣保暖。"),
    ("写作", "写一句关于坚持的名言", "锲而不舍，金石可镂。"),
    ("计算", "100 的一半再乘以 3", "100 ÷ 2 × 3 = 150。"),
    ("翻译", "把'你好'翻译成英文", "Hello."),
    ("解释", "什么是 GPU", "GPU 是图形处理器，擅长大规模并行计算，广泛用于深度学习。"),
    ("改口语化", "本产品具有极高的性价比", "这玩意儿性价比贼高。"),
    ("写标题", "一篇讲如何早起的文章", "标题：告别赖床，从这五步开始。"),
    ("回答", "一年有多少天", "平年 365 天，闰年 366 天。"),
    ("写祝福语", "朋友生日", "祝你生日快乐，岁岁平安，万事胜意！"),
    ("化简", "这句话太啰嗦了，帮我精简：我个人觉得吧，这个事情其实还是蛮重要的", "我认为这件事很重要。"),
    ("翻译", "把'谢谢'翻译成英文", "Thank you."),
    ("分类", "猫、狗、桌子、椅子，哪些是动物", "猫和狗是动物，桌子和椅子是家具。"),
    ("写邀请函", "邀请朋友来家里吃饭", "诚挚邀请你本周六来我家共进晚餐，期待你的到来！"),
    ("解释", "什么是机器学习", "机器学习是让计算机从数据中学习规律、做出预测的技术。"),
    ("问时间", "一小时有多少分钟", "一小时有 60 分钟。"),
    ("改写", "用更委婉的方式说'你错了'", "这个思路或许可以再斟酌一下。"),
    ("写总结", "他今天学了数学、语文和英语", "他今天学习了数学、语文和英语三门课程。"),
    ("回答", "地球绕太阳一圈要多久", "地球绕太阳公转一圈约 365 天，即一年。"),
    ("写文案", "一家健身房开业", "开业钜惠：前 100 名办卡立减 500，健康从今天开始！"),
    ("判断", "7 是质数吗", "是，7 只能被 1 和 7 整除。"),
    ("翻译", "把'再见'翻译成英文", "Goodbye."),
    ("命名", "一个学习英语的 App", "取名：流利说英语。"),
]


def count_trainable(model):
    """返回 (可训练参数量, 总参数量, 占比)。"""
    def size(p):
        # bnb 4bit 量化参数：原始形状存在 quant_state.shape，
        # numel()/shape 是 uint8 打包后大小（约减半）；普通参数直接 numel()
        qs = getattr(p, "quant_state", None)
        if qs is not None and getattr(qs, "shape", None) is not None:
            return math.prod(qs.shape)
        return p.numel()
    trainable = sum(size(p) for p in model.parameters() if p.requires_grad)
    total = sum(size(p) for p in model.parameters())
    return trainable, total, trainable / total


def build_model(mode: str, rank: int, alpha: int):
    """按 mode 加载基座并套上 LoRA。"""
    print(f"\n[加载] 模式={mode}, rank={rank}, alpha={alpha}")

    if mode == "qlora":
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
    elif mode == "lora":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        )
    else:
        raise ValueError(f"未知 mode: {mode}（可选 qlora / lora）")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False  # 训练时关闭 kv cache

    trainable, total, ratio = count_trainable(model)
    print(f"[参数] 可训练 {trainable:,} / 总 {total:,} = {ratio:.2%}")
    return model


def build_dataset(tokenizer):
    """内置玩具数据 → 已 tokenize 的 Dataset。"""
    ds = Dataset.from_dict({
        "instruction": [d[0] for d in SAMPLE_DATA],
        "input": [d[1] for d in SAMPLE_DATA],
        "output": [d[2] for d in SAMPLE_DATA],
    })

    def tokenize_fn(examples):
        texts = []
        for instr, inp, out in zip(
            examples["instruction"], examples["input"], examples["output"],
        ):
            user_content = instr if not inp else f"{instr}\n{inp}"
            msgs = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": out},
            ]
            texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
        result = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)
        # 不手动构造 labels：DataCollatorForLanguageModeling(mlm=False) 会自己
        # 从 input_ids clone 出 labels 并把 pad 位置设为 -100
        return result

    return ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)


def train(model, tokenizer, dataset, steps: int, output_dir: str):
    """用 Trainer 跑 max_steps 步，返回 (平均 loss, 显存峰值)。"""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_steps=steps,
        bf16=True,
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    avg_loss = result.metrics.get("train_loss", float("nan"))
    print(f"[训练] 平均 loss={avg_loss:.4f}, 峰值显存={peak_gb:.2f}GB")
    return avg_loss, peak_gb


def main():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA 实验")
    parser.add_argument("--mode", default="qlora", choices=["qlora", "lora"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true",
                        help="只报告参数量，不训练")
    args = parser.parse_args()

    if args.alpha is None:
        args.alpha = 2 * args.rank

    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = build_model(args.mode, args.rank, args.alpha)
        print("[dry-run] 不训练。")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(args.mode, args.rank, args.alpha)
    dataset = build_dataset(tokenizer)

    output_dir = os.path.join(
        "outputs", f"{args.mode}_r{args.rank}_a{args.alpha}"
    )
    train(model, tokenizer, dataset, args.steps, output_dir)


if __name__ == "__main__":
    main()
