"""Embedding 微调：MultipleNegativesRankingLoss 全参数微调 bge-small-zh-v1.5。

读取 build_data.py 产出的 train.jsonl（anchor/positive 对），
用 in-batch negatives 做对比学习，让 query 向量靠近对应 chunk、远离同 batch 其他 chunk。
微调后模型保存到 finetune/models/bge-kb-v1，供 chroma_store 切换使用。

全参数微调（非 LoRA —— LoRA 是 local-finetuning 子项目的任务）。
bge-small 仅 24M 参数，6GB 显存足够。
"""

import os
from pathlib import Path

# ---- 清代理：bge 走本地缓存，无需外网；避免 Clash 代理干扰 ----
for _k in ("all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
           "https_proxy", "HTTPS_PROXY"):
    os.environ.pop(_k, None)
os.environ["no_proxy"] = os.environ["NO_PROXY"] = "*"

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

# ---- 底层参数（模块内硬编码）----
_BASE_MODEL = "BAAI/bge-small-zh-v1.5"
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "models" / "bge-kb-v1"

_NUM_EPOCHS = 3          # 500 对 / batch16 ≈ 32 step/epoch，3 epoch ≈ 94 step
_BATCH_SIZE = 16
_LEARNING_RATE = 2e-5
_WARMUP_STEPS = 10       # 约 10% 总步数
_MAX_SEQ_LEN = 512


def main():
    train_ds = Dataset.from_json(str(_DATA_DIR / "train.jsonl"))
    test_ds = Dataset.from_json(str(_DATA_DIR / "test.jsonl"))
    # test 里去掉 id 列，只留 anchor/positive，作为训练时的 eval 集
    eval_ds = test_ds.remove_columns(["id"])
    print(f"train={len(train_ds)}，eval={len(eval_ds)}，列={train_ds.column_names}")

    model = SentenceTransformer(_BASE_MODEL, local_files_only=True)
    model.max_seq_length = _MAX_SEQ_LEN

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(_OUTPUT_DIR / "checkpoints"),
        num_train_epochs=_NUM_EPOCHS,
        per_device_train_batch_size=_BATCH_SIZE,
        learning_rate=_LEARNING_RATE,
        warmup_steps=_WARMUP_STEPS,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MNR 关键：避免同 batch 重复 anchor 造成假负例
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],  # 不接 wandb/tensorboard
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
    )

    trainer.train()

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(_OUTPUT_DIR))
    print(f"\n微调完成，模型已保存到 {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
