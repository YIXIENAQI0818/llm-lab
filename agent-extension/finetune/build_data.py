"""合成 embedding 微调训练数据（KB 领域适配）。

从 ChromaDB 的 documents collection 读取 KB 分块（token_chunker 已分好的 chunk），
随机采样 500 训练 + 100 测试，用 DeepSeek 对每个 chunk 生成一条用户检索 query，
产出 (anchor=query, positive=chunk) 数据集，供 train.py / evaluate.py 使用。

绕开 tiktoken（避免 WSL2 Clash 代理下载编码表失败），直接读已建好的向量库，
保证「微调语料 == 检索语料」一致。
"""

import json
import os
import random
import sys
from pathlib import Path

# ---- WSL2 Clash 代理劫持外网：清掉代理直连 DeepSeek（否则 APIConnectionError）----
for _k in ("all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
           "https_proxy", "HTTPS_PROXY"):
    os.environ.pop(_k, None)
os.environ["no_proxy"] = os.environ["NO_PROXY"] = "*"

# 让脚本可从 finetune/ 目录 import 框架代码
ROOT = Path(__file__).resolve().parent.parent  # agent-extension/
sys.path.insert(0, str(ROOT))

import chromadb

from src.agent_framework.llm import LLMClient

# ---- 底层参数（模块内硬编码，文件顶端注释列备选）----
_TRAIN_N = 500        # 训练对数量（备选：300 小规模 / 1000 较大规模）
_TEST_N = 100         # 测试对数量
_SEED = 42            # 采样随机种子，保证可复现
_DATA_DIR = Path(__file__).resolve().parent / "data"

QUERY_PROMPT = (
    "你是一个检索测试集构造助手。下面是一段知识库文档片段。\n"
    "请生成一条「用户检索这段内容时会提出的」自然语言查询。\n"
    "要求：\n"
    "- 中文为主，可保留英文专业术语\n"
    "- 用提问或关键词式表达，符合真实用户习惯\n"
    "- 概括/转述原文，不要整句照抄\n"
    "- 只输出查询本身，不要解释、不要加引号、不要编号"
)


def load_chunks() -> list[dict]:
    """从 ChromaDB documents collection 读取全部 chunk（含稳定 id）。"""
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_data"))
    col = client.get_collection("documents")
    res = col.get(include=["documents", "metadatas"])

    chunks = []
    for text, meta in zip(res["documents"], res["metadatas"]):
        cid = f"{meta['source']}#{meta['chunk_index']}"
        chunks.append({"id": cid, "text": text})
    return chunks


def gen_query(llm: LLMClient, chunk_text: str) -> str:
    """用 LLM 为单个 chunk 生成一条检索 query。"""
    resp = llm.chat(
        [
            {"role": "system", "content": QUERY_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
        thinking=False,
    )
    return resp.choices[0].message.content.strip()


def main():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()
    print(f"documents collection 共 {len(chunks)} 个 chunk")

    rng = random.Random(_SEED)
    sampled = rng.sample(chunks, _TRAIN_N + _TEST_N)
    train = sampled[:_TRAIN_N]
    test = sampled[_TRAIN_N:]

    llm = LLMClient()

    def build(items: list[dict], label: str) -> list[dict]:
        out = []
        for i, item in enumerate(items):
            try:
                q = gen_query(llm, item["text"])
            except Exception as e:
                print(f"  [{label} {i + 1}/{len(items)}] 生成失败，重试一次: {e}")
                try:
                    q = gen_query(llm, item["text"])
                except Exception as e2:
                    print(f"  [{label} {i + 1}/{len(items)}] 仍失败，跳过: {e2}")
                    continue
            out.append({"anchor": q, "positive": item["text"], "id": item["id"]})
            if (i + 1) % 50 == 0:
                print(f"  [{label}] {i + 1}/{len(items)} 完成")
        return out

    print("生成训练集 query ...")
    train_pairs = build(train, "train")
    print("生成测试集 query ...")
    test_pairs = build(test, "test")

    # 写 corpus（全部 chunk，供 evaluate 做检索库）
    with open(_DATA_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c["id"], "text": c["text"]},
                               ensure_ascii=False) + "\n")

    # train 只保留 anchor/positive（训练不需要 id，避免被 data collator 按列序误当 negative）
    with open(_DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps({"anchor": p["anchor"], "positive": p["positive"]},
                               ensure_ascii=False) + "\n")

    # test 保留 id（evaluate 用它构造 relevant_docs 映射）
    with open(_DATA_DIR / "test.jsonl", "w", encoding="utf-8") as f:
        for p in test_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n完成：corpus={len(chunks)}，train={len(train_pairs)}，test={len(test_pairs)}")
    print("样例：")
    for p in test_pairs[:2]:
        print(f"  Q: {p['anchor'][:60]}")
        print(f"  P: {p['positive'][:60]}")
        print()


if __name__ == "__main__":
    main()
