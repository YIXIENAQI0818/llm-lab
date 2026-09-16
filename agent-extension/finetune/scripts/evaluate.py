"""评估：对比 baseline（bge-small-zh-v1.5）与微调后 embedding 的检索指标。

在留出的 100 条测试 query 上，用 InformationRetrievalEvaluator 计算
nDCG@10 / MRR@10 / recall@k，corpus 为全部 KB chunk（5154 条）。
"""

import json
import os
from pathlib import Path

# ---- 清代理：模型均本地加载，避免 Clash 干扰 ----
for _k in ("all_proxy", "ALL_PROXY", "http_proxy", "HTTP_PROXY",
           "https_proxy", "HTTPS_PROXY"):
    os.environ.pop(_k, None)
os.environ["no_proxy"] = os.environ["NO_PROXY"] = "*"

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

_BASE_MODEL = "BAAI/bge-small-zh-v1.5"
_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_MODEL_DIR = _ROOT / "models" / "bge-kb-v1"


def load_eval_data() -> tuple[dict, dict, dict]:
    """读取 corpus + 测试 query + relevant_docs 映射。"""
    corpus: dict[str, str] = {}
    with open(_DATA_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            corpus[o["id"]] = o["text"]

    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    with open(_DATA_DIR / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            cid = o["id"]
            queries[cid] = o["anchor"]
            relevant_docs[cid] = {cid}  # 合成数据：query 唯一对应来源 chunk
    return queries, corpus, relevant_docs


def evaluate(model: SentenceTransformer, name: str,
             queries, corpus, relevant_docs) -> dict:
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        show_progress_bar=True,
        name=name,
    )
    return evaluator(model)  # 返回扁平 metrics dict


def _pick(metrics: dict, name: str, metric: str, k: int) -> float:
    """从 metrics 里提取指定指标（key 形如 {name}_{score_fn}_{metric}@{k}）。"""
    suffix = f"_{metric}@{k}"
    for key, val in metrics.items():
        if key.startswith(name) and key.endswith(suffix):
            return float(val)
    return float("nan")


def main():
    queries, corpus, relevant_docs = load_eval_data()
    print(f"评估集：{len(queries)} 条测试 query，corpus {len(corpus)} 条 chunk\n")

    base_model = SentenceTransformer(_BASE_MODEL, local_files_only=True)
    ft_model = SentenceTransformer(str(_MODEL_DIR))

    print("评估 baseline ...")
    base = evaluate(base_model, "baseline", queries, corpus, relevant_docs)
    print("评估微调后 ...")
    ft = evaluate(ft_model, "finetuned", queries, corpus, relevant_docs)

    rows = [
        ("recall@1", "recall", 1),
        ("recall@5", "recall", 5),
        ("recall@10", "recall", 10),
        ("nDCG@10", "ndcg", 10),
        ("MRR@10", "mrr", 10),
    ]
    print(f"\n{'指标':<10}{'baseline':>12}{'微调后':>12}{'变化':>12}")
    print("-" * 46)
    for label, metric, k in rows:
        b = _pick(base, "baseline", metric, k)
        f = _pick(ft, "finetuned", metric, k)
        delta = f - b
        print(f"{label:<10}{b:>12.4f}{f:>12.4f}{delta:>+12.4f}")


if __name__ == "__main__":
    main()
