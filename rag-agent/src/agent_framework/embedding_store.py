import numpy as np
from sentence_transformers import SentenceTransformer

# 可用 Embedding 模型:
#   BAAI/bge-small-zh-v1.5 — 中文优化，512 维，轻量（当前使用）
#   BAAI/bge-large-zh-v1.5 — 中文优化，1024 维，效果更好但更慢
#   BAAI/bge-m3 — 多语言，支持中英文混合，dense+sparse 双向量


class EmbeddingStore:
    """通用语义存储，支持多 collection 隔离。

    每个 collection 内按余弦相似度检索，跨 collection 可计算文本相似度。
    一个进程内通常只创建一个实例，SentenceTransformer 通过类变量单例共享。
    """

    _singleton_model = None

    def __init__(self):
        if EmbeddingStore._singleton_model is None:
            EmbeddingStore._singleton_model = SentenceTransformer(
                "BAAI/bge-small-zh-v1.5")
        self._model = EmbeddingStore._singleton_model
        self._collections: dict[str, list[dict]] = {}

    # ---- 写入 ----

    def add(self, collection: str, text: str, meta: dict | None = None):
        """往指定 collection 加一条记录。"""
        vec = self._model.encode(text, normalize_embeddings=True)
        if collection not in self._collections:
            self._collections[collection] = []
        self._collections[collection].append({
            "text": text,
            "meta": meta or {},
            "vec": vec,
        })

    def rebuild(self, collection: str, items: list[dict]):
        """批量重建某个 collection（清空 + 批量编码填入）。

        Args:
            collection: collection 名称
            items: [{"text": str, "meta": dict}, ...]
        """
        if not items:
            self._collections.pop(collection, None)
            return

        texts = [item["text"] for item in items]
        vecs = self._model.encode(texts, normalize_embeddings=True)

        self._collections[collection] = [
            {"text": item["text"], "meta": item.get("meta", {}), "vec": vec}
            for item, vec in zip(items, vecs)
        ]

    # ---- 读取 ----

    def search(self, collection: str, query: str, top_k: int = 5,
               threshold: float = 0.0) -> list[dict]:
        """在指定 collection 内按余弦相似度检索。

        Returns:
            [{"score": float, "text": str, "meta": dict}, ...] 按 score 降序
        """
        scores, items = self.batch_similarity(collection, query)
        if not items:
            return []

        # 排序取 top_k
        idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idxs:
            score = float(scores[i])
            if score < threshold:
                continue
            results.append({
                "score": score,
                "text": items[i]["text"],
                "meta": items[i]["meta"],
            })
        return results

    def similarity(self, a: str, b: str) -> float:
        """计算两段文本的余弦相似度（0~1）。"""
        va = self._model.encode(a, normalize_embeddings=True)
        vb = self._model.encode(b, normalize_embeddings=True)
        return float(va @ vb)

    def batch_similarity(self, collection: str, query: str) -> tuple[list[float], list[dict]]:
        """一次编码 query，矩阵乘法计算与 collection 中所有项的相似度。

        Returns:
            (scores, items): scores[i] 为 query 与 items[i] 的余弦相似度
        """
        items = self._collections.get(collection, [])
        if not items:
            return [], []
        query_vec = self._model.encode(query, normalize_embeddings=True)
        all_vecs = np.stack([item["vec"] for item in items])
        scores = (all_vecs @ query_vec).tolist()
        return scores, items

    def delete(self, collection: str, index: int):
        """删除 collection 中第 index 条。"""
        del self._collections[collection][index]

    # ---- 工具 ----

    def collection_size(self, collection: str) -> int:
        return len(self._collections.get(collection, []))

    def list_collections(self) -> list[str]:
        return list(self._collections.keys())

    def __len__(self):
        return sum(len(v) for v in self._collections.values())
