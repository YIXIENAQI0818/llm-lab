import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class EmbeddingStore:
    """通用语义存储，支持多 collection 隔离。

    每个 collection 内按余弦相似度检索，跨 collection 可计算文本相似度。
    一个进程内通常只创建一个实例，SentenceTransformer + CrossEncoder 通过类变量单例共享。

    用法:
        store = EmbeddingStore()
        store.add("tools", "查询天气", {"name": "get_weather"})
        results = store.search("tools", "今天天气怎么样", top_k=3)
        sim = store.similarity("程序员", "写代码")       # bi-encoder
        score = store.cross_similarity("我叫程宣赫", "用户叫程宣赫")  # cross-encoder
    """

    _singleton_model = None
    _singleton_cross_model = None

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 cross_model_name: str = "BAAI/bge-reranker-base"):
        if EmbeddingStore._singleton_model is None:
            EmbeddingStore._singleton_model = SentenceTransformer(model_name)
        self._model = EmbeddingStore._singleton_model

        if EmbeddingStore._singleton_cross_model is None:
            EmbeddingStore._singleton_cross_model = CrossEncoder(cross_model_name)
        self._cross_model = EmbeddingStore._singleton_cross_model

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
        items = self._collections.get(collection, [])
        if not items:
            return []

        query_vec = self._model.encode(query, normalize_embeddings=True)
        all_vecs = np.stack([item["vec"] for item in items])
        scores = (all_vecs @ query_vec)  # 归一化向量，点积即余弦相似度

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
        """bi-encoder 余弦相似度（0~1），快速但粗糙。"""
        va = self._model.encode(a, normalize_embeddings=True)
        vb = self._model.encode(b, normalize_embeddings=True)
        return float(va @ vb)

    def cross_similarity(self, a: str, b: str) -> float:
        """cross-encoder 语义相似度，比 bi-encoder 更准确。

        两段文本同时喂给模型，直接输出相关性分数。
        更慢但更准，适合去重和精排等对准确性要求高的场景。
        """
        return float(self._cross_model.predict([(a, b)])[0])

    # ---- 单条修改 ----

    def update(self, collection: str, index: int, text: str, meta: dict | None = None):
        """更新 collection 中第 index 条（只重编码这一条）。"""
        vec = self._model.encode(text, normalize_embeddings=True)
        self._collections[collection][index] = {
            "text": text, "meta": meta or {}, "vec": vec,
        }

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
