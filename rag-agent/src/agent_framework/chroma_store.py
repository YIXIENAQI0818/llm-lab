"""ChromaDB 向量存储 — 接口对齐 EmbeddingStore，底层持久化。

每个逻辑 collection 映射到一个 ChromaDB collection。
数据存于 chroma_data/ 目录，进程重启后仍在。
"""

import numpy as np
from chromadb import EmbeddingFunction, PersistentClient
from sentence_transformers import SentenceTransformer


_singleton_model = None


def _get_model():
    global _singleton_model
    if _singleton_model is None:
        _singleton_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    return _singleton_model


class BGEEmbedding(EmbeddingFunction):
    """ChromaDB 自定义 embedding function，使用 BGE 模型。"""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        model = _get_model()
        vecs = model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()


class ChromaDBStore:
    """向量存储，接口对齐 EmbeddingStore。"""

    def __init__(self):
        self._client = PersistentClient(path="chroma_data")
        self._ef = BGEEmbedding()
        self._counters: dict[str, int] = {}

    # ---- 写入 ----

    def add(self, collection: str, text: str, meta: dict | None = None):
        col = self._get_or_create(collection)
        idx = self._counters.get(collection, 0)
        col.add(documents=[text], metadatas=[meta or {}], ids=[str(idx)])
        self._counters[collection] = idx + 1

    def rebuild(self, collection: str, items: list[dict]):
        """批量重建 collection（删旧 + 批量写入）。"""
        try:
            self._client.delete_collection(collection)
        except Exception:
            pass

        if not items:
            self._counters.pop(collection, None)
            return

        col = self._client.create_collection(
            collection,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        texts = [item["text"] for item in items]
        metas = [item.get("meta", {}) for item in items]
        ids = [str(i) for i in range(len(items))]

        col.add(documents=texts, metadatas=metas, ids=ids)
        self._counters[collection] = len(items)

    def delete(self, collection: str, index: int):
        col = self._get_or_create(collection)
        col.delete(ids=[str(index)])

    # ---- 读取 ----

    def search(self, collection: str, query: str, top_k: int = 5,
               threshold: float = 0.0) -> list[dict]:
        """余弦相似度检索，返回 [{"score","text","meta"}]。"""
        col = self._get_or_create(collection)
        if col.count() == 0:
            return []

        n = min(top_k, col.count())
        result = col.query(query_texts=[query], n_results=n,
                           include=["documents", "metadatas", "distances"])

        results = []
        for text, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            score = 1.0 - float(dist)
            if score < threshold:
                continue
            results.append({"score": score, "text": text, "meta": meta})
        return results

    def similarity(self, a: str, b: str) -> float:
        va = self._ef([a])[0]
        vb = self._ef([b])[0]
        return float(np.dot(va, vb))

    def batch_similarity(self, collection: str, query: str
                         ) -> tuple[list[float], list[dict]]:
        """获取 query 与 collection 中所有项的相似度。"""
        col = self._get_or_create(collection)
        n = col.count()
        if n == 0:
            return [], []

        result = col.query(query_texts=[query], n_results=n,
                           include=["documents", "metadatas", "distances"])

        scores = []
        items = []
        for text, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            scores.append(1.0 - float(dist))
            items.append({"text": text, "meta": meta})
        return scores, items

    # ---- 工具 ----

    def collection_size(self, collection: str) -> int:
        try:
            return self._client.get_collection(collection).count()
        except Exception:
            return 0

    def list_collections(self) -> list[str]:
        return self._client.list_collections()

    def __len__(self):
        return sum(self.collection_size(c) for c in self.list_collections())

    # ---- 内部 ----

    def _get_or_create(self, name: str):
        """获取已有 collection，不存在则创建。"""
        return self._client.get_or_create_collection(
            name,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
