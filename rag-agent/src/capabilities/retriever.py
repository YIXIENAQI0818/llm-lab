"""混合检索器 — Dense + BM25 + Query Rewriting + RRF 融合。

jieba 分词（中英混合），rank-bm25 做稀疏检索。
"""

import jieba
from rank_bm25 import BM25Okapi


class Retriever:
    """混合检索：Dense（BGE Embedding）+ Sparse（BM25）+ RRF 融合。"""

    RRF_K = 60

    EXPAND_PROMPT = (
        "将以下查询扩展为更丰富的检索关键词，加入同义词和相关术语。"
        "用空格分隔，直接输出关键词，不要解释。"
    )

    DECOMPOSE_PROMPT = (
        "将以下复杂查询拆解为 2-3 个简单的子问题。"
        "用空格分隔输出所有子问题，不要编号，不要解释。"
    )

    def __init__(self, store, llm_client=None):
        self._store = store
        self._llm = llm_client
        self._bm25: dict[str, BM25Okapi] = {}
        self._bm25_texts: dict[str, list[str]] = {}
        self._bm25_metas: dict[str, list[dict]] = {}

    def build_bm25(self, collection: str):
        """从 ChromaDB 读取文本和 meta，首次建 BM25 索引。已有则跳过。"""
        if collection in self._bm25:
            return
        items = self._store.get_all(collection)
        if not items:
            return
        texts = [item["text"] for item in items]
        metas = [item.get("meta", {}) for item in items]
        tokenized = [jieba.lcut(t) for t in texts]
        self._bm25[collection] = BM25Okapi(tokenized)
        self._bm25_texts[collection] = texts
        self._bm25_metas[collection] = metas

    def rebuild_bm25(self, collection: str, texts: list[str],
                     metas: list[dict] | None = None):
        """从给定的文本列表强制重建 BM25 索引（KB reindex 时用）。"""
        tokenized = [jieba.lcut(t) for t in texts]
        self._bm25[collection] = BM25Okapi(tokenized)
        self._bm25_texts[collection] = texts
        self._bm25_metas[collection] = metas or [{} for _ in texts]

    def search(self, collection: str, query: str, threshold: float,
               top_k: int = 5, hybrid: bool = True,
               rewrite: str | bool = "expand") -> list[dict]:
        """混合检索，返回 [{"score","text","meta"}]。"""
        search_query = self._rewrite(query, rewrite) if rewrite and self._llm else query

        dense = self._store.search(collection, search_query,
                                   top_k=top_k * 2, threshold=threshold)

        if not hybrid or collection not in self._bm25:
            return dense[:top_k]

        bm25 = self._bm25_search(collection, query, top_k * 2)
        return self._fuse(dense, bm25, top_k)

    # ---- 内部 ----

    def _rewrite(self, query: str, strategy: str = "expand") -> str:
        try:
            prompt = self.DECOMPOSE_PROMPT if strategy == "decompose" else self.EXPAND_PROMPT
            resp = self._llm.chat([
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ])
            result = resp.choices[0].message.content.strip()
            return f"{query} {result}" if result else query
        except Exception:
            return query

    def _bm25_search(self, collection: str, query: str, top_k: int
                     ) -> list[dict]:
        bm25 = self._bm25[collection]
        texts = self._bm25_texts[collection]
        metas = self._bm25_metas[collection]
        scores = bm25.get_scores(jieba.lcut(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [{"index": i, "score": s, "text": texts[i], "meta": metas[i]}
                for i, s in ranked[:top_k] if s > 0]

    def _fuse(self, dense: list[dict], bm25: list[dict], top_k: int
              ) -> list[dict]:
        merged: dict[str, dict] = {}

        for rank, r in enumerate(dense):
            key = hash(r["text"])
            if key not in merged:
                merged[key] = {"item": r, "rrf": 0.0}
            merged[key]["rrf"] += 1.0 / (self.RRF_K + rank + 1)

        for rank, r in enumerate(bm25):
            key = hash(r["text"])
            if key not in merged:
                merged[key] = {
                    "item": {"score": r["score"], "text": hash(r["text"]),
                             "meta": r["meta"]},
                    "rrf": 0.0,
                }
            merged[key]["rrf"] += 1.0 / (self.RRF_K + rank + 1)

        ranked = sorted(merged.values(), key=lambda x: x["rrf"], reverse=True)
        return [v["item"] for v in ranked[:top_k]]
