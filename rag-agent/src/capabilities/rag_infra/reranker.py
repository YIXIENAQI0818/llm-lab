"""Cross-Encoder 重排序器 — 对粗排结果精排。

使用 BAAI/bge-reranker-v2-m3 模型，中英混合支持。
备选：maidalun1020/bce-reranker-base_v1（约 400MB，中英双语）。
"""

from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-Encoder 精排。"""

    def __init__(self):
        self._model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
        )

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5
               ) -> list[dict]:
        """对粗排候选列表精排，返回 top_k。"""
        if not candidates:
            return []

        pairs = [(str(query), str(c["text"])) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c["score"] = float(s)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
