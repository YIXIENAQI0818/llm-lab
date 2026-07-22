"""知识库 — 离线文档索引 + 在线混合检索。"""

from .rag_infra.token_chunker import TokenChunker, load_markdown_files
from .rag_infra.retriever import Retriever
from .rag_infra.reranker import Reranker


class KnowledgeBase:

    COLLECTION = "documents"

    def __init__(self, es, llm_client=None):
        self._es = es
        self._tc = TokenChunker()
        self._rt = Retriever(es, llm_client, reranker=Reranker())

    # ================================================================
    # 索引
    # ================================================================

    def build_kb_index(self, path: str = "data/", force: bool = False) -> str:
        """写入知识库向量索引。已有数据则跳过，force=True 强制重建。"""
        if not force and not self.is_empty():
            self._rt.build_bm25(self.COLLECTION)
            return "知识库已有数据，跳过索引"

        docs = load_markdown_files(path)
        if not docs:
            return f"在 {path} 下未找到 .md 文件"

        all_chunks = []
        for doc in docs:
            for c in self._tc.chunk(doc["content"], doc["name"]):
                all_chunks.append({"text": c["text"], "meta": c["meta"]})

        self._es.rebuild(self.COLLECTION, all_chunks)
        texts = [c["text"] for c in all_chunks]
        metas = [c["meta"] for c in all_chunks]
        self._rt.rebuild_bm25(self.COLLECTION, texts, metas)

        counts = {}
        for c in all_chunks:
            src = c["meta"]["source"]
            counts[src] = counts.get(src, 0) + 1
        details = "\n".join(
            f"  {src}: {n} chunks" for src, n in counts.items()
        )
        return f"索引完成：{len(docs)} 个文件 → {len(all_chunks)} 个片段\n{details}"

    def is_empty(self) -> bool:
        return self._es.collection_size(self.COLLECTION) == 0

    # ================================================================
    # 检索
    # ================================================================

    def search(self, query: str, strategy: str = "expand",
               top_k: int = 5, threshold: float = 0.3) -> list[dict]:
        """混合检索（Dense + BM25 + RRF + Cross-Encoder 精排）。"""
        rewrite = strategy if strategy in ("expand", "decompose") else "expand"
        return self._rt.search(
            self.COLLECTION, query,
            threshold=threshold, top_k=top_k, rewrite=rewrite,
        )

    # ================================================================
    # 工具
    def _tool_search_docs(self, query: str, strategy: str = "expand") -> str:
        results = self.search(query, strategy=strategy)
        if not results:
            return "未找到相关文档"
        lines = []
        for r in results:
            src = r["meta"].get("source", "?")
            idx = r["meta"].get("chunk_index", "?")
            lines.append(
                f"--- [{src}#{idx}] (score:{r['score']:.3f}) ---\n{r['text']}"
            )
        return "\n".join(lines)
