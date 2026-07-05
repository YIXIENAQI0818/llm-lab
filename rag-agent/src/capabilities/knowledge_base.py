"""知识库 — 封装文档索引与检索，仿 LongTermMemory 模式。

索引由应用层在启动时调用，不暴露为 LLM 工具。
检索通过 search_docs 工具供 LLM 调用。
"""

from .chunker import TokenChunker, load_markdown_files
from ..agent_framework.embedding_store import EmbeddingStore


class KnowledgeBase:
    """离线索引 + 在线检索的知识库。"""

    COLLECTION = "documents"

    def __init__(self, es: EmbeddingStore):
        self._es = es
        self._tc = TokenChunker()

    def build(self, path: str = "data/") -> str:
        """首次索引（启动时调用）。已有数据则跳过。"""
        if not self.is_empty():
            return "知识库已有数据，跳过索引"
        return self.reindex(path)

    def reindex(self, path: str = "data/") -> str:
        """强制重建索引（删旧 + 重新分块写入）。"""
        docs = load_markdown_files(path)
        if not docs:
            return f"在 {path} 下未找到 .md 文件"

        all_chunks = []
        for doc in docs:
            chunks = self._tc.chunk(doc["content"], doc["name"])
            for c in chunks:
                all_chunks.append({"text": c["text"], "meta": c["meta"]})

        # 批量编码 + 重建 collection，比逐个 add 快
        self._es.rebuild(self.COLLECTION, all_chunks)

        counts = {}
        for c in all_chunks:
            src = c["meta"]["source"]
            counts[src] = counts.get(src, 0) + 1
        details = "\n".join(f"  {src}: {n} chunks" for src, n in counts.items())
        return f"索引完成：{len(docs)} 个文件 → {len(all_chunks)} 个片段\n{details}"

    def search(self, query: str, top_k: int = 3,
               threshold: float = 0.3) -> list[dict]:
        """语义检索文档片段（供 search_docs 工具调用）。"""
        return self._es.search(self.COLLECTION, query, top_k=top_k,
                                      threshold=threshold)

    def is_empty(self) -> bool:
        return self._es.collection_size(self.COLLECTION) == 0
