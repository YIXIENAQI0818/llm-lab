"""知识库 — 封装文档索引与检索，仿 LongTermMemory 模式。

索引由应用层在启动时调用，不暴露为 LLM 工具。
检索通过 search_docs 工具供 LLM 调用。
"""

from ..agent_framework.chunker import TokenChunker, load_markdown_files
from ..agent_framework.embedding_store import EmbeddingStore


class KnowledgeBase:
    """离线索引 + 在线检索的知识库。"""

    COLLECTION = "documents"

    def __init__(self, embedding_store: EmbeddingStore,
                 chunk_tokens: int = 256, overlap_tokens: int = 64):
        self._es = embedding_store
        self._chunker = TokenChunker(chunk_tokens=chunk_tokens,
                                     overlap_tokens=overlap_tokens)

    def index(self, path: str) -> str:
        """索引目录下所有 .md 文件（启动时调用，非 LLM 工具）。"""
        docs = load_markdown_files(path)
        if not docs:
            return f"在 {path} 下未找到 .md 文件"

        total_chunks = 0
        details = []
        for doc in docs:
            chunks = self._chunker.chunk(doc["content"], doc["name"])
            for c in chunks:
                self._es.add(self.COLLECTION, c["text"], c["meta"])
            total_chunks += len(chunks)
            details.append(f"  {doc['name']}: {len(chunks)} chunks")

        return f"索引完成：{len(docs)} 个文件 → {total_chunks} 个片段\n" + "\n".join(details)

    def search(self, query: str, top_k: int = 3,
               threshold: float = 0.3) -> list[dict]:
        """语义检索文档片段（供 search_docs 工具调用）。"""
        return self._es.search(self.COLLECTION, query, top_k=top_k,
                               threshold=threshold)

    def is_empty(self) -> bool:
        return self._es.collection_size(self.COLLECTION) == 0
