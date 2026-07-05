"""Token-based 文档分块，使用 tiktoken 精确按 token 数切分。

支持中英文混合文档，tiktoken 的 cl100k_base 编码对两种语言都能正确计数。
每个 chunk 带有 metadata（source, chunk_index, token_start, token_end）。
"""

import tiktoken
from pathlib import Path


class TokenChunker:
    """按 token 数对文档进行滑动窗口分块，支持 overlap。"""

    def __init__(self, chunk_tokens: int = 256, overlap_tokens: int = 64,
                 model: str = "cl100k_base"):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self._enc = tiktoken.get_encoding(model)

    def chunk(self, text: str, source_name: str = "") -> list[dict]:
        """将文本切分为带 overlap 的 token 窗口。

        Args:
            text: 原始文本
            source_name: 来源文件名

        Returns:
            [{"text": str, "meta": {"source": str, "chunk_index": int,
             "token_start": int, "token_end": int}}, ...]
        """
        token_ids = self._enc.encode(text)
        total = len(token_ids)
        if total == 0:
            return []

        chunks = []
        step = self.chunk_tokens - self.overlap_tokens
        if step <= 0:
            raise ValueError("overlap_tokens must be less than chunk_tokens")

        for i, start in enumerate(range(0, total, step)):
            end = min(start + self.chunk_tokens, total)
            chunk_ids = token_ids[start:end]
            chunk_text = self._enc.decode(chunk_ids)
            chunks.append({
                "text": chunk_text,
                "meta": {
                    "source": source_name,
                    "chunk_index": i,
                    "token_start": start,
                    "token_end": end,
                },
            })
            if end >= total:
                break

        return chunks

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))


def load_markdown_files(path: str) -> list[dict]:
    """读取目录下的所有 .md 文件。

    Args:
        path: 文件或目录路径

    Returns:
        [{"name": str, "content": str}, ...]
    """
    p = Path(path)
    files = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.md"))
    else:
        raise FileNotFoundError(f"路径不存在: {path}")

    docs = []
    for f in files:
        content = f.read_text(encoding="utf-8")
        docs.append({"name": f.name, "content": content})
    return docs
