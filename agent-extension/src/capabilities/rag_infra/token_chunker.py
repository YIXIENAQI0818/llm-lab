"""Token-based 文档分块，使用 tiktoken 精确按 token 数切分。

支持中英文混合文档，tiktoken 的 cl100k_base 编码对两种语言都能正确计数。
每个 chunk 带有 metadata（source, chunk_index, token_start, token_end）。
"""

import tiktoken
from pathlib import Path

# tiktoken 编码模型选择:
#   cl100k_base — GPT-4 / GPT-3.5 使用的编码，中英文通用（当前使用）
#   如换用其他模型，修改 _MODEL 即可
_MODEL = "cl100k_base"

# 分块参数
_CHUNK_TOKENS = 256   # 每个 chunk 的 token 数
_OVERLAP_TOKENS = 64  # 窗口重叠 token 数（25%）


class TokenChunker:
    """按 token 数对文档进行滑动窗口分块，支持 overlap。"""

    def __init__(self):
        self._enc = tiktoken.get_encoding(_MODEL)

    def chunk(self, text: str, source_name: str = "") -> list[dict]:
        """将文本切分为带 overlap 的 token 窗口。"""
        token_ids = self._enc.encode(text)
        total = len(token_ids)
        if total == 0:
            return []

        chunks = []
        step = _CHUNK_TOKENS - _OVERLAP_TOKENS

        for i, start in enumerate(range(0, total, step)):
            end = min(start + _CHUNK_TOKENS, total)
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
    """读取目录下的所有 .md 文件。"""
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
