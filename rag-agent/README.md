# rag-agent

Agent + RAG 深度融合实验。将成熟的 Agent 框架与完整 RAG 检索管线结合，实现基于知识库的智能问答。

## 特性

- **混合检索**：BM25 稀疏检索 + Dense 稠密检索，RRF 融合排序
- **查询改写**：LLM 驱动的 Query Expansion（扩展同义词）和 Decomposition（拆分子问题）
- **精排**：Cross-Encoder 对召回结果重新打分（BAAI/bge-reranker-v2-m3）
- **分块**：tiktoken 滑动窗口切分（256 tokens，64 overlap）
- **向量数据库**：ChromaDB PersistentClient 持久化存储
- **组件自供工具**：KB、LTM、ToolRegistry 各自通过 `get_tools()` 暴露能力

## 环境

- Python 3.10+
- DeepSeek API Key
- ChromaDB（持久化向量数据库）
- 向量模型首次运行时自动下载（BGE、Cross-Encoder）

### 安装

```bash
cd rag-agent
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填入 API Key：

```
DEEPSEEK_API_KEY=sk-your-key-here
```

### 运行

```bash
python src/cli.py
```

启动后会自动构建知识库索引（`data/` 目录下的 Markdown 文件）。进入 REPL 后可以提问，Agent 会先检索知识库再回答。

### 索引命令

| 命令 | 功能 |
|------|------|
| `/reindex` | 重建知识库向量索引 |
| `/reindex_memories` | 重建长期记忆索引 |
| `/reindex_tools` | 重建工具索引 |
| `/memories` | 查看所有长期记忆 |
| `/forget <n>` | 删除记忆 |
| `/plan` | 查看当前计划 |

### 实验笔记

`notebooks/` 目录下包含 5 个回合的实验笔记，覆盖从 token 分块到 Cross-Encoder 精排的完整 RAG 管线演进。
