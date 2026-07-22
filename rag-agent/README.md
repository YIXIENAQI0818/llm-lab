# rag-agent

Agent + RAG 深度融合实验。在 agent-advanced 成熟 Agent 框架基础上，集成完整 RAG 检索管线，并将会话记忆、工具描述、知识文档全部迁移至 ChromaDB 向量数据库。

## 实验路线

| 回合 | 主题 | 学习点 |
|------|------|--------|
| 01 | 文档分块 + RAG 工具集成 | tiktoken 滑动窗口分块、index_documents / search_docs 工具 |
| 02 | ChromaDB 全量迁移 | PersistentClient + BGE 嵌入、KB/ToolRegistry/LTM 三集合全部迁入 |
| 03 | 混合检索增强 | BM25 + Dense 并行检索、RRF 融合、Query Rewriting（expand/decompose） |
| 04 | Cross-Encoder 精排 | bge-reranker-v2-m3 重排序、粗 20 精 5 两阶段检索 |
| 05 | 项目收尾 | 旧 EmbeddingStore 废弃、架构文档整理 |

Round 02 超路线图完成：原计划只迁移 KB，实际 LTM + ToolRegistry + KB 三个集合全部迁入 ChromaDB。

## 核心特性

- **混合检索**: BM25 稀疏 + Dense 稠密 → RRF 融合排序
- **查询改写**: LLM 驱动 Query Expansion（扩展同义词）和 Decomposition（拆分子问题）
- **精排**: Cross-Encoder 对粗排 Top-20 重排序取 Top-5
- **ChromaDB**: PersistentClient 持久化，全局单例 BGE 嵌入函数
- **组件自供工具**: KB、LTM、ToolRegistry 通过 `get_tools()` 暴露能力

## 技术栈

- **LLM**: DeepSeek API
- **Embedding**: BAAI/bge-small-zh-v1.5
- **Cross-Encoder**: BAAI/bge-reranker-v2-m3
- **向量数据库**: ChromaDB PersistentClient
- **BM25**: rank-bm25 + jieba 分词
- **分块**: tiktoken cl100k_base（256 tokens, 64 overlap）

## 环境

```bash
cd rag-agent
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-key-here
```

## 使用

```bash
python cli.py
```

启动后自动构建知识库索引（`data/` 目录下的 Markdown 文件）。进入 REPL 后可提问，Agent 会先检索知识库再回答。

### CLI 命令

| 命令 | 功能 |
|------|------|
| `/reindex` | 重建知识库向量索引（force 模式强制覆盖） |
| `/reindex_memories` | 重建长期记忆索引 |
| `/reindex_tools` | 重建工具索引 |
| `/memories` | 查看长期记忆 |
| `/forget <n>` | 删除记忆 |
| `/plan` | 查看当前计划 |

## 项目结构

```
rag-agent/
├── cli.py                     # 单入口（一行创建 Agent + 启动 REPL）
├── src/
│   ├── agent_framework/       # Agent 核心 + ChromaDBStore
│   └── capabilities/
│       ├── rag_infra/         # RAG 管线（chunker / retriever / reranker）
│       ├── knowledge_base.py  # KB（高层，组合 rag_infra）
│       ├── long_term_memory.py
│       ├── plan_manager.py
│       ├── tool_registry.py   # 工具注册 + 语义搜索路由
│       └── demo_tools.py      # 示例纯工具
├── data/                      # 知识库 Markdown 文档
├── agent_memory/              # LTM JSON + Plan 持久化（gitignored）
└── chroma_data/               # ChromaDB 持久化（gitignored）
```

## 实验笔记

`notebooks/` 目录下包含 Rounds 01 和 04 的实验笔记（02-03 和 05 改动直接体现在框架源码中）。
