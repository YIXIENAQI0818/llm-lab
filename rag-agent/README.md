# rag-agent

Agent + RAG 深度融合实验。在成熟的 Agent 框架基础上，集成完整的 RAG 检索管线。

## 特性

- **混合检索**: BM25（稀疏）+ Dense（稠密），RRF 融合
- **查询改写**: LLM 驱动的 Query Expansion / Decomposition
- **精排**: Cross-Encoder（BAAI/bge-reranker-v2-m3）
- **分块**: tiktoken 滑动窗口（256 tokens，64 overlap）
- **向量数据库**: ChromaDB PersistentClient

## 运行

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 DEEPSEEK_API_KEY
python src/cli.py
```
