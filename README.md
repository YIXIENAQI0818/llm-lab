# llm-lab

大模型（LLM）相关学习的实验项目集合。每个子项目围绕一个特定的学习主题或验证想法展开，从 API 调用基础到多 Agent 协作，逐步深入。

## 子项目

| 子项目 | 内容 | 状态 |
|--------|------|------|
| [api-basics](api-basics/) | LLM API 调用基础（OpenAI SDK、参数、流式输出） | ✅ |
| [rag-basics](rag-basics/) | RAG 检索增强生成基础（向量检索、文档问答） | ✅ |
| [agent-basics](agent-basics/) | Agent 与 Function Calling 基础 | ✅ |
| [agent-advanced](agent-advanced/) | Agent 能力深入（记忆、计划、LTM 合并） | ✅ |
| [rag-agent](rag-agent/) | Agent + RAG 融合（ChromaDB、混合检索、Cross-Encoder 精排） | ✅ |
| [multi-agent](multi-agent/) | 多 Agent 协作（Orchestrator + Worker、并行委派） | ✅ |
| [agent-extension](agent-extension/) | Agent 扩展框架（单 Agent + Skill + MCP + 统一工具体系） | 🚧 |

## 技术栈

- **LLM**: DeepSeek API（OpenAI 兼容）
- **Embedding**: sentence-transformers + BAAI/bge-small-zh-v1.5
- **向量数据库**: ChromaDB
- **分词**: tiktoken（cl100k_base）
- **RAG**: BM25 + Dense 混合检索 + RRF 融合 + Cross-Encoder 精排

## 项目结构

```
llm-lab/
├── README.md            # 本文件
├── CLAUDE.md            # 项目级指导（开发用）
├── reports/             # 技术报告
├── <sub-project>/       # 子实验项目
│   ├── README.md
│   ├── src/             # 源代码
│   └── data/            # 数据文件
```

## 环境

每个子项目有独立的 `.env` 和 `requirements.txt`。

```bash
cd <sub-project>
pip install -r requirements.txt
cp .env.example .env    # 编辑 .env 填入 API Key
python src/cli.py
```
