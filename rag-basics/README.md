# rag-basics

RAG（Retrieval-Augmented Generation）检索增强生成基础实验。学习向量检索 + 文档问答的基本流程：将文档分块、向量化存储、根据用户问题检索相关片段、交给 LLM 生成回答。

## 内容

- 文档分块（固定大小 + 滑动窗口）
- 向量化（sentence-transformers / BGE 模型）
- 向量检索（ChromaDB）
- 检索结果注入 LLM 生成

## 环境

- Python 3.10+
- DeepSeek API Key（用于 LLM 生成）
- 向量模型首次运行时自动下载

### 安装

```bash
cd rag-basics
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

详见过 `notebooks/` 目录下的实验笔记。
