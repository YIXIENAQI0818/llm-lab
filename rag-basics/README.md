# rag-basics

RAG（Retrieval-Augmented Generation）检索增强生成基础实验。学习文档向量化 → 语义检索 → 增强生成的完整流程。

## 实验内容

| 回合 | 主题 | 学习点 |
|------|------|--------|
| 01 | Embedding 基础 | 文本 → 向量、余弦相似度、语义距离 |
| 02 | 向量检索 | 文档分块、向量索引、Top-K 检索 |
| 03 | 完整 RAG 管线 | 检索 + 生成、来源引用 |

## 技术栈

- **Embedding**: `sentence-transformers` + `BAAI/bge-small-zh-v1.5`（本地模型）
- **生成**: DeepSeek API（OpenAI 兼容 SDK）
- **向量计算**: `numpy` + `scikit-learn`

## 环境

```bash
cd rag-basics
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env: DEEPSEEK_API_KEY=sk-your-key-here
```

首次运行时会自动下载 BGE 向量模型。

## 使用

`src/client.py` 和 `src/embed.py` 是导入辅助模块（提供 API 客户端和向量化函数），不直接运行。

实验通过 Jupyter 笔记本进行：

```bash
jupyter notebook notebooks/01_embedding.ipynb
```

`notebooks/` 目录下包含 3 个回合的实验笔记，每个可独立运行。
