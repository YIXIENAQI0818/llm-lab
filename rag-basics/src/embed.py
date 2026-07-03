from sentence_transformers import SentenceTransformer

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    return _model


def embed(texts):
    """将文本列表转换为向量数组 (n_texts, embedding_dim)"""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True)
