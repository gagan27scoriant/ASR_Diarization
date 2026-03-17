from __future__ import annotations

import os

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_MODEL: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True)
