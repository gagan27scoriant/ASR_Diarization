from __future__ import annotations

from typing import Any

from app.db import get_db


def _chunks_col():
    return get_db()["document_chunks"]


def replace_document_chunks(doc_id: str, chunks: list[str], embeddings: list[list[float]]) -> None:
    if not doc_id:
        return
    _chunks_col().delete_many({"document_id": doc_id})
    payload = []
    for idx, text in enumerate(chunks):
        emb = embeddings[idx] if idx < len(embeddings) else []
        payload.append(
            {
                "document_id": doc_id,
                "index": idx,
                "text": text,
                "embedding": emb,
            }
        )
    if payload:
        _chunks_col().insert_many(payload, ordered=False)


def load_document_chunks(doc_id: str) -> list[dict[str, Any]]:
    if not doc_id:
        return []
    return list(_chunks_col().find({"document_id": doc_id}).sort("index", 1))
