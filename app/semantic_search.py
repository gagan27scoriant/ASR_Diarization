from __future__ import annotations

from typing import Any

import numpy as np

from app.embeddings import MODEL_NAME, embed_texts
from app.history_store import read_history_item, update_history_embeddings


def search_history_segments(session_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    history = read_history_item(session_id) or {}
    segments = history.get("transcript") or []
    texts = [str(seg.get("text") or "").strip() for seg in segments]
    if not texts or not query:
        return []

    embeddings = history.get("embeddings")
    if (
        not embeddings
        or history.get("embedding_model") != MODEL_NAME
        or not isinstance(embeddings, list)
        or len(embeddings) != len(texts)
    ):
        embeddings = embed_texts(texts).tolist()
        update_history_embeddings(session_id, embeddings, MODEL_NAME)

    emb_matrix = np.array(embeddings, dtype=np.float32)
    query_emb = embed_texts([query]).astype(np.float32)[0]
    scores = emb_matrix @ query_emb
    top_k = max(1, min(int(top_k or 5), len(texts)))
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        seg = segments[int(idx)] or {}
        results.append(
            {
                "segment_index": int(idx),
                "score": float(scores[int(idx)]),
                "text": seg.get("text") or "",
                "speaker": seg.get("speaker") or "",
                "start": seg.get("start") or 0,
                "end": seg.get("end") or 0,
            }
        )
    return results
