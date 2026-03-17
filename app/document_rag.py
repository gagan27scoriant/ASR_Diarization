from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np
import requests

from app.document_store import create_document, read_document, update_document
from app.embeddings import embed_texts


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
DEFAULT_CHUNK_SIZE = int(os.getenv("DOC_CHUNK_SIZE", "2000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DOC_CHUNK_OVERLAP", "80"))
DEFAULT_TOP_K = int(os.getenv("DOC_RETRIEVE_TOP_K", "5"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
RAG_MODEL = os.getenv("RAG_MODEL", os.getenv("SUMMARY_MODEL", "mistral"))
RAG_TIMEOUT_SECONDS = int(os.getenv("RAG_TIMEOUT_SECONDS", "10800"))


def _clean_text(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    if sep:
        parts = text.split(sep)
    else:
        parts = list(text)

    chunks: list[str] = []
    current = ""
    for part in parts:
        if not current:
            candidate = part
        else:
            candidate = f"{current}{sep}{part}" if sep else f"{current}{part}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(part) > chunk_size:
            chunks.extend(_recursive_split(part, chunk_size, separators[1:]))
        else:
            current = part

    if current:
        chunks.append(current)
    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out: list[str] = [chunks[0]]
    for idx in range(1, len(chunks)):
        prev = out[-1]
        prefix = prev[-overlap:] if len(prev) > overlap else prev
        out.append(prefix + chunks[idx])
    return out


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    size = int(chunk_size or DEFAULT_CHUNK_SIZE)
    overlap = int(chunk_overlap or DEFAULT_CHUNK_OVERLAP)
    seps = separators or DEFAULT_SEPARATORS
    raw_chunks = _recursive_split(cleaned, size, seps)
    merged = [c.strip() for c in raw_chunks if c and c.strip()]
    return _apply_overlap(merged, overlap)


def _build_prompt(question: str, context: list[str], history: list[dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        [f"[Chunk {idx + 1}]\n{chunk}" for idx, chunk in enumerate(context)]
    )
    history_block = "\n".join(
        [f"{item.get('role','user').title()}: {item.get('content','')}" for item in history]
    ).strip()
    return (
        "You are a precise assistant. Answer the user using only the provided document context. "
        "If the answer is not in the context, say you don't know and suggest what to look for.\n\n"
        f"Document Context:\n{context_block}\n\n"
        f"Conversation History:\n{history_block}\n\n"
        f"Question: {question}\n"
        "Answer (cite chunk numbers when relevant):"
    )


def _ask_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": RAG_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=RAG_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def ingest_document_text(
    extracted_text: str,
    filename: str,
    owner: dict | None = None,
) -> dict[str, Any]:
    doc_id = uuid.uuid4().hex
    ext = os.path.splitext(filename.lower())[1].lstrip(".")
    chunks = chunk_text(extracted_text)
    if not chunks:
        raise ValueError("No readable text found in document")
    embeddings = embed_texts(chunks).tolist()
    create_document(
        doc_id,
        {
            "filename": filename,
            "document_type": ext,
            "text": extracted_text,
            "chunks": chunks,
            "embeddings": embeddings,
            "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            "chat_history": [],
            "owner": {
                "id": owner.get("id") if owner else None,
                "email": owner.get("email") if owner else None,
                "department": owner.get("department") if owner else None,
                "role": owner.get("role_name") if owner else None,
            },
        },
    )
    return {
        "document_id": doc_id,
        "chunks": chunks,
        "chunk_count": len(chunks),
    }


def retrieve_document_chunks(doc_id: str, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    record = read_document(doc_id) or {}
    chunks = record.get("chunks") or []
    embeddings = record.get("embeddings") or []
    if not chunks or not embeddings:
        return []
    matrix = np.array(embeddings, dtype=np.float32)
    query_vec = embed_texts([query]).astype(np.float32)[0]
    scores = matrix @ query_vec
    k = max(1, min(int(top_k or DEFAULT_TOP_K), len(chunks)))
    top_idx = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_idx:
        results.append(
            {
                "index": int(idx),
                "score": float(scores[int(idx)]),
                "text": chunks[int(idx)],
            }
        )
    return results


def answer_document_question(doc_id: str, question: str, top_k: int | None = None) -> dict[str, Any]:
    record = read_document(doc_id) or {}
    if not record:
        raise ValueError("Document not found")
    history = record.get("chat_history") or []
    trimmed_history = history[-8:]
    results = retrieve_document_chunks(doc_id, question, top_k=top_k)
    context = [hit.get("text") or "" for hit in results]
    if context:
        prompt = _build_prompt(question, context, trimmed_history)
        answer = _ask_ollama(prompt)
    else:
        answer = "I couldn't find relevant content in the document to answer that. Try asking in a different way."

    updated_history = list(trimmed_history)
    updated_history.append({"role": "user", "content": question})
    updated_history.append({"role": "assistant", "content": answer})
    update_document(doc_id, {"chat_history": updated_history})

    return {
        "answer": answer,
        "sources": results,
        "history": updated_history,
    }
