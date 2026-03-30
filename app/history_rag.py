from __future__ import annotations

import os
from typing import Any

import requests

from app.history_store import read_history_item, update_history_chat
from app.semantic_search import search_history_segments

DEFAULT_TOP_K = int(os.getenv("HISTORY_RETRIEVE_TOP_K", "5"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
RAG_MODEL = os.getenv("RAG_MODEL", os.getenv("SUMMARY_MODEL", "mistral"))
RAG_TIMEOUT_SECONDS = int(os.getenv("RAG_TIMEOUT_SECONDS", "10800"))


def _format_time(seconds: float | int) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        total = 0
    hrs = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _build_prompt(question: str, context: list[str], history: list[dict[str, Any]]) -> str:
    context_block = "\n\n".join(
        [f"[Segment {idx + 1}]\n{segment}" for idx, segment in enumerate(context)]
    )
    history_block = "\n".join(
        [f"{item.get('role','user').title()}: {item.get('content','')}" for item in history]
    ).strip()
    return (
        "You are a precise assistant. Answer the user using only the provided transcript context. "
        "Do not mention segment numbers, sources, or phrases like 'mentioned in Segment 1'. "
        "If the answer is not in the context, say you don't know and suggest what to look for.\n\n"
        f"Transcript Context:\n{context_block}\n\n"
        f"Conversation History:\n{history_block}\n\n"
        f"Question: {question}\n"
        "Answer (do not mention segment numbers or sources):"
    )


def _ask_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": RAG_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.2,
                "top_k": 20,
            },
        },
        timeout=RAG_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()

def _collect_context_segments(
    transcript: list[dict[str, Any]],
    hit_indices: list[int],
    window: int = 1,
) -> list[dict[str, Any]]:
    if not transcript or not hit_indices:
        return []
    max_idx = len(transcript) - 1
    indices: list[int] = []
    seen = set()
    for idx in hit_indices:
        start = max(0, idx - window)
        end = min(max_idx, idx + window)
        for i in range(start, end + 1):
            if i in seen:
                continue
            seen.add(i)
            indices.append(i)
    indices.sort()
    return [transcript[i] for i in indices]


def answer_history_question(session_id: str, question: str, top_k: int | None = None) -> dict[str, Any]:
    record = read_history_item(session_id) or {}
    if not record:
        raise ValueError("History not found")

    history = record.get("qa_history") or []
    trimmed_history = history[-8:]

    results = search_history_segments(session_id, question, top_k=int(top_k or DEFAULT_TOP_K))
    transcript_segments = record.get("transcript") or []
    hit_indices = [int(hit.get("segment_index")) for hit in results if hit.get("segment_index") is not None]
    expanded_segments = _collect_context_segments(transcript_segments, hit_indices, window=1)

    context = []
    for seg in expanded_segments:
        speaker = seg.get("speaker") or ""
        start = _format_time(seg.get("start") or 0)
        end = _format_time(seg.get("end") or 0)
        text = seg.get("text") or ""
        context.append(f"Speaker: {speaker}\nTime: {start} - {end}\nText: {text}")

    if context:
        prompt = _build_prompt(question, context, trimmed_history)
        answer = _ask_ollama(prompt)
    else:
        answer = "I couldn't find relevant content in the transcript to answer that. Try asking in a different way."

    updated_history = list(trimmed_history)
    updated_history.append({"role": "user", "content": question})
    updated_history.append({"role": "assistant", "content": answer})
    update_history_chat(session_id, updated_history)

    return {
        "answer": answer,
        "sources": results,
        "history": updated_history,
    }
