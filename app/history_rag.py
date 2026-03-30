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
MIN_CONTEXT_OVERLAP = float(os.getenv("HISTORY_ANSWER_MIN_OVERLAP", "0.12"))


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


def _normalize_tokens(text: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in (text or ""))
    return {tok for tok in cleaned.split() if len(tok) >= 4}


def _is_answer_grounded(answer: str, context: list[str]) -> bool:
    if not answer.strip():
        return False
    ctx_text = " ".join(context)
    ctx_tokens = _normalize_tokens(ctx_text)
    ans_tokens = _normalize_tokens(answer)
    if not ans_tokens:
        return False
    overlap = len(ans_tokens & ctx_tokens) / max(1, len(ans_tokens))
    return overlap >= MIN_CONTEXT_OVERLAP


def answer_history_question(session_id: str, question: str, top_k: int | None = None) -> dict[str, Any]:
    record = read_history_item(session_id) or {}
    if not record:
        raise ValueError("History not found")

    history = record.get("qa_history") or []
    trimmed_history = history[-8:]

    transcript_segments = record.get("transcript") or []
    context = []
    for seg in transcript_segments:
        speaker = seg.get("speaker") or ""
        start = _format_time(seg.get("start") or 0)
        end = _format_time(seg.get("end") or 0)
        text = seg.get("text") or ""
        context.append(f"Speaker: {speaker}\nTime: {start} - {end}\nText: {text}")

    # Keep semantic search results for sources/UX, but do not use them for context.
    results = search_history_segments(session_id, question, top_k=int(top_k or DEFAULT_TOP_K))

    if context:
        prompt = _build_prompt(question, context, trimmed_history)
        answer = _ask_ollama(prompt)
        if not _is_answer_grounded(answer, context):
            answer = (
                "I don't know based on the provided transcript context. "
                "Try asking about a specific speaker or time range."
            )
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
