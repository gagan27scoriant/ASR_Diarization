from __future__ import annotations

import json
import os
from typing import Any

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
AGENT_MODEL = os.getenv("AGENT_MODEL", os.getenv("SUMMARY_MODEL", "mistral"))
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "10800"))


def _ask_ollama(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": AGENT_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=AGENT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def chat_with_agent(query: str, payload: dict[str, Any] | None = None) -> str:
    clean_query = (query or "").strip()
    context_bits: list[str] = []
    payload = payload or {}

    if payload.get("session_id"):
        context_bits.append(f"Transcript session is available: {payload.get('session_id')}")
    if payload.get("document_id"):
        context_bits.append(f"Document is available: {payload.get('document_id')}")
    if payload.get("content"):
        context_bits.append("Additional text content is attached.")

    context_block = "\n".join(context_bits) if context_bits else "No uploaded file or working context is attached."

    prompt = (
        "You are the conversational layer for this specific AI knowledge workspace application.\n"
        "You must only help with operations that this application can actually perform.\n"
        "Do not claim you can help with general world knowledge, essays, recipes, movies, life advice, current events, reminders, or unrelated chatbot abilities.\n"
        "Be direct, useful, and concise.\n\n"
        "This application can help with:\n"
        "- Uploading audio or video and processing it for transcription and speaker diarization.\n"
        "- Summarizing transcripts or meeting content.\n"
        "- Asking questions over an existing transcript session.\n"
        "- Semantic or keyword-like search over transcript history.\n"
        "- Uploading documents and asking questions about them.\n"
        "- Translating text, summaries, or transcript/document-derived content.\n"
        "- Converting supported text content into speech.\n"
        "- Explaining how to use these app features.\n\n"
        "If the user asks what you can do, answer only with the app features above.\n"
        "If the user asks for something outside these app capabilities, politely say that this assistant is limited to the application's transcription, summary, document, translation, search, and text-to-speech workflows.\n"
        "If a request requires an uploaded file, transcript, or document, clearly say what is missing and what the user should upload or open.\n\n"
        f"Context:\n{context_block}\n\n"
        f"User Query:\n{clean_query}\n\n"
        "Assistant Response:"
    )
    return _ask_ollama(prompt)


def infer_agent_action(query: str, payload: dict[str, Any], tool_names: list[str]) -> dict[str, Any] | None:
    clean_query = (query or "").strip()
    if not clean_query:
        return None

    prompt = (
        "You are an agent planner. Decide whether the user needs a direct chatbot reply or one tool.\n"
        "Available tools:\n"
        f"{', '.join(tool_names)}\n\n"
        "Return strict JSON only with this shape:\n"
        "{"
        "\"mode\":\"chat\"|\"tool\","
        "\"tool\":\"chat_response\" or tool name,"
        "\"reason\":\"short reason\","
        "\"arguments\":{"
        "\"question\":\"...\","
        "\"text\":\"...\","
        "\"target_lang\":\"...\","
        "\"tts_lang\":\"...\""
        "}"
        "}\n\n"
        "Rules:\n"
        "- Use mode=chat when the user is asking a normal conversational question and no tool is required.\n"
        "- Use a tool only when the request clearly needs one.\n"
        "- If the user asks to translate text written in the query itself, extract that text into arguments.text.\n"
        "- If the user asks text-to-speech for text written in the query itself, extract that text into arguments.text.\n"
        "- If the user asks about an existing transcript or document, prefer answer_history or answer_document when context exists.\n"
        "- If unsure, use mode=chat.\n\n"
        f"Payload Context: {json.dumps(payload, ensure_ascii=True)}\n"
        f"User Query: {clean_query}\n"
    )

    raw = _ask_ollama(prompt)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None

    mode = str(parsed.get("mode") or "").strip().lower()
    tool = str(parsed.get("tool") or "").strip()
    arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
    reason = str(parsed.get("reason") or "").strip()

    if mode not in {"chat", "tool"}:
        return None
    if mode == "tool" and tool not in tool_names:
        return None

    return {
        "mode": mode,
        "tool": "chat_response" if mode == "chat" else tool,
        "reason": reason or "Planned by LLM query understanding.",
        "arguments": arguments,
    }
