from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.agent_llm import chat_with_agent
from app.document_rag import answer_document_question
from app.history_rag import answer_history_question
from app.processing_service import process_media_pipeline, summarize_and_persist
from app.semantic_search import search_history_segments
from app.tts import synthesize_speech
from app.translation import get_translator


ToolHandler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    required_permission: str
    required_fields: tuple[str, ...]
    handler: ToolHandler


def _missing_fields(payload: dict[str, Any], required_fields: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for field in required_fields:
        value = payload.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
    return missing


def summarize_transcript_tool(payload: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_and_persist(
        payload.get("content") or "",
        (payload.get("meeting_title") or "Meeting").strip(),
        (payload.get("meeting_date") or "").strip(),
        (payload.get("meeting_place") or "").strip(),
        (payload.get("session_id") or "").strip(),
    )
    return {"summary": summary}


def translate_text_tool(payload: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    translator = get_translator()
    target_lang = translator.resolve_lang_code((payload.get("target_lang") or "").strip(), is_target=True)
    source_lang = translator.resolve_lang_code((payload.get("source_lang") or "").strip(), is_target=False)

    if isinstance(payload.get("texts"), list):
        translated = translator.translate_lines(payload.get("texts") or [], target_lang=target_lang, source_lang=source_lang)
        return {
            "texts": translated,
            "target_lang": target_lang,
            "source_lang": source_lang,
        }

    text = str(payload.get("text") or payload.get("content") or "")
    translated = translator.translate_text(text, target_lang=target_lang, source_lang=source_lang)
    return {
        "text": translated,
        "target_lang": target_lang,
        "source_lang": source_lang,
    }


def answer_document_tool(payload: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return answer_document_question(
        (payload.get("document_id") or "").strip(),
        (payload.get("question") or payload.get("query") or "").strip(),
        top_k=int(payload.get("top_k") or 5),
    )


def answer_history_tool(payload: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return answer_history_question(
        (payload.get("session_id") or "").strip(),
        (payload.get("question") or payload.get("query") or "").strip(),
        top_k=int(payload.get("top_k") or 5),
    )


def search_history_tool(payload: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return {
        "results": search_history_segments(
            (payload.get("session_id") or "").strip(),
            (payload.get("query") or "").strip(),
            top_k=int(payload.get("top_k") or 5),
        )
    }


def process_media_tool(payload: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return process_media_pipeline(
        (payload.get("file_path") or "").strip(),
        (payload.get("filename") or "").strip(),
        deps["asr_model"],
        deps["diarization_pipeline"],
        owner=deps.get("owner"),
    )


def text_to_speech_tool(payload: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return synthesize_speech(
        payload.get("text") or payload.get("content") or "",
        payload.get("tts_lang") or payload.get("target_lang") or "",
        slow=str(payload.get("tts_slow") or "").strip().lower() in {"1", "true", "yes", "on"},
    )


def chat_response_tool(payload: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": chat_with_agent(
            (payload.get("query") or payload.get("question") or "").strip(),
            payload,
        )
    }


TOOLS: dict[str, AgentTool] = {
    "chat_response": AgentTool(
        name="chat_response",
        description="Respond like a normal chatbot when no external tool is required.",
        required_permission="",
        required_fields=("query",),
        handler=chat_response_tool,
    ),
    "summarize_transcript": AgentTool(
        name="summarize_transcript",
        description="Generate a structured summary from transcript text.",
        required_permission="summary:generate",
        required_fields=("content",),
        handler=summarize_transcript_tool,
    ),
    "translate_text": AgentTool(
        name="translate_text",
        description="Translate text or a list of texts into a target language.",
        required_permission="translate:run",
        required_fields=("target_lang",),
        handler=translate_text_tool,
    ),
    "answer_document": AgentTool(
        name="answer_document",
        description="Answer a question using an indexed document.",
        required_permission="rag:ask",
        required_fields=("document_id", "question"),
        handler=answer_document_tool,
    ),
    "answer_history": AgentTool(
        name="answer_history",
        description="Answer a question using a past transcript session.",
        required_permission="rag:ask",
        required_fields=("session_id", "question"),
        handler=answer_history_tool,
    ),
    "search_history": AgentTool(
        name="search_history",
        description="Run semantic search over a past transcript session.",
        required_permission="history:read",
        required_fields=("session_id", "query"),
        handler=search_history_tool,
    ),
    "process_media": AgentTool(
        name="process_media",
        description="Run the media pipeline over an audio or video file path.",
        required_permission="process:media",
        required_fields=("file_path", "filename"),
        handler=process_media_tool,
    ),
    "text_to_speech": AgentTool(
        name="text_to_speech",
        description="Convert text into spoken audio using gTTS and return an MP3 file.",
        required_permission="translate:run",
        required_fields=(),
        handler=text_to_speech_tool,
    ),
}


def get_tool(name: str) -> AgentTool | None:
    return TOOLS.get((name or "").strip())


def list_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "required_permission": tool.required_permission,
            "required_fields": list(tool.required_fields),
        }
        for tool in TOOLS.values()
    ]


def validate_tool_payload(tool: AgentTool, payload: dict[str, Any]) -> list[str]:
    missing = _missing_fields(payload, tool.required_fields)

    if tool.name in {"answer_document", "answer_history"}:
        has_question = bool((payload.get("question") or "").strip()) or bool((payload.get("query") or "").strip())
        missing = [field for field in missing if field != "question"]
        if not has_question:
            missing.append("question_or_query")

    if tool.name == "translate_text":
        has_text = isinstance(payload.get("texts"), list) or bool(str(payload.get("text") or payload.get("content") or "").strip())
        if not has_text:
            missing.append("text_or_texts")

    if tool.name == "text_to_speech":
        has_text = bool(str(payload.get("text") or payload.get("content") or "").strip())
        if not has_text:
            missing.append("text_or_content")

    return missing
