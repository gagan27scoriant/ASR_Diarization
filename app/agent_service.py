from __future__ import annotations

from typing import Any

from app.document_rag import answer_document_question, ingest_document_text
from app.history_rag import answer_history_question
from app.history_store import update_history_chat
from app.processing_service import process_document_upload, process_media_pipeline, summarize_and_persist


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _transcript_to_text(transcript: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for seg in transcript or []:
        speaker = (seg.get("speaker") or "Speaker").strip()
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _query_wants_summary(query: str) -> bool:
    q = _norm(query)
    return any(term in q for term in ("summar", "minutes", "recap", "meeting notes"))


def _query_wants_document_answer(query: str, payload: dict[str, Any]) -> bool:
    q = _norm(query)
    explicit_question = bool((payload.get("question") or "").strip())
    question_starters = ("ask ", "what ", "who ", "why ", "how ", "when ", "where ", "tell me ", "explain ")
    return explicit_question or q.endswith("?") or any(q.startswith(term) for term in question_starters)


def _query_wants_transcript_answer(query: str, payload: dict[str, Any]) -> bool:
    return _query_wants_document_answer(query, payload)


def handle_uploaded_file(
    uploaded_file,
    query: str,
    payload: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    filename = (getattr(uploaded_file, "filename", None) or "").strip()
    if not filename:
        raise ValueError("Uploaded file is missing a filename")

    if deps["is_supported_document"](filename):
        processed = process_document_upload(
            uploaded_file,
            meeting_title="Document Summary",
            meeting_date="",
            meeting_place="",
            owner=deps.get("owner"),
        )
        rag_meta = ingest_document_text(
            processed.get("document_text") or "",
            processed.get("document_filename") or filename,
            owner=deps.get("owner"),
        )
        deps["update_document"](
            rag_meta.get("document_id"),
            {"summary": processed.get("summary") or ""},
        )

        response: dict[str, Any] = {
            "ok": True,
            "selected_tool": "process_document",
            "plan": [
                {"tool": "process_document", "reason": "Extract text and generate a document summary."},
                {"tool": "answer_document", "reason": "Enable query-based answers over indexed document chunks."},
            ],
            "result": {
                **processed,
                "document_id": rag_meta.get("document_id"),
                "chunk_count": rag_meta.get("chunk_count"),
                "chat_history": [],
            },
        }

        if _query_wants_document_answer(query, payload):
            question = (payload.get("question") or query or "").strip()
            if question:
                answer = answer_document_question(
                    rag_meta.get("document_id") or "",
                    question,
                    top_k=int(payload.get("top_k") or 5),
                )
                response["plan"].append(
                    {"tool": "answer_document", "reason": "Answer the user's first question about the uploaded document."}
                )
                response["result"]["answer"] = answer.get("answer") or ""
                response["result"]["sources"] = answer.get("sources") or []
                response["result"]["chat_history"] = answer.get("history") or []

        return response

    media_result = process_media_pipeline(
        deps["source_path"],
        filename,
        deps["asr_model"],
        deps["diarization_pipeline"],
        owner=deps.get("owner"),
    )

    response = {
        "ok": True,
        "selected_tool": "process_media",
        "plan": [
            {"tool": "process_media", "reason": "Transcribe, diarize, and map speakers for the uploaded media."}
        ],
        "result": media_result,
    }

    question = (payload.get("question") or query or "").strip()
    if question and media_result.get("session_id"):
        if _query_wants_transcript_answer(query, payload):
            answer = answer_history_question(
                media_result.get("session_id") or "",
                question,
                top_k=int(payload.get("top_k") or 5),
            )
            response["plan"].append(
                {"tool": "answer_history", "reason": "Answer the user's first question about the uploaded transcript."}
            )
            response["result"]["answer"] = answer.get("answer") or ""
            response["result"]["sources"] = answer.get("sources") or []
            response["result"]["history"] = answer.get("history") or []
        else:
            seed_history = [{"role": "user", "content": question}]
            update_history_chat(media_result.get("session_id") or "", seed_history)
            response["result"]["history"] = seed_history

    if _query_wants_summary(query):
        transcript_text = _transcript_to_text(media_result.get("transcript") or [])
        summary = summarize_and_persist(
            transcript_text,
            (payload.get("meeting_title") or "Meeting").strip(),
            (payload.get("meeting_date") or "").strip(),
            (payload.get("meeting_place") or "").strip(),
            (media_result.get("session_id") or "").strip(),
        )
        response["plan"].append(
            {"tool": "summarize_transcript", "reason": "Generate a summary because the upload query requested it."}
        )
        response["result"]["summary"] = summary
        if question and media_result.get("session_id"):
            seeded_history = [{"role": "user", "content": question}, {"role": "assistant", "content": summary}]
            update_history_chat(media_result.get("session_id") or "", seeded_history)
            response["result"]["history"] = seeded_history

    return response
