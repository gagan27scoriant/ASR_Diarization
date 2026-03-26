from __future__ import annotations

from typing import Any

from app.agent_llm import infer_agent_action
from app.agent_tools import TOOLS, AgentTool, get_tool, validate_tool_payload


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _wants_translation(query: str, payload: dict[str, Any]) -> bool:
    if payload.get("target_lang"):
        return True
    return any(term in query for term in ("translate", "translation", "convert to", "in hindi", "in tamil", "in english"))


def _wants_tts(query: str, payload: dict[str, Any]) -> bool:
    if payload.get("tts_lang"):
        return True
    return any(term in query for term in ("text to speech", "tts", "speak", "voice", "audio version", "read aloud"))


def _wants_summary(query: str, payload: dict[str, Any]) -> bool:
    if payload.get("content") and not payload.get("target_lang"):
        return any(term in query for term in ("summar", "minutes", "meeting notes", "mom", "recap"))
    return any(term in query for term in ("summar", "minutes", "meeting notes", "recap"))


def _wants_document_qa(query: str, payload: dict[str, Any]) -> bool:
    return bool(payload.get("document_id")) and (
        "?" in query or any(term in query for term in ("what ", "who ", "why ", "how ", "ask ", "tell me ", "explain "))
    )


def _wants_history_qa(query: str, payload: dict[str, Any]) -> bool:
    return bool(payload.get("session_id")) and (
        "?" in query or any(term in query for term in ("who said", "what did", "when did", "why did", "how did", "ask "))
    )


def _wants_history_search(query: str, payload: dict[str, Any]) -> bool:
    return bool(payload.get("session_id")) and any(term in query for term in ("search", "find", "lookup", "keyword"))


def _wants_media_processing(query: str, payload: dict[str, Any]) -> bool:
    return bool(payload.get("file_path")) and any(term in query for term in ("transcribe", "diar", "process", "speaker", "recording", "audio", "video"))


def _choose_primary_tool(query: str, payload: dict[str, Any]) -> AgentTool | None:
    explicit = get_tool(payload.get("tool") or "")
    if explicit:
        return explicit

    if _wants_media_processing(query, payload):
        return TOOLS["process_media"]
    if _wants_tts(query, payload):
        return TOOLS["text_to_speech"]
    if _wants_summary(query, payload):
        return TOOLS["summarize_transcript"]
    if _wants_translation(query, payload):
        return TOOLS["translate_text"]
    if _wants_document_qa(query, payload):
        return TOOLS["answer_document"]
    if _wants_history_search(query, payload):
        return TOOLS["search_history"]
    if _wants_history_qa(query, payload):
        return TOOLS["answer_history"]
    return None


def _merge_planner_arguments(payload: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload or {})
    for key in ("question", "text", "target_lang", "tts_lang"):
        value = arguments.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if not merged.get(key):
            merged[key] = value
    return merged


def _build_plan(tool: AgentTool, payload: dict[str, Any]) -> list[dict[str, Any]]:
    query = _norm(payload.get("query") or "")
    plan: list[dict[str, Any]] = [{"tool": tool.name, "reason": f"Primary match for query intent: '{query or tool.name}'"}]

    if tool.name == "summarize_transcript" and payload.get("target_lang") and "translate" in query:
        plan.append({"tool": "translate_text", "reason": "Translate the generated summary into the requested language."})

    return plan


def plan_agent_query(payload: dict[str, Any]) -> dict[str, Any]:
    query = _norm(payload.get("query") or "")
    planning_payload = dict(payload or {})
    planner_result: dict[str, Any] | None = None

    try:
        planner_result = infer_agent_action(query, planning_payload, list(TOOLS.keys()))
    except Exception:
        planner_result = None

    if planner_result and planner_result.get("mode") == "tool":
        planning_payload = _merge_planner_arguments(planning_payload, planner_result.get("arguments") or {})
        tool = get_tool(planner_result.get("tool") or "")
    elif planner_result and planner_result.get("mode") == "chat":
        tool = TOOLS["chat_response"]
    else:
        tool = _choose_primary_tool(query, planning_payload)

    if not tool and not any(
        planning_payload.get(key) for key in ("file_path", "session_id", "document_id", "content", "text")
    ) and query:
        tool = TOOLS["chat_response"]

    if not tool:
        return {
            "ok": False,
            "error": "No matching tool found for the query. Provide tool explicitly or include query/context such as session_id, document_id, or target_lang.",
        }

    plan = _build_plan(tool, planning_payload)
    if planner_result and planner_result.get("reason"):
        plan[0]["reason"] = planner_result["reason"]

    missing = validate_tool_payload(tool, planning_payload)
    if missing:
        return {
            "ok": False,
            "tool": tool.name,
            "required_permission": tool.required_permission,
            "missing_fields": missing,
            "plan": plan,
            "error": f"Missing required fields for tool '{tool.name}': {', '.join(missing)}",
        }

    required_permission = tool.required_permission
    if tool.name == "summarize_transcript" and planning_payload.get("target_lang") and "translate" in query:
        required_permission = "summary:generate + translate:run"

    return {
        "ok": True,
        "tool": tool.name,
        "required_permission": required_permission,
        "plan": plan,
        "resolved_payload": planning_payload,
    }


def run_agent_query(payload: dict[str, Any], deps: dict[str, Any]) -> dict[str, Any]:
    route = plan_agent_query(payload)
    if not route.get("ok"):
        return route

    resolved_payload = route.get("resolved_payload") or payload
    tool = TOOLS[route["tool"]]
    result = tool.handler(resolved_payload, deps)

    if route["tool"] == "summarize_transcript" and resolved_payload.get("target_lang") and "translate" in _norm(resolved_payload.get("query") or ""):
        translated = TOOLS["translate_text"].handler(
            {
                "text": result.get("summary") or "",
                "target_lang": resolved_payload.get("target_lang"),
                "source_lang": resolved_payload.get("source_lang") or "eng_Latn",
            },
            deps,
        )
        result["translated_summary"] = translated.get("text") or ""
        route["required_permission"] = "summary:generate + translate:run"

    return {
        "ok": True,
        "selected_tool": tool.name,
        "required_permission": route.get("required_permission"),
        "plan": route.get("plan") or [],
        "result": result,
    }
