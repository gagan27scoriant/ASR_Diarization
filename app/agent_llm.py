from __future__ import annotations

from datetime import datetime
import json
import os
import re
from typing import Any
from zoneinfo import ZoneInfo

import requests
from sympy import Eq, SympifyError, factor, simplify, solve, sqrt
from sympy import symbols as sympy_symbols
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
AGENT_MODEL = os.getenv("AGENT_MODEL", os.getenv("SUMMARY_MODEL", "mistral"))
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "10800"))
AGENT_TIMEZONE = os.getenv("AGENT_TIMEZONE", "Asia/Kolkata")
_X, _Y, _Z = sympy_symbols("x y z")
_SYMPY_LOCALS = {"x": _X, "y": _Y, "z": _Z, "sqrt": sqrt}
_PARSE_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)


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


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _now_local() -> datetime:
    try:
        return datetime.now(ZoneInfo(AGENT_TIMEZONE))
    except Exception:
        return datetime.now()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _format_math_value(value: Any) -> str:
    text = str(value)
    if text.endswith(".00000000000000"):
        text = text.split(".", 1)[0]
    return text


def _normalize_math_text(query: str) -> str:
    q = _norm(query)
    replacements = (
        ("multiplied by", "*"),
        ("times", "*"),
        ("x", "x"),
        ("divided by", "/"),
        ("over", "/"),
        ("plus", "+"),
        ("minus", "-"),
        ("to the power of", "^"),
        ("raised to", "^"),
        ("squared", "^2"),
        ("cubed", "^3"),
    )
    for src, dst in replacements:
        q = q.replace(src, dst)
    q = re.sub(r"(\d)\s*x\s*(\d)", r"\1*\2", q)
    q = re.sub(r"(\d+)\s*%\s*of\s*(\d+(?:\.\d+)?)", r"(\1/100)*(\2)", q)
    q = re.sub(r"square root of ([a-z0-9\(\)\+\-\*/\^\.\s]+)", r"sqrt(\1)", q)
    q = re.sub(r"\b(what is|what's|calculate|compute|evaluate|find|answer)\b", "", q)
    q = re.sub(r"\s+", " ", q).strip(" ?")
    return q


def _extract_math_target(query: str) -> tuple[str | None, str]:
    q = _normalize_math_text(query)
    if not q:
        return None, ""

    if q.startswith("solve "):
        return "solve", q[6:].strip()
    if q.startswith("simplify "):
        return "simplify", q[9:].strip()
    if q.startswith("factor "):
        return "factor", q[7:].strip()
    if _contains_any(q, ("+", "-", "*", "/", "^", "sqrt(", "=")) or re.search(r"\d", q):
        return "evaluate", q
    return None, ""
    


def _answer_math_question(query: str) -> str | None:
    action, expr_text = _extract_math_target(query)
    if not action or not expr_text:
        return None

    try:
        if action == "solve":
            if "=" in expr_text:
                left, right = expr_text.split("=", 1)
                equation = Eq(
                    parse_expr(left.strip(), local_dict=_SYMPY_LOCALS, transformations=_PARSE_TRANSFORMATIONS),
                    parse_expr(right.strip(), local_dict=_SYMPY_LOCALS, transformations=_PARSE_TRANSFORMATIONS),
                )
            else:
                equation = Eq(
                    parse_expr(expr_text, local_dict=_SYMPY_LOCALS, transformations=_PARSE_TRANSFORMATIONS),
                    0,
                )
            free_symbols = sorted(equation.free_symbols, key=lambda sym: sym.name)
            target_symbol = free_symbols[0] if free_symbols else _X
            solutions = solve(equation, target_symbol)
            if not solutions:
                return "I could not find a solution."
            if len(solutions) == 1:
                return f"The solution is {target_symbol} = {_format_math_value(solutions[0])}."
            joined = ", ".join(_format_math_value(sol) for sol in solutions)
            return f"The solutions are {target_symbol} = {joined}."

        expr = parse_expr(expr_text, local_dict=_SYMPY_LOCALS, transformations=_PARSE_TRANSFORMATIONS)

        if action == "simplify":
            result = simplify(expr)
            return f"The simplified result is {_format_math_value(result)}."

        if action == "factor":
            result = factor(expr)
            return f"The factored form is {_format_math_value(result)}."

        result = simplify(expr)
        return f"The answer is {_format_math_value(result)}."
    except (SympifyError, SyntaxError, TypeError, ValueError, ZeroDivisionError):
        return None


def _has_time_intent(q: str) -> bool:
    return (
        _contains_any(q, ("time", "clock"))
        and _contains_any(q, ("what", "current", "now", "tell", "give", "show"))
    )


def _has_date_intent(q: str) -> bool:
    return (
        "date" in q
        and _contains_any(q, ("what", "today", "current", "now", "tell", "give", "show"))
    )


def _has_day_intent(q: str) -> bool:
    if "day" in q and _contains_any(q, ("today", "todays", "current", "now")):
        return True
    if "today" in q and _contains_any(q, ("what day", "which day", "day is", "day today")):
        return True
    if re.search(r"\bwt\b.*\bday\b.*\btoday\b", q):
        return True
    return False


def _answer_time_question(query: str) -> str | None:
    q = _norm(query)
    now = _now_local()

    if _contains_any(q, ("date and time", "time and date", "current date and time")):
        return f"Today is {now.strftime('%A, %B')} {now.day}, {now.year}, and the current time is {now.strftime('%I:%M %p')} ({AGENT_TIMEZONE})."

    if _has_day_intent(q):
        return f"Today is {now.strftime('%A, %B')} {now.day}, {now.year}."

    if _has_date_intent(q):
        return f"Today's date is {now.strftime('%B')} {now.day}, {now.year}."

    if _has_time_intent(q):
        return f"The current time is {now.strftime('%I:%M %p')} ({AGENT_TIMEZONE})."

    return None


def chat_with_agent(query: str, payload: dict[str, Any] | None = None) -> str:
    clean_query = (query or "").strip()
    direct_time_answer = _answer_time_question(clean_query)
    if direct_time_answer:
        return direct_time_answer
    direct_math_answer = _answer_math_question(clean_query)
    if direct_math_answer:
        return direct_math_answer

    context_bits: list[str] = []
    payload = payload or {}

    if payload.get("session_id"):
        context_bits.append(f"Transcript session is available: {payload.get('session_id')}")
    if payload.get("document_id"):
        context_bits.append(f"Document is available: {payload.get('document_id')}")
    if payload.get("content"):
        content_text = str(payload.get("content") or "").strip()
        if content_text:
            if len(content_text) > 4000:
                content_text = content_text[:4000].rstrip() + "..."
            context_bits.append(f"Additional text content is attached:\n{content_text}")

    context_block = "\n".join(context_bits) if context_bits else "No uploaded file or working context is attached."
    history_items = payload.get("chat_history") if isinstance(payload.get("chat_history"), list) else []
    trimmed_history = history_items[-10:]
    history_block = "\n".join(
        f"{('User' if item.get('role') == 'user' else 'Assistant')}: {str(item.get('content') or '').strip()}"
        for item in trimmed_history
        if str(item.get("content") or "").strip()
    ).strip() or "No earlier conversation."
    now = _now_local()
    now_context = f"{now.strftime('%A, %B')} {now.day}, {now.year} {now.strftime('%I:%M %p')} ({AGENT_TIMEZONE})"

    prompt = (
        "You are a helpful, natural conversational assistant inside an AI knowledge workspace.\n"
        "Respond like a normal chatbot: clear, direct, relevant, and concise.\n"
        "When the user is just chatting or asking a general question, answer normally.\n"
        "Use the Conversation History as memory for follow-up questions.\n"
        "If the user refers to something said earlier with words like 'my', 'that', 'it', 'he', 'she', 'they', or 'earlier', resolve it from the Conversation History.\n"
        "If the answer is available in the earlier conversation, use it directly instead of saying you do not know.\n"
        "When the user asks for something that would work better with an uploaded file, transcript, or document, explain what context is missing.\n"
        "Do not mention internal tools, routing, payloads, or system logic.\n\n"
        f"Current local date and time: {now_context}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Conversation History:\n{history_block}\n\n"
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
