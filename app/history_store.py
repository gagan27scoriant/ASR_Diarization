from datetime import datetime, timezone
from typing import Any

from app.db import get_db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _history_col():
    return get_db()["history"]


def history_json_path(session_id: str) -> str:
    return session_id or ""


def _history_doc_to_entry(doc: dict[str, Any]) -> dict[str, Any]:
    transcript = doc.get("transcript") or []
    summary = doc.get("summary") or ""
    return {
        "session_id": doc.get("session_id"),
        "title": doc.get("title") or doc.get("session_id"),
        "processed_file": doc.get("processed_file") or "",
        "before_audio_file": doc.get("before_audio_file") or doc.get("processed_file") or "",
        "after_audio_file": doc.get("after_audio_file") or doc.get("processed_file") or "",
        "source_video": doc.get("source_video") or "",
        "segments": len(transcript),
        "has_summary": bool(str(summary).strip()),
        "updated_at": doc.get("updated_at") or doc.get("created_at") or "",
        "owner": doc.get("owner") or {},
    }


def list_history_entries(owner: dict | None = None) -> list[dict]:
    query: dict[str, Any] = {}
    if owner:
        query["owner.email"] = owner.get("email")
    docs = list(_history_col().find(query).sort("updated_at", -1))
    return [_history_doc_to_entry(doc) for doc in docs]


def read_history_item(session_id: str) -> dict | None:
    if not session_id:
        return None
    return _history_col().find_one({"session_id": session_id})


def write_history_item(session_id: str, data: dict) -> bool:
    if not session_id:
        return False
    payload = dict(data or {})
    payload["session_id"] = session_id
    payload.setdefault("created_at", _now_iso())
    payload["updated_at"] = _now_iso()
    _history_col().update_one({"session_id": session_id}, {"$set": payload}, upsert=True)
    return True


def update_history_transcript(session_id: str, transcript=None, summary=None) -> bool:
    if not session_id:
        return False
    update: dict[str, Any] = {"updated_at": _now_iso()}
    if transcript is not None:
        update["transcript"] = transcript
    if summary is not None:
        update["summary"] = summary
    res = _history_col().update_one({"session_id": session_id}, {"$set": update})
    return res.matched_count > 0


def update_history_embeddings(session_id: str, embeddings: list[list[float]], model_name: str) -> bool:
    if not session_id:
        return False
    update = {
        "embeddings": embeddings,
        "embedding_model": model_name,
        "updated_at": _now_iso(),
    }
    res = _history_col().update_one({"session_id": session_id}, {"$set": update})
    return res.matched_count > 0


def rename_history_item(session_id: str, new_title: str) -> bool:
    if not session_id:
        return False
    res = _history_col().update_one(
        {"session_id": session_id},
        {"$set": {"title": new_title, "updated_at": _now_iso()}},
    )
    return res.matched_count > 0


def delete_history_item(session_id: str) -> bool:
    if not session_id:
        return False
    res = _history_col().delete_one({"session_id": session_id})
    return res.deleted_count > 0
