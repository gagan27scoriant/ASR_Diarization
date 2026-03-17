from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db import get_db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _doc_col():
    return get_db()["document_store"]


def create_document(doc_id: str, payload: dict[str, Any]) -> bool:
    if not doc_id:
        return False
    record = dict(payload or {})
    record["document_id"] = doc_id
    record.setdefault("created_at", _now_iso())
    record["updated_at"] = _now_iso()
    _doc_col().update_one({"document_id": doc_id}, {"$set": record}, upsert=True)
    return True


def read_document(doc_id: str) -> dict | None:
    if not doc_id:
        return None
    return _doc_col().find_one({"document_id": doc_id})


def update_document(doc_id: str, update: dict[str, Any]) -> bool:
    if not doc_id:
        return False
    payload = dict(update or {})
    payload["updated_at"] = _now_iso()
    res = _doc_col().update_one({"document_id": doc_id}, {"$set": payload})
    return res.matched_count > 0


def _document_doc_to_entry(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": doc.get("document_id"),
        "filename": doc.get("filename") or "",
        "document_type": doc.get("document_type") or "",
        "chunk_count": int(doc.get("chunk_count") or 0),
        "updated_at": doc.get("updated_at") or doc.get("created_at") or "",
        "owner": doc.get("owner") or {},
        "summary": doc.get("summary") or "",
    }


def list_documents() -> list[dict[str, Any]]:
    docs = list(_doc_col().find().sort("updated_at", -1))
    return [_document_doc_to_entry(doc) for doc in docs]
