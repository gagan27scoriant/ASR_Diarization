import json
import os
from datetime import datetime
from glob import glob

from werkzeug.utils import secure_filename

from app.config import OUTPUT_FOLDER


def history_json_path(session_id: str) -> str:
    safe_id = secure_filename(session_id or "").strip()
    if not safe_id:
        return ""
    return os.path.join(OUTPUT_FOLDER, f"{safe_id}.json")


def history_entry_from_file(json_path: str) -> dict | None:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    transcript = data.get("transcript") or []
    summary = data.get("summary") or ""
    session_id = os.path.splitext(os.path.basename(json_path))[0]
    ts = os.path.getmtime(json_path)
    updated_at = datetime.fromtimestamp(ts).isoformat(timespec="seconds")

    return {
        "session_id": session_id,
        "title": data.get("title") or session_id,
        "processed_file": data.get("processed_file") or "",
        "before_audio_file": data.get("before_audio_file") or data.get("processed_file") or "",
        "after_audio_file": data.get("after_audio_file") or data.get("processed_file") or "",
        "source_video": data.get("source_video") or "",
        "segments": len(transcript),
        "has_summary": bool(str(summary).strip()),
        "updated_at": updated_at,
    }


def list_history_entries() -> list[dict]:
    entries = []
    for path in sorted(glob(os.path.join(OUTPUT_FOLDER, "*.json")), key=os.path.getmtime, reverse=True):
        entry = history_entry_from_file(path)
        if entry:
            entries.append(entry)
    return entries


def read_history_item(session_id: str) -> dict | None:
    path = history_json_path(session_id)
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_history_item(session_id: str, data: dict) -> bool:
    path = history_json_path(session_id)
    if not path:
        return False
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return True


def update_history_transcript(session_id: str, transcript=None, summary=None) -> bool:
    path = history_json_path(session_id)
    if not path or not os.path.isfile(path):
        return False

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if transcript is not None:
        data["transcript"] = transcript
    if summary is not None:
        data["summary"] = summary

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return True


def rename_history_item(session_id: str, new_title: str) -> bool:
    path = history_json_path(session_id)
    if not path or not os.path.isfile(path):
        return False

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["title"] = new_title
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return True


def delete_history_item(session_id: str) -> bool:
    path = history_json_path(session_id)
    if not path or not os.path.isfile(path):
        return False
    os.remove(path)
    return True
