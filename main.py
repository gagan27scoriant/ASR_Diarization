import os
import time
import warnings

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for, g, make_response
from werkzeug.utils import secure_filename

from app.auth import (
    ROLE_USER,
    ROLE_ADMIN,
    ROLE_SUPER_ADMIN,
    authenticate_user,
    create_user,
    delete_user,
    get_policy,
    has_permission,
    issue_token,
    decode_token,
    get_user_by_id,
    list_users,
    update_user_admin,
    update_user_profile,
    log_activity,
    list_activity,
    list_departments,
    create_department,
)
from app.asr import load_asr
from app.config import (
    AUDIO_FOLDER,
    DOCUMENT_FOLDER,
    DOCUMENT_FORMAT_ERROR,
    VIDEO_FOLDER,
    ensure_workspace_folders,
    is_supported_document,
)
from app.db import init_db
from app.diarization import load_diarization
from app.history_store import (
    delete_history_item as remove_history_item,
    history_json_path,
    list_history_entries,
    read_history_item,
    rename_history_item as rename_history_record,
    update_history_transcript,
)
from app.processing_service import (
    process_document_upload,
    process_media_pipeline,
    resolve_uploaded_or_path_media,
    summarize_and_persist,
    transcribe_live_audio_chunk,
)
from app.translation import get_translator


warnings.filterwarnings(
    "ignore",
    message=r"(?s).*torchcodec is not installed correctly.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*implementation will be changed to use torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Module 'speechbrain\.pretrained' was deprecated.*",
    category=UserWarning,
)


app = Flask(__name__, template_folder="templates", static_folder="static")
ensure_workspace_folders()
init_db()
APP_START_EPOCH = time.time()


print("🚀 Loading ASR model...")
preferred_asr = os.getenv("ASR_MODEL_SIZE", "medium")
try:
    asr_model = load_asr(preferred_asr)
except Exception as asr_err:
    print(f"⚠️ Failed to load ASR model '{preferred_asr}': {asr_err}")
    print("↪ Falling back to ASR model 'medium'...")
    asr_model = load_asr("medium")

print("🚀 Loading diarization model...")
diarization_pipeline = load_diarization()
print("✅ Models ready\n")


def _current_user():
    return getattr(g, "user", None)


def _is_authenticated() -> bool:
    return _current_user() is not None


def _get_token():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1].strip()
    return request.cookies.get("access_token")


@app.before_request
def enforce_login():
    if request.endpoint in {"login", "api_login", "logout", "static"}:
        return None
    token = _get_token()
    if token:
        try:
            payload = decode_token(token)
            issued_at = float(payload.get("iat") or 0)
            if issued_at < APP_START_EPOCH:
                raise ValueError("Stale token")
            user_id = int(payload.get("sub"))
            user = get_user_by_id(user_id)
            if user:
                g.user = user
                return None
        except Exception:
            pass
    accepts_html = "text/html" in request.accept_mimetypes
    if accepts_html and request.method == "GET":
        return redirect(url_for("login"))
    return jsonify({"error": "Unauthorized"}), 401


def _require_permission(permission: str):
    role = (_current_user() or {}).get("role_name", ROLE_USER)
    if not has_permission(role, permission):
        return jsonify({"error": "Forbidden"}), 403
    return None


def _is_admin(user: dict) -> bool:
    return user.get("role_name") == ROLE_ADMIN


def _is_super_admin(user: dict) -> bool:
    return user.get("role_name") == ROLE_SUPER_ADMIN


def _can_manage_role(actor: dict, target_role: str) -> bool:
    if _is_super_admin(actor):
        return target_role in {ROLE_ADMIN, ROLE_USER}
    if _is_admin(actor):
        return target_role == ROLE_USER
    return False


def _can_manage_department(actor: dict, target_department: str) -> bool:
    if _is_super_admin(actor):
        return True
    if _is_admin(actor):
        return (actor.get("department") or "") == (target_department or "")
    return False


def _history_visible(entry: dict, actor: dict) -> bool:
    owner = (entry or {}).get("owner") or {}
    if _is_super_admin(actor):
        return True
    if _is_admin(actor):
        return owner.get("department") == actor.get("department")
    return owner.get("email") == actor.get("email")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    payload = request.form or request.get_json(silent=True) or {}
    email = (payload.get("email") or payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    record = authenticate_user(email, password)
    if not record:
        return render_template("login.html", error="Invalid credentials"), 401
    token = issue_token(record)
    resp = make_response(redirect(url_for("home")))
    resp.set_cookie("access_token", token, httponly=True, samesite="Lax")
    log_activity(record["id"], "auth:login", {"email": record["email"]})
    return resp


@app.route("/api/login", methods=["POST"])
def api_login():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip()
    password = (payload.get("password") or "").strip()
    record = authenticate_user(email, password)
    if not record:
        return jsonify({"error": "Invalid credentials"}), 401
    token = issue_token(record)
    log_activity(record["id"], "auth:login", {"email": record["email"]})
    return jsonify({"token": token})


@app.route("/logout", methods=["POST", "GET"])
def logout():
    user = _current_user() or {}
    if user:
        log_activity(user["id"], "auth:logout", {"email": user["email"]})
    resp = make_response(redirect(url_for("login")))
    resp.set_cookie("access_token", "", expires=0)
    return resp


@app.route("/me", methods=["GET"])
def me():
    if not _is_authenticated():
        return jsonify({"error": "Unauthorized"}), 401
    user = _current_user()
    policy = get_policy(user.get("role_name", ROLE_USER))
    return jsonify({"user": user, "policy": policy})


@app.route("/api/users", methods=["GET", "POST"])
def api_users():
    forbidden = _require_permission("user:manage")
    if forbidden:
        return forbidden
    actor = _current_user()
    if request.method == "GET":
        users = []
        for record in list_users():
            if not _can_manage_department(actor, record["department"]):
                continue
            if _is_admin(actor) and record["role_name"] == ROLE_SUPER_ADMIN:
                continue
            users.append(record)
        return jsonify({"users": users})

    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip()
    password = (payload.get("password") or "").strip()
    role = (payload.get("role") or "").strip() or ROLE_USER
    department = (payload.get("department") or "").strip() or (actor.get("department") or "general")
    if not _can_manage_role(actor, role):
        return jsonify({"error": "Forbidden"}), 403
    if not _can_manage_department(actor, department):
        return jsonify({"error": "Forbidden"}), 403
    record = create_user(name, email, password, role, department)
    log_activity(actor["id"], "user:create", {"target": record["email"], "role": record["role_name"]})
    return jsonify(record), 201


@app.route("/api/admins", methods=["POST"])
def api_admins():
    forbidden = _require_permission("admin:manage")
    if forbidden:
        return forbidden
    actor = _current_user()
    if not _is_super_admin(actor):
        return jsonify({"error": "Forbidden"}), 403
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip()
    password = (payload.get("password") or "").strip()
    department = (payload.get("department") or "").strip() or "general"
    record = create_user(name, email, password, ROLE_ADMIN, department)
    log_activity(actor["id"], "admin:create", {"target": record["email"], "department": record["department"]})
    return jsonify(record), 201


@app.route("/api/users/<int:user_id>", methods=["DELETE", "PATCH"])
def api_user_detail(user_id: int):
    forbidden = _require_permission("user:manage")
    if forbidden:
        return forbidden
    actor = _current_user()
    target = get_user_by_id(user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404
    if not _can_manage_department(actor, target["department"]):
        return jsonify({"error": "Forbidden"}), 403
    if request.method == "DELETE":
        if target["role_name"] == ROLE_SUPER_ADMIN:
            return jsonify({"error": "Forbidden"}), 403
        if target["role_name"] == ROLE_ADMIN and not _is_super_admin(actor):
            return jsonify({"error": "Forbidden"}), 403
        delete_user(user_id)
        log_activity(actor["id"], "user:delete", {"target": target["email"]})
        return jsonify({"ok": True})

    payload = request.get_json(silent=True) or {}
    role = (payload.get("role") or "").strip() or None
    department = payload.get("department")
    password = (payload.get("password") or "").strip() or None
    if role:
        if not _can_manage_role(actor, role):
            return jsonify({"error": "Forbidden"}), 403
    if department is not None and not _can_manage_department(actor, department):
        return jsonify({"error": "Forbidden"}), 403
    update_user_admin(user_id, role, department, password)
    log_activity(actor["id"], "user:update", {"target": target["email"]})
    return jsonify({"ok": True})


@app.route("/api/me", methods=["PATCH"])
def update_profile():
    if not _is_authenticated():
        return jsonify({"error": "Unauthorized"}), 401
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip() or None
    password = (payload.get("password") or "").strip() or None
    user = _current_user()
    update_user_profile(user["id"], name, password)
    log_activity(user["id"], "profile:update", {})
    return jsonify({"ok": True})


@app.route("/api/audit", methods=["GET"])
def api_audit():
    forbidden = _require_permission("audit:read")
    if forbidden:
        return forbidden
    actor = _current_user()
    limit = int((request.args.get("limit") or "200").strip())
    events = list_activity(limit=limit)
    if _is_super_admin(actor):
        return jsonify({"events": events})
    dept = actor.get("department") or ""
    filtered = [e for e in events if e.get("department") == dept and e.get("role") != ROLE_SUPER_ADMIN]
    return jsonify({"events": filtered})


@app.route("/api/departments", methods=["GET", "POST"])
def api_departments():
    if request.method == "GET":
        return jsonify({"departments": list_departments()})
    forbidden = _require_permission("settings:manage")
    if forbidden:
        return forbidden
    actor = _current_user()
    if not _is_super_admin(actor):
        return jsonify({"error": "Forbidden"}), 403
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    create_department(name)
    log_activity(actor["id"], "department:create", {"name": name})
    return jsonify({"ok": True}), 201


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"error": "Invalid audio filename"}), 400
    return send_from_directory(AUDIO_FOLDER, safe_name)


@app.route("/videos/<path:filename>")
def serve_video(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"error": "Invalid video filename"}), 400
    return send_from_directory(VIDEO_FOLDER, safe_name)


@app.route("/documents/<path:filename>")
def serve_document(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        return jsonify({"error": "Invalid document filename"}), 400
    return send_from_directory(DOCUMENT_FOLDER, safe_name)


@app.route("/process", methods=["POST"])
def process_audio():
    try:
        forbidden = _require_permission("process:media")
        if forbidden:
            return forbidden
        uploaded_file = request.files.get("audio_file")
        payload = request.get_json(silent=True) if not uploaded_file else None
        source_path, filename = resolve_uploaded_or_path_media(uploaded_file, payload)
        user = _current_user()
        result = process_media_pipeline(source_path, filename, asr_model, diarization_pipeline, owner=user)
        log_activity(user["id"], "process:media", {"filename": filename, "session_id": result.get("session_id")})
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/transcribe_chunk", methods=["POST"])
def transcribe_chunk():
    try:
        forbidden = _require_permission("process:media")
        if forbidden:
            return forbidden
        uploaded_chunk = request.files.get("audio_chunk")
        if not uploaded_chunk or not uploaded_chunk.filename:
            return jsonify({"error": "Audio chunk missing"}), 400
        text = transcribe_live_audio_chunk(uploaded_chunk, asr_model)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarize_text", methods=["POST"])
def summarize_from_text():
    try:
        forbidden = _require_permission("summary:generate")
        if forbidden:
            return forbidden
        data = request.get_json() or {}
        text = data.get("content", "")
        meeting_title = (data.get("meeting_title") or "").strip()
        meeting_date = (data.get("meeting_date") or "").strip()
        meeting_place = (data.get("meeting_place") or "").strip()

        if not text or not text.strip():
            return jsonify({"error": "Content missing"}), 400
        if not meeting_title or not meeting_date or not meeting_place:
            return jsonify({"error": "Meeting title, date, and place are required"}), 400

        print("→ Generating summary from exported text...")
        summary = summarize_and_persist(
            text,
            meeting_title,
            meeting_date,
            meeting_place,
            (data.get("session_id") or "").strip(),
        )
        user = _current_user()
        log_activity(user["id"], "summary:generate", {"session_id": (data.get("session_id") or "").strip()})
        return jsonify({"summary": summary})
    except Exception as e:
        print("❌ Summary Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/process_document", methods=["POST"])
def process_document():
    try:
        forbidden = _require_permission("process:document")
        if forbidden:
            return forbidden
        uploaded_file = request.files.get("document_file")
        if not uploaded_file or not uploaded_file.filename:
            return jsonify({"error": "Document file missing"}), 400
        if not is_supported_document(uploaded_file.filename):
            return jsonify({"error": DOCUMENT_FORMAT_ERROR}), 400

        meeting_title = (request.form.get("meeting_title") or "").strip()
        meeting_date = (request.form.get("meeting_date") or "").strip()
        meeting_place = (request.form.get("meeting_place") or "").strip()
        if not meeting_title or not meeting_date or not meeting_place:
            return jsonify({"error": "Meeting title, date, and place are required"}), 400

        print("→ Generating summary from uploaded document...")
        user = _current_user()
        result = process_document_upload(
            uploaded_file,
            meeting_title=meeting_title,
            meeting_date=meeting_date,
            meeting_place=meeting_place,
            owner=user,
        )
        log_activity(user["id"], "process:document", {"filename": uploaded_file.filename})
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print("❌ Document Processing Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate_content():
    try:
        forbidden = _require_permission("translate:run")
        if forbidden:
            return forbidden
        data = request.get_json(silent=True) or {}
        target_lang = (data.get("target_lang") or "").strip()
        source_lang = (data.get("source_lang") or "").strip()
        text = data.get("text")
        texts = data.get("texts")

        if not target_lang:
            return jsonify({"error": "target_lang is required"}), 400

        translator = get_translator()
        resolved_target = translator.resolve_lang_code(target_lang, is_target=True)
        resolved_source = translator.resolve_lang_code(source_lang or "", is_target=False)

        if texts is not None:
            if not isinstance(texts, list):
                return jsonify({"error": "texts must be an array"}), 400
            translated_texts = translator.translate_lines(
                [str(x or "") for x in texts],
                target_lang=resolved_target,
                source_lang=resolved_source,
            )
            return jsonify(
                {
                    "texts": translated_texts,
                    "target_lang": resolved_target,
                    "source_lang": resolved_source,
                }
            )

        if not isinstance(text, str):
            return jsonify({"error": "text must be a string"}), 400

        translated_text = translator.translate_text(
            text,
            target_lang=resolved_target,
            source_lang=resolved_source,
        )
        user = _current_user()
        log_activity(user["id"], "translate:run", {"target_lang": resolved_target})
        return jsonify(
            {
                "text": translated_text,
                "target_lang": resolved_target,
                "source_lang": resolved_source,
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def list_history():
    try:
        forbidden = _require_permission("history:read")
        if forbidden:
            return forbidden
        user = _current_user()
        entries = list_history_entries()
        filtered = [e for e in entries if _history_visible(e, user)]
        log_activity(user["id"], "history:read", {})
        return jsonify({"history": filtered})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<session_id>", methods=["GET"])
def get_history_item(session_id):
    try:
        forbidden = _require_permission("history:read")
        if forbidden:
            return forbidden
        data = read_history_item(session_id)
        if data is None:
            return jsonify({"error": "History not found"}), 404
        user = _current_user()
        if not _history_visible(data, user):
            return jsonify({"error": "Forbidden"}), 403
        log_activity(user["id"], "history:item", {"session_id": session_id})

        return jsonify(
            {
                "session_id": session_id,
                "title": data.get("title") or session_id,
                "processed_file": data.get("processed_file") or "",
                "before_audio_file": data.get("before_audio_file") or data.get("processed_file") or "",
                "after_audio_file": data.get("after_audio_file") or data.get("processed_file") or "",
                "source_video": data.get("source_video") or "",
                "transcript": data.get("transcript") or [],
                "summary": data.get("summary") or "",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<session_id>/transcript", methods=["POST"])
def save_history_transcript(session_id):
    try:
        forbidden = _require_permission("history:read")
        if forbidden:
            return forbidden
        if not history_json_path(session_id) or read_history_item(session_id) is None:
            return jsonify({"error": "History not found"}), 404

        payload = request.get_json(silent=True) or {}
        transcript = payload.get("transcript")
        summary = payload.get("summary")

        if transcript is not None and not isinstance(transcript, list):
            return jsonify({"error": "Invalid transcript payload"}), 400
        if summary is not None and not isinstance(summary, str):
            return jsonify({"error": "Invalid summary payload"}), 400

        update_history_transcript(session_id, transcript=transcript, summary=summary)
        user = _current_user()
        item = read_history_item(session_id) or {}
        if not _history_visible(item, user):
            return jsonify({"error": "Forbidden"}), 403
        log_activity(user["id"], "history:update", {"session_id": session_id})
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<session_id>", methods=["PATCH"])
def rename_history_item(session_id):
    try:
        forbidden = _require_permission("history:rename")
        if forbidden:
            return forbidden
        if not history_json_path(session_id) or read_history_item(session_id) is None:
            return jsonify({"error": "History not found"}), 404

        payload = request.get_json(silent=True) or {}
        new_title = (payload.get("title") or "").strip()
        if not new_title:
            return jsonify({"error": "Title is required"}), 400

        rename_history_record(session_id, new_title)
        user = _current_user()
        item = read_history_item(session_id) or {}
        if not _history_visible(item, user):
            return jsonify({"error": "Forbidden"}), 403
        log_activity(user["id"], "history:rename", {"session_id": session_id, "title": new_title})
        return jsonify({"ok": True, "title": new_title})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<session_id>", methods=["DELETE"])
def delete_history_item(session_id):
    try:
        forbidden = _require_permission("history:delete")
        if forbidden:
            return forbidden
        item = read_history_item(session_id) or {}
        if not _history_visible(item, _current_user()):
            return jsonify({"error": "Forbidden"}), 403
        if not remove_history_item(session_id):
            return jsonify({"error": "History not found"}), 404
        user = _current_user()
        log_activity(user["id"], "history:delete", {"session_id": session_id})
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    use_reloader = os.getenv("FLASK_USE_RELOADER", "0").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=use_reloader)
