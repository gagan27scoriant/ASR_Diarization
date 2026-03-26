import os
import time
import uuid
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
    get_user_by_email,
    list_users,
    update_user_admin,
    update_user_profile,
    log_activity,
    list_activity,
    list_departments,
    create_department,
    delete_activity,
    clear_activity,
    delete_department,
    update_department,
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
    list_history_entries,
    read_history_item,
    rename_history_item as rename_history_record,
    update_history_transcript,
)
from app.document_store import list_documents, read_document, update_document, rename_document, delete_document
from app.document_chunks import delete_document_chunks, load_document_chunks
from app.processing_service import (
    process_document_upload,
    process_media_pipeline,
    resolve_uploaded_or_path_media,
    summarize_and_persist,
    transcribe_live_audio_chunk,
)
from app.document_rag import answer_document_question, ingest_document_text
from app.history_rag import answer_history_question
from app.semantic_search import search_history_segments
from app.translation import get_translator
from app.agent_router import plan_agent_query, run_agent_query
from app.agent_tools import list_tools
from app.agent_service import handle_uploaded_file


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
APP_SESSION_ID = uuid.uuid4().hex


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
            if payload.get("sid") != APP_SESSION_ID:
                raise ValueError("Stale session")
            user_id = payload.get("sub")
            user = get_user_by_id(user_id)
            if user:
                g.user = user
                g.auth_payload = payload
                return None
        except Exception:
            pass
    wants_json = request.path.startswith("/api/") or request.path.startswith("/history")
    accepts_html = "text/html" in request.accept_mimetypes
    if not wants_json and accepts_html and request.method == "GET":
        return redirect(url_for("login"))
    return jsonify({"error": "Unauthorized"}), 401


def _require_permission(permission: str):
    if not permission:
        return None
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


def _document_visible(entry: dict, actor: dict) -> bool:
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
    token = issue_token(record, session_id=APP_SESSION_ID)
    resp = make_response(redirect(url_for("home")))
    resp.set_cookie("access_token", token, httponly=True, samesite="Lax")
    resp.set_cookie("impersonator_token", "", expires=0)
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
    token = issue_token(record, session_id=APP_SESSION_ID)
    log_activity(record["id"], "auth:login", {"email": record["email"]})
    return jsonify({"token": token})


@app.route("/api/impersonate", methods=["POST"])
def api_impersonate():
    actor = _current_user()
    if not actor or actor.get("role_name") not in {ROLE_ADMIN, ROLE_SUPER_ADMIN}:
        return jsonify({"error": "Forbidden"}), 403
    payload = getattr(g, "auth_payload", {}) or {}
    if payload.get("impersonated_by"):
        return jsonify({"error": "Already impersonating"}), 400
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip()
    if not email:
        return jsonify({"error": "Email required"}), 400
    target = get_user_by_email(email)
    if not target:
        return jsonify({"error": "User not found"}), 404
    if target.get("role_name") == ROLE_SUPER_ADMIN:
        return jsonify({"error": "Forbidden"}), 403
    if _is_admin(actor):
        if target.get("role_name") != ROLE_USER:
            return jsonify({"error": "Forbidden"}), 403
        if target.get("department") != actor.get("department"):
            return jsonify({"error": "Forbidden"}), 403
    token = issue_token(target, impersonator=actor, session_id=APP_SESSION_ID)
    actor_token = _get_token()
    if not actor_token:
        return jsonify({"error": "Missing session token"}), 400
    resp = make_response(jsonify({"ok": True, "user": target}))
    resp.set_cookie("impersonator_token", actor_token, httponly=True, samesite="Lax")
    resp.set_cookie("access_token", token, httponly=True, samesite="Lax")
    log_activity(actor["id"], "auth:impersonate", {"target": target["email"]})
    return resp


@app.route("/api/impersonate/stop", methods=["POST"])
def api_impersonate_stop():
    token = request.cookies.get("impersonator_token")
    if not token:
        return jsonify({"error": "No impersonation active"}), 400
    try:
        payload = decode_token(token)
        issued_at = float(payload.get("iat") or 0)
        if issued_at < APP_START_EPOCH:
            raise ValueError("Stale token")
        if payload.get("sid") != APP_SESSION_ID:
            raise ValueError("Stale session")
        user_id = payload.get("sub")
        user = get_user_by_id(user_id)
    except Exception:
        return jsonify({"error": "Invalid impersonation token"}), 400
    if not user or user.get("role_name") not in {ROLE_ADMIN, ROLE_SUPER_ADMIN}:
        return jsonify({"error": "Forbidden"}), 403
    new_token = issue_token(user, session_id=APP_SESSION_ID)
    resp = make_response(jsonify({"ok": True}))
    resp.set_cookie("access_token", new_token, httponly=True, samesite="Lax")
    resp.set_cookie("impersonator_token", "", expires=0)
    log_activity(user["id"], "auth:impersonate_end", {})
    return resp


@app.route("/logout", methods=["POST", "GET"])
def logout():
    user = _current_user() or {}
    if user:
        log_activity(user["id"], "auth:logout", {"email": user["email"]})
    resp = make_response(redirect(url_for("login")))
    resp.set_cookie("access_token", "", expires=0)
    resp.set_cookie("impersonator_token", "", expires=0)
    return resp


@app.route("/me", methods=["GET"])
def me():
    if not _is_authenticated():
        return jsonify({"error": "Unauthorized"}), 401
    user = _current_user()
    policy = get_policy(user.get("role_name", ROLE_USER))
    payload = getattr(g, "auth_payload", {}) or {}
    return jsonify({"user": user, "policy": policy, "impersonator": payload.get("impersonated_by")})


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


@app.route("/api/users/<user_id>", methods=["DELETE", "PATCH"])
def api_user_detail(user_id: str):
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


@app.route("/api/audit/<activity_id>", methods=["DELETE"])
def api_audit_delete(activity_id: str):
    forbidden = _require_permission("audit:read")
    if forbidden:
        return forbidden
    actor = _current_user()
    if _is_super_admin(actor):
        delete_activity(activity_id)
        return jsonify({"ok": True})
    # Admin can delete only within their department
    events = list_activity(limit=500)
    for ev in events:
        if ev.get("id") == activity_id:
            if ev.get("department") != actor.get("department"):
                return jsonify({"error": "Forbidden"}), 403
            if ev.get("role") == ROLE_SUPER_ADMIN:
                return jsonify({"error": "Forbidden"}), 403
            delete_activity(activity_id)
            return jsonify({"ok": True})
    return jsonify({"error": "Not found"}), 404


@app.route("/api/audit", methods=["DELETE"])
def api_audit_clear():
    forbidden = _require_permission("audit:read")
    if forbidden:
        return forbidden
    actor = _current_user()
    if not _is_super_admin(actor):
        return jsonify({"error": "Forbidden"}), 403
    clear_activity()
    return jsonify({"ok": True})


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


@app.route("/api/departments/<department_id>", methods=["DELETE"])
def api_department_delete(department_id: str):
    forbidden = _require_permission("settings:manage")
    if forbidden:
        return forbidden
    actor = _current_user()
    if not _is_super_admin(actor):
        return jsonify({"error": "Forbidden"}), 403
    delete_department(department_id)
    log_activity(actor["id"], "department:delete", {"id": department_id})
    return jsonify({"ok": True})


@app.route("/api/departments/<department_id>", methods=["PATCH"])
def api_department_update(department_id: str):
    forbidden = _require_permission("settings:manage")
    if forbidden:
        return forbidden
    actor = _current_user()
    if not _is_super_admin(actor):
        return jsonify({"error": "Forbidden"}), 403
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    update_department(department_id, name)
    log_activity(actor["id"], "department:update", {"id": department_id, "name": name})
    return jsonify({"ok": True})


@app.route("/api/search", methods=["POST"])
def api_search():
    forbidden = _require_permission("history:read")
    if forbidden:
        return forbidden
    payload = request.get_json(silent=True) or {}
    session_id = (payload.get("session_id") or "").strip()
    query = (payload.get("query") or "").strip()
    top_k = int(payload.get("top_k") or 5)
    if not session_id or not query:
        return jsonify({"error": "session_id and query are required"}), 400
    item = read_history_item(session_id) or {}
    if not item:
        return jsonify({"error": "History not found"}), 404
    user = _current_user()
    if not _history_visible(item, user):
        return jsonify({"error": "Forbidden"}), 403
    results = search_history_segments(session_id, query, top_k=top_k)
    log_activity(user["id"], "history:search", {"session_id": session_id})
    return jsonify({"results": results})


@app.route("/api/document/ask", methods=["POST"])
def api_document_ask():
    forbidden = _require_permission("rag:ask")
    if forbidden:
        return forbidden
    payload = request.get_json(silent=True) or {}
    document_id = (payload.get("document_id") or "").strip()
    question = (payload.get("question") or "").strip()
    top_k = int(payload.get("top_k") or 5)
    if not document_id or not question:
        return jsonify({"error": "document_id and question are required"}), 400
    try:
        result = answer_document_question(document_id, question, top_k=top_k)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print("❌ Document QA Error:", e)
        return jsonify({"error": str(e)}), 500
    user = _current_user()
    log_activity(user["id"], "document:ask", {"document_id": document_id})
    return jsonify(result)


@app.route("/api/history/ask", methods=["POST"])
def api_history_ask():
    forbidden = _require_permission("rag:ask")
    if forbidden:
        return forbidden
    payload = request.get_json(silent=True) or {}
    session_id = (payload.get("session_id") or "").strip()
    question = (payload.get("question") or "").strip()
    top_k = int(payload.get("top_k") or 5)
    if not session_id or not question:
        return jsonify({"error": "session_id and question are required"}), 400
    record = read_history_item(session_id) or {}
    if not record:
        return jsonify({"error": "History not found"}), 404
    user = _current_user()
    if not _history_visible(record, user):
        return jsonify({"error": "Forbidden"}), 403
    try:
        result = answer_history_question(session_id, question, top_k=top_k)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print("❌ History QA Error:", e)
        return jsonify({"error": str(e)}), 500
    log_activity(user["id"], "history:ask", {"session_id": session_id})
    return jsonify(result)


@app.route("/api/agent/tools", methods=["GET"])
def api_agent_tools():
    return jsonify({"tools": list_tools()})


@app.route("/api/agent/query", methods=["POST"])
def api_agent_query():
    payload = request.get_json(silent=True) or {}
    route = plan_agent_query(payload)
    if not route.get("ok"):
        return jsonify(route), 400

    permission = route.get("required_permission") or ""
    if " + " in permission:
        for single_permission in [p.strip() for p in permission.split("+") if p.strip()]:
            forbidden = _require_permission(single_permission)
            if forbidden:
                return forbidden
    else:
        forbidden = _require_permission(permission)
        if forbidden:
            return forbidden

    user = _current_user()
    session_id = (payload.get("session_id") or "").strip()
    document_id = (payload.get("document_id") or "").strip()

    if session_id:
        record = read_history_item(session_id) or {}
        if not record:
            return jsonify({"error": "History not found"}), 404
        if not _history_visible(record, user):
            return jsonify({"error": "Forbidden"}), 403

    if document_id:
        record = read_document(document_id) or {}
        if not record:
            return jsonify({"error": "Document not found"}), 404
        if not _document_visible(record, user):
            return jsonify({"error": "Forbidden"}), 403

    try:
        result = run_agent_query(
            payload,
            {
                "asr_model": asr_model,
                "diarization_pipeline": diarization_pipeline,
                "owner": user,
            },
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print("❌ Agent Query Error:", e)
        return jsonify({"error": str(e)}), 500

    log_activity(
        user["id"],
        "agent:query",
        {
            "tool": result.get("selected_tool"),
            "session_id": session_id,
            "document_id": document_id,
        },
    )
    return jsonify(result)


@app.route("/api/agent/chat", methods=["POST"])
def api_agent_chat():
    payload = request.form.to_dict() if request.files else (request.get_json(silent=True) or {})
    query = (payload.get("query") or "").strip()
    uploaded_file = request.files.get("file") or request.files.get("audio_file") or request.files.get("document_file")

    if uploaded_file and uploaded_file.filename:
        user = _current_user()
        filename = secure_filename(uploaded_file.filename.strip())
        if not filename:
            return jsonify({"error": "Invalid uploaded filename"}), 400
        forbidden = _require_permission("process:document" if is_supported_document(filename) else "process:media")
        if forbidden:
            return forbidden

        try:
            if is_supported_document(filename):
                result = handle_uploaded_file(
                    uploaded_file,
                    query,
                    payload,
                    {
                        "owner": user,
                        "is_supported_document": is_supported_document,
                        "update_document": update_document,
                        "asr_model": asr_model,
                        "diarization_pipeline": diarization_pipeline,
                        "source_path": "",
                    },
                )
            else:
                source_path, resolved_name = resolve_uploaded_or_path_media(uploaded_file, None)
                result = handle_uploaded_file(
                    uploaded_file,
                    query,
                    payload,
                    {
                        "owner": user,
                        "is_supported_document": is_supported_document,
                        "update_document": update_document,
                        "asr_model": asr_model,
                        "diarization_pipeline": diarization_pipeline,
                        "source_path": source_path,
                    },
                )
                if resolved_name and resolved_name != filename:
                    result.setdefault("result", {})["uploaded_filename"] = resolved_name
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            print("❌ Agent Upload Error:", e)
            return jsonify({"error": str(e)}), 500

        log_activity(
            user["id"],
            "agent:chat",
            {
                "tool": result.get("selected_tool"),
                "filename": filename,
                "session_id": (result.get("result") or {}).get("session_id"),
                "document_id": (result.get("result") or {}).get("document_id"),
            },
        )
        return jsonify(result)

    return api_agent_query()


@app.route("/api/documents", methods=["GET"])
def api_document_list():
    forbidden = _require_permission("rag:ask")
    if forbidden:
        return forbidden
    user = _current_user()
    entries = list_documents()
    filtered = [e for e in entries if _document_visible(e, user)]
    return jsonify({"documents": filtered})


@app.route("/api/documents/<doc_id>", methods=["GET"])
def api_document_get(doc_id: str):
    forbidden = _require_permission("rag:ask")
    if forbidden:
        return forbidden
    record = read_document(doc_id) or {}
    if not record:
        return jsonify({"error": "Document not found"}), 404
    user = _current_user()
    if not _document_visible(record, user):
        return jsonify({"error": "Forbidden"}), 403
    preview = record.get("text_preview") or ""
    return jsonify(
        {
            "document_id": record.get("document_id"),
            "filename": record.get("filename") or "",
            "document_type": record.get("document_type") or "",
            "text_preview": preview,
            "summary": record.get("summary") or "",
            "chat_history": record.get("chat_history") or [],
        }
    )


@app.route("/api/documents/<doc_id>/chunks/<int:chunk_idx>", methods=["GET"])
def api_document_chunk(doc_id: str, chunk_idx: int):
    forbidden = _require_permission("rag:ask")
    if forbidden:
        return forbidden
    record = read_document(doc_id) or {}
    if not record:
        return jsonify({"error": "Document not found"}), 404
    user = _current_user()
    if not _document_visible(record, user):
        return jsonify({"error": "Forbidden"}), 403

    rows = load_document_chunks(doc_id)
    if chunk_idx < 0 or chunk_idx >= len(rows):
        return jsonify({"error": "Chunk not found"}), 404
    row = rows[chunk_idx] if chunk_idx < len(rows) else {}
    return jsonify({
        "index": int(row.get("index") if row.get("index") is not None else chunk_idx),
        "text": row.get("text") or "",
    })


@app.route("/api/documents/<doc_id>", methods=["PATCH"])
def api_document_rename(doc_id: str):
    forbidden = _require_permission("history:rename")
    if forbidden:
        return forbidden
    record = read_document(doc_id) or {}
    if not record:
        return jsonify({"error": "Document not found"}), 404
    user = _current_user()
    if not _document_visible(record, user):
        return jsonify({"error": "Forbidden"}), 403
    payload = request.get_json(silent=True) or {}
    name = (payload.get("filename") or "").strip()
    if not name:
        return jsonify({"error": "filename is required"}), 400
    if not rename_document(doc_id, name):
        return jsonify({"error": "Rename failed"}), 400
    log_activity(user["id"], "document:rename", {"document_id": doc_id, "filename": name})
    return jsonify({"ok": True, "filename": name})


@app.route("/api/documents/<doc_id>", methods=["DELETE"])
def api_document_delete(doc_id: str):
    forbidden = _require_permission("history:delete")
    if forbidden:
        return forbidden
    record = read_document(doc_id) or {}
    if not record:
        return jsonify({"error": "Document not found"}), 404
    user = _current_user()
    if not _document_visible(record, user):
        return jsonify({"error": "Forbidden"}), 403
    delete_document_chunks(doc_id)
    if not delete_document(doc_id):
        return jsonify({"error": "Delete failed"}), 400
    log_activity(
        user["id"],
        "document:delete",
        {"document_id": doc_id, "filename": record.get("filename") or ""},
    )
    return jsonify({"ok": True})


@app.route("/api/documents", methods=["DELETE"])
def api_document_clear():
    forbidden = _require_permission("history:delete")
    if forbidden:
        return forbidden
    user = _current_user()
    entries = list_documents()
    targets = [e for e in entries if _document_visible(e, user)]
    deleted = 0
    for entry in targets:
        doc_id = entry.get("document_id")
        if not doc_id:
            continue
        delete_document_chunks(doc_id)
        if delete_document(doc_id):
            deleted += 1
    log_activity(user["id"], "document:clear", {"count": deleted})
    return jsonify({"ok": True, "deleted": deleted})


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

        print("→ Generating summary from uploaded document...")
        user = _current_user()
        result = process_document_upload(
            uploaded_file,
            meeting_title="Document Summary",
            meeting_date="",
            meeting_place="",
            owner=user,
        )
        rag_meta = ingest_document_text(
            result.get("document_text") or "",
            result.get("document_filename") or uploaded_file.filename,
            owner=user,
        )
        update_document(
            rag_meta.get("document_id"),
            {"summary": result.get("summary") or ""},
        )
        result["document_id"] = rag_meta.get("document_id")
        result["chunk_count"] = rag_meta.get("chunk_count")
        result["chat_history"] = []
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
                "qa_history": data.get("qa_history") or [],
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
        if read_history_item(session_id) is None:
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
        if read_history_item(session_id) is None:
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
