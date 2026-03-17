import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from bson import ObjectId
from pymongo.errors import DuplicateKeyError
from werkzeug.security import check_password_hash, generate_password_hash

from app.db import get_db


ROLE_USER = "user"
ROLE_ADMIN = "admin"
ROLE_SUPER_ADMIN = "super_admin"

ROLE_IDS = {
    ROLE_USER: 1,
    ROLE_ADMIN: 2,
    ROLE_SUPER_ADMIN: 3,
}

POLICIES: dict[str, dict[str, Any]] = {
    ROLE_SUPER_ADMIN: {
        "permissions": [
            "process:media",
            "process:document",
            "rag:ask",
            "summary:generate",
            "translate:run",
            "history:read",
            "history:rename",
            "history:delete",
            "export:transcript",
            "export:summary",
            "user:manage",
            "admin:manage",
            "audit:read",
            "settings:manage",
        ],
    },
    ROLE_ADMIN: {
        "permissions": [
            "process:media",
            "process:document",
            "rag:ask",
            "summary:generate",
            "translate:run",
            "history:read",
            "history:rename",
            "history:delete",
            "export:transcript",
            "export:summary",
            "user:manage",
            "audit:read",
        ],
    },
    ROLE_USER: {
        "permissions": [
            "process:media",
            "process:document",
            "rag:ask",
            "summary:generate",
            "translate:run",
            "history:read",
            "history:delete",
            "export:transcript",
            "profile:update",
        ],
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _users_col():
    return get_db()["users"]


def _departments_col():
    return get_db()["departments"]


def _activity_col():
    return get_db()["activity_log"]


def _history_col():
    return get_db()["history"]


def _oid(value: Any) -> ObjectId | None:
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _user_doc_to_dict(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    return {
        "id": str(doc.get("_id")),
        "name": doc.get("name"),
        "email": doc.get("email"),
        "department": doc.get("department"),
        "role_name": doc.get("role_name"),
    }


def get_policy(role: str) -> dict[str, Any]:
    return POLICIES.get(role, POLICIES[ROLE_USER])


def has_permission(role: str, permission: str) -> bool:
    policy = get_policy(role)
    return permission in set(policy.get("permissions", []))


def _jwt_secret() -> str:
    return os.getenv("JWT_SECRET", "change_me")


def issue_token(
    user: dict[str, Any],
    impersonator: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> str:
    exp_minutes = int(os.getenv("JWT_EXPIRES_MIN", "720"))
    payload = {
        "sub": str(user["id"]),
        "role": user["role_name"],
        "department": user.get("department"),
        "name": user.get("name"),
        "email": user.get("email"),
        "exp": datetime.utcnow() + timedelta(minutes=exp_minutes),
        "iat": datetime.utcnow(),
    }
    if session_id:
        payload["sid"] = session_id
    if impersonator:
        payload["impersonated_by"] = {
            "id": impersonator.get("id"),
            "email": impersonator.get("email"),
            "name": impersonator.get("name"),
            "role": impersonator.get("role_name"),
        }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, _jwt_secret(), algorithms=["HS256"])


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    if not email or not password:
        return None
    row = _users_col().find_one({"email": email.lower().strip()})
    if not row:
        return None
    if not check_password_hash(row.get("password_hash", ""), password):
        return None
    return _user_doc_to_dict(row)


def get_user_by_id(user_id: Any) -> dict[str, Any] | None:
    oid = _oid(user_id)
    if not oid:
        return None
    row = _users_col().find_one({"_id": oid})
    return _user_doc_to_dict(row)


def get_user_by_email(email: str) -> dict[str, Any] | None:
    if not email:
        return None
    row = _users_col().find_one({"email": email.lower().strip()})
    return _user_doc_to_dict(row)


def list_users() -> list[dict[str, Any]]:
    rows = list(_users_col().find({}).sort("created_at", -1))
    return [
        {
            "id": str(row["_id"]),
            "name": row.get("name"),
            "email": row.get("email"),
            "department": row.get("department"),
            "role_name": row.get("role_name"),
            "created_at": row.get("created_at"),
        }
        for row in rows
    ]


def create_user(name: str, email: str, password: str, role_name: str, department: str) -> dict[str, Any]:
    if role_name not in ROLE_IDS:
        raise ValueError("Invalid role")
    if not password:
        raise ValueError("Password required")
    dep_value = (department or "general").strip() or "general"
    dep = _departments_col().find_one({"name": dep_value})
    if not dep:
        raise ValueError("Department not found")
    try:
        result = _users_col().insert_one(
            {
                "name": name.strip(),
                "email": email.lower().strip(),
                "password_hash": generate_password_hash(password),
                "role_name": role_name,
                "department": dep_value,
                "created_at": _now_iso(),
            }
        )
    except DuplicateKeyError as exc:
        raise ValueError("Email already exists") from exc
    return {
        "id": str(result.inserted_id),
        "name": name.strip(),
        "email": email.lower().strip(),
        "department": dep_value,
        "role_name": role_name,
    }


def update_user_profile(user_id: Any, name: str | None, password: str | None) -> None:
    oid = _oid(user_id)
    if not oid:
        raise ValueError("Invalid user id")
    update: dict[str, Any] = {}
    if name:
        update["name"] = name.strip()
    if password:
        update["password_hash"] = generate_password_hash(password)
    if update:
        _users_col().update_one({"_id": oid}, {"$set": update})


def update_user_admin(user_id: Any, role_name: str | None, department: str | None, password: str | None) -> None:
    oid = _oid(user_id)
    if not oid:
        raise ValueError("Invalid user id")
    update: dict[str, Any] = {}
    if role_name:
        if role_name not in ROLE_IDS:
            raise ValueError("Invalid role")
        update["role_name"] = role_name
    if department is not None:
        dep_value = (department or "general").strip() or "general"
        dep = _departments_col().find_one({"name": dep_value})
        if not dep:
            raise ValueError("Department not found")
        update["department"] = dep_value
    if password:
        update["password_hash"] = generate_password_hash(password)
    if update:
        _users_col().update_one({"_id": oid}, {"$set": update})


def delete_user(user_id: Any) -> None:
    oid = _oid(user_id)
    if not oid:
        raise ValueError("Invalid user id")
    _users_col().delete_one({"_id": oid})


def log_activity(user_id: Any, action: str, meta: dict[str, Any] | None = None) -> None:
    oid = _oid(user_id)
    user = _users_col().find_one({"_id": oid}) if oid else None
    _activity_col().insert_one(
        {
            "user_id": oid,
            "action": action,
            "meta": meta or {},
            "created_at": _now_iso(),
            "email": user.get("email") if user else None,
            "department": user.get("department") if user else None,
            "role": user.get("role_name") if user else None,
        }
    )


def list_activity(limit: int = 200) -> list[dict[str, Any]]:
    rows = list(_activity_col().find({}).sort("created_at", -1).limit(limit))
    events = []
    for row in rows:
        events.append(
            {
                "id": str(row.get("_id")),
                "action": row.get("action"),
                "meta": row.get("meta") or {},
                "created_at": row.get("created_at"),
                "email": row.get("email"),
                "department": row.get("department"),
                "role": row.get("role"),
            }
        )
    return events


def delete_activity(activity_id: Any) -> None:
    oid = _oid(activity_id)
    if not oid:
        raise ValueError("Invalid activity id")
    _activity_col().delete_one({"_id": oid})


def clear_activity() -> None:
    _activity_col().delete_many({})


def list_departments() -> list[dict[str, Any]]:
    rows = list(_departments_col().find({}).sort("name", 1))
    return [{"id": str(row.get("_id")), "name": row.get("name")} for row in rows]


def create_department(name: str) -> None:
    dep = (name or "").strip()
    if not dep:
        raise ValueError("Department name required")
    try:
        _departments_col().insert_one({"name": dep, "created_at": _now_iso()})
    except DuplicateKeyError as exc:
        raise ValueError("Department already exists") from exc


def delete_department(department_id: Any) -> None:
    oid = _oid(department_id)
    if not oid:
        raise ValueError("Department not found")
    row = _departments_col().find_one({"_id": oid})
    if not row:
        raise ValueError("Department not found")
    dept_name = row.get("name")
    users = list(_users_col().find({"department": dept_name}, {"_id": 1}))
    user_ids = [u["_id"] for u in users]
    if user_ids:
        _activity_col().delete_many({"user_id": {"$in": user_ids}})
        _users_col().delete_many({"_id": {"$in": user_ids}})
    _history_col().delete_many({"owner.department": dept_name})
    _departments_col().delete_one({"_id": oid})


def update_department(department_id: Any, name: str) -> None:
    dep = (name or "").strip()
    if not dep:
        raise ValueError("Department name required")
    oid = _oid(department_id)
    if not oid:
        raise ValueError("Department not found")
    row = _departments_col().find_one({"_id": oid})
    if not row:
        raise ValueError("Department not found")
    old_name = row.get("name")
    _departments_col().update_one({"_id": oid}, {"$set": {"name": dep}})
    _users_col().update_many({"department": old_name}, {"$set": {"department": dep}})
    _activity_col().update_many({"department": old_name}, {"$set": {"department": dep}})
    _history_col().update_many({"owner.department": old_name}, {"$set": {"owner.department": dep}})
