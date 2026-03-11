import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from werkzeug.security import check_password_hash, generate_password_hash

from app.db import get_conn


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


def get_policy(role: str) -> dict[str, Any]:
    return POLICIES.get(role, POLICIES[ROLE_USER])


def has_permission(role: str, permission: str) -> bool:
    policy = get_policy(role)
    return permission in set(policy.get("permissions", []))


def _jwt_secret() -> str:
    return os.getenv("JWT_SECRET", "change_me")


def issue_token(user: dict[str, Any]) -> str:
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
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def decode_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, _jwt_secret(), algorithms=["HS256"])


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    if not email or not password:
        return None
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.name, u.email, u.password_hash, u.department, r.role_name
        FROM users u
        JOIN roles r ON r.id = u.role_id
        WHERE u.email = ?
        """,
        (email.lower().strip(),),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if not check_password_hash(row["password_hash"], password):
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "department": row["department"],
        "role_name": row["role_name"],
    }


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.name, u.email, u.department, r.role_name
        FROM users u
        JOIN roles r ON r.id = u.role_id
        WHERE u.id = ?
        """,
        (user_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "department": row["department"],
        "role_name": row["role_name"],
    }


def list_users() -> list[dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.name, u.email, u.department, u.created_at, r.role_name
        FROM users u
        JOIN roles r ON r.id = u.role_id
        ORDER BY u.created_at DESC
        """
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "department": row["department"],
            "role_name": row["role_name"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def create_user(name: str, email: str, password: str, role_name: str, department: str) -> dict[str, Any]:
    if role_name not in ROLE_IDS:
        raise ValueError("Invalid role")
    if not password:
        raise ValueError("Password required")
    conn = get_conn()
    cur = conn.cursor()
    dep_value = (department or "general").strip() or "general"
    cur.execute("SELECT id FROM departments WHERE name = ?", (dep_value,))
    dep = cur.fetchone()
    if not dep:
        raise ValueError("Department not found")
    cur.execute(
        """
        INSERT INTO users (name, email, password_hash, role_id, department, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            name.strip(),
            email.lower().strip(),
            generate_password_hash(password),
            ROLE_IDS[role_name],
            dep_value,
            _now_iso(),
        ),
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()
    return {
        "id": user_id,
        "name": name.strip(),
        "email": email.lower().strip(),
        "department": (department or "general").strip() or "general",
        "role_name": role_name,
    }


def update_user_profile(user_id: int, name: str | None, password: str | None) -> None:
    conn = get_conn()
    cur = conn.cursor()
    if name:
        cur.execute("UPDATE users SET name = ? WHERE id = ?", (name.strip(), user_id))
    if password:
        cur.execute("UPDATE users SET password_hash = ? WHERE id = ?", (generate_password_hash(password), user_id))
    conn.commit()
    conn.close()


def update_user_admin(user_id: int, role_name: str | None, department: str | None, password: str | None) -> None:
    conn = get_conn()
    cur = conn.cursor()
    if role_name:
        if role_name not in ROLE_IDS:
            raise ValueError("Invalid role")
        cur.execute("UPDATE users SET role_id = ? WHERE id = ?", (ROLE_IDS[role_name], user_id))
    if department is not None:
        dep_value = (department or "general").strip() or "general"
        cur.execute("SELECT id FROM departments WHERE name = ?", (dep_value,))
        dep = cur.fetchone()
        if not dep:
            raise ValueError("Department not found")
        cur.execute("UPDATE users SET department = ? WHERE id = ?", (dep_value, user_id))
    if password:
        cur.execute("UPDATE users SET password_hash = ? WHERE id = ?", (generate_password_hash(password), user_id))
    conn.commit()
    conn.close()


def delete_user(user_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def log_activity(user_id: int, action: str, meta: dict[str, Any] | None = None) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO activity_log (user_id, action, meta, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, action, json.dumps(meta or {}), _now_iso()),
    )
    conn.commit()
    conn.close()


def list_activity(limit: int = 200) -> list[dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT a.id, a.action, a.meta, a.created_at, u.email, u.department, r.role_name
        FROM activity_log a
        JOIN users u ON u.id = a.user_id
        JOIN roles r ON r.id = u.role_id
        ORDER BY a.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    events = []
    for row in rows:
        events.append(
            {
                "action": row["action"],
                "meta": json.loads(row["meta"] or "{}"),
                "created_at": row["created_at"],
                "email": row["email"],
                "department": row["department"],
                "role": row["role_name"],
            }
        )
    return events


def list_departments() -> list[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM departments ORDER BY name")
    rows = cur.fetchall()
    conn.close()
    return [row["name"] for row in rows]


def create_department(name: str) -> None:
    dep = (name or "").strip()
    if not dep:
        raise ValueError("Department name required")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO departments (name, created_at) VALUES (?, ?)", (dep, _now_iso()))
    conn.commit()
    conn.close()
