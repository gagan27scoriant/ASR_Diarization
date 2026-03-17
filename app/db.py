import os
from datetime import datetime, timezone

from pymongo import ASCENDING, MongoClient


_client: MongoClient | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db():
    global _client
    if _client is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        _client = MongoClient(uri)
    db_name = os.getenv("MONGODB_DB", "asr_studio")
    return _client[db_name]


def init_db() -> None:
    db = get_db()
    db.roles.create_index("role_name", unique=True)
    db.roles.create_index("role_id", unique=True)
    db.users.create_index("email", unique=True)
    db.departments.create_index("name", unique=True)
    db.activity_log.create_index([("created_at", ASCENDING)])
    db.history.create_index("session_id", unique=True)
    db.document_store.create_index("document_id", unique=True)

    for role_id, role_name in ((1, "user"), (2, "admin"), (3, "super_admin")):
        db.roles.update_one(
            {"role_name": role_name},
            {"$setOnInsert": {"role_id": role_id, "role_name": role_name}},
            upsert=True,
        )

    db.departments.update_one(
        {"name": "general"},
        {"$setOnInsert": {"name": "general", "created_at": _now_iso()}},
        upsert=True,
    )
