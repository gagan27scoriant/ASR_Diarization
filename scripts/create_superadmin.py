#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db import init_db
from app.auth import create_user, ROLE_SUPER_ADMIN


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the initial Super Admin account.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--department", default="global")
    args = parser.parse_args()

    init_db()
    create_user(args.name, args.email, args.password, ROLE_SUPER_ADMIN, args.department)
    print("Super Admin created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
