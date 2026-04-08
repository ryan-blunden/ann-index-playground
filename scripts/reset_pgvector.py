from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import psycopg
from psycopg import sql

ENV_PATH = Path(".env")
DEFAULT_ADMIN_URL = "postgresql:///postgres"
DEFAULT_DATABASE_URL = "postgresql:///ann_indexes_pgvector"


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def db_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path and parsed.path != "/":
        return parsed.path.rsplit("/", maxsplit=1)[-1]
    raise ValueError(f"Could not determine database name from {url!r}")


def main() -> None:
    load_env_file()
    admin_url = os.environ.get("PGVECTOR_ADMIN_DATABASE_URL", DEFAULT_ADMIN_URL)
    database_url = os.environ.get("PGVECTOR_DATABASE_URL", DEFAULT_DATABASE_URL)
    database_name = db_name_from_url(database_url)

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (database_name,))
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database_name)))
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))

    print(f"Reset pgvector database: {database_name}")


if __name__ == "__main__":
    main()
