"""Environment and provider readiness checks."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import Settings
from .providers.base import Fetcher, SearchClient


def run_doctor(settings: Settings, search_client: SearchClient, fetcher: Fetcher, live: bool = False) -> dict[str, Any]:
    report = {
        "status": "ok",
        "configuration": settings.sanitized_diagnostics(),
        "database": _database_check(settings.expanded_database_path),
        "search": search_client.doctor_check(live=live),
        "fetch": fetcher.doctor_check(live=live),
    }
    if any(section.get("status") != "ok" for section in (report["database"], report["search"], report["fetch"])):
        report["status"] = "error"
    return report


def _database_check(database_path: Path) -> dict[str, object]:
    parent = database_path.parent
    writable = parent.exists() and parent.is_dir() and _dir_writable(parent)
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
            writable = _dir_writable(parent)
        except OSError:
            writable = False

    sqlite_ok = False
    try:
        with sqlite3.connect(":memory:") as connection:
            connection.execute("SELECT 1")
            sqlite_ok = True
    except sqlite3.OperationalError:
        sqlite_ok = False

    return {
        "status": "ok" if writable and sqlite_ok else "error",
        "database_path": str(database_path),
        "parent_writable": writable,
        "sqlite": sqlite_ok,
    }


def _dir_writable(path: Path) -> bool:
    probe = path / ".magpie-write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False
