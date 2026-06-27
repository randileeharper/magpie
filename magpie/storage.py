"""Versioned SQLite storage for durable runs and reusable source snapshots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .errors import StorageError
from .models import EvidenceItem, FreshnessClass, Reference, SourceKind, StopReason, to_jsonable, utc_now
from .text import valid_unicode, valid_unicode_tree


SCHEMA_VERSION = 4
TRACKING_QUERY_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid",
}


def canonicalize_url(url: str) -> str:
    parts = urlsplit(valid_unicode(url).strip())
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
         if key.lower() not in TRACKING_QUERY_PARAMS],
        doseq=True,
    )
    return urlunsplit((parts.scheme.lower() or "https", parts.netloc.lower(), parts.path or "/", query, ""))


def content_hash(text: str) -> str:
    return hashlib.sha256(valid_unicode(text).strip().encode("utf-8")).hexdigest()


def normalize_query(query: str) -> str:
    return " ".join(valid_unicode(query).lower().split())


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


class SQLiteStorage:
    """Short-lived connections keep storage safe across A2A worker threads.

    Write-heavy sequences (a research round) can instead hold a single
    connection open via :meth:`transaction` so that all writes in the round
    commit together or roll back together on failure.
    """

    def __init__(self, database_path: Path):
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_lock = threading.Lock()
        self._local = threading.local()

    def _shared_connection(self) -> sqlite3.Connection | None:
        return getattr(self._local, "connection", None)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        shared = self._shared_connection()
        if shared is not None:
            yield shared
            return
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold a single connection open for a sequence of writes.

        All storage operations performed inside the ``with`` block share one
        SQLite connection. On clean exit the connection is committed; on any
        exception it is rolled back. This lets a caller group independent
        write methods (``add_query``, ``add_search_results``,
        ``upsert_source``, ``add_evidence_item`` ...) into one atomic unit
        without changing the methods themselves.

        Transactions are per-thread and may not be nested.
        """
        if self._shared_connection() is not None:
            raise StorageError("Nested storage transactions are not supported.")
        connection = sqlite3.connect(self._database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        self._local.connection = connection
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            self._local.connection = None
            connection.close()

    def initialize(self) -> None:
        with self._initialize_lock:
            if self._database_path.exists():
                if self._schema_is_current():
                    with self._connect() as connection:
                        connection.execute("PRAGMA journal_mode = WAL")
                        connection.executescript(_AUXILIARY_SCHEMA)
                    return
                self._delete_database_files()
            try:
                with self._connect() as connection:
                    connection.execute("PRAGMA journal_mode = WAL")
                    connection.executescript(_SCHEMA)
                    connection.executescript(_AUXILIARY_SCHEMA)
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            except sqlite3.Error as exc:
                raise StorageError(f"SQLite initialization failed for {self._database_path}: {exc}") from exc

    def _schema_is_current(self) -> bool:
        try:
            with sqlite3.connect(self._database_path) as connection:
                return connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        except sqlite3.Error:
            return False

    def _delete_database_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self._database_path) + suffix)
            if path.exists():
                path.unlink()

    def close(self) -> None:
        return None

    def create_run(
        self, question: str, run_label: str | None, freshness_class: FreshnessClass,
        response_detail: str, run_id: str | None = None,
    ) -> str:
        question = valid_unicode(question)
        run_label = valid_unicode(run_label) if run_label is not None else None
        run_id = run_id or str(uuid.uuid4())
        now = utc_now()
        normalized = normalize_query(question)
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO research_runs
                   (run_id, question, normalized_query, run_label, freshness_class, response_detail, status,
                    stop_reason, cancel_requested, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'running', NULL, 0, ?, ?)""",
                (run_id, question, normalized, run_label, freshness_class.value, response_detail, now, now),
            )
        return run_id

    def update_run_status(self, run_id: str, status: str, stop_reason: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE research_runs SET status=?, stop_reason=?, updated_at=? WHERE run_id=?",
                (status, stop_reason, utc_now(), run_id),
            )

    def request_cancel(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE research_runs SET cancel_requested=1, updated_at=?
                   WHERE run_id=? AND status='running'""",
                (utc_now(), run_id),
            )
        return cursor.rowcount > 0

    def mark_run_cancelled(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE research_runs
                   SET status='cancelled', stop_reason=?, updated_at=?
                   WHERE run_id=? AND status='running' AND cancel_requested=1""",
                (StopReason.CANCELLED.value, utc_now(), run_id),
            )
        return cursor.rowcount > 0

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM research_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return bool(row["cancel_requested"]) if row else False

    def append_event(self, run_id: str, stage: str, payload: dict[str, Any]) -> str:
        stage = valid_unicode(stage)
        payload = valid_unicode_tree(payload)
        event_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO run_events VALUES (?, ?, ?, ?, ?)",
                (event_id, run_id, stage, json.dumps(payload, sort_keys=True), utc_now()),
            )
        return event_id

    def add_query(self, run_id: str, normalized_query: str, provider: str, freshness_class: FreshnessClass) -> str:
        normalized_query = valid_unicode(normalized_query)
        provider = valid_unicode(provider)
        query_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO research_queries VALUES (?, ?, ?, ?, ?, ?)",
                (query_id, run_id, normalized_query, provider, freshness_class.value, utc_now()),
            )
        return query_id

    def list_queries_for_run(self, run_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT normalized_query FROM research_queries WHERE run_id=? ORDER BY created_at", (run_id,)
            ).fetchall()
        return [row["normalized_query"] for row in rows]

    def add_search_results(self, query_id: str, results: list[dict[str, Any]]) -> dict[str, str]:
        results = valid_unicode_tree(results)
        result_ids: dict[str, str] = {}
        with self._connect() as connection:
            for rank, result in enumerate(results):
                result_id = str(uuid.uuid4())
                connection.execute(
                    """INSERT INTO search_results
                       (result_id, query_id, rank_index, title, url, snippet, site_name, published_at,
                        provider, author, inline_text, highlights_json, raw_result_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (result_id, query_id, rank, result["title"], result["url"], result["snippet"],
                     result.get("site_name"), result.get("published_at"), result.get("provider"),
                     result.get("author"), result.get("inline_text"),
                     json.dumps(result.get("highlights", []), sort_keys=True),
                     json.dumps(result.get("raw_result", {}), sort_keys=True), utc_now()),
                )
                result_ids[result["url"]] = result_id
        return result_ids

    def upsert_source(
        self, run_id: str, raw_url: str, title: str, site_name: str | None,
        published_at: str | None, text: str, raw_payload: dict[str, Any],
        source_kind: SourceKind = SourceKind.PAGE_FETCH, search_result_id: str | None = None,
        fetch_error: str | None = None,
    ) -> str:
        raw_url = valid_unicode(raw_url)
        title = valid_unicode(title)
        site_name = valid_unicode(site_name) if site_name is not None else None
        text = valid_unicode(text)
        raw_payload = valid_unicode_tree(raw_payload)
        canonical_url = canonicalize_url(raw_url)
        digest = content_hash(text)
        with self._connect() as connection:
            document = connection.execute(
                "SELECT document_id FROM documents WHERE content_hash=?", (digest,)
            ).fetchone()
            document_id = document["document_id"] if document else str(uuid.uuid4())
            if not document:
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?)", (document_id, digest, text, utc_now())
                )
            source_id = str(uuid.uuid4())
            connection.execute(
                """INSERT INTO sources
                   (source_id, document_id, canonical_url, raw_url, title, site_name, published_at,
                    fetched_at, source_kind, search_result_id, fetch_error, raw_payload)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (source_id, document_id, canonical_url, raw_url, title, site_name, published_at,
                 utc_now(), source_kind.value, search_result_id, fetch_error,
                 json.dumps(raw_payload, sort_keys=True)),
            )
            connection.execute(
                "INSERT OR IGNORE INTO run_source_links VALUES (?, ?)", (run_id, source_id)
            )
        return source_id

    def link_run_source(self, run_id: str, source_id: str) -> None:
        with self._connect() as connection:
            connection.execute("INSERT OR IGNORE INTO run_source_links VALUES (?, ?)", (run_id, source_id))

    def add_extract(self, source_id: str, text: str, extraction_version: str = "v2") -> str:
        return source_id

    def get_extract_text(self, source_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT d.text FROM sources s JOIN documents d ON d.document_id=s.document_id WHERE s.source_id=?",
                (source_id,),
            ).fetchone()
        return row["text"] if row else None

    def get_cached_source_by_url(self, raw_url: str, min_fetched_at: str | None = None) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT s.*, d.text FROM sources s JOIN documents d ON d.document_id=s.document_id
                   WHERE s.canonical_url=? ORDER BY s.fetched_at DESC, s.source_id DESC LIMIT 1""",
                (canonicalize_url(raw_url),),
            ).fetchone()
        if not row or (min_fetched_at and _parse_utc(row["fetched_at"]) < _parse_utc(min_fetched_at)):
            return None
        return dict(row)

    def find_fresh_source_ids_for_exact_query(self, normalized_query: str, min_fetched_at: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT f.references_json
                   FROM research_runs r
                   JOIN final_answers f ON f.run_id=r.run_id
                   WHERE r.status IN ('completed', 'partial') AND r.normalized_query=?
                   ORDER BY f.created_at DESC""",
                (normalized_query,),
            ).fetchall()
            ordered_ids: list[str] = []
            for row in rows:
                for reference in json.loads(row["references_json"]):
                    source_id = reference.get("source_id")
                    if source_id and source_id not in ordered_ids:
                        ordered_ids.append(source_id)
            if not ordered_ids:
                return []
            min_dt = _parse_utc(min_fetched_at)
            placeholders = ",".join("?" for _ in ordered_ids)
            candidates = connection.execute(
                f"""SELECT s.source_id, s.canonical_url, s.fetched_at,
                          sr.source_id AS rejected
                   FROM sources s
                   LEFT JOIN source_rejections sr
                     ON sr.normalized_query=? AND sr.source_id=s.source_id
                   WHERE s.source_id IN ({placeholders})""",
                (normalized_query, *ordered_ids),
            ).fetchall()
            by_id = {row["source_id"]: row for row in candidates}
            source_ids: list[str] = []
            seen_urls: set[str] = set()
            for source_id in ordered_ids:
                source = by_id.get(source_id)
                if source is None or source["rejected"] is not None:
                    continue
                if _parse_utc(source["fetched_at"]) < min_dt:
                    continue
                if source["canonical_url"] in seen_urls:
                    continue
                seen_urls.add(source["canonical_url"])
                source_ids.append(source_id)
            return source_ids

    def reject_source_for_query(self, normalized_query: str, source_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO source_rejections VALUES (?, ?, ?)",
                (normalize_query(normalized_query), source_id, utc_now()),
            )

    def get_canonical_urls(self, source_ids: list[str]) -> set[str]:
        if not source_ids:
            return set()
        placeholders = ",".join("?" for _ in source_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT canonical_url FROM sources WHERE source_id IN ({placeholders})",
                source_ids,
            ).fetchall()
        return {row["canonical_url"] for row in rows}

    def add_evidence_item(self, run_id: str, source_id: str, excerpt: str, note: str) -> EvidenceItem:
        excerpt = valid_unicode(excerpt)
        note = valid_unicode(note)
        item = EvidenceItem(str(uuid.uuid4()), source_id, excerpt, note)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evidence_items VALUES (?, ?, ?, ?, ?, ?)",
                (item.evidence_id, run_id, source_id, excerpt, note, utc_now()),
            )
        return item

    def get_source_references(self, source_ids: list[str]) -> list[Reference]:
        if not source_ids:
            return []
        # Batch the lookup in one query rather than one SELECT per source_id,
        # then preserve the caller's requested order in the returned list.
        placeholders = ",".join("?" for _ in source_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM sources WHERE source_id IN ({placeholders})",
                tuple(source_ids),
            ).fetchall()
        by_id = {row["source_id"]: row for row in rows}
        references: list[Reference] = []
        for source_id in source_ids:
            row = by_id.get(source_id)
            if row:
                references.append(Reference(
                    row["source_id"], row["title"], row["raw_url"], row["site_name"],
                    row["published_at"], row["fetched_at"], SourceKind(row["source_kind"]),
                ))
        return references

    def save_final_answer(self, run_id: str, summary: str, answer: str, references: list[Reference]) -> None:
        summary = valid_unicode(summary)
        answer = valid_unicode(answer)
        with self._connect() as connection:
            self._save_final_answer(connection, run_id, summary, answer, references)

    def finalize_run(
        self,
        run_id: str,
        summary: str,
        answer: str,
        references: list[Reference],
        status: str,
        stop_reason: str,
    ) -> bool:
        """Persist an answer only when the run is still active and not canceled."""
        summary = valid_unicode(summary)
        answer = valid_unicode(answer)
        now = utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE research_runs
                   SET status=?, stop_reason=?, updated_at=?
                   WHERE run_id=? AND status='running' AND cancel_requested=0""",
                (status, stop_reason, now, run_id),
            )
            if cursor.rowcount == 0:
                return False
            self._save_final_answer(connection, run_id, summary, answer, references, created_at=now)
        return True

    def _save_final_answer(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        summary: str,
        answer: str,
        references: list[Reference],
        *,
        created_at: str | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO final_answers VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(run_id) DO UPDATE SET summary=excluded.summary, answer=excluded.answer,
               references_json=excluded.references_json, created_at=excluded.created_at""",
            (
                str(uuid.uuid4()),
                run_id,
                summary,
                answer,
                json.dumps([to_jsonable(r) for r in references], sort_keys=True),
                created_at or utc_now(),
            ),
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_event_payloads(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT stage, payload_json FROM run_events WHERE run_id=? ORDER BY created_at, event_id", (run_id,)
            ).fetchall()
        return [{"stage": row["stage"], **json.loads(row["payload_json"])} for row in rows]


_SCHEMA = """
CREATE TABLE research_runs (
 run_id TEXT PRIMARY KEY, question TEXT NOT NULL, normalized_query TEXT NOT NULL, run_label TEXT, freshness_class TEXT NOT NULL,
 response_detail TEXT NOT NULL, status TEXT NOT NULL, stop_reason TEXT, cancel_requested INTEGER NOT NULL,
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE research_queries (
 query_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
 normalized_query TEXT NOT NULL, provider TEXT NOT NULL, freshness_class TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE search_results (
 result_id TEXT PRIMARY KEY, query_id TEXT NOT NULL REFERENCES research_queries(query_id) ON DELETE CASCADE,
 rank_index INTEGER NOT NULL, title TEXT NOT NULL, url TEXT NOT NULL, snippet TEXT NOT NULL, site_name TEXT,
 published_at TEXT, provider TEXT, author TEXT, inline_text TEXT, highlights_json TEXT, raw_result_json TEXT,
 created_at TEXT NOT NULL
);
CREATE TABLE documents (
 document_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL UNIQUE, text TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE sources (
 source_id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(document_id), canonical_url TEXT NOT NULL,
 raw_url TEXT NOT NULL, title TEXT NOT NULL, site_name TEXT, published_at TEXT, fetched_at TEXT NOT NULL,
 source_kind TEXT NOT NULL, search_result_id TEXT, fetch_error TEXT, raw_payload TEXT NOT NULL
);
CREATE TABLE run_source_links (
 run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
 source_id TEXT NOT NULL REFERENCES sources(source_id), PRIMARY KEY(run_id, source_id)
);
CREATE TABLE evidence_items (
 evidence_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
 source_id TEXT NOT NULL REFERENCES sources(source_id), excerpt TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE run_events (
 event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
 stage TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE final_answers (
 answer_id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE REFERENCES research_runs(run_id) ON DELETE CASCADE,
 summary TEXT NOT NULL, answer TEXT NOT NULL, references_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX research_runs_normq_idx ON research_runs(normalized_query);
CREATE INDEX research_queries_run_idx ON research_queries(run_id);
CREATE INDEX sources_fetched_idx ON sources(fetched_at);
CREATE INDEX sources_url_fetched_idx ON sources(canonical_url, fetched_at);
"""

_AUXILIARY_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_rejections (
 normalized_query TEXT NOT NULL,
 source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
 created_at TEXT NOT NULL,
 PRIMARY KEY(normalized_query, source_id)
);
"""
