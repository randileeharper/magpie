from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from magpie.errors import StorageError
from magpie.models import FreshnessClass, Reference, SourceKind
from magpie.storage import SQLiteStorage, canonicalize_url


class StorageTests(unittest.TestCase):
    def test_url_canonicalization_strips_tracking_query_params(self) -> None:
        url = "https://Example.com/news?id=1&utm_source=test#fragment"
        self.assertEqual(canonicalize_url(url), "https://example.com/news?id=1")

    def test_shared_document_keeps_distinct_source_urls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SQLiteStorage(Path(tmpdir) / "magpie.db")
            storage.initialize()
            run_id = storage.create_run("q", None, FreshnessClass.EVERGREEN, "compact")
            first = storage.upsert_source(
                run_id,
                "https://example.com/a?utm_source=x",
                "A",
                "Example",
                None,
                "same body",
                {"k": "v"},
            )
            second = storage.upsert_source(
                run_id,
                "https://example.com/b",
                "B",
                "Example",
                None,
                "same body",
                {"k": "v"},
            )
            first_cached = storage.get_cached_source_by_url("https://example.com/a")
            second_cached = storage.get_cached_source_by_url("https://example.com/b")
            connection = storage._connection
            document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            connection.close()
            storage.close()
        self.assertNotEqual(first, second)
        self.assertEqual(first_cached["raw_url"], "https://example.com/a?utm_source=x")
        self.assertEqual(second_cached["raw_url"], "https://example.com/b")
        self.assertEqual(document_count, 1)

    def test_initialize_replaces_incompatible_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "magpie.db"
            import sqlite3
            with sqlite3.connect(database_path) as connection:
                connection.execute("CREATE TABLE old_schema (value TEXT)")
            storage = SQLiteStorage(database_path)
            storage.initialize()
            connection = storage._connection
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            old_table = connection.execute(
                "SELECT name FROM sqlite_master WHERE name='old_schema'"
            ).fetchone()
            connection.close()
            storage.close()
        self.assertEqual(version, 3)
        self.assertIsNone(old_table)

    def test_refreshing_url_creates_append_only_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SQLiteStorage(Path(tmpdir) / "magpie.db")
            storage.initialize()
            run_id = storage.create_run("q", None, FreshnessClass.EVERGREEN, "compact")
            first = storage.upsert_source(run_id, "https://example.com/a", "A", None, None, "old", {})
            second = storage.upsert_source(run_id, "https://example.com/a", "A", None, None, "new", {})
            connection = storage._connection
            source_count = connection.execute(
                "SELECT COUNT(*) FROM sources WHERE canonical_url='https://example.com/a'"
            ).fetchone()[0]
            connection.close()
            self.assertNotEqual(first, second)
            self.assertEqual(source_count, 2)
            self.assertEqual(storage.get_extract_text(first), "old")
            self.assertEqual(storage.get_extract_text(second), "new")

    def test_storage_replaces_unpaired_surrogates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SQLiteStorage(Path(tmpdir) / "magpie.db")
            storage.initialize()
            run_id = storage.create_run("bad \ud8a2 question", None, FreshnessClass.EVERGREEN, "compact")
            source_id = storage.upsert_source(
                run_id,
                "https://example.com/",
                "bad \ud8a2 title",
                None,
                None,
                "bad \ud8a2 body",
                {"bad": "\ud8a2"},
            )

            run = storage.get_run(run_id)
            body = storage.get_extract_text(source_id)

        self.assertEqual(run["question"], "bad ? question")
        self.assertEqual(body, "bad ? body")

    def test_exact_query_cache_only_returns_final_answer_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SQLiteStorage(Path(tmpdir) / "magpie.db")
            storage.initialize()
            run_id = storage.create_run("question", None, FreshnessClass.EVERGREEN, "compact")
            rejected = storage.upsert_source(
                run_id, "https://example.com/rejected", "Rejected", None, None, "no answer", {}
            )
            accepted = storage.upsert_source(
                run_id, "https://example.com/accepted", "Accepted", None, None, "answer", {}
            )
            reference = Reference(
                accepted, "Accepted", "https://example.com/accepted", None, None, None, SourceKind.PAGE_FETCH
            )
            storage.save_final_answer(run_id, "summary", "answer", [reference])
            storage.update_run_status(run_id, "completed")

            cached = storage.find_fresh_source_ids_for_exact_query("question", "2000-01-01T00:00:00Z")
            storage.reject_source_for_query("question", accepted)
            rejected_cached = storage.find_fresh_source_ids_for_exact_query("question", "2000-01-01T00:00:00Z")

        self.assertEqual(cached, [accepted])
        self.assertNotIn(rejected, cached)
        self.assertEqual(rejected_cached, [])


if __name__ == "__main__":
    unittest.main()
