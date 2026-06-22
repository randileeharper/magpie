from __future__ import annotations

import tempfile
import threading
import time
import unittest
import asyncio
from pathlib import Path
from unittest import mock

import httpx
from google.protobuf.json_format import MessageToDict
from a2a.helpers import new_text_part
from a2a.types import Message, Role, SendMessageRequest

from magpie.a2a import build_fastapi_app
from magpie.config import Settings
from magpie.models import (
    FetchedSource, PlanningContext, QueryProposal, ResearchRequest, SearchResultRecord, SynthesisDraft,
)
from magpie.providers.fake import FakeFetcher, FakeResolverClient
from magpie.service import ResearchService
from magpie.storage import SQLiteStorage


class IterativeResolver(FakeResolverClient):
    def __init__(self) -> None:
        super().__init__()
        self.query_index = 0
        self.contexts: list[PlanningContext] = []

    def propose_query(self, question: str, context: PlanningContext) -> QueryProposal:
        self.contexts.append(context)
        self.query_index += 1
        return QueryProposal(f"query {self.query_index}")

    def synthesize(self, question, evidence, prior_draft=None):
        cited = [*(prior_draft.cited_source_ids if prior_draft else []), evidence[0].source_id]
        return SynthesisDraft("partial", "grounded partial", cited, ["more"])


class ManySearch:
    def search(self, request):
        return [
            SearchResultRecord(f"title {index}", f"https://example.com/{request.query}/{index}", "snippet")
            for index in range(10)
        ]


class AnyFetcher(FakeFetcher):
    def fetch(self, url: str):
        return FetchedSource(url, url, "Example", f"Evidence from {url}.")


class BlockingSearch:
    def __init__(self) -> None:
        self.started = threading.Event()

    def search(self, request):
        self.started.set()
        time.sleep(0.1)
        return []


class InvariantTests(unittest.TestCase):
    def _service(self, tmpdir: str, resolver, search) -> ResearchService:
        storage = SQLiteStorage(Path(tmpdir) / "magpie.db")
        storage.initialize()
        settings = Settings(
            database_path=str(Path(tmpdir) / "magpie.db"),
            max_search_queries_per_run=3,
            max_sources_per_query=2,
            max_sources_per_run=3,
            max_evidence_items_per_run=2,
        )
        return ResearchService(storage, resolver, search, AnyFetcher(), settings)

    def test_run_wide_budgets_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = IterativeResolver()
            service = self._service(tmpdir, resolver, ManySearch())
            result = service.research(ResearchRequest("question"))
            with service.storage._connect() as connection:
                queries = connection.execute(
                    "SELECT COUNT(*) FROM research_queries WHERE run_id=?", (result.run_id,)
                ).fetchone()[0]
                sources = connection.execute(
                    "SELECT COUNT(*) FROM run_source_links WHERE run_id=?", (result.run_id,)
                ).fetchone()[0]
                evidence = connection.execute(
                    "SELECT COUNT(*) FROM evidence_items WHERE run_id=?", (result.run_id,)
                ).fetchone()[0]
        self.assertLessEqual(queries, 3)
        self.assertLessEqual(sources, 3)
        self.assertLessEqual(evidence, 2)
        self.assertEqual(result.status, "partial")
        self.assertTrue(resolver.contexts)

    def test_explicit_run_id_and_cancellation_are_durable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            search = BlockingSearch()
            service = self._service(tmpdir, IterativeResolver(), search)
            holder = {}

            def run() -> None:
                holder["result"] = service.research(ResearchRequest("question"), run_id="task-123")

            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(search.started.wait(1))
            service.cancel_run("task-123")
            thread.join(2)
            run_row = service.storage.get_run("task-123")
        self.assertEqual(run_row["status"], "cancelled")
        self.assertEqual(holder["result"].stop_reason.value, "cancelled")

    def test_a2a_task_id_is_durable_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, FakeResolverClient(), ManySearch())
            app = build_fastapi_app(service, "http://testserver")
            message = Message(
                role=Role.ROLE_USER,
                message_id="message-1",
                parts=[new_text_part("question", media_type="text/plain")],
            )

            async def send() -> dict:
                async def inline_to_thread(function, *args, **kwargs):
                    return function(*args, **kwargs)

                with mock.patch("magpie.a2a.asyncio.to_thread", side_effect=inline_to_thread):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
                    ) as client:
                        response = await client.post(
                            "/message:send",
                            headers={"A2A-Version": "1.0"},
                            json=MessageToDict(SendMessageRequest(
                                message=message,
                                configuration={"return_immediately": False},
                            )),
                        )
                        self.assertEqual(response.status_code, 200)
                        return response.json()

            task = asyncio.run(send())["task"]
            run = service.storage.get_run(task["id"])
        self.assertEqual(task["status"]["state"], "TASK_STATE_COMPLETED")
        self.assertEqual(run["run_id"], task["id"])
        self.assertEqual(
            task["status"]["message"]["parts"][0]["text"],
            "Evidence from https://example.com/question/0.",
        )

    def test_agent_card_alias_without_json_suffix_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, FakeResolverClient(), ManySearch())
            app = build_fastapi_app(service, "http://testserver")

            async def fetch() -> dict:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://testserver"
                ) as client:
                    response = await client.get("/.well-known/agent-card")
                    self.assertEqual(response.status_code, 200)
                    return response.json()

            payload = asyncio.run(fetch())
        self.assertEqual(payload["name"], "Magpie")
        self.assertEqual(payload["skills"][0]["id"], "magpie_ask")


if __name__ == "__main__":
    unittest.main()
