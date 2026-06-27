from __future__ import annotations

import tempfile
import threading
import time
import unittest
import asyncio
from pathlib import Path
from unittest import mock

import httpx
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict
from a2a.helpers import new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.types import Message, Role, SendMessageRequest, TaskState

from magpie.a2a import build_fastapi_app, SDKResearchAgentExecutor
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

    def test_a2a_message_metadata_selects_skill_over_http_transport(self) -> None:
        # Regression for issue #50: clients set `skill` on the message
        # (Message.metadata), but RequestContext.metadata reads the request-level
        # SendMessageRequest.metadata field, so the skill was lost over the HTTP
        # transport and every request fell back to magpie_ask. The executor must
        # honor message-level metadata so the magpie_search route is reachable.
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, FakeResolverClient(), ManySearch())
            app = build_fastapi_app(service, "http://testserver")
            message = Message(
                role=Role.ROLE_USER,
                message_id="message-1",
                parts=[new_text_part("question", media_type="text/plain")],
                metadata={"skill": "magpie_search"},
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
        self.assertEqual(task["status"]["state"], "TASK_STATE_COMPLETED")
        # The search route produces a "Found N results" summary; the ask route
        # (the previous default) would produce an evidence-grounded answer.
        self.assertEqual(task["status"]["message"]["parts"][0]["text"], "Found 5 results for: question")

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

    def test_a2a_task_transitions_to_failed_when_service_raises(self) -> None:
        # Regression for issue #32: a service call that raises must transition
        # the A2A task to a terminal failed state with an error message, rather
        # than propagating uncaught and leaving the task non-terminal. The
        # executor is driven directly so request-level metadata (the `skill`)
        # is honored; `search()` re-raises on failure (unlike `research()`,
        # which returns a ResearchErrorResult), exercising the bug path.
        class ExplodingSearch(ManySearch):
            def search(self, request):
                raise RuntimeError("search blew up")

        class CapturingQueue(EventQueue):
            def __init__(self) -> None:
                self.events: list = []

            async def enqueue_event(self, event: object) -> None:
                self.events.append(event)

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, FakeResolverClient(), ExplodingSearch())
            executor = SDKResearchAgentExecutor(service)
            message = Message(
                role=Role.ROLE_USER,
                message_id="message-1",
                parts=[new_text_part("question", media_type="text/plain")],
            )
            request = SendMessageRequest(message=message)
            skill_struct = struct_pb2.Struct()
            skill_struct.fields["skill"].string_value = "magpie_search"
            request.metadata.CopyFrom(skill_struct)
            context = RequestContext(
                call_context=ServerCallContext(),
                request=request,
                task_id="task-1",
                context_id="ctx-1",
            )

            async def run() -> list:
                queue = CapturingQueue()

                async def inline_to_thread(function, *args, **kwargs):
                    return function(*args, **kwargs)

                with mock.patch("magpie.a2a.asyncio.to_thread", side_effect=inline_to_thread):
                    await executor.execute(context, queue)
                return queue.events

            events = asyncio.run(run())

        failed = [
            event for event in events
            if type(event).__name__ == "TaskStatusUpdateEvent" and event.status.state == TaskState.TASK_STATE_FAILED
        ]
        self.assertEqual(len(failed), 1, "expected exactly one failed status event")
        self.assertTrue(failed[0].status.HasField("message"))
        self.assertIn("search blew up", failed[0].status.message.parts[0].text)

    def test_distinct_services_resolve_concurrently(self) -> None:
        # Regression for issue #38: the resolver gate must be per-service, not a
        # global singleton, so two concurrent runs against distinct services
        # (with distinct resolvers) proceed in parallel. With the old global
        # semaphore, run B would block until run A released the gate, so the two
        # `propose_query` calls could never overlap in time.
        class TrackingResolver(FakeResolverClient):
            def __init__(self, state: dict) -> None:
                super().__init__()
                self._state = state
                self._lock = state["lock"]

            def propose_query(self, question, context):
                with self._lock:
                    self._state["active"] += 1
                    self._state["peak"] = max(self._state["peak"], self._state["active"])
                # Hold the call long enough for a concurrent run to observe it.
                time.sleep(0.1)
                with self._lock:
                    self._state["active"] -= 1
                return QueryProposal(question)

        state: dict = {"lock": threading.Lock(), "active": 0, "peak": 0}

        def build_service(tmpdir: str, resolver: TrackingResolver) -> ResearchService:
            storage = SQLiteStorage(Path(tmpdir) / "magpie.db")
            storage.initialize()
            settings = Settings(database_path=str(Path(tmpdir) / "magpie.db"))
            return ResearchService(storage, resolver, ManySearch(), AnyFetcher(), settings)

        with tempfile.TemporaryDirectory() as tmpdir:
            resolver_a = TrackingResolver(state)
            resolver_b = TrackingResolver(state)
            service_a = build_service(str(Path(tmpdir) / "a"), resolver_a)
            service_b = build_service(str(Path(tmpdir) / "b"), resolver_b)

            def run(service: ResearchService, question: str) -> None:
                service.research(ResearchRequest(question))

            thread_a = threading.Thread(target=run, args=(service_a, "alpha"))
            thread_b = threading.Thread(target=run, args=(service_b, "beta"))
            thread_a.start()
            thread_b.start()
            thread_a.join(5)
            thread_b.join(5)

        self.assertFalse(thread_a.is_alive(), "service A run did not finish")
        self.assertFalse(thread_b.is_alive(), "service B run did not finish")
        # With a per-service lock, the two resolver calls overlap (peak >= 2).
        # With the old global semaphore they serialize (peak == 1).
        self.assertGreaterEqual(
            state["peak"], 2,
            "resolver calls did not run concurrently across distinct services",
        )


if __name__ == "__main__":
    unittest.main()
