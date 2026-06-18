from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from a2a.helpers import new_text_part
from a2a.types import Message, Role, SendMessageRequest
from google.protobuf.json_format import MessageToDict
from jsonschema import Draft202012Validator

from magpie import cli
from magpie.a2a import build_fastapi_app
from magpie.config import Settings
from magpie.errors import A2AUnavailableError, SearchError
from magpie.historian import (
    FakeHistorianSink,
    HistorianDeliveryError,
    HttpHistorianSink,
    NullHistorianSink,
    build_event,
)
from magpie.models import FreshnessClass, ResearchRequest, SynthesisDraft
from magpie.providers.fake import FakeFetcher, FakeResolverClient, FakeSearchClient
from magpie.service import ResearchService
from magpie.storage import SQLiteStorage


def _enabled_settings(tmpdir: str, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_path": str(Path(tmpdir) / "magpie.db"),
        "search_provider": "fake",
        "fetch_provider": "fake",
        "resolver_backend": "fake",
        "weather_enabled": False,
        "anime_enabled": False,
        "news_enabled": False,
        "historian_enabled": True,
        "historian_base_url": "https://historian.test",
        "historian_token": "hist_test_token",
        "historian_timeout_seconds": 3.0,
        "historian_verify_tls": False,
        "historian_retry_count": 2,
        "fetch_debug_log_path": str(Path(tmpdir) / "fetch.log"),
    }
    values.update(overrides)
    settings = Settings(**values)
    settings.validate()
    return settings


def _service(tmpdir: str, sink: object, settings: Settings | None = None) -> ResearchService:
    configured = settings or _enabled_settings(tmpdir)
    storage = SQLiteStorage(configured.expanded_database_path)
    storage.initialize()
    return ResearchService(
        storage=storage,
        resolver=FakeResolverClient(),
        search_client=FakeSearchClient(),
        fetcher=FakeFetcher(),
        settings=configured,
        historian_sink=sink,
    )


def _validate_manifest_events(events: list[dict[str, object]]) -> None:
    manifest = json.loads(
        (Path(__file__).parents[1] / "historian.manifest.json").read_text(encoding="utf-8")
    )
    schemas = {
        (item["event_type"], item["version"]): item["json_schema"]
        for item in manifest["schemas"]
    }
    for event in events:
        if str(event["type"]).startswith("core."):
            continue
        Draft202012Validator(
            schemas[(event["type"], event["schemaversion"])]
        ).validate(event["data"])


class HistorianSinkTests(unittest.TestCase):
    def test_null_sink_has_no_side_effects(self) -> None:
        sink = NullHistorianSink()
        sink.emit({"id": "one"})
        sink.emit_batch([{"id": "two"}])
        sink.close()

    def test_http_sink_uses_bearer_endpoints_and_payloads(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"status": "ok"})

        with tempfile.TemporaryDirectory() as tmpdir:
            client = httpx.Client(
                base_url="https://historian.test",
                transport=httpx.MockTransport(handler),
                timeout=3.0,
                verify=False,
            )
            sink = HttpHistorianSink(_enabled_settings(tmpdir), client=client)
            event = build_event(
                "research.run.canceled",
                {"run_id": "run-1", "status": "canceled", "stop_reason": "cancelled"},
            )
            sink.emit(event)
            sink.emit_batch([event])

        self.assertEqual(
            [request.url.path for request in requests],
            ["/v1/events", "/v1/events:batch"],
        )
        self.assertTrue(all(
            request.headers["Authorization"] == "Bearer hist_test_token"
            for request in requests
        ))
        self.assertEqual(json.loads(requests[0].content), event)
        self.assertEqual(json.loads(requests[1].content), {"events": [event]})

    def test_http_sink_builds_client_with_configured_transport_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _enabled_settings(tmpdir)
            fake_client = mock.Mock()
            with mock.patch("magpie.historian.httpx.Client", return_value=fake_client) as factory:
                sink = HttpHistorianSink(settings)

        factory.assert_called_once_with(
            base_url="https://historian.test",
            timeout=3.0,
            verify=False,
        )
        sink.close()
        fake_client.close.assert_called_once()

    def test_http_sink_retries_connection_and_5xx_with_stable_id(self) -> None:
        attempts: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(json.loads(request.content))
            if len(attempts) == 1:
                raise httpx.ConnectError("offline", request=request)
            if len(attempts) == 2:
                return httpx.Response(503, json={"status": "error"})
            return httpx.Response(200, json={"status": "ok"})

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = HttpHistorianSink(
                _enabled_settings(tmpdir),
                client=httpx.Client(
                    base_url="https://historian.test",
                    transport=httpx.MockTransport(handler),
                ),
                sleep=lambda _: None,
            )
            event = build_event("research.route.selected", {
                "run_id": "run-1", "route": "web_research", "fallback_reason": None,
            })
            sink.emit(event)

        self.assertEqual(len(attempts), 3)
        self.assertEqual({attempt["id"] for attempt in attempts}, {event["id"]})

    def test_http_sink_does_not_retry_client_failures(self) -> None:
        for status_code in (400, 401, 403, 409, 422):
            attempts = 0

            def handler(request: httpx.Request) -> httpx.Response:
                nonlocal attempts
                attempts += 1
                return httpx.Response(status_code, request=request, json={"status": "error"})

            with tempfile.TemporaryDirectory() as tmpdir:
                sink = HttpHistorianSink(
                    _enabled_settings(tmpdir),
                    client=httpx.Client(
                        base_url="https://historian.test",
                        transport=httpx.MockTransport(handler),
                    ),
                    sleep=lambda _: None,
                )
                with self.assertRaises(httpx.HTTPStatusError):
                    sink.emit(build_event("research.route.selected", {
                        "run_id": "run-1", "route": "web_research", "fallback_reason": None,
                    }))
            self.assertEqual(attempts, 1)


class HistorianServiceTests(unittest.TestCase):
    def test_service_events_match_manifest_and_share_correlation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = FakeHistorianSink()
            service = _service(tmpdir, sink)
            result = service.research(ResearchRequest("Who is the mayor of New York?"))

        self.assertEqual(result.status, "ok")
        event_types = [event["type"] for event in sink.events]
        self.assertIn("research.run.started", event_types)
        self.assertIn("research.route.selected", event_types)
        self.assertIn("research.query.executed", event_types)
        self.assertIn("research.source.discovered", event_types)
        self.assertIn("research.source.fetched", event_types)
        self.assertIn("research.synthesis.completed", event_types)
        self.assertEqual(event_types.count("research.run.completed"), 1)
        self.assertTrue(all(event.get("correlationid") == result.run_id for event in sink.events))

        _validate_manifest_events(sink.events)
        for event in sink.events:
            encoded = json.dumps(event)
            self.assertNotIn("hist_test_token", encoded)

    def test_delivery_failure_does_not_change_success(self) -> None:
        class FailingSink:
            def emit(self, event):
                raise HistorianDeliveryError("offline")

            def emit_batch(self, events):
                raise HistorianDeliveryError("offline")

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            service = _service(tmpdir, FailingSink())
            with self.assertLogs("magpie.service", level="WARNING") as logs:
                result = service.research(ResearchRequest("Who is the mayor of New York?"))

        self.assertEqual(result.status, "ok")
        self.assertTrue(any("Historian delivery failed" in line for line in logs.output))

    def test_provider_failure_emits_operation_and_terminal_errors(self) -> None:
        class ExplodingSearch:
            def search(self, request):
                raise SearchError("provider unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = FakeHistorianSink()
            settings = _enabled_settings(tmpdir)
            storage = SQLiteStorage(settings.expanded_database_path)
            storage.initialize()
            service = ResearchService(
                storage=storage,
                resolver=FakeResolverClient(),
                search_client=ExplodingSearch(),
                fetcher=FakeFetcher(),
                settings=settings,
                historian_sink=sink,
            )
            result = service.research(ResearchRequest("Who is the mayor of New York?"))

        self.assertEqual(result.status, "error")
        operation_error = next(
            event for event in sink.events if event["type"] == "core.operation.error"
        )
        self.assertEqual(operation_error["data"]["component"], "search")
        self.assertEqual(
            [event["type"] for event in sink.events].count("research.run.failed"),
            1,
        )
        _validate_manifest_events(sink.events)

    def test_partial_run_event_matches_manifest(self) -> None:
        class PartialResolver(FakeResolverClient):
            def synthesize(self, question, evidence, prior_draft=None):
                return SynthesisDraft(
                    "partial",
                    "grounded partial answer",
                    [evidence[0].source_id],
                    ["Need another source"],
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            sink = FakeHistorianSink()
            settings = _enabled_settings(
                tmpdir,
                max_search_queries_per_run=1,
                max_sources_per_query=1,
                max_sources_per_run=1,
                max_evidence_items_per_run=1,
            )
            storage = SQLiteStorage(settings.expanded_database_path)
            storage.initialize()
            service = ResearchService(
                storage=storage,
                resolver=PartialResolver(),
                search_client=FakeSearchClient(),
                fetcher=FakeFetcher(),
                settings=settings,
                historian_sink=sink,
            )
            result = service.research(ResearchRequest("Who is the mayor of New York?"))

        self.assertEqual(result.status, "partial")
        self.assertEqual(
            [event["type"] for event in sink.events].count("research.run.partial"),
            1,
        )
        _validate_manifest_events(sink.events)

    def test_event_data_recursively_redacts_configured_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = FakeHistorianSink()
            settings = _enabled_settings(
                tmpdir,
                search_api_key="search-secret",
                resolver_api_key="resolver-secret",
            )
            service = _service(tmpdir, sink, settings)
            service._emit(
                "core.operation.error",
                {
                    "app_id": "magpie",
                    "component": "test",
                    "error_type": "Example",
                    "message": "search-secret resolver-secret hist_test_token",
                    "operation": "test",
                    "details": {"nested": "resolver-secret"},
                },
            )

        encoded = json.dumps(sink.events)
        self.assertNotIn("search-secret", encoded)
        self.assertNotIn("resolver-secret", encoded)
        self.assertNotIn("hist_test_token", encoded)

    def test_cancellation_emits_once_after_durable_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sink = FakeHistorianSink()
            service = _service(tmpdir, sink)
            run_id = service.storage.create_run(
                "question",
                None,
                FreshnessClass.EVERGREEN,
                "compact",
            )
            service.cancel_run(run_id)
            service.cancel_run(run_id)
            run = service.storage.get_run(run_id)

        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(
            [event["type"] for event in sink.events].count("research.run.canceled"),
            1,
        )
        _validate_manifest_events(sink.events)

    def test_a2a_and_cli_fallback_each_emit_one_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            a2a_sink = FakeHistorianSink()
            a2a_service = _service(str(Path(tmpdir) / "a2a"), a2a_sink)
            app = build_fastapi_app(a2a_service, "http://testserver")
            message = Message(
                role=Role.ROLE_USER,
                message_id="message-1",
                parts=[new_text_part("Who is the mayor of New York?", media_type="text/plain")],
            )

            async def send() -> None:
                async def inline_to_thread(function, *args, **kwargs):
                    return function(*args, **kwargs)

                with mock.patch("magpie.a2a.asyncio.to_thread", side_effect=inline_to_thread):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://testserver",
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

            asyncio.run(send())

            cli_dir = Path(tmpdir) / "cli"
            cli_dir.mkdir()
            cli_sink = FakeHistorianSink()
            cli_service = _service(str(cli_dir), cli_sink)
            config_path = cli_dir / "config.json"
            config_path.write_text(
                json.dumps({
                    "database_path": str(cli_dir / "magpie.db"),
                    "search_provider": "fake",
                    "fetch_provider": "fake",
                    "resolver_backend": "fake",
                    "weather_enabled": False,
                    "anime_enabled": False,
                    "news_enabled": False,
                }),
                encoding="utf-8",
            )
            context = SimpleNamespace(service=cli_service, storage=cli_service.storage)
            with (
                mock.patch(
                    "magpie.cli.LocalA2AClient.send",
                    side_effect=A2AUnavailableError("offline"),
                ),
                mock.patch("magpie.cli.build_app", return_value=context),
                redirect_stdout(io.StringIO()),
            ):
                exit_code = cli.main([
                    "--config", str(config_path), "ask",
                    "Who is the mayor of New York?", "--json",
                ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [event["type"] for event in a2a_sink.events].count("research.run.completed"),
            1,
        )
        self.assertEqual(
            [event["type"] for event in cli_sink.events].count("research.run.completed"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
