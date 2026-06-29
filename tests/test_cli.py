from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest import mock

from magpie import cli
from magpie.app import build_app
from magpie.a2a import LocalA2AClient
from magpie.models import ResearchRequest
from magpie.errors import A2ARequestError, A2AUnavailableError, FetchError, SearchError


class CLITests(unittest.TestCase):
    def test_local_a2a_client_builds_sdk_message_with_message_id(self) -> None:
        captured: dict[str, object] = {}

        class FakeMessage:
            def __init__(self, **kwargs) -> None:
                captured["message_kwargs"] = kwargs
                self.metadata = {}

        class FakeResponse:
            def HasField(self, _name: str) -> bool:
                return False

        class FakeClient:
            async def send_message(self, _request):
                if False:
                    yield None
                return
                yield  # pragma: no cover

            async def close(self) -> None:
                return None

        class FakeFactory:
            def __init__(self, _config) -> None:
                return None

            async def create_from_url(self, _url: str):
                return FakeClient()

        sdk = {
            "Message": FakeMessage,
            "Role": mock.Mock(ROLE_USER="ROLE_USER"),
            "SendMessageRequest": lambda **kwargs: kwargs,
            "ClientConfig": lambda **kwargs: kwargs,
            "ClientFactory": lambda config: FakeFactory(config),
            "AgentCardResolutionError": RuntimeError,
            "Task": object,
            "TaskState": mock.Mock(),
            "new_text_part": lambda text, media_type=None: {"text": text, "media_type": media_type},
        }

        client = LocalA2AClient("http://127.0.0.1:8766")
        with mock.patch.multiple("magpie.a2a", **sdk):
            with self.assertRaises(A2ARequestError):
                client.send(ResearchRequest(question="Who is the mayor of Seattle?"))

        self.assertEqual(captured["message_kwargs"]["role"], "ROLE_USER")
        self.assertTrue(captured["message_kwargs"]["message_id"])

    def test_local_a2a_client_uses_sdk_message_and_task_payload(self) -> None:
        class FakePart:
            def __init__(self, *, data=None, text: str | None = None) -> None:
                self.data = data
                self.text = text or ""

            def HasField(self, name: str) -> bool:
                if name == "data":
                    return self.data is not None
                if name == "text":
                    return bool(self.text)
                return False

        class FakeArtifact:
            def __init__(self, parts) -> None:
                self.parts = parts

        class FakeStatus:
            def __init__(self) -> None:
                self.state = "TASK_STATE_COMPLETED"

            def HasField(self, _name: str) -> bool:
                return False

        class FakeTask:
            def __init__(self) -> None:
                self.id = "task-123"
                self.artifacts = [FakeArtifact([FakePart(data={"status": "ok", "run_id": "task-123"})])]
                self.status = FakeStatus()
                self.history = []

        client = LocalA2AClient("http://127.0.0.1:8766")
        with mock.patch("magpie.a2a.json_format.MessageToDict", return_value={"status": "ok", "run_id": "task-123"}):
            result = client._task_to_payload(FakeTask(), mock.Mock(
                TASK_STATE_COMPLETED="TASK_STATE_COMPLETED",
                TASK_STATE_FAILED="TASK_STATE_FAILED",
                TASK_STATE_REJECTED="TASK_STATE_REJECTED",
                TASK_STATE_CANCELED="TASK_STATE_CANCELED",
            ))
        self.assertEqual(result["run_id"], "task-123")

    def test_local_a2a_client_raises_for_non_terminal_task(self) -> None:
        class FakeStatus:
            def __init__(self) -> None:
                self.state = "TASK_STATE_WORKING"

            def HasField(self, _name: str) -> bool:
                return False

        class FakeTask:
            def __init__(self) -> None:
                self.id = "task-123"
                self.artifacts = []
                self.status = FakeStatus()
                self.history = []

        client = LocalA2AClient("http://127.0.0.1:8766")
        with self.assertRaises(A2ARequestError):
            client._task_to_payload(FakeTask(), mock.Mock(
                TASK_STATE_COMPLETED="TASK_STATE_COMPLETED",
                TASK_STATE_FAILED="TASK_STATE_FAILED",
                TASK_STATE_REJECTED="TASK_STATE_REJECTED",
                TASK_STATE_CANCELED="TASK_STATE_CANCELED",
            ))

    def test_post_acceptance_a2a_failure_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"database_path": str(Path(tmpdir) / "magpie.db")}))
            with mock.patch(
                "magpie.cli.LocalA2AClient.send",
                side_effect=A2ARequestError("accepted request failed"),
            ), mock.patch("magpie.cli.build_app") as build_app, redirect_stderr(io.StringIO()):
                exit_code = cli.main([
                    "--config", str(config_path), "ask", "question", "--json",
                ])
        self.assertEqual(exit_code, 1)
        build_app.assert_not_called()

    def test_ask_outputs_json_via_a2a(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database_path": str(Path(tmpdir) / "magpie.db"),
                        "search_provider": "fake",
                        "fetch_provider": "fake",
                        "resolver_backend": "fake",
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with mock.patch(
                "magpie.cli.LocalA2AClient.send",
                return_value={
                    "status": "ok",
                    "run_id": "run-123",
                    "summary": "summary",
                    "answer": "answer",
                    "stop_reason": "needed_new_search",
                    "references": [],
                },
            ) as mock_send, redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "--config",
                        str(config_path),
                        "ask",
                        "Who is the mayor of New York?",
                        "--json",
                    ]
                )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "ok")
        mock_send.assert_called_once()

    def test_a2a_failure_falls_back_to_direct_service(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database_path": str(Path(tmpdir) / "magpie.db"),
                        "search_provider": "fake",
                        "fetch_provider": "fake",
                        "resolver_backend": "fake",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "magpie.cli.LocalA2AClient.send",
                side_effect=A2AUnavailableError("boom"),
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = cli.main(
                        [
                            "--config",
                            str(config_path),
                            "ask",
                            "Who is the mayor of New York?",
                            "--json",
                        ]
                    )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "ok")

    def test_serve_invokes_uvicorn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database_path": str(Path(tmpdir) / "magpie.db"),
                        "search_provider": "fake",
                        "fetch_provider": "fake",
                        "resolver_backend": "fake",
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("uvicorn.run") as mock_run:
                exit_code = cli.main(
                    [
                        "--config",
                        str(config_path),
                        "serve",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "9001",
                    ]
                )
        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once()

    def test_doctor_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database_path": str(Path(tmpdir) / "magpie.db"),
                        "search_provider": "fake",
                        "fetch_provider": "fake",
                        "resolver_backend": "fake",
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "--config",
                        str(config_path),
                        "doctor",
                        "--json",
                    ]
                )
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "ok")
        self.assertIn("news", payload)

    def test_human_output_renders_multiline_answer_on_own_block(self) -> None:
        rendered = cli._human_output(
            {
                "run_id": "run-123",
                "summary": "short summary",
                "answer": "Line 1\nLine 2",
                "references": [],
            }
        )
        self.assertIn("summary: short summary", rendered)
        self.assertIn("answer:\nLine 1\nLine 2", rendered)

    def test_clear_cache_deletes_database_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "magpie.db"
            database_path.write_text("junk", encoding="utf-8")
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database_path": str(database_path),
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "--config",
                        str(config_path),
                        "clear-cache",
                        "--json",
                    ]
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["deleted"], True)
        self.assertFalse(database_path.exists())

    def test_clear_cache_reports_missing_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            database_path = Path(tmpdir) / "magpie.db"
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database_path": str(database_path),
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "--config",
                        str(config_path),
                        "clear-cache",
                        "--json",
                    ]
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["deleted"], False)

    def test_config_init_writes_default_config(self) -> None:
        from importlib.resources import files

        expected = files("magpie").joinpath("config.example.json").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "config.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli.main(["config", "init", "--path", str(target)])
            self.assertEqual(exit_code, 0)
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(encoding="utf-8"), expected)
            self.assertIn("Wrote config", stdout.getvalue())

    def test_config_init_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "config.json"
            target.write_text("existing", encoding="utf-8")
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = cli.main(["config", "init", "--path", str(target)])
            self.assertEqual(exit_code, 2)
            self.assertIn("--force", stderr.getvalue())
            self.assertEqual(target.read_text(encoding="utf-8"), "existing")

    def test_config_init_force_overwrites(self) -> None:
        from importlib.resources import files

        expected = files("magpie").joinpath("config.example.json").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "config.json"
            target.write_text("existing", encoding="utf-8")
            exit_code = cli.main(["config", "init", "--path", str(target), "--force"])
            self.assertEqual(exit_code, 0)
            self.assertEqual(target.read_text(encoding="utf-8"), expected)

    def test_config_init_print_outputs_template(self) -> None:
        from importlib.resources import files

        expected = files("magpie").joinpath("config.example.json").read_text(encoding="utf-8")
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main(["config", "init", "--print"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), expected)

    def test_config_init_print_does_not_write_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    cli.main(["config", "init", "--print"])
            finally:
                os.chdir(previous_cwd)
        self.assertFalse((Path(tmpdir) / "config.json").exists())

    def test_config_init_print_conflicts_with_path(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["config", "init", "--print", "--path", "/tmp/foo"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--print", stderr.getvalue())

    def test_config_init_print_conflicts_with_force(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["config", "init", "--print", "--force"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--print", stderr.getvalue())

    def test_config_init_print_conflicts_with_path_and_force(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(
                    ["config", "init", "--print", "--path", "/tmp/foo", "--force"]
                )
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("--print", stderr.getvalue())

    def test_config_path_prints_loaded_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"database_path": str(Path(tmpdir) / "magpie.db")}), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = cli.main(["--config", str(config_path), "config", "path"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), str(config_path.resolve()))

    def test_config_path_reports_none_when_no_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = Path.cwd()
            previous_home = os.environ.get("HOME")
            os.chdir(tmpdir)
            os.environ["HOME"] = tmpdir
            try:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = cli.main(["config", "path"])
            finally:
                os.chdir(previous_cwd)
                if previous_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = previous_home
        self.assertEqual(exit_code, 0)
        self.assertIn("none", stdout.getvalue())

    def _tmp_config(self) -> str:
        """Write a minimal fake-provider config to a temp dir and return its path."""
        tmpdir = tempfile.mkdtemp()
        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "database_path": str(Path(tmpdir) / "magpie.db"),
                    "search_provider": "fake",
                    "fetch_provider": "fake",
                    "resolver_backend": "fake",
                }
            ),
            encoding="utf-8",
        )
        return str(config_path)

    def _fake_app(self, *, search_exc=None, fetch_exc=None):
        """An AppContext stand-in whose service raises on search/fetch."""

        class _FakeService:
            def search(self, _query, max_results=5):
                if search_exc:
                    raise search_exc
                return {"results": []}

            def fetch(self, **_kwargs):
                if fetch_exc:
                    raise fetch_exc
                return {}

            def close(self) -> None:
                pass

        class _FakeStorage:
            def close(self) -> None:
                pass

        class _FakeApp:
            service = _FakeService()
            storage = _FakeStorage()
            settings = mock.Mock()

        return _FakeApp()

    def test_search_known_error_returns_exit_2(self) -> None:
        config_path = Path(self._tmp_config())
        with mock.patch(
            "magpie.app.build_app",
            return_value=self._fake_app(search_exc=SearchError("provider unavailable")),
        ), redirect_stderr(io.StringIO()):
            exit_code = cli.main(["--config", str(config_path), "search", "query"])
        self.assertEqual(exit_code, 2)

    def test_search_unexpected_error_returns_exit_3(self) -> None:
        config_path = Path(self._tmp_config())
        with mock.patch(
            "magpie.app.build_app",
            return_value=self._fake_app(search_exc=TypeError("boom")),
        ), redirect_stderr(io.StringIO()):
            exit_code = cli.main(["--config", str(config_path), "search", "query"])
        self.assertEqual(exit_code, 3)

    def test_fetch_known_error_returns_exit_2(self) -> None:
        config_path = Path(self._tmp_config())
        with mock.patch(
            "magpie.app.build_app",
            return_value=self._fake_app(fetch_exc=FetchError("fetch failed")),
        ), redirect_stderr(io.StringIO()):
            exit_code = cli.main(["--config", str(config_path), "fetch", "https://example.com/x"])
        self.assertEqual(exit_code, 2)

    def test_fetch_non_numeric_index_returns_exit_2_without_traceback(self) -> None:
        # A common user typo: `magpie fetch abc` (neither URL nor integer)
        # must produce a clean error message + exit 2, not a traceback/exit 3.
        config_path = Path(self._tmp_config())
        stderr = io.StringIO()
        with mock.patch(
            "magpie.app.build_app", return_value=self._fake_app()
        ), redirect_stderr(stderr):
            exit_code = cli.main(["--config", str(config_path), "fetch", "not-a-url-or-int"])
        self.assertEqual(exit_code, 2)
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertIn("integer", stderr.getvalue())

    def test_build_app_warns_when_verify_tls_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database_path": str(Path(tmpdir) / "magpie.db"),
                        "search_provider": "fake",
                        "fetch_provider": "fake",
                        "resolver_backend": "fake",
                        "verify_tls": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertLogs("magpie.app", level="WARNING") as captured:
                app = build_app(str(config_path))
            try:
                self.assertFalse(app.settings.verify_tls)
                self.assertTrue(
                    any("verify_tls" in message for message in captured.output),
                    f"expected verify_tls warning, got: {captured.output}",
                )
            finally:
                app.storage.close()

    def test_build_app_does_not_warn_when_verify_tls_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database_path": str(Path(tmpdir) / "magpie.db"),
                        "search_provider": "fake",
                        "fetch_provider": "fake",
                        "resolver_backend": "fake",
                        "verify_tls": True,
                    }
                ),
                encoding="utf-8",
            )
            # assertLogs requires at least one record; assertNoLogs is the
            # correct inverse (3.10+).
            with self.assertNoLogs("magpie.app", level="WARNING"):
                app = build_app(str(config_path))
            app.storage.close()


if __name__ == "__main__":
    unittest.main()
