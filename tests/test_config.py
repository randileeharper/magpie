from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from magpie.config import Settings
from magpie.errors import ConfigError
from magpie.models import ResponseDetail


class SettingsTests(unittest.TestCase):
    def test_load_from_json_and_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = f"{tmpdir}/config.json"
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "http_port": 9000,
                        "resolver_include_reasoning": False,
                        "response_detail": "compact",
                    },
                    handle,
                )
            os.environ["MAGPIE_HTTP_PORT"] = "8123"
            try:
                settings = Settings.load(config_path)
            finally:
                os.environ.pop("MAGPIE_HTTP_PORT", None)

        self.assertEqual(settings.http_port, 8123)
        self.assertEqual(settings.response_detail, ResponseDetail.COMPACT)
        diagnostics = settings.sanitized_diagnostics()
        self.assertNotIn("resolver_api_key", diagnostics)
        self.assertIn("fetch_debug_log_path", diagnostics)

    def test_debug_log_paths_default_to_private_location(self) -> None:
        settings = Settings()
        self.assertNotIn("/tmp/", settings.resolver_debug_log_path)
        self.assertNotIn("/tmp/", settings.fetch_debug_log_path)
        self.assertTrue(settings.expanded_resolver_debug_log_path.as_posix().startswith(str(Path.home())))
        self.assertTrue(settings.expanded_fetch_debug_log_path.as_posix().startswith(str(Path.home())))

    def test_load_discovers_local_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "http_port": 9011,
                        "response_detail": "debug",
                    }
                ),
                encoding="utf-8",
            )
            previous_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                settings = Settings.load()
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(settings.http_port, 9011)
        self.assertEqual(settings.response_detail, ResponseDetail.DEBUG)

    def test_explicit_config_path_wins_over_discovered_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_config = Path(tmpdir) / "config.json"
            explicit_config = Path(tmpdir) / "other.json"
            local_config.write_text(json.dumps({"http_port": 7001}), encoding="utf-8")
            explicit_config.write_text(json.dumps({"http_port": 8124}), encoding="utf-8")
            previous_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                settings = Settings.load(str(explicit_config))
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(settings.http_port, 8124)
        self.assertEqual(settings.loaded_config_path, str(explicit_config.resolve()))

    def test_config_root_must_be_an_object(self) -> None:
        for content in ("[]", "null", "42", '"value"'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "config.json"
                config_path.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ConfigError, "JSON object"):
                    Settings.load(str(config_path))

    def test_news_settings_are_exposed_in_diagnostics(self) -> None:
        settings = Settings(news_enabled=True, news_digest_size=4, news_summary_max_characters=200)
        diagnostics = settings.sanitized_diagnostics()
        self.assertEqual(diagnostics["news_digest_size"], 4)
        self.assertEqual(diagnostics["news_summary_max_characters"], 200)

    def test_historian_defaults_and_token_redaction(self) -> None:
        settings = Settings()
        settings.validate()

        self.assertFalse(settings.historian_enabled)
        self.assertEqual(settings.historian_base_url, "http://127.0.0.1:8768")
        self.assertEqual(settings.historian_timeout_seconds, 5.0)
        self.assertTrue(settings.historian_verify_tls)
        self.assertEqual(settings.historian_retry_count, 2)
        diagnostics = settings.sanitized_diagnostics()
        self.assertFalse(diagnostics["has_historian_token"])
        self.assertNotIn("historian_token", diagnostics)

    def test_historian_environment_overrides_and_normalizes_url(self) -> None:
        overrides = {
            "MAGPIE_HISTORIAN_ENABLED": "true",
            "MAGPIE_HISTORIAN_BASE_URL": "https://historian.test///",
            "MAGPIE_HISTORIAN_TOKEN": "hist_secret",
            "MAGPIE_HISTORIAN_TIMEOUT_SECONDS": "2.5",
            "MAGPIE_HISTORIAN_VERIFY_TLS": "false",
            "MAGPIE_HISTORIAN_RETRY_COUNT": "4",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{}", encoding="utf-8")
            previous = {name: os.environ.get(name) for name in overrides}
            os.environ.update(overrides)
            try:
                settings = Settings.load(str(config_path))
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

        self.assertTrue(settings.historian_enabled)
        self.assertEqual(settings.historian_base_url, "https://historian.test")
        self.assertEqual(settings.historian_token, "hist_secret")
        self.assertEqual(settings.historian_timeout_seconds, 2.5)
        self.assertFalse(settings.historian_verify_tls)
        self.assertEqual(settings.historian_retry_count, 4)

    def test_enabled_historian_requires_token(self) -> None:
        with self.assertRaisesRegex(ConfigError, "historian_token"):
            Settings(historian_enabled=True).validate()

    def test_historian_bounds_are_validated(self) -> None:
        with self.assertRaisesRegex(ConfigError, "timeout"):
            Settings(historian_timeout_seconds=0).validate()
        with self.assertRaisesRegex(ConfigError, "retry"):
            Settings(historian_retry_count=-1).validate()


if __name__ == "__main__":
    unittest.main()
