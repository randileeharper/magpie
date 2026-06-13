from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from magpie.config import Settings
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


if __name__ == "__main__":
    unittest.main()
