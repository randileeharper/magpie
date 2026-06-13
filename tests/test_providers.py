from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from magpie.config import Settings
from magpie.models import FreshnessClass, SearchRequest, WeatherKind
from magpie.providers.exa import ExaSearchClient
from magpie.providers.neonhail import NeonHailWeatherClient


class ExaProviderTests(unittest.TestCase):
    def _settings(self, tmpdir: str, **overrides) -> Settings:
        data = {
            "database_path": str(Path(tmpdir) / "magpie.db"),
            "search_provider": "exa",
            "fetch_provider": "fake",
        }
        data.update(overrides)
        path = Path(tmpdir) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return Settings.load(str(path))

    def test_mcp_sse_response_is_parsed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = (
                'data: {"result":{"content":[{"type":"text","text":"'
                'Title: Example\\nURL: https://example.com\\nText: Useful content here"}]}}\n\n'
            )
            return httpx.Response(200, text=body)

        with tempfile.TemporaryDirectory() as tmpdir:
            client = ExaSearchClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            results = client.search(SearchRequest(query="test", limit=5, freshness_class=FreshnessClass.EVERGREEN))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.com")
        self.assertEqual(results[0].provider, "exa_mcp")

    def test_api_fallback_is_used_when_mcp_fails(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if request.url.path.endswith("/mcp"):
                return httpx.Response(500, text="nope")
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Example",
                            "url": "https://example.com",
                            "text": "Fallback content",
                            "highlights": ["Fallback content"],
                            "author": "Example",
                            "publishedDate": "2025-01-01",
                        }
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = ExaSearchClient(
                settings=self._settings(tmpdir, search_api_key="secret"),
                transport=httpx.MockTransport(handler),
            )
            results = client.search(SearchRequest(query="test", limit=5, freshness_class=FreshnessClass.RECENT))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].provider, "exa_api")
        self.assertEqual(len(calls), 2)


class NeonHailProviderTests(unittest.TestCase):
    def test_conditions_are_formatted_without_resolver_synthesis(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v0/conditions/98230")
            return httpx.Response(200, json={
                "temperature": 71.006,
                "relativeHumidity": 65.4,
                "windSpeed": 1.62,
                "windGust": 8.064,
            })

        client = NeonHailWeatherClient(
            Settings(weather_base_url="https://weather.test/v0"),
            transport=httpx.MockTransport(handler),
        )
        report = client.get_weather("98230", WeatherKind.CONDITIONS)

        self.assertIn("Temperature: 71°F", report.answer)
        self.assertIn("Humidity: 65.4%", report.answer)
        self.assertEqual(report.reference.source_kind.value, "weather_api")

    def test_forecast_is_bounded_to_four_periods(self) -> None:
        periods = [
            {"name": f"Period {index}", "detailedForecast": f"Details {index}"}
            for index in range(6)
        ]

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"periods": periods})

        client = NeonHailWeatherClient(
            Settings(weather_base_url="https://weather.test/v0"),
            transport=httpx.MockTransport(handler),
        )
        report = client.get_weather("98230", WeatherKind.FORECAST)

        self.assertIn("Period 3", report.answer)
        self.assertNotIn("Period 4", report.answer)


if __name__ == "__main__":
    unittest.main()
