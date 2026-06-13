from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from magpie.config import Settings
from magpie.models import AnimeField, FreshnessClass, SearchRequest, WeatherKind
from magpie.providers.anilist import AniListClient
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


class AniListProviderTests(unittest.TestCase):
    def test_title_search_uses_jikan_only_as_discovery_fallback(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            if request.method == "GET":
                return httpx.Response(200, json={"data": [{
                    "title": "Raise wa Tanin ga Ii",
                    "title_english": "Yakuza Fiancé: Raise wa Tanin ga Ii",
                    "synopsis": "must not be retained",
                }]})
            search = json.loads(request.content)["variables"]["search"]
            if search == "Raise wa Tanin ga Ii":
                return httpx.Response(200, json={"data": {"Page": {"media": [{
                    "id": 170468,
                    "title": {"english": "Yakuza Fiancé", "romaji": "Raise wa Tanin ga Ii", "native": "来世は他人がいい"},
                    "format": "TV",
                    "seasonYear": 2024,
                }]}}})
            return httpx.Response(200, json={"data": {"Page": {"media": []}}})

        client = AniListClient(
            Settings(
                anime_base_url="https://anilist.test",
                anime_title_search_fallback_url="https://jikan.test/anime",
            ),
            httpx.MockTransport(handler),
        )
        candidates = client.search_anime("Yakuza Fiancée")

        self.assertEqual(candidates[0].anime_id, 170468)
        self.assertTrue(any("jikan.test" in call for call in calls))

    def test_media_info_requests_and_returns_only_compact_fields(self) -> None:
        captured_query = ""

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_query
            captured_query = json.loads(request.content)["query"]
            return httpx.Response(200, json={"data": {"Media": {
                "id": 170468,
                "title": {"english": "Yakuza Fiancé", "romaji": "Raise wa Tanin ga Ii"},
                "description": "A <b>yakuza romance</b>.",
                "format": "TV",
                "status": "FINISHED",
                "episodes": 12,
                "seasonYear": 2024,
            }}})

        client = AniListClient(Settings(anime_base_url="https://anilist.test"), httpx.MockTransport(handler))
        report = client.get_anime_info(
            170468,
            [AnimeField.DESCRIPTION, AnimeField.FORMAT, AnimeField.SEASON_YEAR, AnimeField.EPISODES, AnimeField.STATUS],
        )

        self.assertEqual(
            report.answer,
            "Yakuza Fiancé\nA yakuza romance.\nFormat: Tv\nSeason year: 2024\nEpisodes: 12\nStatus: Finished",
        )
        self.assertNotIn("score", captured_query)
        self.assertNotIn("image", captured_query)
        self.assertEqual(report.reference.source_kind.value, "anilist_api")

    def test_lookup_requests_only_selected_field(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            query = json.loads(request.content)["query"]
            self.assertIn("episodes", query)
            self.assertNotIn("description", query)
            self.assertNotIn("averageScore", query)
            return httpx.Response(200, json={"data": {"Media": {
                "id": 1,
                "title": {"english": "Bookworm Season 2", "romaji": "Honzuki 2"},
                "episodes": 12,
            }}})

        client = AniListClient(Settings(anime_base_url="https://anilist.test"), httpx.MockTransport(handler))
        report = client.get_anime_info(1, [AnimeField.EPISODES])

        self.assertEqual(report.answer, "Bookworm Season 2\nEpisodes: 12")

    def test_schedule_converts_to_system_timezone_and_omits_metadata(self) -> None:
        previous_tz = os.environ.get("TZ")
        os.environ["TZ"] = "America/Los_Angeles"
        time.tzset()
        try:
            def handler(request: httpx.Request) -> httpx.Response:
                body = json.loads(request.content)
                self.assertNotIn("description", body["query"])
                return httpx.Response(200, json={"data": {"Page": {"airingSchedules": [{
                    "airingAt": 1781395200,
                    "episode": 3,
                    "media": {"id": 1, "title": {"english": "Example Anime", "romaji": "Example"}},
                }]}}})

            client = AniListClient(
                Settings(anime_base_url="https://anilist.test"),
                httpx.MockTransport(handler),
            )
            report = client.get_daily_schedule()
        finally:
            if previous_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = previous_tz
            time.tzset()

        self.assertIn("Example Anime, episode 3", report.answer)
        self.assertNotIn("anilist:", report.answer)


if __name__ == "__main__":
    unittest.main()
