"""Failure-mode tests for provider HTTP and parse error branches.

These tests backfill the branches called out in issue #93. They use
``httpx.MockTransport`` or simple fake result objects so no network access is
required. Each test exercises a real guard in the provider source rather than
a copy of its logic.
"""

import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import httpx

from magpie.config import Settings
from magpie.errors import (
    AnimeError,
    FetchError,
    NewsError,
    ResolverError,
    SearchError,
    WeatherError,
)
from magpie.models import (
    AnimeField,
    FreshnessClass,
    NewsCategory,
    NewsRequest,
    NewsRequestKind,
    NewsTimeScope,
    SearchRequest,
    WeatherKind,
)
from magpie.providers.anilist import AniListClient
from magpie.providers.crawl4ai_fetcher import Crawl4AIFetcher, _LoopWorker
from magpie.providers.exa import ExaSearchClient
from magpie.providers.neonhail import NeonHailWeatherClient
from magpie.providers.news_rss import NewsRSSClient
from magpie.providers.openai_compatible import OpenAICompatibleResolverClient


def _settings_from_json(tmpdir: str, **overrides: object) -> Settings:
    """Build Settings from a JSON config file so env-var defaults are avoided."""
    data: dict[str, object] = {
        "database_path": str(Path(tmpdir) / "magpie.db"),
        "search_provider": "exa",
        "fetch_provider": "fake",
    }
    data.update(overrides)
    path = Path(tmpdir) / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return Settings.load(str(path))


def _feed_registry(tmpdir: str, url: str = "https://feeds.test/feed") -> str:
    path = Path(tmpdir) / "feeds.json"
    path.write_text(
        json.dumps([
            {"id": "ars-ai", "name": "Test Feed", "url": url, "categories": ["ai"], "enabled": True},
            {"id": "techcrunch-ai", "name": "Off", "url": "https://techcrunch.com/feed/", "categories": ["ai"], "enabled": False},
        ]),
        encoding="utf-8",
    )
    return str(path)


@contextmanager
def _reset_loop_worker():
    """Ensure the _LoopWorker singleton does not leak between Crawl4AI tests."""
    _LoopWorker._singleton = None
    try:
        yield
    finally:
        worker = _LoopWorker._singleton
        if worker is not None:
            worker.close()
        _LoopWorker._singleton = None


# ---------------------------------------------------------------------------
# 1. AniList – GraphQL errors field (anilist.py:272-273)
# ---------------------------------------------------------------------------

class AniListGraphQLErrorTests(unittest.TestCase):
    def _client(self, handler) -> AniListClient:
        return AniListClient(
            Settings(anime_base_url="https://anilist.test"),
            httpx.MockTransport(handler),
        )

    def test_graphql_errors_field_raises_anime_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "data": {"Page": {"media": []}},
                "errors": [{"message": "Internal server error"}],
            })

        client = self._client(handler)
        with self.assertRaises(AnimeError) as ctx:
            client.search_anime("anything")
        self.assertIn("GraphQL", str(ctx.exception))

    def test_graphql_errors_on_get_anime_info_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "data": {"Media": None},
                "errors": [{"message": "Not found"}],
            })

        client = self._client(handler)
        with self.assertRaises(AnimeError):
            client.get_anime_info(99999, [AnimeField.DESCRIPTION])

    def test_http_error_raises_anime_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        client = self._client(handler)
        with self.assertRaises(AnimeError):
            client.search_anime("test")


# ---------------------------------------------------------------------------
# 2. AniList – malformed response shapes (anilist.py:115, 185)
# ---------------------------------------------------------------------------

class AniListMalformedResponseTests(unittest.TestCase):
    def _client(self, handler) -> AniListClient:
        return AniListClient(
            Settings(anime_base_url="https://anilist.test"),
            httpx.MockTransport(handler),
        )

    def test_get_anime_info_with_null_media_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": {"Media": None}})

        client = self._client(handler)
        with self.assertRaises(AnimeError) as ctx:
            client.get_anime_info(1, [AnimeField.DESCRIPTION])
        self.assertIn("did not return", str(ctx.exception))

    def test_get_anime_info_with_string_media_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": {"Media": "unexpected"}})

        client = self._client(handler)
        with self.assertRaises(AnimeError):
            client.get_anime_info(1, [AnimeField.EPISODES])

    def test_get_credits_with_null_media_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": {"Media": None}})

        client = self._client(handler)
        with self.assertRaises(AnimeError) as ctx:
            client.get_credits(1)
        self.assertIn("did not return", str(ctx.exception))

    def test_search_skips_non_dict_media_items(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "data": {"Page": {"media": [
                    "not a dict",
                    None,
                    42,
                    {"id": 1, "title": {"english": "OK Anime"}, "format": "TV", "seasonYear": 2024},
                ]}}
            })

        client = self._client(handler)
        results = client.search_anime("test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].anime_id, 1)

    def test_get_anime_info_raises_when_all_fields_empty(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": {"Media": {
                "id": 1,
                "title": {"english": "Empty Anime"},
                "episodes": None,
            }}})

        client = self._client(handler)
        with self.assertRaises(AnimeError) as ctx:
            client.get_anime_info(1, [AnimeField.EPISODES])
        self.assertIn("none of the requested", str(ctx.exception))


# ---------------------------------------------------------------------------
# 3. NeonHail – raise_for_status failures and empty-conditions path
# ---------------------------------------------------------------------------

class NeonHailFailureTests(unittest.TestCase):
    def _client(self, handler) -> NeonHailWeatherClient:
        return NeonHailWeatherClient(
            Settings(weather_base_url="https://weather.test/v0"),
            httpx.MockTransport(handler),
        )

    def test_http_404_raises_weather_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        client = self._client(handler)
        with self.assertRaises(WeatherError) as ctx:
            client.get_weather("98230", WeatherKind.CONDITIONS)
        self.assertIn("weather request failed", str(ctx.exception))

    def test_http_500_raises_weather_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        client = self._client(handler)
        with self.assertRaises(WeatherError):
            client.get_weather("98230", WeatherKind.FORECAST)

    def test_empty_conditions_raises_weather_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "temperature": None,
                "relativeHumidity": None,
                "windSpeed": None,
                "windGust": None,
                "textDescription": "",
            })

        client = self._client(handler)
        with self.assertRaises(WeatherError) as ctx:
            client.get_weather("98230", WeatherKind.CONDITIONS)
        self.assertIn("no usable", str(ctx.exception))

    def test_empty_forecast_periods_raises_weather_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"periods": []})

        client = self._client(handler)
        with self.assertRaises(WeatherError) as ctx:
            client.get_weather("98230", WeatherKind.FORECAST)
        self.assertIn("no forecast periods", str(ctx.exception))

    def test_forecast_periods_not_list_raises_weather_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"periods": "oops"})

        client = self._client(handler)
        with self.assertRaises(WeatherError):
            client.get_weather("98230", WeatherKind.FORECAST)

    def test_connect_error_raises_weather_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        client = self._client(handler)
        with self.assertRaises(WeatherError):
            client.get_weather("98230", WeatherKind.CONDITIONS)


# ---------------------------------------------------------------------------
# 4. Exa – malformed / empty MCP SSE body (exa.py:159-219)
# ---------------------------------------------------------------------------

class ExaMalformedSSETests(unittest.TestCase):
    def _client(self, tmpdir: str, handler) -> ExaSearchClient:
        return ExaSearchClient(
            settings=_settings_from_json(tmpdir, search_transport="mcp_only"),
            transport=httpx.MockTransport(handler),
        )

    def _search(self, client: ExaSearchClient) -> list:
        return client.search(SearchRequest(query="test", limit=5, freshness_class=FreshnessClass.EVERGREEN))

    def test_mcp_sse_with_error_field_raises_search_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            body = 'data: {"error": {"message": "Rate limit exceeded"}}\n\n'
            return httpx.Response(200, text=body)

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir, handler)
            with self.assertRaises(SearchError) as ctx:
                self._search(client)
        self.assertIn("Rate limit", str(ctx.exception))

    def test_mcp_sse_with_is_error_flag_raises_search_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            body = (
                'data: {"result":{"isError":true,'
                '"content":[{"type":"text","text":"Tool execution failed"}]}}\n\n'
            )
            return httpx.Response(200, text=body)

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir, handler)
            with self.assertRaises(SearchError) as ctx:
                self._search(client)
        self.assertIn("Tool execution failed", str(ctx.exception))

    def test_mcp_sse_with_empty_content_raises_search_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            body = 'data: {"result":{"content":[]}}\n\n'
            return httpx.Response(200, text=body)

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir, handler)
            with self.assertRaises(SearchError) as ctx:
                self._search(client)
        self.assertIn("empty content", str(ctx.exception))

    def test_mcp_sse_with_unparseable_body_raises_search_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not valid sse or json ~~~")

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir, handler)
            with self.assertRaises(SearchError) as ctx:
                self._search(client)
        self.assertIn("unreadable", str(ctx.exception))


# ---------------------------------------------------------------------------
# 5. OpenAI-compatible resolver – HTTP 4xx/5xx responses
# ---------------------------------------------------------------------------

class OpenAICompatibleHTTPErrorTests(unittest.TestCase):
    def _settings(self, tmpdir: str) -> Settings:
        return Settings(
            resolver_base_url="https://llm.test",
            resolver_model="test-model",
            resolver_debug_log_path=str(Path(tmpdir) / "resolver.log"),
        )

    def _client(self, tmpdir: str, handler) -> OpenAICompatibleResolverClient:
        return OpenAICompatibleResolverClient(
            settings=self._settings(tmpdir),
            transport=httpx.MockTransport(handler),
        )

    def test_http_401_raises_resolver_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="Unauthorized")

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir, handler)
            with self.assertRaises(ResolverError) as ctx:
                client.route_request("what is the weather?")
        self.assertIn("401", str(ctx.exception))

    def test_http_500_raises_resolver_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir, handler)
            with self.assertRaises(ResolverError) as ctx:
                client.route_request("what is the weather?")
        self.assertIn("500", str(ctx.exception))

    def test_non_json_response_raises_resolver_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>Not JSON</html>")

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir, handler)
            with self.assertRaises(ResolverError) as ctx:
                client.route_request("what is the weather?")
        self.assertIn("non-JSON", str(ctx.exception))

    def test_connect_error_propagates_as_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir, handler)
            with self.assertRaises(httpx.HTTPError):
                client.route_request("what is the weather?")


# ---------------------------------------------------------------------------
# 6. NewsRSS – all-feeds-failed, max_items <= 0, LAST_7_DAYS / YESTERDAY
# ---------------------------------------------------------------------------

class NewsRSSFailureModeTests(unittest.TestCase):
    def _news_request(self, scope: NewsTimeScope = NewsTimeScope.LAST_24_HOURS) -> NewsRequest:
        return NewsRequest(NewsRequestKind.CATEGORY, NewsCategory.AI, scope)

    def test_all_feeds_failed_raises_news_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Down")

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _feed_registry(tmpdir)
            client = NewsRSSClient(
                _settings_from_json(tmpdir, news_feed_registry_path=registry, news_cache_ttl_seconds=0),
                transport=httpx.MockTransport(handler),
            )
            with self.assertRaises(NewsError) as ctx:
                client.get_news(self._news_request(), max_items=5)
        self.assertIn("All configured", str(ctx.exception))

    def test_max_items_zero_returns_empty_report_without_fetching(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, text="<?xml version='1.0'?><rss><channel><title>X</title></channel></rss>")

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _feed_registry(tmpdir)
            client = NewsRSSClient(
                _settings_from_json(tmpdir, news_feed_registry_path=registry),
                transport=httpx.MockTransport(handler),
            )
            report = client.get_news(self._news_request(), max_items=0)

        self.assertEqual(calls, [])
        self.assertEqual(report.references, [])

    def test_max_items_negative_returns_empty_report(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<?xml version='1.0'?><rss><channel><title>X</title></channel></rss>")

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _feed_registry(tmpdir)
            client = NewsRSSClient(
                _settings_from_json(tmpdir, news_feed_registry_path=registry),
                transport=httpx.MockTransport(handler),
            )
            report = client.get_news(self._news_request(), max_items=-3)

        self.assertEqual(report.references, [])

    def test_last_7_days_includes_items_within_window(self) -> None:
        tz = datetime.now().astimezone().tzinfo
        six_days_ago = (datetime.now(tz) - timedelta(days=6)).strftime("%a, %d %b %Y %H:%M:%S %z")
        feed = (
            '<?xml version="1.0"?><rss><channel><title>Feed</title>'
            f'<item><title>Recent Story</title><link>https://example.com/recent</link>'
            f'<pubDate>{six_days_ago}</pubDate><description>Within window</description></item>'
            '</channel></rss>'
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=feed)

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _feed_registry(tmpdir)
            client = NewsRSSClient(
                _settings_from_json(tmpdir, news_feed_registry_path=registry, news_cache_ttl_seconds=0),
                transport=httpx.MockTransport(handler),
            )
            report = client.get_news(
                NewsRequest(NewsRequestKind.CATEGORY, NewsCategory.AI, NewsTimeScope.LAST_7_DAYS),
                max_items=5,
            )
        self.assertIn("Recent Story", report.answer)

    def test_last_7_days_excludes_items_older_than_window(self) -> None:
        tz = datetime.now().astimezone().tzinfo
        eight_days_ago = (datetime.now(tz) - timedelta(days=8)).strftime("%a, %d %b %Y %H:%M:%S %z")
        feed = (
            '<?xml version="1.0"?><rss><channel><title>Feed</title>'
            f'<item><title>Old Story</title><link>https://example.com/old</link>'
            f'<pubDate>{eight_days_ago}</pubDate><description>Too old</description></item>'
            '</channel></rss>'
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=feed)

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _feed_registry(tmpdir)
            client = NewsRSSClient(
                _settings_from_json(tmpdir, news_feed_registry_path=registry, news_cache_ttl_seconds=0),
                transport=httpx.MockTransport(handler),
            )
            report = client.get_news(
                NewsRequest(NewsRequestKind.CATEGORY, NewsCategory.AI, NewsTimeScope.LAST_7_DAYS),
                max_items=5,
            )
        self.assertNotIn("Old Story", report.answer)
        self.assertEqual(report.references, [])

    def test_yesterday_window_includes_yesterday_items(self) -> None:
        tz = datetime.now().astimezone().tzinfo
        yesterday = (
            datetime.now(tz).replace(hour=12, minute=0, second=0, microsecond=0)
            - timedelta(days=1)
        ).strftime("%a, %d %b %Y %H:%M:%S %z")
        feed = (
            '<?xml version="1.0"?><rss><channel><title>Feed</title>'
            f'<item><title>Yesterday Story</title><link>https://example.com/yesterday</link>'
            f'<pubDate>{yesterday}</pubDate><description>From yesterday</description></item>'
            '</channel></rss>'
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=feed)

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _feed_registry(tmpdir)
            client = NewsRSSClient(
                _settings_from_json(tmpdir, news_feed_registry_path=registry, news_cache_ttl_seconds=0),
                transport=httpx.MockTransport(handler),
            )
            report = client.get_news(
                NewsRequest(NewsRequestKind.CATEGORY, NewsCategory.AI, NewsTimeScope.YESTERDAY),
                max_items=5,
            )
        self.assertIn("Yesterday Story", report.answer)

    def test_yesterday_window_excludes_today_items(self) -> None:
        tz = datetime.now().astimezone().tzinfo
        now = datetime.now(tz).strftime("%a, %d %b %Y %H:%M:%S %z")
        feed = (
            '<?xml version="1.0"?><rss><channel><title>Feed</title>'
            f'<item><title>Today Story</title><link>https://example.com/today</link>'
            f'<pubDate>{now}</pubDate><description>Published today</description></item>'
            '</channel></rss>'
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=feed)

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = _feed_registry(tmpdir)
            client = NewsRSSClient(
                _settings_from_json(tmpdir, news_feed_registry_path=registry, news_cache_ttl_seconds=0),
                transport=httpx.MockTransport(handler),
            )
            report = client.get_news(
                NewsRequest(NewsRequestKind.CATEGORY, NewsCategory.AI, NewsTimeScope.YESTERDAY),
                max_items=5,
            )
        self.assertNotIn("Today Story", report.answer)
        self.assertEqual(report.references, [])


# ---------------------------------------------------------------------------
# 7. Crawl4AI – _extract_markdown logic and no-usable-content FetchError
# ---------------------------------------------------------------------------

class Crawl4AIExtractMarkdownTests(unittest.TestCase):
    def _fetcher(self) -> Crawl4AIFetcher:
        with tempfile.TemporaryDirectory() as tmpdir:
            return Crawl4AIFetcher(
                settings=Settings(
                    database_path=str(Path(tmpdir) / "magpie.db"),
                    fetch_provider="crawl4ai",
                )
            )

    def test_extract_markdown_prefers_raw_markdown_attribute(self) -> None:
        fetcher = self._fetcher()
        result = MagicMock()
        result.markdown = MagicMock()
        result.markdown.raw_markdown = "  # Real markdown\n\nContent here.  "
        self.assertEqual(fetcher._extract_markdown(result), "# Real markdown\n\nContent here.")

    def test_extract_markdown_falls_back_to_markdown_string(self) -> None:
        fetcher = self._fetcher()
        result = MagicMock()
        result.markdown = "  Fallback markdown.  "
        self.assertEqual(fetcher._extract_markdown(result), "Fallback markdown.")

    def test_extract_markdown_returns_none_when_markdown_is_none(self) -> None:
        fetcher = self._fetcher()
        result = MagicMock()
        result.markdown = None
        self.assertIsNone(fetcher._extract_markdown(result))

    def test_extract_markdown_returns_none_when_all_empty(self) -> None:
        fetcher = self._fetcher()
        result = MagicMock()
        result.markdown = MagicMock()
        result.markdown.raw_markdown = "   "
        self.assertIsNone(fetcher._extract_markdown(result))


class Crawl4AINoUsableContentTests(unittest.TestCase):
    """The no-usable-content FetchError guard lives inside _fetch_async.

    Rather than reconstruct the guard logic, these tests stub _imports() to
    return a fake AsyncWebCrawler whose arun() yields a result with no usable
    content, so the real guard at crawl4ai_fetcher.py:109-110 is exercised.
    """

    @staticmethod
    def _empty_result() -> MagicMock:
        result = MagicMock()
        result.markdown = None
        result.html = None
        result.cleaned_html = None
        result.title = ""
        result.success = True
        return result

    def _make_fake_imports(self, result: MagicMock):
        class _FakeCrawler:
            def __init__(self, config=None):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return False

            async def arun(self, url, config):
                return result

        def fake_imports(_self):
            return {
                "AsyncWebCrawler": _FakeCrawler,
                "BrowserConfig": lambda **kw: object(),
                "CrawlerRunConfig": lambda **kw: object(),
                "CacheMode": MagicMock(),
            }

        return fake_imports

    def _fetcher(self) -> Crawl4AIFetcher:
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = Crawl4AIFetcher(
                settings=Settings(
                    database_path=str(Path(tmpdir) / "magpie.db"),
                    fetch_provider="crawl4ai",
                )
            )
        # _worker is a slot, so assignment is allowed. Replace the shared
        # singleton with a mock that drives coroutines synchronously.
        fetcher._worker = MagicMock()
        fetcher._worker.run = lambda coro, timeout=None: _run_async(coro)
        # _fetch_async now asks the worker for the long-lived crawler. Mirror
        # the real get_crawler() so the _FakeCrawler from the patched imports
        # is entered once and reused.
        async def _get_crawler(imports: dict[str, Any]) -> Any:
            crawler = imports["AsyncWebCrawler"](config=imports["BrowserConfig"](verbose=False))
            await crawler.__aenter__()
            return crawler
        fetcher._worker.get_crawler = _get_crawler
        return fetcher

    def test_fetch_raises_fetch_error_when_no_usable_content(self) -> None:
        with _reset_loop_worker():
            fetcher = self._fetcher()
            fake_imports = self._make_fake_imports(self._empty_result())
            with patch.object(Crawl4AIFetcher, "_imports", fake_imports):
                with self.assertRaises(FetchError) as ctx:
                    fetcher.fetch("https://example.com/empty")
        self.assertIn("no usable content", str(ctx.exception))
        self.assertIn("https://example.com/empty", str(ctx.exception))

    def test_fetch_raises_fetch_error_when_only_whitespace(self) -> None:
        result = MagicMock()
        result.markdown = MagicMock()
        result.markdown.raw_markdown = "   \n   "
        result.html = "  "
        result.cleaned_html = None
        result.title = ""
        result.success = True

        with _reset_loop_worker():
            fetcher = self._fetcher()
            fake_imports = self._make_fake_imports(result)
            with patch.object(Crawl4AIFetcher, "_imports", fake_imports):
                with self.assertRaises(FetchError):
                    fetcher.fetch("https://example.com/blank")


class Crawl4AITimeoutTests(unittest.TestCase):
    """A stuck crawl must time out instead of blocking the caller forever."""

    def test_stuck_coroutine_raises_fetch_error_on_timeout(self) -> None:
        import asyncio

        started = threading.Event()

        async def _stuck_coro(*_args: object) -> object:
            started.set()
            await asyncio.Event().wait()  # never completes
            return MagicMock()

        with _reset_loop_worker():
            with tempfile.TemporaryDirectory() as tmpdir:
                fetcher = Crawl4AIFetcher(
                    settings=Settings(
                        database_path=str(Path(tmpdir) / "magpie.db"),
                        fetch_provider="crawl4ai",
                        fetch_timeout_seconds=0.25,
                    )
                )
            # Use the real shared _LoopWorker so future.result(timeout=...) runs.
            fetcher._worker = _LoopWorker.shared()
            # Patch _fetch_async on the class (slots=True forbids instance
            # assignment) to a coroutine that never completes, so only the
            # timeout path is exercised.
            with patch.object(Crawl4AIFetcher, "_fetch_async", _stuck_coro):
                with self.assertRaises(FetchError) as ctx:
                    fetcher.fetch("https://example.com/hang")
        self.assertTrue(started.is_set())
        self.assertIn("timed out", str(ctx.exception))
        self.assertIn("https://example.com/hang", str(ctx.exception))


class Crawl4AICrawlerReuseTests(unittest.TestCase):
    """The long-lived crawler must be created once and reused across fetches."""

    def test_crawler_is_entered_once_and_reused_across_fetches(self) -> None:
        enter_count = 0

        def fake_imports_factory(result: MagicMock):
            class _FakeCrawler:
                def __init__(self, config=None):
                    self.arun_count = 0

                async def __aenter__(self):
                    nonlocal enter_count
                    enter_count += 1
                    return self

                async def __aexit__(self, *_exc):
                    return False

                async def arun(self, url, config):
                    self.arun_count += 1
                    return result

            def fake_imports(_self):
                return {
                    "AsyncWebCrawler": _FakeCrawler,
                    "BrowserConfig": lambda **kw: object(),
                    "CrawlerRunConfig": lambda **kw: object(),
                    "CacheMode": MagicMock(),
                }

            return fake_imports

        result = MagicMock()
        result.markdown = "# content"
        result.html = None
        result.cleaned_html = None
        result.title = "title"
        result.success = True

        with _reset_loop_worker():
            with tempfile.TemporaryDirectory() as tmpdir:
                fetcher = Crawl4AIFetcher(
                    settings=Settings(
                        database_path=str(Path(tmpdir) / "magpie.db"),
                        fetch_provider="crawl4ai",
                    )
                )
            fetcher._worker = _LoopWorker.shared()
            fake_imports = fake_imports_factory(result)
            with patch.object(Crawl4AIFetcher, "_imports", fake_imports):
                fetcher.fetch("https://example.com/a")
                fetcher.fetch("https://example.com/b")

        # One browser/crawler entered, reused for both fetches.
        self.assertEqual(enter_count, 1)


def _run_async(coro):
    """Drive a coroutine to completion without a running event loop."""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


if __name__ == "__main__":
    unittest.main()
