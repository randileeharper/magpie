"""Crawl4AI-backed fetcher."""

from __future__ import annotations

import asyncio
import atexit
import threading
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from ..errors import DependencyError, FetchError
from ..models import FetchedSource, SourceKind


class _LoopWorker:
    """Keep a long-lived event loop alive for Crawl4AI subprocess resources.

    A process-wide singleton so that repeated ``Crawl4AIFetcher`` constructions
    (e.g. repeated ``build_app()`` calls in tests or hot-reloads in a long-lived
    worker) share one daemon thread + event loop instead of accumulating one
    per instance. Use :meth:`shared` to obtain the singleton.

    The worker also owns a lazily-created long-lived ``AsyncWebCrawler`` so that
    a research run that fetches many sources reuses a single browser process
    instead of launching/tearing one down per URL.
    """

    _singleton: _LoopWorker | None = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self.close)
        self._crawler: Any = None
        self._crawler_lock: asyncio.Lock | None = None

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Any, *, timeout: float | None = None) -> Any:
        """Schedule *coro* on the worker loop and block until it finishes.

        If *timeout* is given and the coroutine does not complete in time, the
        underlying asyncio task is cancelled and ``TimeoutError`` is raised.
        Without a timeout the call blocks indefinitely (legacy behaviour).
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if timeout is None:
            return future.result()
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def get_crawler(self, imports: dict[str, Any]) -> Any:
        """Return the long-lived ``AsyncWebCrawler``, creating it on first use.

        *imports* is the dict produced by
        :meth:`Crawl4AIFetcher._imports` — the worker stays crawl4ai-agnostic
        by receiving the classes rather than importing them itself.
        """
        if self._crawler is not None:
            return self._crawler
        if self._crawler_lock is None:
            self._crawler_lock = asyncio.Lock()
        async with self._crawler_lock:
            if self._crawler is not None:
                return self._crawler
            browser_config = imports["BrowserConfig"](verbose=False)
            crawler = imports["AsyncWebCrawler"](config=browser_config)
            await crawler.__aenter__()
            self._crawler = crawler
            return self._crawler

    async def close_crawler(self) -> None:
        """Shut down the long-lived crawler if one was created."""
        if self._crawler is not None:
            try:
                await self._crawler.__aexit__(None, None, None)
            except Exception:
                pass
            self._crawler = None

    def close(self) -> None:
        if self._loop.is_closed():
            return
        # Close the long-lived crawler before stopping the loop so the browser
        # subprocess is cleaned up. Best-effort: never block shutdown on a
        # stuck crawler.
        if self._crawler is not None:
            try:
                cleanup = asyncio.run_coroutine_threadsafe(self.close_crawler(), self._loop)
                cleanup.result(timeout=5)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)
        if not self._loop.is_closed():
            self._loop.close()

    @classmethod
    def shared(cls) -> _LoopWorker:
        """Return the process-wide loop worker, creating it once if needed."""
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton


@dataclass(slots=True)
class Crawl4AIFetcher:
    """Fetch and normalize pages using Crawl4AI."""

    settings: Settings
    _worker: _LoopWorker | None = None

    def fetch(self, url: str) -> FetchedSource:
        try:
            if self._worker is None:
                self._worker = _LoopWorker.shared()
            return self._worker.run(
                self._fetch_async(url),
                timeout=self.settings.fetch_timeout_seconds,
            )
        except DependencyError:
            raise
        except TimeoutError as exc:
            raise FetchError(
                f"Crawl4AI timed out after {self.settings.fetch_timeout_seconds}s for {url}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise FetchError(f"Crawl4AI failed for {url}: {exc}") from exc

    def doctor_check(self, live: bool = False) -> dict[str, object]:
        try:
            self._imports()
        except DependencyError as exc:
            return {"status": "error", "provider": "crawl4ai", "live": live, "message": str(exc)}
        return {
            "status": "ok",
            "provider": "crawl4ai",
            "live": live,
            "message": "Crawl4AI importable. Run `crawl4ai-setup` if browser assets are not installed yet.",
        }

    async def _fetch_async(self, url: str) -> FetchedSource:
        crawl4ai = self._imports()
        CrawlerRunConfig = crawl4ai["CrawlerRunConfig"]
        CacheMode = crawl4ai["CacheMode"]

        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            check_robots_txt=False,
            verbose=False,
        )
        # Reuse the long-lived crawler owned by the worker instead of creating
        # (and tearing down) a fresh browser process per URL.
        assert self._worker is not None
        crawler = await self._worker.get_crawler(crawl4ai)
        result = await crawler.arun(url=url, config=run_config)

        markdown = self._extract_markdown(result)
        raw_html = getattr(result, "html", None)
        cleaned_html = getattr(result, "cleaned_html", None)
        text = markdown or cleaned_html or raw_html
        if not isinstance(text, str) or not text.strip():
            raise FetchError(f"Crawl4AI returned no usable content for {url}.")

        return FetchedSource(
            url=url,
            title=getattr(result, "title", "") or url,
            site_name=None,
            text=text.strip(),
            published_at=None,
            metadata={"success": getattr(result, "success", None)},
            markdown=markdown,
            raw_html=raw_html,
            retrieved_via="crawl4ai",
            source_kind=SourceKind.PAGE_FETCH,
        )

    def _extract_markdown(self, result: Any) -> str | None:
        markdown = getattr(result, "markdown", None)
        if markdown is None:
            return None
        raw_markdown = getattr(markdown, "raw_markdown", None)
        if isinstance(raw_markdown, str) and raw_markdown.strip():
            return raw_markdown.strip()
        if isinstance(markdown, str) and markdown.strip():
            return markdown.strip()
        return None

    def _imports(self) -> dict[str, Any]:
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
        except ImportError as exc:
            raise DependencyError(
                "Crawl4AI is unavailable. Install `crawl4ai` and run `crawl4ai-setup` before using the real fetcher."
            ) from exc
        return {
            "AsyncWebCrawler": AsyncWebCrawler,
            "BrowserConfig": BrowserConfig,
            "CrawlerRunConfig": CrawlerRunConfig,
            "CacheMode": CacheMode,
        }
