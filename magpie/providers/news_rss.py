"""Specialized RSS and Atom news aggregation."""

from __future__ import annotations

import json
import time
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html import unescape
from html.parser import HTMLParser
from importlib import resources
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

import feedparser
import httpx

from ..config import Settings
from ..errors import ConfigError, NewsError
from ..models import (
    NewsCategory,
    NewsItem,
    NewsReport,
    NewsRequest,
    NewsRequestKind,
    NewsTimeScope,
    Reference,
    SourceKind,
)
from ..storage import canonicalize_url
from ..text import valid_unicode


class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return " ".join(part.strip() for part in self.parts if part.strip())


@dataclass(slots=True)
class FeedDefinition:
    feed_id: str
    name: str
    url: str
    categories: list[NewsCategory]
    enabled: bool = True


class NewsRSSClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport
        self._local_tz = datetime.now().astimezone().tzinfo or UTC
        self._feeds = self._load_registry()
        self._cache: dict[str, tuple[float, list[NewsItem]]] = {}
        self._cache_lock = Lock()

    def get_news(self, request: NewsRequest, max_items: int) -> NewsReport:
        if request.kind != NewsRequestKind.CATEGORY or request.category is None:
            raise NewsError("Unsupported news request.")
        if max_items <= 0:
            return self._no_results_report(request, [])

        category_feeds = [feed for feed in self._feeds if request.category in feed.categories]
        if not category_feeds:
            raise NewsError(f"No enabled RSS feeds are configured for {request.category.value}.")

        started_at, ended_at = self._time_window(request.time_scope)
        fetched_items: list[NewsItem] = []
        warnings: list[str] = []
        failures = 0

        with ThreadPoolExecutor(max_workers=self.settings.news_fetch_concurrency) as executor:
            future_map = {
                executor.submit(self._load_feed_items, feed, request.category): feed
                for feed in category_feeds
            }
            for future in as_completed(future_map):
                feed = future_map[future]
                try:
                    items = future.result()
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    warnings.append(f"RSS feed failed for {feed.name}: {exc}")
                    continue
                fetched_items.extend(items)

        if failures == len(category_feeds):
            raise NewsError("All configured RSS feeds failed.")

        windowed: list[NewsItem] = []
        discarded_future = 0
        for item in fetched_items:
            published = self._parse_iso(item.published_at)
            if published > datetime.now(self._local_tz) + timedelta(minutes=1):
                discarded_future += 1
                continue
            if started_at <= published <= ended_at:
                windowed.append(item)
        if discarded_future:
            warnings.append(f"Discarded {discarded_future} future-dated feed items.")

        selected = self._select_items(windowed, max_items)
        if not selected:
            return self._no_results_report(request, warnings)

        references = [
            Reference(
                source_id=f"rss:{index}:{canonicalize_url(item.url)}",
                title=item.title,
                url=item.url,
                site_name=item.source_name,
                published_at=item.published_at,
                fetched_at=None,
                source_kind=SourceKind.RSS_FEED,
            )
            for index, item in enumerate(selected, start=1)
        ]
        lines = [
            (
                f"{index}. {self._format_local_time(item.published_at)} | {item.title} | {item.summary} | "
                f"{item.source_name} | {item.url}"
            )
            for index, item in enumerate(selected, start=1)
        ]
        summary = self._summary_label(request, len(selected))
        return NewsReport(summary=summary, answer="\n".join(lines), references=references, warnings=warnings)

    def doctor_check(self, live: bool = False) -> dict[str, object]:
        enabled_feeds = [feed for feed in self._feeds if feed.enabled]
        report: dict[str, object] = {
            "status": "ok",
            "provider": "rss_feed",
            "enabled": self.settings.news_enabled,
            "registry_path": self.settings.news_feed_registry_path or "builtin",
            "enabled_feed_count": len(enabled_feeds),
            "categories": sorted({category.value for feed in enabled_feeds for category in feed.categories}),
            "live": live,
        }
        if not self.settings.news_enabled:
            return report
        if not enabled_feeds:
            report["status"] = "error"
            report["error"] = "No enabled RSS feeds are configured."
            return report
        if live:
            checks: list[dict[str, object]] = []
            failures = 0
            for feed in enabled_feeds:
                try:
                    count = len(self._load_feed_items(feed, feed.categories[0]))
                    checks.append({"feed_id": feed.feed_id, "status": "ok", "item_count": count})
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    checks.append({"feed_id": feed.feed_id, "status": "error", "error": str(exc)})
            report["checks"] = checks
            if failures == len(enabled_feeds):
                report["status"] = "error"
        return report

    def _load_registry(self) -> list[FeedDefinition]:
        builtin = json.loads(resources.files("magpie").joinpath("news_feeds.json").read_text(encoding="utf-8"))
        feeds = self._parse_registry_entries(builtin)
        if self.settings.news_feed_registry_path:
            override_path = Path(self.settings.news_feed_registry_path).expanduser()
            override_entries = json.loads(override_path.read_text(encoding="utf-8"))
            override_feeds = self._parse_registry_entries(override_entries)
            merged: dict[str, FeedDefinition] = {feed.feed_id: feed for feed in feeds}
            order = [feed.feed_id for feed in feeds]
            for feed in override_feeds:
                if feed.feed_id not in merged:
                    order.append(feed.feed_id)
                merged[feed.feed_id] = feed
            feeds = [merged[feed_id] for feed_id in order]
        enabled = [feed for feed in feeds if feed.enabled]
        if not enabled:
            raise ConfigError("At least one enabled news feed is required.")
        return enabled

    def _parse_registry_entries(self, entries: Any) -> list[FeedDefinition]:
        if not isinstance(entries, list):
            raise ConfigError("News feed registry must be a JSON array.")
        seen_ids: set[str] = set()
        parsed: list[FeedDefinition] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ConfigError("News feed registry entries must be objects.")
            feed_id = valid_unicode(str(entry.get("id", "")).strip())
            name = valid_unicode(str(entry.get("name", "")).strip())
            url = valid_unicode(str(entry.get("url", "")).strip())
            enabled = bool(entry.get("enabled", True))
            raw_categories = entry.get("categories", [])
            if not feed_id or not name or not url:
                raise ConfigError("News feed registry entries require id, name, and url.")
            if feed_id in seen_ids:
                raise ConfigError(f"Duplicate news feed id: {feed_id}")
            seen_ids.add(feed_id)
            parsed_url = urlsplit(url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise ConfigError(f"Malformed news feed URL for {feed_id}: {url}")
            if not isinstance(raw_categories, list) or not raw_categories:
                raise ConfigError(f"News feed {feed_id} must declare at least one category.")
            categories: list[NewsCategory] = []
            for raw_category in raw_categories:
                try:
                    category = NewsCategory(str(raw_category))
                except ValueError as exc:
                    raise ConfigError(f"Invalid news category for {feed_id}: {raw_category}") from exc
                if category not in categories:
                    categories.append(category)
            parsed.append(FeedDefinition(feed_id, name, url, categories, enabled))
        return parsed

    def _load_feed_items(self, feed: FeedDefinition, category: NewsCategory) -> list[NewsItem]:
        cached = self._cache_get(feed.feed_id)
        if cached is not None:
            return cached
        body = self._fetch_feed_bytes(feed.url)
        parsed = feedparser.parse(body)
        items: list[NewsItem] = []
        discarded_undated = 0
        feed_name = valid_unicode(str(parsed.feed.get("title") or feed.name)).strip() or feed.name
        for entry in parsed.entries:
            published = self._entry_datetime(entry)
            if published is None:
                discarded_undated += 1
                continue
            title = self._clean_text(str(entry.get("title", "")).strip())
            url = valid_unicode(str(entry.get("link", "")).strip())
            if not title or not url:
                continue
            summary = self._entry_summary(entry)
            items.append(
                NewsItem(
                    title=title,
                    url=url,
                    source_name=feed_name,
                    published_at=published.astimezone(self._local_tz).isoformat(),
                    summary=summary,
                    category=category,
                )
            )
        if discarded_undated:
            pass
        self._cache_put(feed.feed_id, items)
        return items

    def _fetch_feed_bytes(self, url: str) -> bytes:
        with httpx.Client(
            transport=self.transport,
            timeout=self.settings.news_timeout_seconds,
            verify=self.settings.verify_tls,
            follow_redirects=True,
        ) as client:
            with client.stream("GET", url, headers={"Accept": "application/rss+xml, application/atom+xml, application/xml"}) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > self.settings.news_max_feed_bytes:
                        raise NewsError(f"Feed exceeds byte limit: {url}")
                    chunks.append(chunk)
        return b"".join(chunks)

    def _entry_datetime(self, entry: Any) -> datetime | None:
        candidate = entry.get("published_parsed") or entry.get("updated_parsed")
        if candidate is None:
            return None
        stamp = calendar.timegm(candidate)
        return datetime.fromtimestamp(stamp, tz=UTC)

    def _entry_summary(self, entry: Any) -> str:
        raw = ""
        if entry.get("summary"):
            raw = str(entry.get("summary"))
        elif entry.get("description"):
            raw = str(entry.get("description"))
        elif entry.get("content"):
            content = entry.get("content")
            if isinstance(content, list) and content:
                raw = str(content[0].get("value", ""))
        cleaned = self._clean_text(raw)
        if not cleaned:
            return "No feed summary provided."
        if len(cleaned) <= self.settings.news_summary_max_characters:
            return cleaned
        bounded = cleaned[: self.settings.news_summary_max_characters - 1].rstrip()
        if " " in bounded:
            bounded = bounded.rsplit(" ", 1)[0]
        return bounded.rstrip(".,;: ") + "…"

    def _clean_text(self, text: str) -> str:
        stripper = _HTMLStripper()
        stripper.feed(unescape(valid_unicode(text)))
        return " ".join(stripper.text().split())

    def _select_items(self, items: list[NewsItem], max_items: int) -> list[NewsItem]:
        deduped: list[NewsItem] = []
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        for item in sorted(items, key=lambda value: self._parse_iso(value.published_at), reverse=True):
            canonical_url = canonicalize_url(item.url)
            normalized_title = " ".join(item.title.lower().split())
            if canonical_url in seen_urls or normalized_title in seen_titles:
                continue
            seen_urls.add(canonical_url)
            seen_titles.add(normalized_title)
            deduped.append(item)

        selected: list[NewsItem] = []
        per_source: dict[str, int] = {}
        for item in deduped:
            if len(selected) >= max_items:
                break
            count = per_source.get(item.source_name, 0)
            if count >= self.settings.news_per_source_limit:
                continue
            selected.append(item)
            per_source[item.source_name] = count + 1
        if len(selected) < max_items:
            used_urls = {canonicalize_url(item.url) for item in selected}
            for item in deduped:
                if len(selected) >= max_items:
                    break
                if canonicalize_url(item.url) in used_urls:
                    continue
                selected.append(item)
                used_urls.add(canonicalize_url(item.url))
        return selected

    def _time_window(self, scope: NewsTimeScope) -> tuple[datetime, datetime]:
        now = datetime.now(self._local_tz)
        if scope == NewsTimeScope.TODAY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return start, now
        if scope == NewsTimeScope.YESTERDAY:
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start = today_start - timedelta(days=1)
            end = today_start - timedelta(microseconds=1)
            return start, end
        if scope == NewsTimeScope.LAST_7_DAYS:
            return now - timedelta(days=7), now
        return now - timedelta(hours=24), now

    def _no_results_report(self, request: NewsRequest, warnings: list[str]) -> NewsReport:
        category = request.category.value if request.category else "news"
        label = request.time_scope.value.replace("_", " ")
        summary = f"No {category} news found for {label}."
        answer = f"No {category} news items were found in the configured RSS feeds for {label}."
        return NewsReport(summary=summary, answer=answer, references=[], warnings=warnings)

    def _summary_label(self, request: NewsRequest, count: int) -> str:
        category = request.category.value if request.category else "news"
        label = request.time_scope.value.replace("_", " ")
        return f"{count} {category} news stories for {label}."

    def _format_local_time(self, value: str) -> str:
        dt = self._parse_iso(value)
        return dt.strftime("%Y-%m-%d %I:%M %p %Z").replace(" 0", " ")

    def _parse_iso(self, value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(self._local_tz)

    def _cache_get(self, feed_id: str) -> list[NewsItem] | None:
        if self.settings.news_cache_ttl_seconds == 0:
            return None
        with self._cache_lock:
            cached = self._cache.get(feed_id)
            if cached is None:
                return None
            expires_at, items = cached
            if expires_at < time.monotonic():
                self._cache.pop(feed_id, None)
                return None
            return items

    def _cache_put(self, feed_id: str, items: list[NewsItem]) -> None:
        if self.settings.news_cache_ttl_seconds == 0:
            return
        with self._cache_lock:
            self._cache[feed_id] = (time.monotonic() + self.settings.news_cache_ttl_seconds, items)
