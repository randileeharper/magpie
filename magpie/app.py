"""Application assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .providers.crawl4ai_fetcher import Crawl4AIFetcher
from .providers.exa import ExaSearchClient
from .providers.fake import FakeFetcher, FakeResolverClient, FakeSearchClient
from .providers.openai_compatible import OpenAICompatibleResolverClient
from .providers.neonhail import NeonHailWeatherClient
from .providers.base import Fetcher, SearchClient, WeatherClient
from .service import ResearchService
from .storage import SQLiteStorage


@dataclass(slots=True)
class AppContext:
    settings: Settings
    storage: SQLiteStorage
    service: ResearchService
    search_client: SearchClient
    fetcher: Fetcher
    weather_client: WeatherClient | None


def build_app(config_path: str | None = None) -> AppContext:
    settings = Settings.load(config_path)
    for log_path in (settings.resolver_debug_log_path, settings.fetch_debug_log_path):
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    storage = SQLiteStorage(settings.expanded_database_path)
    storage.initialize()

    if settings.search_provider == "fake":
        search_client = FakeSearchClient()
    else:
        search_client = ExaSearchClient(settings=settings)

    if settings.fetch_provider == "fake":
        fetcher = FakeFetcher()
    else:
        fetcher = Crawl4AIFetcher(settings=settings)

    if settings.resolver_backend == "fake":
        resolver = FakeResolverClient(include_reasoning=settings.resolver_include_reasoning)
    else:
        resolver = OpenAICompatibleResolverClient(settings=settings)

    weather_client = NeonHailWeatherClient(settings=settings) if settings.weather_enabled else None
    service = ResearchService(
        storage=storage,
        resolver=resolver,
        search_client=search_client,
        fetcher=fetcher,
        settings=settings,
        weather_client=weather_client,
    )
    return AppContext(
        settings=settings,
        storage=storage,
        service=service,
        search_client=search_client,
        fetcher=fetcher,
        weather_client=weather_client,
    )
