"""Typed data models for Magpie."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ResponseDetail(str, Enum):
    COMPACT = "compact"
    DEBUG = "debug"


class FreshnessClass(str, Enum):
    RECENT = "recent"
    EVERGREEN = "evergreen"


class StopReason(str, Enum):
    ANSWERED_FROM_CACHE = "answered_from_cache"
    NEEDED_NEW_SEARCH = "needed_new_search"
    SPECIALIZED_ROUTE = "specialized_route"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_PROGRESS = "no_progress"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERNAL_ERROR = "internal_error"


class SourceKind(str, Enum):
    PAGE_FETCH = "page_fetch"
    SEARCH_RESULT_FALLBACK = "search_result_fallback"
    WEATHER_API = "weather_api"
    ANILIST_API = "anilist_api"
    RSS_FEED = "rss_feed"


class RequestRoute(str, Enum):
    WEB_RESEARCH = "web_research"
    WEATHER = "weather"
    ANIME = "anime"
    NEWS = "news"


class WeatherKind(str, Enum):
    CONDITIONS = "conditions"
    FORECAST = "forecast"


class AnimeRequestKind(str, Enum):
    LOOKUP = "lookup"
    CREDITS = "credits"
    SCHEDULE = "schedule"


class AnimeField(str, Enum):
    DESCRIPTION = "description"
    EPISODES = "episodes"
    DURATION = "duration"
    STATUS = "status"
    FORMAT = "format"
    SEASON = "season"
    SEASON_YEAR = "season_year"
    START_DATE = "start_date"
    END_DATE = "end_date"
    GENRES = "genres"
    STUDIOS = "studios"
    SOURCE_MATERIAL = "source_material"
    AVERAGE_SCORE = "average_score"
    NEXT_AIRING_EPISODE = "next_airing_episode"


class NewsCategory(str, Enum):
    GENERAL = "general"
    WORLD = "world"
    US = "us"
    POLITICS = "politics"
    BUSINESS = "business"
    TECHNOLOGY = "technology"
    AI = "ai"
    SCIENCE = "science"
    HEALTH = "health"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"


class NewsTimeScope(str, Enum):
    LAST_24_HOURS = "last_24_hours"
    TODAY = "today"
    YESTERDAY = "yesterday"
    LAST_7_DAYS = "last_7_days"


class NewsRequestKind(str, Enum):
    CATEGORY = "category"
    UNSUPPORTED_TOPIC = "unsupported_topic"


@dataclass(slots=True)
class Reference:
    source_id: str
    title: str
    url: str
    site_name: str | None
    published_at: str | None
    fetched_at: str | None
    source_kind: SourceKind = SourceKind.PAGE_FETCH


@dataclass(slots=True)
class ResearchRequest:
    question: str
    max_references: int = 5
    response_detail: ResponseDetail = ResponseDetail.COMPACT
    run_label: str | None = None


@dataclass(slots=True)
class RunBudget:
    queries_remaining: int
    sources_remaining: int
    evidence_remaining: int


@dataclass(slots=True)
class PlanningContext:
    prior_queries: list[str]
    seen_urls: list[str]
    remaining_questions: list[str]
    budget: RunBudget


@dataclass(slots=True)
class ResearchResult:
    status: str
    run_id: str
    summary: str
    answer: str
    references: list[Reference]
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    stop_reason: StopReason = StopReason.NEEDED_NEW_SEARCH
    debug: dict[str, Any] | None = None


@dataclass(slots=True)
class ResearchErrorResult:
    status: str
    run_id: str
    stage: str
    message: str
    stop_reason: StopReason
    partial_answer: str | None = None
    partial_references: list[Reference] = field(default_factory=list)
    debug: dict[str, Any] | None = None


@dataclass(slots=True)
class SpecializedRouteResult:
    """Outcome of a specialized (weather/anime/news) route.

    Routes return this instead of building a full :class:`ResearchResult` so
    they stay decoupled from :class:`ResearchService`'s persistence and event
    helpers. The service owns finalization via
    :meth:`ResearchService.finalize_specialized_route`.

    ``elapsed_ms`` is the total time spent on the specialized fetch for this
    route (e.g. one weather lookup, one anime lookup, or one batched news
    fetch). When ``references`` contains more than one item (news), the
    finalize seam amortizes it across references.

    ``provider`` is the source-recording label (e.g. ``"neonhail"``,
    ``"anilist"``, ``"rss"``) fed to per-source telemetry. ``route_name`` is
    the human route label (``"weather"``/``"anime"``/``"news"``) used in the
    completion trace; it differs from ``provider``.
    """

    summary: str
    answer: str
    references: list[Reference]
    stop_reason: StopReason
    provider: str
    route_name: str
    elapsed_ms: float


@dataclass(slots=True)
class IndexedSearchResultItem:
    index: int
    title: str
    url: str
    site_name: str | None
    published_at: str | None
    summary: str


@dataclass(slots=True)
class IndexedSearchResult:
    run_id: str
    query: str
    results: list[IndexedSearchResultItem]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FetchResult:
    run_id: str
    index: int | None
    url: str
    title: str
    content: str
    fetched_via: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryProposal:
    query: str


@dataclass(slots=True)
class RouteDecision:
    route: RequestRoute
    weather_kind: WeatherKind | None = None
    zip_code: str | None = None


@dataclass(slots=True)
class AnimeRequest:
    kind: AnimeRequestKind
    title_query: str | None = None
    character_query: str | None = None
    requested_fields: list[AnimeField] = field(default_factory=list)


@dataclass(slots=True)
class NewsRequest:
    kind: NewsRequestKind
    category: NewsCategory | None = None
    time_scope: NewsTimeScope = NewsTimeScope.LAST_24_HOURS


@dataclass(slots=True)
class NewsItem:
    title: str
    url: str
    source_name: str
    published_at: str
    summary: str
    category: NewsCategory


@dataclass(slots=True)
class NewsReport:
    summary: str
    answer: str
    references: list[Reference]
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnimeCandidate:
    anime_id: int
    english_title: str | None
    romaji_title: str | None
    native_title: str | None
    format: str | None
    season_year: int | None


@dataclass(slots=True)
class CharacterCredit:
    character_name: str
    voice_actor_names: list[str]


@dataclass(slots=True)
class AnimeReport:
    summary: str
    answer: str
    reference: Reference


@dataclass(slots=True)
class WeatherReport:
    summary: str
    answer: str
    reference: Reference


@dataclass(slots=True)
class SearchRequest:
    query: str
    limit: int
    freshness_class: FreshnessClass


@dataclass(slots=True)
class SearchResultRecord:
    title: str
    url: str
    snippet: str
    site_name: str | None = None
    published_at: str | None = None
    provider: str = "unknown"
    author: str | None = None
    inline_text: str | None = None
    highlights: list[str] = field(default_factory=list)
    raw_result: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FetchedSource:
    url: str
    title: str
    site_name: str | None
    text: str
    published_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    markdown: str | None = None
    raw_html: str | None = None
    retrieved_via: str = "unknown"
    fetch_error: str | None = None
    source_kind: SourceKind = SourceKind.PAGE_FETCH


@dataclass(slots=True)
class EvidenceItem:
    evidence_id: str
    source_id: str
    excerpt: str
    note: str


@dataclass(slots=True)
class SynthesisDraft:
    summary: str
    answer: str
    cited_source_ids: list[str]
    remaining_questions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    source_answers_question: bool = True


def utc_now() -> str:
    """Return a UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and enums into JSON-safe structures."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value
