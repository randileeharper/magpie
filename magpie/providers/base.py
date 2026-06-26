"""Provider interfaces."""

from __future__ import annotations

from typing import Protocol

from ..models import (
    EvidenceItem,
    AnimeCandidate,
    AnimeField,
    AnimeReport,
    AnimeRequest,
    NewsReport,
    NewsRequest,
    CharacterCredit,
    FetchedSource,
    PlanningContext,
    QueryProposal,
    Reference,
    RouteDecision,
    SearchRequest,
    SearchResultRecord,
    SynthesisDraft,
    WeatherKind,
    WeatherReport,
)


def reasoning_request_options(include_reasoning: bool) -> dict[str, object]:
    """Build the reasoning-effort payload sent to OpenAI-compatible resolvers."""
    effort = "medium" if include_reasoning else "none"
    return {
        "think": include_reasoning,
        "reasoning_effort": effort,
        "reasoning": {"effort": effort},
    }


class ResolverClient(Protocol):
    def route_request(self, question: str) -> RouteDecision: ...

    def classify_anime_request(self, question: str) -> AnimeRequest: ...

    def classify_news_request(self, question: str) -> NewsRequest: ...

    def refine_anime_title_queries(self, question: str, attempted_query: str) -> list[str]: ...

    def select_anime_candidate(self, question: str, candidates: list[AnimeCandidate]) -> int | None: ...

    def select_character(self, query: str, credits: list[CharacterCredit]) -> str | None: ...

    def propose_query(self, question: str, context: PlanningContext) -> QueryProposal: ...

    def synthesize(
        self,
        question: str,
        evidence: list[EvidenceItem],
        prior_draft: SynthesisDraft | None = None,
    ) -> SynthesisDraft: ...

    def compose(
        self,
        question: str,
        evidence: list[EvidenceItem],
        prior_draft: SynthesisDraft,
    ) -> SynthesisDraft: ...

    def reasoning_request_options(self) -> dict[str, object]: ...


class SearchClient(Protocol):
    def search(self, request: SearchRequest) -> list[SearchResultRecord]: ...

    def doctor_check(self, live: bool = False) -> dict[str, object]: ...


class Fetcher(Protocol):
    def fetch(self, url: str) -> FetchedSource: ...

    def doctor_check(self, live: bool = False) -> dict[str, object]: ...


class WeatherClient(Protocol):
    def get_weather(self, zip_code: str, kind: WeatherKind) -> WeatherReport: ...


class AnimeClient(Protocol):
    def search_anime(self, title_query: str) -> list[AnimeCandidate]: ...

    def get_anime_info(self, anime_id: int, requested_fields: list[AnimeField]) -> AnimeReport: ...

    def get_credits(self, anime_id: int) -> tuple[str, list[CharacterCredit], Reference]: ...

    def get_daily_schedule(self) -> AnimeReport: ...


class NewsClient(Protocol):
    def get_news(self, request: NewsRequest, max_items: int) -> NewsReport: ...

    def doctor_check(self, live: bool = False) -> dict[str, object]: ...
