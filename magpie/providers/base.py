"""Provider interfaces."""

from __future__ import annotations

from typing import Protocol

from ..models import (
    EvidenceItem,
    FetchedSource,
    PlanningContext,
    QueryProposal,
    RouteDecision,
    SearchRequest,
    SearchResultRecord,
    SynthesisDraft,
    WeatherKind,
    WeatherReport,
)


class ResolverClient(Protocol):
    def route_request(self, question: str) -> RouteDecision: ...

    def propose_query(self, question: str, context: PlanningContext) -> QueryProposal: ...

    def synthesize(
        self,
        question: str,
        evidence: list[EvidenceItem],
        prior_draft: SynthesisDraft | None = None,
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
