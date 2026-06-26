"""Deterministic fake providers for local development and tests."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..models import (
    EvidenceItem,
    AnimeCandidate,
    AnimeField,
    AnimeRequest,
    AnimeRequestKind,
    CharacterCredit,
    FetchedSource,
    NewsCategory,
    NewsReport,
    NewsRequest,
    NewsRequestKind,
    NewsTimeScope,
    PlanningContext,
    QueryProposal,
    Reference,
    RequestRoute,
    RouteDecision,
    SearchRequest,
    SearchResultRecord,
    SourceKind,
    SynthesisDraft,
)
from .base import reasoning_request_options


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in {"the", "and", "for", "with", "that"}
    }


def _overlap_score(left: str, right: str) -> int:
    return len(_tokens(left) & _tokens(right))


@dataclass(slots=True)
class FakeResolverClient:
    """A very small deterministic resolver that keeps the workflow testable."""

    include_reasoning: bool = False

    def route_request(self, question: str) -> RouteDecision:
        return RouteDecision(route=RequestRoute.WEB_RESEARCH)

    def classify_anime_request(self, question: str) -> AnimeRequest:
        return AnimeRequest(AnimeRequestKind.LOOKUP, title_query=question, requested_fields=[AnimeField.DESCRIPTION])

    def classify_news_request(self, question: str) -> NewsRequest:
        return NewsRequest(NewsRequestKind.CATEGORY, NewsCategory.GENERAL, NewsTimeScope.LAST_24_HOURS)

    def refine_anime_title_queries(self, question: str, attempted_query: str) -> list[str]:
        return [attempted_query]

    def select_anime_candidate(self, question: str, candidates: list[AnimeCandidate]) -> int | None:
        return candidates[0].anime_id if candidates else None

    def select_character(self, query: str, credits: list[CharacterCredit]) -> str | None:
        return credits[0].character_name if credits else None

    def propose_query(self, question: str, context: PlanningContext) -> QueryProposal:
        if context.remaining_questions:
            return QueryProposal(query=context.remaining_questions[0])
        return QueryProposal(query=question)

    def synthesize(
        self,
        question: str,
        evidence: list[EvidenceItem],
        prior_draft: SynthesisDraft | None = None,
    ) -> SynthesisDraft:
        if not evidence:
            return SynthesisDraft(
                summary="No evidence found.",
                answer="I could not find enough evidence to answer the question.",
                cited_source_ids=[],
                remaining_questions=["What evidence answers the question?"],
                source_answers_question=False,
            )
        best = evidence[0]
        return SynthesisDraft(
            summary=f"Grounded answer for: {question}",
            answer=best.excerpt,
            cited_source_ids=[*(prior_draft.cited_source_ids if prior_draft else []), best.source_id],
            remaining_questions=[],
        )

    def compose(
        self,
        question: str,
        evidence: list[EvidenceItem],
        prior_draft: SynthesisDraft,
    ) -> SynthesisDraft:
        if not evidence:
            return prior_draft
        return SynthesisDraft(
            summary=prior_draft.summary or f"Composed answer for: {question}",
            answer=prior_draft.answer,
            cited_source_ids=list(prior_draft.cited_source_ids),
            remaining_questions=[],
        )

    def reasoning_request_options(self) -> dict[str, object]:
        return reasoning_request_options(self.include_reasoning)


@dataclass(slots=True)
class FakeSearchClient:
    """A fake search provider backed by an in-memory index."""

    index: dict[str, list[SearchResultRecord]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index:
            return
        self.index = {
            "who is the mayor of new york": [
                SearchResultRecord(
                    title="Zohran Mamdani wins New York City mayoral election",
                    url="https://example.com/mayor-election",
                    snippet="Zohran Mamdani was elected mayor of New York City in 2025.",
                    site_name="Example News",
                    published_at="2025-11-04",
                    provider="fake",
                    inline_text="Zohran Mamdani was elected mayor of New York City on November 4, 2025.",
                )
            ],
            "when was zohran mamdani elected mayor": [
                SearchResultRecord(
                    title="Zohran Mamdani wins New York City mayoral election",
                    url="https://example.com/mayor-election",
                    snippet="Zohran Mamdani was elected mayor of New York City on November 4, 2025.",
                    site_name="Example News",
                    published_at="2025-11-04",
                    provider="fake",
                    inline_text="Zohran Mamdani was elected mayor of New York City on November 4, 2025.",
                )
            ],
        }

    def search(self, request: SearchRequest) -> list[SearchResultRecord]:
        query_key = request.query.strip().lower()
        if query_key in self.index:
            return self.index[query_key][: request.limit]

        ranked: list[tuple[int, SearchResultRecord]] = []
        for known_query, results in self.index.items():
            score = _overlap_score(query_key, known_query)
            for result in results:
                ranked.append((score, result))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[: request.limit] if item[0] > 0]

    def doctor_check(self, live: bool = False) -> dict[str, object]:
        return {"status": "ok", "provider": "fake", "live": live}


@dataclass(slots=True)
class FakeNewsClient:
    def get_news(self, request: NewsRequest, max_items: int) -> NewsReport:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        references = [
            Reference(
                f"rss:{index}",
                f"Story {index}",
                f"https://example.com/story-{index}",
                "Example Feed",
                f"{today}T0{index}:00:00-07:00",
                None,
                SourceKind.RSS_FEED,
            )
            for index in range(1, max_items + 1)
        ]
        lines = [
            (
                f"{index}. {today} 0{index}:00 PDT | Story {index} | Example summary {index}. | "
                f"Example Feed | https://example.com/story-{index}"
            )
            for index in range(1, max_items + 1)
        ]
        return NewsReport(
            summary="Latest general news.",
            answer="\n".join(lines),
            references=references,
        )

    def doctor_check(self, live: bool = False) -> dict[str, object]:
        return {"status": "ok", "provider": "fake_news", "live": live}


@dataclass(slots=True)
class FakeFetcher:
    """A fake fetcher backed by URL text fixtures."""

    pages: dict[str, FetchedSource] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pages:
            return
        self.pages = {
            "https://example.com/mayor-election": FetchedSource(
                url="https://example.com/mayor-election",
                title="Zohran Mamdani wins New York City mayoral election",
                site_name="Example News",
                text=(
                    "Zohran Mamdani was elected mayor of New York City on November 4, 2025. "
                    "The race concluded after a citywide general election."
                ),
                markdown=(
                    "Zohran Mamdani was elected mayor of New York City on November 4, 2025. "
                    "The race concluded after a citywide general election."
                ),
                published_at="2025-11-04",
                retrieved_via="fake",
                source_kind=SourceKind.PAGE_FETCH,
            )
        }

    def fetch(self, url: str) -> FetchedSource:
        if url not in self.pages:
            return FetchedSource(
                url=url,
                title="Unknown result",
                site_name="Unknown",
                text="No fixture content was configured for this URL.",
                retrieved_via="fake",
                source_kind=SourceKind.PAGE_FETCH,
            )
        return self.pages[url]

    def doctor_check(self, live: bool = False) -> dict[str, object]:
        return {"status": "ok", "provider": "fake", "live": live}
