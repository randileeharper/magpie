"""Bounded, state-aware research orchestration."""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import perf_counter
from pathlib import Path
from typing import Any

from .errors import AnimeError, FetchError, NewsError, ResearchCancelled, ResolverError, WeatherError
from .config import Settings
from .historian import HistorianSink, NullHistorianSink, build_event
from .models import (
    AnimeReport, AnimeRequestKind, EvidenceItem, FreshnessClass, NewsRequestKind, PlanningContext,
    ResearchErrorResult, ResearchRequest, ResearchResult, RequestRoute, ResponseDetail, RunBudget,
    SearchRequest, SearchResultRecord, SourceKind, StopReason, SynthesisDraft, WeatherKind, to_jsonable,
)
from .providers.base import AnimeClient, Fetcher, NewsClient, ResolverClient, SearchClient, WeatherClient
from .storage import SQLiteStorage, canonicalize_url, normalize_query
from .text import valid_unicode


RECENT_SIGNALS = {"latest", "current", "today", "yesterday", "this week", "this month", "this year"}
PROCEDURAL_SIGNALS = ("how do i ", "how to ", "steps to ", "guide to ")
RECIPE_SIGNALS = ("recipe", "cook", "bake", "bread", "dough")
ACTIONABLE_SECTION_SIGNALS = (
    "ingredient", "instruction", "method", "step", "directions", "preparation",
    "how to make", "recipe", "bake", "mix", "fold", "proof", "ferment",
)
IMPERATIVE_SIGNALS = (
    "add ", "bake ", "combine ", "cover ", "fold ", "heat ", "knead ", "mix ",
    "place ", "preheat ", "rest ", "shape ", "stir ", "transfer ",
)
_GLOBAL_RESOLVER_GATE = threading.BoundedSemaphore(1)
_FETCH_LOG_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RunTelemetry:
    run_id: str
    started_at: float
    started_event_id: str
    freshness_class: FreshnessClass
    route: str = "unselected"
    stage: str = "run"
    queries: int = 0
    sources_discovered: int = 0
    sources_fetched: int = 0
    sources_rejected: int = 0
    cache_hits: int = 0
    syntheses: int = 0
    operation_error_recorded: bool = False


def detect_freshness_class(question: str) -> FreshnessClass:
    lowered = question.lower()
    if any(signal in lowered for signal in RECENT_SIGNALS):
        return FreshnessClass.RECENT
    years = {int(value) for value in re.findall(r"\b20\d{2}\b", question)}
    if any(year >= datetime.now(UTC).year - 1 for year in years):
        return FreshnessClass.RECENT
    return FreshnessClass.EVERGREEN


@dataclass(slots=True)
class ResearchService:
    storage: SQLiteStorage
    resolver: ResolverClient
    search_client: SearchClient
    fetcher: Fetcher
    settings: Settings
    weather_client: WeatherClient | None = None
    anime_client: AnimeClient | None = None
    news_client: NewsClient | None = None
    historian_sink: HistorianSink = field(default_factory=NullHistorianSink)
    _resolver_semaphore: threading.BoundedSemaphore = field(default=_GLOBAL_RESOLVER_GATE)
    _telemetry: dict[str, RunTelemetry] = field(default_factory=dict)
    _telemetry_lock: threading.Lock = field(default_factory=threading.Lock)

    def cancel_run(self, run_id: str) -> None:
        self.storage.request_cancel(run_id)
        if self.storage.mark_run_cancelled(run_id):
            self.storage.append_event(run_id, "run_cancelled", {"run_id": run_id})
            self._emit_run_event(
                run_id,
                "research.run.canceled",
                {
                    "run_id": run_id,
                    "status": "canceled",
                    "stop_reason": StopReason.CANCELLED.value,
                },
            )

    def close(self) -> None:
        self.historian_sink.close()

    def _emit(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        subject: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        source: str = "app://magpie/research",
    ) -> str:
        event = build_event(
            event_type,
            self._sanitize_event_data(data),
            source=source,
            subject=subject,
            correlation_id=correlation_id,
            causation_id=causation_id,
        )
        try:
            self.historian_sink.emit(event)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Historian delivery failed for event_id=%s type=%s: %s",
                event["id"],
                event_type,
                exc,
            )
        return str(event["id"])

    def _emit_run_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        subject: str | None = None,
        source: str = "app://magpie/research",
    ) -> str:
        telemetry = self._get_telemetry(run_id)
        return self._emit(
            event_type,
            data,
            subject=subject or run_id,
            correlation_id=run_id,
            causation_id=telemetry.started_event_id if telemetry else None,
            source=source,
        )

    def _sanitize_event_data(self, value: Any) -> Any:
        secrets = [
            secret
            for secret in (
                self.settings.search_api_key,
                self.settings.resolver_api_key,
                self.settings.historian_token,
            )
            if secret
        ]
        if isinstance(value, str):
            sanitized = valid_unicode(value)
            for secret in secrets:
                sanitized = sanitized.replace(secret, "[REDACTED]")
            return sanitized
        if isinstance(value, dict):
            return {str(key): self._sanitize_event_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize_event_data(item) for item in value]
        return value

    def _get_telemetry(self, run_id: str) -> RunTelemetry | None:
        with self._telemetry_lock:
            return self._telemetry.get(run_id)

    def _set_stage(self, run_id: str, stage: str) -> None:
        telemetry = self._get_telemetry(run_id)
        if telemetry:
            telemetry.stage = stage

    def _stage(self, run_id: str) -> str:
        telemetry = self._get_telemetry(run_id)
        return telemetry.stage if telemetry else "research"

    def _select_route(self, run_id: str, route: str, fallback_reason: str | None = None) -> None:
        telemetry = self._get_telemetry(run_id)
        if telemetry and telemetry.route == route and fallback_reason is None:
            return
        if telemetry:
            telemetry.route = route
            telemetry.stage = "route"
        self._emit_run_event(
            run_id,
            "research.route.selected",
            {
                "run_id": run_id,
                "route": route,
                "fallback_reason": fallback_reason,
            },
        )

    def _record_operation_error(
        self, run_id: str, component: str, operation: str | None, exc: Exception
    ) -> None:
        telemetry = self._get_telemetry(run_id)
        if telemetry:
            telemetry.operation_error_recorded = True
        self._emit_run_event(
            run_id,
            "core.operation.error",
            {
                "app_id": "magpie",
                "component": component,
                "error_type": exc.__class__.__name__,
                "message": str(exc),
                "operation": operation,
                "details": {"run_id": run_id},
            },
            source=f"app://magpie/{component}",
        )

    def _record_query_executed(
        self,
        run_id: str,
        query_id: str,
        query: str,
        freshness: FreshnessClass,
        result_count: int,
        elapsed_ms: float,
    ) -> None:
        telemetry = self._get_telemetry(run_id)
        if telemetry:
            telemetry.queries += 1
        self._emit_run_event(
            run_id,
            "research.query.executed",
            {
                "run_id": run_id,
                "query_id": query_id,
                "normalized_query": query,
                "provider": self.settings.search_provider,
                "freshness_class": freshness.value,
                "result_count": result_count,
                "duration_ms": elapsed_ms,
            },
            subject=query_id,
        )

    def _record_source_discovered(
        self,
        run_id: str,
        search_result_id: str | None,
        result: SearchResultRecord,
        canonical_url: str,
    ) -> None:
        telemetry = self._get_telemetry(run_id)
        if telemetry:
            telemetry.sources_discovered += 1
        self._emit_run_event(
            run_id,
            "research.source.discovered",
            {
                "run_id": run_id,
                "search_result_id": search_result_id,
                "canonical_url": canonical_url,
                "title": result.title,
                "provider": result.provider or self.settings.search_provider,
                "published_at": result.published_at,
            },
            subject=search_result_id or canonical_url,
        )

    def _record_source_fetched(
        self,
        run_id: str,
        source_id: str,
        *,
        search_result_id: str | None,
        url: str,
        title: str,
        provider: str,
        source_kind: str,
        published_at: str | None,
        duration_ms: float,
        fallback_content: bool,
    ) -> None:
        telemetry = self._get_telemetry(run_id)
        if telemetry:
            telemetry.sources_fetched += 1
        self._emit_run_event(
            run_id,
            "research.source.fetched",
            {
                "run_id": run_id,
                "source_id": source_id,
                "search_result_id": search_result_id,
                "canonical_url": canonicalize_url(url),
                "title": title,
                "provider": provider,
                "source_kind": source_kind,
                "published_at": published_at,
                "duration_ms": duration_ms,
                "fallback_content": fallback_content,
            },
            subject=source_id,
        )

    def _record_specialized_source(
        self, run_id: str, reference: Any, provider: str, duration_ms: float
    ) -> None:
        self._record_source_fetched(
            run_id,
            reference.source_id,
            search_result_id=None,
            url=reference.url,
            title=reference.title,
            provider=provider,
            source_kind=reference.source_kind.value,
            published_at=reference.published_at,
            duration_ms=duration_ms,
            fallback_content=False,
        )

    def _record_source_rejected(
        self, run_id: str, source_id: str | None, query: str | None, reason: str
    ) -> None:
        telemetry = self._get_telemetry(run_id)
        if telemetry:
            telemetry.sources_rejected += 1
        self._emit_run_event(
            run_id,
            "research.source.rejected",
            {
                "run_id": run_id,
                "source_id": source_id,
                "normalized_query": query,
                "reason": reason,
            },
            subject=source_id or run_id,
        )

    def _record_cache_hit(self, run_id: str, reference: Any, cache_kind: str) -> None:
        telemetry = self._get_telemetry(run_id)
        if telemetry:
            telemetry.cache_hits += 1
        self._emit_run_event(
            run_id,
            "research.cache.hit",
            {
                "run_id": run_id,
                "source_id": reference.source_id,
                "canonical_url": canonicalize_url(reference.url),
                "title": reference.title,
                "cache_kind": cache_kind,
                "freshness_class": telemetry.freshness_class.value if telemetry else "evergreen",
            },
            subject=reference.source_id,
        )

    def _record_run_finished(
        self,
        run_id: str,
        event_type: str,
        status: str,
        reason: StopReason,
        timings: dict[str, list[float]],
        *,
        reference_ids: list[str] | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        telemetry = self._get_telemetry(run_id)
        counters = {
            "queries": telemetry.queries if telemetry else 0,
            "sources_discovered": telemetry.sources_discovered if telemetry else 0,
            "sources_fetched": telemetry.sources_fetched if telemetry else 0,
            "sources_rejected": telemetry.sources_rejected if telemetry else 0,
            "cache_hits": telemetry.cache_hits if telemetry else 0,
            "syntheses": telemetry.syntheses if telemetry else 0,
        }
        data: dict[str, Any] = {
            "run_id": run_id,
            "status": status,
            "route": telemetry.route if telemetry else RequestRoute.WEB_RESEARCH.value,
            "stop_reason": reason.value,
            "reference_ids": reference_ids or [],
            "counts": counters,
            "timings_ms": {
                key: round(sum(values), 2)
                for key, values in timings.items()
            },
            "duration_ms": round(
                (perf_counter() - telemetry.started_at) * 1000, 2
            ) if telemetry else 0.0,
        }
        if event_type == "research.run.failed":
            data["error_type"] = error_type or "ResearchError"
            data["error_message"] = error_message or ""
            data["stage"] = telemetry.stage if telemetry else "research"
        self._emit_run_event(run_id, event_type, data)

    def research(
        self, request: ResearchRequest, *, run_id: str | None = None
    ) -> ResearchResult | ResearchErrorResult:
        freshness = detect_freshness_class(request.question)
        timings: dict[str, list[float]] = {}
        run_id = self.storage.create_run(
            request.question, request.run_label, freshness, request.response_detail.value, run_id=run_id
        )
        self._begin_logs(run_id, request.question)
        self.storage.append_event(run_id, "run_started", {"freshness_class": freshness.value})
        started_event_id = self._emit(
            "research.run.started",
            {
                "run_id": run_id,
                "question": request.question,
                "run_label": request.run_label,
                "freshness_class": freshness.value,
                "response_detail": request.response_detail.value,
            },
            subject=run_id,
            correlation_id=run_id,
        )
        with self._telemetry_lock:
            self._telemetry[run_id] = RunTelemetry(run_id, perf_counter(), started_event_id, freshness)
        budget = RunBudget(
            queries_remaining=getattr(self.settings, "max_search_queries_per_run"),
            sources_remaining=getattr(self.settings, "max_sources_per_run"),
            evidence_remaining=getattr(self.settings, "max_evidence_items_per_run"),
        )
        evidence: list[EvidenceItem] = []
        seen_urls: set[str] = set()
        warnings: list[str] = []
        limitations: list[str] = []
        remaining_questions: list[str] = []
        last_draft: SynthesisDraft | None = None

        try:
            self._raise_if_cancelled(run_id)
            specialized_result = self._try_specialized_route(run_id, request, timings, warnings)
            if specialized_result is not None:
                return specialized_result
            self._select_route(run_id, RequestRoute.WEB_RESEARCH.value)
            cached_ids = self.storage.find_fresh_source_ids_for_exact_query(
                normalize_query(request.question), self._min_fetched_at(freshness)
            )
            if cached_ids:
                seen_urls.update(self.storage.get_canonical_urls(cached_ids))
                for reference in self.storage.get_source_references(cached_ids):
                    self._record_cache_hit(run_id, reference, "exact_query")
                evidence.extend(self._evidence_from_sources(
                    run_id, cached_ids, request.question, budget, "Reused from exact-query cache"
                ))
                for item in evidence:
                    last_draft = self._synthesize(run_id, request.question, item, last_draft, timings)
                    if not last_draft.source_answers_question:
                        self.storage.reject_source_for_query(normalize_query(request.question), item.source_id)
                        self._record_source_rejected(
                            run_id, item.source_id, normalize_query(request.question),
                            "source_did_not_answer_question",
                        )
                    remaining_questions = self._remaining_questions_after_quality(
                        request.question, last_draft, limitations
                    )
                    if not last_draft.remaining_questions:
                        return self._finalize(run_id, request, last_draft, StopReason.ANSWERED_FROM_CACHE, timings)

            while budget.queries_remaining > 0 and budget.sources_remaining > 0 and budget.evidence_remaining > 0:
                self._raise_if_cancelled(run_id)
                prior_queries = self.storage.list_queries_for_run(run_id)
                context = PlanningContext(prior_queries, sorted(seen_urls), remaining_questions, budget)
                proposal, elapsed = self._call_resolver("propose_query", request.question, context)
                self._trace(run_id, "QUERY PROPOSED", [f"query: {proposal.query}", f"elapsed_ms: {elapsed}"])
                self._record_timing(timings, "resolver.propose_query", elapsed)
                query = normalize_query(proposal.query)
                if not query or query in prior_queries:
                    limitations.append("Planner could not produce a new useful query.")
                    return self._finish_incomplete(
                        run_id, request, last_draft, warnings, limitations, StopReason.NO_PROGRESS, timings
                    )
                budget.queries_remaining -= 1
                query_id = self.storage.add_query(
                    run_id, query, getattr(self.settings, "search_provider"), freshness
                )
                self._set_stage(run_id, "search")
                results, elapsed = self._search(run_id, proposal.query, freshness)
                self._trace(run_id, "SEARCH RESULTS", [f"query: {proposal.query}", f"result_count: {len(results)}"])
                self._record_timing(timings, "search", elapsed)
                result_ids = self.storage.add_search_results(query_id, [to_jsonable(result) for result in results])
                self._record_query_executed(run_id, query_id, query, freshness, len(results), elapsed)
                candidates: list[SearchResultRecord] = []
                for result in results:
                    canonical = canonicalize_url(result.url)
                    if canonical not in seen_urls:
                        seen_urls.add(canonical)
                        candidates.append(result)
                        self._record_source_discovered(
                            run_id, result_ids.get(result.url), result, canonical
                        )
                    if len(candidates) >= min(
                        getattr(self.settings, "max_sources_per_query"), budget.sources_remaining
                    ):
                        break
                if not candidates:
                    limitations.append(f"No new sources found for query: {proposal.query}")
                    continue

                for result in candidates:
                    self._raise_if_cancelled(run_id)
                    budget.sources_remaining -= 1
                    source_id, text, new_warnings, new_limitations, elapsed = self._acquire(
                        run_id, result, result_ids.get(result.url), freshness
                    )
                    self._record_timing(timings, "fetch", elapsed)
                    warnings.extend(new_warnings)
                    limitations.extend(new_limitations)
                    item = self._select_evidence(
                        run_id, source_id, text, request.question, remaining_questions, budget, []
                    )
                    if item:
                        evidence.append(item)
                        self._trace(run_id, "SYNTHESIS CHECK", [
                            f"source_id: {item.source_id}",
                            f"source_characters: {len(item.excerpt)}",
                        ])
                        last_draft = self._synthesize(run_id, request.question, item, last_draft, timings)
                        if not last_draft.source_answers_question:
                            self.storage.reject_source_for_query(query, item.source_id)
                            self._record_source_rejected(
                                run_id, item.source_id, query, "source_did_not_answer_question"
                            )
                        remaining_questions = self._remaining_questions_after_quality(
                            request.question, last_draft, limitations
                        )
                        if not remaining_questions:
                            return self._finalize(
                                run_id, request, last_draft, StopReason.NEEDED_NEW_SEARCH,
                                timings, warnings, limitations,
                            )

            return self._finish_incomplete(
                run_id, request, last_draft, warnings, limitations, StopReason.BUDGET_EXHAUSTED, timings
            )
        except ResearchCancelled as exc:
            if self.storage.mark_run_cancelled(run_id):
                self.storage.append_event(run_id, "run_cancelled", {"run_id": run_id})
                self._emit_run_event(
                    run_id,
                    "research.run.canceled",
                    {
                        "run_id": run_id,
                        "status": "canceled",
                        "stop_reason": StopReason.CANCELLED.value,
                    },
                )
            return ResearchErrorResult("error", run_id, "research", str(exc), StopReason.CANCELLED)
        except Exception as exc:  # noqa: BLE001
            self.storage.update_run_status(run_id, "failed", StopReason.FAILED.value)
            self.storage.append_event(run_id, "run_failed", {"error": str(exc)})
            telemetry = self._get_telemetry(run_id)
            if telemetry is None or not telemetry.operation_error_recorded:
                stage = self._stage(run_id)
                self._record_operation_error(run_id, stage, stage, exc)
            self._record_run_finished(
                run_id, "research.run.failed", "error", StopReason.FAILED, timings,
                error_type=exc.__class__.__name__, error_message=str(exc),
            )
            return ResearchErrorResult(
                "error", run_id, "research", str(exc), StopReason.FAILED,
                debug=self._build_debug(run_id, request, timings),
            )
        finally:
            with self._telemetry_lock:
                self._telemetry.pop(run_id, None)

    def _try_specialized_route(
        self, run_id: str, request: ResearchRequest, timings: dict[str, list[float]], warnings: list[str]
    ) -> ResearchResult | None:
        if self.weather_client is None and self.anime_client is None and self.news_client is None:
            return None
        try:
            self._set_stage(run_id, "route")
            decision, elapsed = self._call_resolver("route_request", request.question)
            self._record_timing(timings, "resolver.route_request", elapsed)
            self._trace(run_id, "REQUEST ROUTED", [
                f"route: {decision.route.value}",
                f"weather_kind: {decision.weather_kind.value if decision.weather_kind else ''}",
                f"zip_code: {decision.zip_code or ''}",
                f"elapsed_ms: {elapsed}",
            ])
        except Exception as exc:  # noqa: BLE001
            self._record_operation_error(run_id, "resolver", "route_request", exc)
            warnings.append(f"Request routing failed; used web research instead: {exc}")
            self._trace(run_id, "REQUEST ROUTING FALLBACK", [f"error: {exc}"])
            self._select_route(run_id, RequestRoute.WEB_RESEARCH.value, str(exc))
            return None
        self._select_route(run_id, decision.route.value)
        if decision.route == RequestRoute.ANIME and self.anime_client is not None:
            return self._try_anime_route(run_id, request, timings, warnings)
        if decision.route == RequestRoute.NEWS and self.news_client is not None:
            return self._try_news_route(run_id, request, timings, warnings)
        if decision.route != RequestRoute.WEATHER or self.weather_client is None:
            return None
        if not decision.zip_code:
            warnings.append("Weather route could not determine a US ZIP code; used web research instead.")
            self._select_route(
                run_id, RequestRoute.WEB_RESEARCH.value, "weather_zip_code_unavailable"
            )
            return None

        started = perf_counter()
        try:
            self._set_stage(run_id, "weather")
            report = self.weather_client.get_weather(
                decision.zip_code, decision.weather_kind or WeatherKind.CONDITIONS
            )
        except WeatherError as exc:
            self._record_operation_error(run_id, "weather", "get_weather", exc)
            warnings.append(f"Specialized weather lookup failed; used web research instead: {exc}")
            self._trace(run_id, "WEATHER ROUTE FALLBACK", [f"error: {exc}"])
            self._select_route(run_id, RequestRoute.WEB_RESEARCH.value, str(exc))
            return None
        elapsed = round((perf_counter() - started) * 1000, 2)
        self._record_timing(timings, "weather", elapsed)
        references = [report.reference][: max(0, request.max_references)]
        self.storage.save_final_answer(run_id, report.summary, report.answer, references)
        self.storage.update_run_status(run_id, "completed", StopReason.SPECIALIZED_ROUTE.value)
        self._record_specialized_source(run_id, report.reference, "neonhail", elapsed)
        self._record_run_finished(
            run_id,
            "research.run.completed",
            "ok",
            StopReason.SPECIALIZED_ROUTE,
            timings,
            reference_ids=[item.source_id for item in references],
        )
        self._trace(run_id, "COMPLETED", ["status: ok", "route: weather"])
        return ResearchResult(
            status="ok",
            run_id=run_id,
            summary=report.summary,
            answer=report.answer,
            references=references,
            warnings=warnings,
            stop_reason=StopReason.SPECIALIZED_ROUTE,
            debug=self._build_debug(run_id, request, timings),
        )

    def _try_anime_route(
        self, run_id: str, request: ResearchRequest, timings: dict[str, list[float]], warnings: list[str]
    ) -> ResearchResult | None:
        try:
            self._set_stage(run_id, "anime")
            anime_request, elapsed = self._call_resolver("classify_anime_request", request.question)
            self._record_timing(timings, "resolver.classify_anime_request", elapsed)
            self._trace(run_id, "ANIME REQUEST CLASSIFIED", [
                f"kind: {anime_request.kind.value}",
                f"title_query: {anime_request.title_query or ''}",
                f"character_query: {anime_request.character_query or ''}",
                f"requested_fields: {', '.join(item.value for item in anime_request.requested_fields)}",
                f"elapsed_ms: {elapsed}",
            ])
            started = perf_counter()
            if anime_request.kind == AnimeRequestKind.SCHEDULE:
                report = self.anime_client.get_daily_schedule()
            else:
                if not anime_request.title_query:
                    raise AnimeError("Anime title could not be determined.")
                candidates = self.anime_client.search_anime(anime_request.title_query)
                if not candidates:
                    refined_queries, elapsed = self._call_resolver(
                        "refine_anime_title_queries", request.question, anime_request.title_query
                    )
                    self._record_timing(timings, "resolver.refine_anime_title_queries", elapsed)
                    for refined_query in refined_queries:
                        if refined_query == anime_request.title_query:
                            continue
                        candidates = self.anime_client.search_anime(refined_query)
                        if candidates:
                            break
                if len(candidates) == 1:
                    selected_id = candidates[0].anime_id
                else:
                    selected_id, elapsed = self._call_resolver(
                        "select_anime_candidate", request.question, candidates
                    )
                    self._record_timing(timings, "resolver.select_anime_candidate", elapsed)
                if selected_id is None:
                    raise AnimeError("No AniList title candidate matched the request.")
                if anime_request.kind == AnimeRequestKind.LOOKUP:
                    report = self.anime_client.get_anime_info(selected_id, anime_request.requested_fields)
                else:
                    title, credits, reference = self.anime_client.get_credits(selected_id)
                    if anime_request.character_query:
                        character_name, elapsed = self._call_resolver(
                            "select_character", anime_request.character_query, credits
                        )
                        self._record_timing(timings, "resolver.select_anime_character", elapsed)
                        credit = next(
                            (item for item in credits if item.character_name == character_name), None
                        )
                        if credit is None:
                            raise AnimeError("No character matched the requested name.")
                        answer = (
                            f"{credit.character_name} in {title} is voiced in Japanese by "
                            f"{', '.join(credit.voice_actor_names)}."
                        )
                    else:
                        answer = f"Japanese voice cast for {title}:\n" + "\n".join(
                            f"{item.character_name} - {', '.join(item.voice_actor_names)}"
                            for item in credits[:15]
                        )
                    report = AnimeReport(
                        f"Japanese voice cast information for {title}.", answer, reference
                    )
            self._record_timing(timings, "anime", round((perf_counter() - started) * 1000, 2))
        except Exception as exc:  # noqa: BLE001
            self._record_operation_error(run_id, "anime", "specialized_lookup", exc)
            warnings.append(f"Specialized anime lookup failed; used web research instead: {exc}")
            self._trace(run_id, "ANIME ROUTE FALLBACK", [f"error: {exc}"])
            self._select_route(run_id, RequestRoute.WEB_RESEARCH.value, str(exc))
            return None
        references = [report.reference][: max(0, request.max_references)]
        self.storage.save_final_answer(run_id, report.summary, report.answer, references)
        self.storage.update_run_status(run_id, "completed", StopReason.SPECIALIZED_ROUTE.value)
        self._record_specialized_source(
            run_id, report.reference, "anilist", timings.get("anime", [0.0])[-1]
        )
        self._record_run_finished(
            run_id,
            "research.run.completed",
            "ok",
            StopReason.SPECIALIZED_ROUTE,
            timings,
            reference_ids=[item.source_id for item in references],
        )
        self._trace(run_id, "COMPLETED", ["status: ok", "route: anime"])
        return ResearchResult(
            "ok", run_id, report.summary, report.answer, references, warnings=warnings,
            stop_reason=StopReason.SPECIALIZED_ROUTE,
            debug=self._build_debug(run_id, request, timings),
        )

    def _try_news_route(
        self, run_id: str, request: ResearchRequest, timings: dict[str, list[float]], warnings: list[str]
    ) -> ResearchResult | None:
        try:
            self._set_stage(run_id, "news")
            news_request, elapsed = self._call_resolver("classify_news_request", request.question)
            self._record_timing(timings, "resolver.classify_news_request", elapsed)
            self._trace(run_id, "NEWS REQUEST CLASSIFIED", [
                f"kind: {news_request.kind.value}",
                f"category: {news_request.category.value if news_request.category else ''}",
                f"time_scope: {news_request.time_scope.value}",
                f"elapsed_ms: {elapsed}",
            ])
            if news_request.kind == NewsRequestKind.UNSUPPORTED_TOPIC:
                self._trace(run_id, "NEWS ROUTE FALLBACK", ["reason: unsupported_topic"])
                self._select_route(run_id, RequestRoute.WEB_RESEARCH.value, "unsupported_news_topic")
                return None
            started = perf_counter()
            limit = min(self.settings.news_digest_size, max(0, request.max_references))
            report = self.news_client.get_news(news_request, limit)
            self._record_timing(timings, "news", round((perf_counter() - started) * 1000, 2))
        except NewsError as exc:
            self._record_operation_error(run_id, "news", "get_news", exc)
            warnings.append(f"Specialized news lookup failed; used web research instead: {exc}")
            self._trace(run_id, "NEWS ROUTE FALLBACK", [f"error: {exc}"])
            self._select_route(run_id, RequestRoute.WEB_RESEARCH.value, str(exc))
            return None
        warnings.extend(report.warnings)
        references = report.references[: max(0, request.max_references)]
        self.storage.save_final_answer(run_id, report.summary, report.answer, references)
        self.storage.update_run_status(run_id, "completed", StopReason.SPECIALIZED_ROUTE.value)
        for reference in references:
            self._record_specialized_source(run_id, reference, "rss", 0.0)
        self._record_run_finished(
            run_id,
            "research.run.completed",
            "ok",
            StopReason.SPECIALIZED_ROUTE,
            timings,
            reference_ids=[item.source_id for item in references],
        )
        self._trace(run_id, "COMPLETED", [
            "status: ok",
            "route: news",
            f"reference_count: {len(references)}",
        ])
        return ResearchResult(
            "ok",
            run_id,
            report.summary,
            report.answer,
            references,
            warnings=warnings,
            stop_reason=StopReason.SPECIALIZED_ROUTE,
            debug=self._build_debug(run_id, request, timings),
        )

    def _call_resolver(self, method: str, *args: object) -> tuple[object, float]:
        started = perf_counter()
        with self._resolver_semaphore:
            result = getattr(self.resolver, method)(*args)
        return result, round((perf_counter() - started) * 1000, 2)

    def _search(
        self, run_id: str, query: str, freshness: FreshnessClass
    ) -> tuple[list[SearchResultRecord], float]:
        started = perf_counter()
        try:
            results = self.search_client.search(SearchRequest(
                query, getattr(self.settings, "max_search_results_per_query"), freshness
            ))
        except Exception as exc:
            self._record_operation_error(run_id, "search", "search", exc)
            raise
        return results, round((perf_counter() - started) * 1000, 2)

    def _acquire(
        self, run_id: str, result: SearchResultRecord, search_result_id: str | None, freshness: FreshnessClass
    ) -> tuple[str, str, list[str], list[str], float]:
        cached = self.storage.get_cached_source_by_url(result.url, self._min_fetched_at(freshness))
        if cached:
            self.storage.link_run_source(run_id, cached["source_id"])
            reference = self.storage.get_source_references([cached["source_id"]])[0]
            self._record_cache_hit(run_id, reference, "url")
            return cached["source_id"], cached["text"], [], [], 0.0
        started = perf_counter()
        try:
            self._set_stage(run_id, "fetch")
            fetched = self.fetcher.fetch(result.url)
            elapsed = round((perf_counter() - started) * 1000, 2)
            source_id = self.storage.upsert_source(
                run_id, fetched.url, fetched.title, fetched.site_name, fetched.published_at, fetched.text,
                {"metadata": fetched.metadata, "markdown": fetched.markdown, "raw_html": fetched.raw_html,
                 "retrieved_via": fetched.retrieved_via},
                fetched.source_kind, search_result_id, fetched.fetch_error,
            )
            self._record_source_fetched(
                run_id,
                source_id,
                search_result_id=search_result_id,
                url=fetched.url,
                title=fetched.title,
                provider=fetched.retrieved_via or self.settings.fetch_provider,
                source_kind=fetched.source_kind.value,
                published_at=fetched.published_at,
                duration_ms=elapsed,
                fallback_content=False,
            )
            return source_id, fetched.text, [], [], elapsed
        except FetchError as exc:
            self._record_operation_error(run_id, "fetch", "fetch", exc)
            fallback = result.inline_text or "\n".join(result.highlights)
            if not fallback:
                raise
            elapsed = round((perf_counter() - started) * 1000, 2)
            source_id = self.storage.upsert_source(
                run_id, result.url, result.title, result.site_name, result.published_at, fallback,
                {"provider_result": result.raw_result, "provider": result.provider},
                SourceKind.SEARCH_RESULT_FALLBACK, search_result_id, str(exc),
            )
            self._record_source_fetched(
                run_id,
                source_id,
                search_result_id=search_result_id,
                url=result.url,
                title=result.title,
                provider=result.provider or self.settings.search_provider,
                source_kind=SourceKind.SEARCH_RESULT_FALLBACK.value,
                published_at=result.published_at,
                duration_ms=elapsed,
                fallback_content=True,
            )
            return (
                source_id, fallback,
                [f"Used search-provider content for {result.url} because page fetch failed."],
                [f"Citation for {result.url} came from search-provider content after fetch failure."],
                elapsed,
            )

    def _select_evidence(
        self, run_id: str, source_id: str, text: str, question: str,
        remaining_questions: list[str], budget: RunBudget, current_evidence: list[EvidenceItem],
        note: str = "Selected relevant extract",
    ) -> EvidenceItem | None:
        if budget.evidence_remaining <= 0:
            return None
        max_chars = getattr(self.settings, "max_evidence_characters_per_item")
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if chunk.strip()]
        terms = set(re.findall(r"[a-z0-9]+", " ".join([question, *remaining_questions]).lower()))
        procedural = self._is_procedural(question)
        scored: list[tuple[int, int, str]] = []
        for index, chunk in enumerate(chunks):
            lowered = chunk.lower()
            tokens = set(re.findall(r"[a-z0-9]+", lowered))
            score = len(terms & tokens) * 3
            if procedural:
                score += sum(3 for signal in ACTIONABLE_SECTION_SIGNALS if signal in lowered)
                score += sum(2 for signal in IMPERATIVE_SIGNALS if signal in lowered)
                score += min(4, len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:g|kg|ml|cup|cups|tsp|tbsp|minutes?|mins?|hours?|hrs?|°f|°c)\b", lowered)))
            link_count = chunk.count("](")
            if link_count >= 4:
                score -= 20
            if len(chunk) < 40:
                score -= 2
            scored.append((score, index, chunk))
        useful = [item for item in scored if item[0] > 0]
        candidates = useful or scored
        selected = sorted(sorted(candidates, key=lambda item: item[0], reverse=True)[:12], key=lambda item: item[1])
        excerpt = "\n\n".join(chunk for _score, _index, chunk in selected)[:max_chars].strip()
        if not excerpt:
            return None
        source_limit = min(max_chars, getattr(self.settings, "max_synthesis_input_characters"))
        budget.evidence_remaining -= 1
        return self.storage.add_evidence_item(run_id, source_id, excerpt[:source_limit], note)

    def _is_procedural(self, question: str) -> bool:
        lowered = question.lower().strip()
        return any(signal in lowered for signal in PROCEDURAL_SIGNALS)

    def _answer_quality_issue(self, question: str, answer: str) -> str | None:
        lowered = answer.lower().strip()
        if not lowered:
            return f"Find a source that directly answers: {question}"
        if not self._is_procedural(question):
            return None
        step_count = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*]\s+)", answer))
        imperative_count = sum(1 for signal in IMPERATIVE_SIGNALS if signal in lowered)
        if step_count < 3 and imperative_count < 3:
            return "Find enough evidence to provide a concrete, ordered set of instructions."
        if any(signal in question.lower() for signal in RECIPE_SIGNALS):
            quantities = re.findall(
                r"\b\d+(?:\.\d+)?\s*(?:g|kg|ml|cup|cups|tsp|tbsp|minutes?|mins?|hours?|hrs?|°f|°c)\b",
                lowered,
            )
            if len(quantities) < 4:
                return "Find a complete recipe with ingredient quantities, timings, and baking temperature."
        return None

    def _remaining_questions_after_quality(
        self, question: str, draft: SynthesisDraft, limitations: list[str]
    ) -> list[str]:
        if not draft.source_answers_question:
            if not draft.remaining_questions:
                draft.remaining_questions = [f"Find a source that directly answers: {question}"]
            return draft.remaining_questions
        quality_issue = self._answer_quality_issue(question, draft.answer)
        if not draft.remaining_questions and quality_issue:
            draft.remaining_questions = [quality_issue]
            if quality_issue not in limitations:
                limitations.append(quality_issue)
        return draft.remaining_questions

    def _evidence_from_sources(
        self, run_id: str, source_ids: list[str], question: str, budget: RunBudget, note: str
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for source_id in source_ids:
            if budget.evidence_remaining <= 0:
                break
            text = self.storage.get_extract_text(source_id)
            if not text:
                continue
            self.storage.link_run_source(run_id, source_id)
            item = self._select_evidence(run_id, source_id, text, question, [], budget, evidence, note)
            if item:
                evidence.append(item)
        return evidence

    def _synthesize(
        self, run_id: str, question: str, evidence: EvidenceItem,
        prior_draft: SynthesisDraft | None, timings: dict[str, list[float]],
    ) -> SynthesisDraft:
        self._raise_if_cancelled(run_id)
        max_chars = getattr(self.settings, "max_synthesis_input_characters")
        bounded = EvidenceItem(
            evidence.evidence_id, evidence.source_id, evidence.excerpt[:max_chars], evidence.note
        )
        self._set_stage(run_id, "synthesis")
        draft, elapsed = self._call_resolver("synthesize", question, [bounded], prior_draft)
        self._record_timing(timings, "resolver.synthesize", elapsed)
        if not draft.source_answers_question:
            draft = SynthesisDraft(
                summary=prior_draft.summary if prior_draft else "",
                answer=prior_draft.answer if prior_draft else "",
                cited_source_ids=list(prior_draft.cited_source_ids) if prior_draft else [],
                remaining_questions=draft.remaining_questions or (
                    list(prior_draft.remaining_questions) if prior_draft else [question]
                ),
                source_answers_question=False,
            )
        allowed = {bounded.source_id, *(prior_draft.cited_source_ids if prior_draft else [])}
        illegal = set(draft.cited_source_ids) - allowed
        if illegal:
            raise ResolverError(f"Synthesis cited unknown source IDs: {sorted(illegal)}")
        if draft.answer and not draft.cited_source_ids:
            raise ResolverError("Synthesis returned an answer without citing grounded evidence.")
        telemetry = self._get_telemetry(run_id)
        if telemetry:
            telemetry.syntheses += 1
        self._emit_run_event(
            run_id,
            "research.synthesis.completed",
            {
                "run_id": run_id,
                "source_id": bounded.source_id,
                "cited_source_ids": draft.cited_source_ids,
                "source_answers_question": draft.source_answers_question,
                "remaining_question_count": len(draft.remaining_questions),
                "summary_characters": len(draft.summary),
                "answer_characters": len(draft.answer),
                "duration_ms": elapsed,
            },
            subject=bounded.source_id,
        )
        return draft

    def _finish_incomplete(
        self, run_id: str, request: ResearchRequest, draft: SynthesisDraft | None,
        warnings: list[str], limitations: list[str], reason: StopReason, timings: dict[str, list[float]],
    ) -> ResearchResult | ResearchErrorResult:
        if draft and draft.answer and draft.cited_source_ids and not self._answer_quality_issue(request.question, draft.answer):
            limitations.append("Research stopped before all remaining questions were answered.")
            return self._finalize(run_id, request, draft, reason, timings, warnings, limitations, status="partial")
        self.storage.update_run_status(run_id, "failed", StopReason.INSUFFICIENT_EVIDENCE.value)
        message = "The service could not gather enough evidence to answer the question."
        self._record_run_finished(
            run_id,
            "research.run.failed",
            "error",
            StopReason.INSUFFICIENT_EVIDENCE,
            timings,
            error_type="InsufficientEvidence",
            error_message=message,
        )
        return ResearchErrorResult(
            "error", run_id, "research", message,
            StopReason.INSUFFICIENT_EVIDENCE, debug=self._build_debug(run_id, request, timings),
        )

    def _finalize(
        self, run_id: str, request: ResearchRequest, draft: SynthesisDraft, reason: StopReason,
        timings: dict[str, list[float]], warnings: list[str] | None = None,
        limitations: list[str] | None = None, status: str = "ok",
    ) -> ResearchResult:
        self._raise_if_cancelled(run_id)
        references = self.storage.get_source_references(draft.cited_source_ids)[: max(0, request.max_references)]
        self.storage.save_final_answer(run_id, draft.summary, draft.answer, references)
        self.storage.update_run_status(run_id, "completed" if status == "ok" else "partial", reason.value)
        self._record_run_finished(
            run_id,
            "research.run.completed" if status == "ok" else "research.run.partial",
            status,
            reason,
            timings,
            reference_ids=[item.source_id for item in references],
        )
        self._trace(run_id, "COMPLETED", [f"status: {status}", f"stop_reason: {reason.value}"])
        return ResearchResult(
            status, run_id, draft.summary, draft.answer, references,
            draft.warnings + (warnings or []), draft.limitations + (limitations or []),
            reason, self._build_debug(run_id, request, timings),
        )

    def _raise_if_cancelled(self, run_id: str) -> None:
        if self.storage.is_cancel_requested(run_id):
            raise ResearchCancelled("Run was cancelled before completion.")

    def _min_fetched_at(self, freshness: FreshnessClass) -> str:
        ttl = getattr(self.settings, "cache_recent_ttl_seconds" if freshness == FreshnessClass.RECENT else "cache_evergreen_ttl_seconds")
        return (datetime.now(UTC) - timedelta(seconds=ttl)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _record_timing(self, timings: dict[str, list[float]], key: str, elapsed: float) -> None:
        timings.setdefault(key, []).append(elapsed)

    def _build_debug(self, run_id: str, request: ResearchRequest, timings: dict[str, list[float]]) -> dict[str, object] | None:
        if request.response_detail != ResponseDetail.DEBUG and not getattr(self.settings, "include_timing_debug", False):
            return None
        payload: dict[str, object] = {
            "reasoning_options": self.resolver.reasoning_request_options(),
            "timings": {key: {"count": len(values), "total_ms": round(sum(values), 2)} for key, values in timings.items()},
        }
        if request.response_detail == ResponseDetail.DEBUG:
            payload["run_events"] = self.storage.get_event_payloads(run_id)
        return payload

    def _begin_logs(self, run_id: str, question: str) -> None:
        begin = getattr(self.resolver, "begin_request_debug_log", None)
        if callable(begin):
            begin(run_id, question)
        self._trace(run_id, "RUN STARTED", [f"question: {question}"])

    def _trace(self, run_id: str, section: str, lines: list[str]) -> None:
        path = Path(getattr(self.settings, "fetch_debug_log_path"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with _FETCH_LOG_LOCK, path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(f"=== {section} ===\nrun_id: {run_id}\n")
            for line in lines:
                handle.write(f"{valid_unicode(line)}\n")
            handle.write("\n")
