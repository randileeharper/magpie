"""Bounded, state-aware research orchestration."""

from __future__ import annotations

import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

from .errors import FetchError, ResearchCancelled, ResolverError
from .config import Settings
from .historian import HistorianSink, NullHistorianSink, build_event
from .models import (
    EvidenceItem, FetchResult, FreshnessClass, IndexedSearchResult,
    IndexedSearchResultItem, PlanningContext,
    ResearchErrorResult, ResearchRequest, ResearchResult, RequestRoute, ResponseDetail, RunBudget,
    SearchRequest, SearchResultRecord, SourceKind, StopReason, SynthesisDraft, to_jsonable,
)
from .providers.base import AnimeClient, Fetcher, NewsClient, ResolverClient, SearchClient, WeatherClient
from .storage import SQLiteStorage, canonicalize_url, normalize_query
from .text import valid_unicode
from .routes import try_specialized_route


RECENT_SIGNALS = {"latest", "current", "today", "yesterday", "this week", "this month", "this year"}
PROCEDURAL_SIGNALS = ("how do i ", "how to ", "steps to ", "guide to ")
# Tokenize for evidence overlap scoring: word runs for space-delimited scripts
# (Latin, Cyrillic, …) plus individual characters for CJK scripts that don't
# use word boundaries, so Japanese/Chinese/Korean queries score correctly.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]", re.IGNORECASE)
ACTIONABLE_SECTION_SIGNALS = (
    "instruction", "method", "step", "directions", "preparation", "procedure",
    "how to", "process",
)
IMPERATIVE_SIGNALS = (
    "add ", "combine ", "connect ", "cover ", "create ", "enter ", "install ", "mix ",
    "place ", "press ", "remove ", "run ", "set ", "turn ", "type ", "wait ",
)
# Matches any measurement a procedural answer might cite, not only cooking ones.
_MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:"
    r"g|kg|mg|ml|l|cup|cups|tsp|tbsp|oz|lb|"
    r"minutes?|mins?|seconds?|secs?|hours?|hrs?|days?|"
    r"°f|°c|"
    r"px|pt|em|rem|"
    r"mb|gb|tb|kb"
    r")\b"
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
    current_year = datetime.now(UTC).year
    years = {int(value) for value in re.findall(r"\b20\d{2}\b", question)}
    if any(current_year - 1 <= year <= current_year for year in years):
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
        if self.storage.request_cancel(run_id) and self.storage.mark_run_cancelled(run_id):
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
            queries_remaining=self.settings.max_search_queries_per_run,
            sources_remaining=self.settings.max_sources_per_run,
            evidence_remaining=self.settings.max_evidence_items_per_run,
        )
        evidence: list[EvidenceItem] = []
        seen_urls: set[str] = set()
        warnings: list[str] = []
        limitations: list[str] = []
        remaining_questions: list[str] = []
        last_draft: SynthesisDraft | None = None

        try:
            self._raise_if_cancelled(run_id)
            specialized_result = try_specialized_route(self, run_id, request, timings, warnings)
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
                with self.storage.transaction():
                    cached_evidence = self._evidence_from_sources(
                        run_id, cached_ids, request.question, budget, "Reused from exact-query cache"
                    )
                    if cached_evidence:
                        evidence.extend(cached_evidence)
                        last_draft = self._synthesize(run_id, request.question, cached_evidence, last_draft, timings)
                        remaining_questions = self._remaining_questions_after_quality(
                            request.question, last_draft, limitations
                        )
                        if not last_draft.source_answers_question:
                            for item in cached_evidence:
                                self.storage.reject_source_for_query(normalize_query(request.question), item.source_id)
                                self._record_source_rejected(
                                    run_id, item.source_id, normalize_query(request.question),
                                    "source_did_not_answer_question",
                                )
                    if not remaining_questions:
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
                self._set_stage(run_id, "search")
                results, elapsed = self._search(run_id, proposal.query, freshness)
                self._raise_if_cancelled(run_id)
                self._trace(run_id, "SEARCH RESULTS", [f"query: {proposal.query}", f"result_count: {len(results)}"])
                self._record_timing(timings, "search", elapsed)
                with self.storage.transaction():
                    query_id = self.storage.add_query(
                        run_id, query, self.settings.search_provider, freshness
                    )
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
                            self.settings.max_sources_per_query, budget.sources_remaining
                        ):
                            break
                    if not candidates:
                        limitations.append(f"No new sources found for query: {proposal.query}")
                        continue

                    round_evidence: list[EvidenceItem] = []
                    for result in candidates:
                        if budget.evidence_remaining <= 0:
                            break
                        self._raise_if_cancelled(run_id)
                        budget.sources_remaining -= 1
                        source_id, text, new_warnings, new_limitations, elapsed = self._acquire(
                            run_id, result, result_ids.get(result.url), freshness
                        )
                        self._record_timing(timings, "fetch", elapsed)
                        warnings.extend(new_warnings)
                        limitations.extend(new_limitations)
                        item = self._select_evidence(
                            run_id, source_id, text, request.question, remaining_questions, budget, [],
                        )
                        if item:
                            round_evidence.append(item)
                            self._trace(run_id, "EVIDENCE SELECTED", [
                                f"source_id: {item.source_id}",
                                f"source_characters: {len(item.excerpt)}",
                            ])
                    if not round_evidence:
                        continue
                    evidence.extend(round_evidence)
                    last_draft = self._synthesize(run_id, request.question, round_evidence, last_draft, timings)
                    if not last_draft.source_answers_question:
                        for item in round_evidence:
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

    def search(self, query: str, *, max_results: int = 5, run_id: str | None = None) -> IndexedSearchResult:
        freshness = detect_freshness_class(query)
        run_id = self.storage.create_run(query, None, freshness, ResponseDetail.COMPACT.value, run_id=run_id)
        self._begin_logs(run_id, query)
        self._select_route(run_id, RequestRoute.WEB_RESEARCH.value)
        warnings: list[str] = []
        timings: dict[str, list[float]] = {}
        try:
            self._raise_if_cancelled(run_id)
            budget = RunBudget(
                queries_remaining=1,
                sources_remaining=max_results,
                evidence_remaining=max_results,
            )
            context = PlanningContext([], [], [], budget)
            proposal, elapsed = self._call_resolver("propose_query", query, context)
            self._record_timing(timings, "resolver.propose_query", elapsed)
            normalized = normalize_query(proposal.query)
            self._set_stage(run_id, "search")
            results, elapsed = self._search(run_id, proposal.query, freshness)
            self._record_timing(timings, "search", elapsed)
            with self.storage.transaction():
                query_id = self.storage.add_query(run_id, normalized, self.settings.search_provider, freshness)
                result_ids = self.storage.add_search_results(query_id, [to_jsonable(r) for r in results])
                seen_urls: set[str] = set()
                items: list[IndexedSearchResultItem] = []
                for result in results[:max_results]:
                    canonical = canonicalize_url(result.url)
                    if canonical in seen_urls:
                        continue
                    seen_urls.add(canonical)
                    self._record_source_discovered(run_id, result_ids.get(result.url), result, canonical)
                    content = result.inline_text or "\n".join(result.highlights) or result.snippet
                    if not content.strip():
                        warnings.append(f"No content extracted for {result.url}")
                        continue
                    budget.sources_remaining -= 1
                    source_id = self.storage.upsert_source(
                        run_id, result.url, result.title, result.site_name, result.published_at, content,
                        {"provider_result": result.raw_result, "provider": result.provider},
                        SourceKind.SEARCH_RESULT_FALLBACK, result_ids.get(result.url), None,
                    )
                    summary = result.snippet or content[:500]
                    items.append(IndexedSearchResultItem(
                        index=len(items),
                        title=result.title,
                        url=result.url,
                        site_name=result.site_name,
                        published_at=result.published_at,
                        summary=summary,
                    ))
                self.storage.update_run_status(run_id, "completed", StopReason.NEEDED_NEW_SEARCH.value)
                self._trace(run_id, "SEARCH COMPLETED", [f"result_count: {len(items)}"])
                return IndexedSearchResult(run_id=run_id, query=proposal.query, results=items, warnings=warnings)
        except Exception as exc:  # noqa: BLE001
            self.storage.update_run_status(run_id, "failed", StopReason.FAILED.value)
            self.storage.append_event(run_id, "run_failed", {"error": str(exc)})
            raise

    def fetch(
        self, *, run_id: str | None = None, index: int | None = None,
        url: str | None = None, full: bool = False,
    ) -> FetchResult:
        warnings: list[str] = []
        if index is not None and run_id is not None:
            stored = self._fetch_stored_index(run_id, index)
            if stored and not full:
                return stored
            if stored and full:
                url = stored.url
                title = stored.title
            else:
                raise ResolverError(f"No source found at index {index} for run {run_id}.")
        if not url:
            raise ResolverError("fetch requires either (run_id + index) or url.")
        fetch_run_id = run_id or str(uuid.uuid4())
        if not run_id:
            freshness = detect_freshness_class(url)
            self.storage.create_run(url, None, freshness, ResponseDetail.COMPACT.value, run_id=fetch_run_id)
            self._begin_logs(fetch_run_id, url)
        started = perf_counter()
        try:
            fetched = self.fetcher.fetch(url)
            elapsed = round((perf_counter() - started) * 1000, 2)
            source_id = self.storage.upsert_source(
                fetch_run_id, fetched.url, fetched.title, fetched.site_name, fetched.published_at, fetched.text,
                {"metadata": fetched.metadata, "markdown": fetched.markdown, "raw_html": fetched.raw_html,
                 "retrieved_via": fetched.retrieved_via},
                fetched.source_kind, None, fetched.fetch_error,
            )
            content = fetched.markdown or fetched.text
            self._trace(fetch_run_id, "FETCH COMPLETED", [
                f"url: {url}", f"source_id: {source_id}", f"characters: {len(content)}", f"elapsed_ms: {elapsed}",
            ])
            return FetchResult(
                run_id=fetch_run_id, index=index, url=fetched.url,
                title=fetched.title, content=content, fetched_via="crawl4ai", warnings=warnings,
            )
        except FetchError as exc:
            raise ResolverError(f"Failed to fetch {url}: {exc}") from exc

    def _fetch_stored_index(self, run_id: str, index: int) -> FetchResult | None:
        with self.storage._connect() as connection:
            rows = connection.execute(
                """SELECT s.source_id, s.raw_url, s.title, s.site_name, s.source_kind, d.text
                   FROM run_source_links rsl
                   JOIN sources s ON s.source_id = rsl.source_id
                   JOIN documents d ON d.document_id = s.document_id
                   WHERE rsl.run_id = ?
                   ORDER BY s.fetched_at""",
                (run_id,),
            ).fetchall()
        if index >= len(rows):
            return None
        row = rows[index]
        return FetchResult(
            run_id=run_id, index=index, url=row["raw_url"],
            title=row["title"], content=row["text"], fetched_via="stored",
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
                query, self.settings.max_search_results_per_query, freshness
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
        if result.inline_text and len(result.inline_text.strip()) >= 500:
            started = perf_counter()
            source_id = self.storage.upsert_source(
                run_id, result.url, result.title, result.site_name, result.published_at, result.inline_text,
                {"provider_result": result.raw_result, "provider": result.provider},
                SourceKind.SEARCH_RESULT_FALLBACK, search_result_id, None,
            )
            elapsed = round((perf_counter() - started) * 1000, 2)
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
                fallback_content=False,
            )
            return source_id, result.inline_text, [], [], elapsed
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
        max_chars = self.settings.max_evidence_characters_per_item
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n|(?<=[.!?])\s+", text) if chunk.strip()]
        terms = set(_TOKEN_PATTERN.findall(" ".join([question, *remaining_questions]).lower()))
        procedural = self._is_procedural(question)
        scored: list[tuple[int, int, str]] = []
        for index, chunk in enumerate(chunks):
            lowered = chunk.lower()
            tokens = set(_TOKEN_PATTERN.findall(lowered))
            score = len(terms & tokens) * 3
            if procedural:
                score += sum(3 for signal in ACTIONABLE_SECTION_SIGNALS if signal in lowered)
                score += sum(2 for signal in IMPERATIVE_SIGNALS if signal in lowered)
                score += min(4, len(_MEASUREMENT_PATTERN.findall(lowered)))
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
        source_limit = min(max_chars, self.settings.max_synthesis_input_characters)
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
        self, run_id: str, question: str, evidence: list[EvidenceItem],
        prior_draft: SynthesisDraft | None, timings: dict[str, list[float]],
    ) -> SynthesisDraft:
        self._raise_if_cancelled(run_id)
        self._set_stage(run_id, "synthesis")
        draft, elapsed = self._call_resolver("synthesize", question, evidence, prior_draft)
        self._record_timing(timings, "resolver.synthesize", elapsed)
        allowed = {
            *(item.source_id for item in evidence),
            *(prior_draft.cited_source_ids if prior_draft else []),
        }
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
                "evidence_count": len(evidence),
                "cited_source_ids": draft.cited_source_ids,
                "source_answers_question": draft.source_answers_question,
                "remaining_question_count": len(draft.remaining_questions),
                "summary_characters": len(draft.summary),
                "answer_characters": len(draft.answer),
                "duration_ms": elapsed,
            },
            subject=",".join(draft.cited_source_ids),
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
        self._finalize_storage(
            run_id,
            draft.summary,
            draft.answer,
            references,
            "completed" if status == "ok" else "partial",
            reason,
        )
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

    def _finalize_storage(
        self,
        run_id: str,
        summary: str,
        answer: str,
        references: list[Any],
        status: str,
        reason: StopReason,
    ) -> None:
        if not self.storage.finalize_run(
            run_id, summary, answer, references, status, reason.value
        ):
            raise ResearchCancelled("Run was cancelled before completion.")

    def _min_fetched_at(self, freshness: FreshnessClass) -> str:
        ttl = self.settings.cache_recent_ttl_seconds if freshness == FreshnessClass.RECENT else self.settings.cache_evergreen_ttl_seconds
        return (datetime.now(UTC) - timedelta(seconds=ttl)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _record_timing(self, timings: dict[str, list[float]], key: str, elapsed: float) -> None:
        timings.setdefault(key, []).append(elapsed)

    def _build_debug(self, run_id: str, request: ResearchRequest, timings: dict[str, list[float]]) -> dict[str, object] | None:
        if request.response_detail != ResponseDetail.DEBUG and not self.settings.include_timing_debug:
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
        path = self.settings.expanded_fetch_debug_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with _FETCH_LOG_LOCK, path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(f"=== {section} ===\nrun_id: {run_id}\n")
            for line in lines:
                handle.write(f"{self._sanitize_event_data(line)}\n")
            handle.write("\n")
