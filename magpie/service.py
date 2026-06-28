"""Bounded, state-aware research orchestration."""

from __future__ import annotations

import logging
import re
import threading
import uuid
import httpx
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

from .errors import FetchError, ResearchCancelled, ResolverError, SearchError, StorageError
from .evidence import EvidenceSelector
from .telemetry import RunTelemetry, TelemetryEmitter
from .runcontext import RunContext
from .config import Settings
from .historian import HistorianSink, NullHistorianSink
from .models import (
    EvidenceItem, FetchResult, FreshnessClass, IndexedSearchResult,
    IndexedSearchResultItem, PlanningContext,
    ResearchErrorResult, ResearchRequest, ResearchResult, RequestRoute, ResponseDetail, RunBudget,
    SearchRequest, SearchResultRecord, SourceKind, SpecializedRouteResult, StopReason, SynthesisDraft, to_jsonable,
)
from .providers.base import AnimeClient, Fetcher, NewsClient, ResolverClient, SearchClient, WeatherClient
from .storage import SQLiteStorage, canonicalize_url, normalize_query
from .routes import try_specialized_route


RECENT_SIGNALS = {"latest", "current", "today", "yesterday", "this week", "this month", "this year"}
_FETCH_LOG_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)

# Exceptions that represent expected, provider/infrastructure-level failures of
# a research run. They finalize the run as a normal (terminal) failure rather
# than an internal bug. `httpx.HTTPError` is included because the resolver
# adapter intentionally re-raises network errors (e.g. timeouts) from
# `_ask_json`; those are provider failures, not programming bugs.
_DOMAIN_RUN_ERRORS: tuple[type[BaseException], ...] = (
    ResolverError, SearchError, FetchError, StorageError, httpx.HTTPError,
)


def detect_freshness_class(question: str) -> FreshnessClass:
    lowered = question.lower()
    if any(signal in lowered for signal in RECENT_SIGNALS):
        return FreshnessClass.RECENT
    current_year = datetime.now(UTC).year
    years = {int(value) for value in re.findall(r"\b20\d{2}\b", question)}
    if any(current_year - 1 <= year <= current_year for year in years):
        return FreshnessClass.RECENT
    return FreshnessClass.EVERGREEN


class RouteContext:
    """Narrow interface handed to specialized routes.

    Exposes only the mid-flow plumbing and specialized clients that routes
    need, so :mod:`magpie.routes` depends on this named interface rather than
    reaching into :class:`ResearchService` private helpers. The service
    builds one per run and passes it to :func:`try_specialized_route`.

    Terminal persistence and event emission stay on the service via
    :meth:`ResearchService.finalize_specialized_route`; routes return a
    :class:`~magpie.models.SpecializedRouteResult` and the service finalizes.
    """

    __slots__ = (
        "_service",
        "weather_client",
        "anime_client",
        "news_client",
        "settings",
    )

    def __init__(self, service: "ResearchService") -> None:
        self._service = service
        # Specialized clients are immutable for the lifetime of a run.
        self.weather_client = service.weather_client
        self.anime_client = service.anime_client
        self.news_client = service.news_client
        self.settings = service.settings

    def set_stage(self, run_id: str, stage: str) -> None:
        self._service._set_stage(run_id, stage)

    def select_route(self, run_id: str, route: str, fallback_reason: str | None = None) -> None:
        self._service._select_route(run_id, route, fallback_reason)

    def call_resolver(self, operation: str, *args: Any) -> Any:
        return self._service._call_resolver(operation, *args)

    def record_timing(self, timings: dict[str, list[float]], key: str, elapsed: float) -> None:
        self._service._record_timing(timings, key, elapsed)

    def trace(self, run_id: str, section: str, lines: list[str]) -> None:
        self._service._trace(run_id, section, lines)

    def record_operation_error(
        self, run_id: str, component: str, operation: str | None, exc: Exception
    ) -> None:
        self._service._record_operation_error(run_id, component, operation, exc)


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
    _resolver_semaphore: threading.Lock = field(default_factory=threading.Lock)
    # Derived collaborators, built in __post_init__ from the fields above so
    # the constructor signature (a public contract used by tests, app.py, and
    # a2a.py) stays unchanged. ``init=False`` keeps them out of __init__.
    _evidence: EvidenceSelector = field(init=False)
    _telemetry_emitter: TelemetryEmitter = field(init=False)

    def __post_init__(self) -> None:
        self._evidence = EvidenceSelector(self.storage, self.settings)
        self._telemetry_emitter = TelemetryEmitter(self.historian_sink, self.settings)

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
        # Delegated to TelemetryEmitter. Kept as a thin shim so existing
        # callers (including tests reaching this private method) keep working.
        return self._telemetry_emitter.emit(
            event_type, data,
            subject=subject, correlation_id=correlation_id,
            causation_id=causation_id, source=source,
        )

    def _emit_run_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        subject: str | None = None,
        source: str = "app://magpie/research",
    ) -> str:
        return self._telemetry_emitter.emit_run_event(run_id, event_type, data, subject=subject, source=source)

    def _sanitize_event_data(self, value: Any) -> Any:
        return self._telemetry_emitter.sanitize_event_data(value)

    def _get_telemetry(self, run_id: str) -> RunTelemetry | None:
        return self._telemetry_emitter.get_telemetry(run_id)

    def _set_stage(self, run_id: str, stage: str) -> None:
        self._telemetry_emitter.set_stage(run_id, stage)

    def _stage(self, run_id: str) -> str:
        return self._telemetry_emitter.stage(run_id)

    def _select_route(self, run_id: str, route: str, fallback_reason: str | None = None) -> None:
        self._telemetry_emitter.select_route(run_id, route, fallback_reason)

    def _record_operation_error(
        self, run_id: str, component: str, operation: str | None, exc: Exception
    ) -> None:
        self._telemetry_emitter.record_operation_error(run_id, component, operation, exc)

    def _record_query_executed(
        self,
        run_id: str,
        query_id: str,
        query: str,
        freshness: FreshnessClass,
        result_count: int,
        elapsed_ms: float,
    ) -> None:
        self._telemetry_emitter.record_query_executed(
            run_id, query_id, query, freshness, result_count, elapsed_ms
        )

    def _record_source_discovered(
        self,
        run_id: str,
        search_result_id: str | None,
        result: SearchResultRecord,
        canonical_url: str,
    ) -> None:
        self._telemetry_emitter.record_source_discovered(
            run_id, search_result_id, result, canonical_url
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
        self._telemetry_emitter.record_source_fetched(
            run_id, source_id,
            search_result_id=search_result_id, url=url, title=title, provider=provider,
            source_kind=source_kind, published_at=published_at,
            duration_ms=duration_ms, fallback_content=fallback_content,
        )

    def _record_specialized_source(
        self, run_id: str, reference: Any, provider: str, duration_ms: float
    ) -> None:
        self._telemetry_emitter.record_specialized_source(run_id, reference, provider, duration_ms)

    def _record_source_rejected(
        self, run_id: str, source_id: str | None, query: str | None, reason: str
    ) -> None:
        self._telemetry_emitter.record_source_rejected(run_id, source_id, query, reason)

    def _record_cache_hit(self, run_id: str, reference: Any, cache_kind: str) -> None:
        self._telemetry_emitter.record_cache_hit(run_id, reference, cache_kind)

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
        self._telemetry_emitter.record_run_finished(
            run_id, event_type, status, reason, timings,
            reference_ids=reference_ids, error_type=error_type, error_message=error_message,
        )

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
        self._telemetry_emitter.register_run(run_id, started_event_id, freshness)
        budget = RunBudget(
            queries_remaining=self.settings.max_search_queries_per_run,
            sources_remaining=self.settings.max_sources_per_run,
            evidence_remaining=self.settings.max_evidence_items_per_run,
        )
        ctx = RunContext(budget=budget)

        try:
            self._raise_if_cancelled(run_id)
            route_ctx = RouteContext(self)
            specialized_result = try_specialized_route(route_ctx, run_id, request, timings, ctx.warnings)
            if specialized_result is not None:
                return self.finalize_specialized_route(
                    run_id, request, timings, ctx.warnings, specialized_result
                )
            self._select_route(run_id, RequestRoute.WEB_RESEARCH.value)
            cached_ids = self.storage.find_fresh_source_ids_for_exact_query(
                normalize_query(request.question), self._min_fetched_at(freshness)
            )
            if cached_ids:
                ctx.seen_urls.update(self.storage.get_canonical_urls(cached_ids))
                for reference in self.storage.get_source_references(cached_ids):
                    self._record_cache_hit(run_id, reference, "exact_query")
                with self.storage.transaction():
                    cached_evidence = self._evidence_from_sources(
                        run_id, cached_ids, request.question, ctx, "Reused from exact-query cache"
                    )
                    if cached_evidence:
                        ctx.evidence.extend(cached_evidence)
                        ctx.last_draft = self._synthesize(run_id, request.question, cached_evidence, ctx.last_draft, timings)
                        ctx.remaining_questions = self._remaining_questions_after_quality(
                            run_id, request, ctx
                        )
                        if not ctx.last_draft.source_answers_question:
                            for item in cached_evidence:
                                self.storage.reject_source_for_query(normalize_query(request.question), item.source_id)
                                self._record_source_rejected(
                                    run_id, item.source_id, normalize_query(request.question),
                                    "source_did_not_answer_question",
                                )
                    if not ctx.remaining_questions:
                        return self._finalize(run_id, request, ctx, StopReason.ANSWERED_FROM_CACHE, timings)

            while ctx.budget.queries_remaining > 0 and ctx.budget.sources_remaining > 0 and ctx.budget.evidence_remaining > 0:
                self._raise_if_cancelled(run_id)
                prior_queries = self.storage.list_queries_for_run(run_id)
                planning = PlanningContext(prior_queries, sorted(ctx.seen_urls), ctx.remaining_questions, ctx.budget)
                proposal, elapsed = self._call_resolver("propose_query", request.question, planning)
                self._trace(run_id, "QUERY PROPOSED", [f"query: {proposal.query}", f"elapsed_ms: {elapsed}"])
                self._record_timing(timings, "resolver.propose_query", elapsed)
                query = normalize_query(proposal.query)
                if not query or query in prior_queries:
                    ctx.limitations.append("Planner could not produce a new useful query.")
                    return self._finish_incomplete(
                        run_id, request, ctx, StopReason.NO_PROGRESS, timings
                    )
                ctx.budget.queries_remaining -= 1
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
                        if canonical not in ctx.seen_urls:
                            ctx.seen_urls.add(canonical)
                            candidates.append(result)
                            self._record_source_discovered(
                                run_id, result_ids.get(result.url), result, canonical
                            )
                        if len(candidates) >= min(
                            self.settings.max_sources_per_query, ctx.budget.sources_remaining
                        ):
                            break
                    if not candidates:
                        ctx.limitations.append(f"No new sources found for query: {proposal.query}")
                        continue

                    round_evidence: list[EvidenceItem] = []
                    for result in candidates:
                        if ctx.budget.evidence_remaining <= 0:
                            break
                        self._raise_if_cancelled(run_id)
                        ctx.budget.sources_remaining -= 1
                        source_id, text, new_warnings, new_limitations, elapsed = self._acquire(
                            run_id, result, result_ids.get(result.url), freshness
                        )
                        self._record_timing(timings, "fetch", elapsed)
                        ctx.warnings.extend(new_warnings)
                        ctx.limitations.extend(new_limitations)
                        item = self._select_evidence(
                            run_id, source_id, text, request.question, ctx.remaining_questions, ctx.budget, [],
                        )
                        if item:
                            round_evidence.append(item)
                            self._trace(run_id, "EVIDENCE SELECTED", [
                                f"source_id: {item.source_id}",
                                f"source_characters: {len(item.excerpt)}",
                            ])
                    if not round_evidence:
                        continue
                    ctx.evidence.extend(round_evidence)
                    ctx.last_draft = self._synthesize(run_id, request.question, round_evidence, ctx.last_draft, timings)
                    if not ctx.last_draft.source_answers_question:
                        for item in round_evidence:
                            self.storage.reject_source_for_query(query, item.source_id)
                            self._record_source_rejected(
                                run_id, item.source_id, query, "source_did_not_answer_question"
                            )
                    ctx.remaining_questions = self._remaining_questions_after_quality(
                        run_id, request, ctx
                    )
                    if not ctx.remaining_questions:
                        return self._finalize(
                            run_id, request, ctx, StopReason.NEEDED_NEW_SEARCH, timings,
                        )

            return self._finish_incomplete(
                run_id, request, ctx, StopReason.BUDGET_EXHAUSTED, timings
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
        except _DOMAIN_RUN_ERRORS as exc:
            return self._finish_run_failure(run_id, request, timings, exc, StopReason.FAILED)
        except Exception as exc:  # noqa: BLE001
            # Unexpected (non-domain) exception: a logic bug, storage failure, or
            # other programming error. Surface it as a distinct internal-error
            # terminal so it is distinguishable from provider failures, while
            # still honoring the durable-run contract (finalize + return a result).
            LOGGER.exception("Internal error during research run %s", run_id)
            return self._finish_run_failure(run_id, request, timings, exc, StopReason.INTERNAL_ERROR)
        finally:
            self._clear_run_state(run_id)

    def _clear_run_state(self, run_id: str) -> None:
        """Drop in-memory run state (telemetry, terminal-event guard) after a run ends."""
        self._telemetry_emitter.clear_run(run_id)

    def _finish_run_failure(
        self,
        run_id: str,
        request: ResearchRequest,
        timings: dict[str, list[float]],
        exc: BaseException,
        reason: StopReason,
    ) -> ResearchErrorResult:
        """Finalize a failed run and return its error result.

        Used by both the domain-failure (``FAILED``) and internal-error
        (``INTERNAL_ERROR``) terminal paths so both honor the durable-run
        contract: update run status, append a failure event, record the
        operation error if it was not already attributed, and emit the
        ``research.run.failed`` terminal with the exception's type/message.
        """
        self.storage.update_run_status(run_id, "failed", reason.value)
        self.storage.append_event(run_id, "run_failed", {"error": str(exc)})
        telemetry = self._get_telemetry(run_id)
        if telemetry is None or not telemetry.operation_error_recorded:
            stage = self._stage(run_id)
            self._record_operation_error(run_id, stage, stage, exc)
        self._record_run_finished(
            run_id, "research.run.failed", "error", reason, timings,
            error_type=exc.__class__.__name__, error_message=str(exc),
        )
        return ResearchErrorResult(
            "error", run_id, "research", str(exc), reason,
            debug=self._build_debug(run_id, request, timings),
        )

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
        except _DOMAIN_RUN_ERRORS as exc:
            self.storage.update_run_status(run_id, "failed", StopReason.FAILED.value)
            self.storage.append_event(run_id, "run_failed", {"error": str(exc)})
            raise
        except Exception as exc:  # noqa: BLE001
            # Unexpected (non-domain) failure: finalize as an internal error so it
            # is distinguishable from provider failures, then re-raise to keep
            # search()'s "raise on failure" contract intact for callers.
            LOGGER.exception("Internal error during search run %s", run_id)
            self.storage.update_run_status(run_id, "failed", StopReason.INTERNAL_ERROR.value)
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
        owns_run = run_id is None
        fetch_run_id = run_id or str(uuid.uuid4())
        timings: dict[str, list[float]] = {}
        if owns_run:
            freshness = detect_freshness_class(url)
            self.storage.create_run(url, None, freshness, ResponseDetail.COMPACT.value, run_id=fetch_run_id)
            self._begin_logs(fetch_run_id, url)
            self._start_run(fetch_run_id, url, None, freshness, ResponseDetail.COMPACT)
            self._select_route(fetch_run_id, RequestRoute.WEB_RESEARCH.value)
        started = perf_counter()
        try:
            fetched = self.fetcher.fetch(url)
            elapsed = round((perf_counter() - started) * 1000, 2)
            self._record_timing(timings, "fetch", elapsed)
            source_id = self.storage.upsert_source(
                fetch_run_id, fetched.url, fetched.title, fetched.site_name, fetched.published_at, fetched.text,
                {"metadata": fetched.metadata, "markdown": fetched.markdown, "raw_html": fetched.raw_html,
                 "retrieved_via": fetched.retrieved_via},
                fetched.source_kind, None, fetched.fetch_error,
            )
            content = fetched.markdown or fetched.text
            if owns_run:
                self.storage.update_run_status(
                    fetch_run_id, "completed", StopReason.NEEDED_NEW_SEARCH.value
                )
                self._record_run_finished(
                    fetch_run_id, "research.run.completed", "ok",
                    StopReason.NEEDED_NEW_SEARCH, timings, reference_ids=[source_id],
                )
            self._trace(fetch_run_id, "FETCH COMPLETED", [
                f"url: {url}", f"source_id: {source_id}", f"characters: {len(content)}", f"elapsed_ms: {elapsed}",
            ])
            return FetchResult(
                run_id=fetch_run_id, index=index, url=fetched.url,
                title=fetched.title, content=content, fetched_via="crawl4ai", warnings=warnings,
            )
        except FetchError as exc:
            if owns_run:
                self._finalize_fetch_failure(fetch_run_id, exc, timings)
            raise ResolverError(f"Failed to fetch {url}: {exc}") from exc
        finally:
            if owns_run:
                self._clear_run_state(fetch_run_id)

    def _start_run(
        self, run_id: str, question: str, run_label: str | None,
        freshness: FreshnessClass, response_detail: ResponseDetail,
    ) -> None:
        """Emit the run-started event and register run telemetry.

        Used by ``fetch`` (and available to other owned-run entry points) so a
        run has a ``research.run.started`` event and telemetry backing its
        terminal events. Called after the run row already exists in storage.
        """
        self.storage.append_event(run_id, "run_started", {"freshness_class": freshness.value})
        started_event_id = self._emit(
            "research.run.started",
            {
                "run_id": run_id,
                "question": question,
                "run_label": run_label,
                "freshness_class": freshness.value,
                "response_detail": response_detail.value,
            },
            subject=run_id,
            correlation_id=run_id,
        )
        self._telemetry_emitter.register_run(run_id, started_event_id, freshness)

    def _finalize_fetch_failure(
        self, run_id: str, exc: BaseException, timings: dict[str, list[float]],
    ) -> None:
        """Mark a fetch-owned run failed and emit the ``research.run.failed`` terminal."""
        self.storage.update_run_status(run_id, "failed", StopReason.FAILED.value)
        self.storage.append_event(run_id, "run_failed", {"error": str(exc)})
        self._record_run_finished(
            run_id, "research.run.failed", "error", StopReason.FAILED, timings,
            error_type=exc.__class__.__name__, error_message=str(exc),
        )

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
        # Delegated to EvidenceSelector. Kept as a thin shim so existing
        # callers (and tests reaching this private method) keep working.
        return self._evidence.select_evidence(
            run_id, source_id, text, question, remaining_questions, budget, current_evidence, note
        )

    def _is_procedural(self, question: str) -> bool:
        return self._evidence.is_procedural(question)

    def _answer_quality_issue(self, question: str, answer: str) -> str | None:
        return self._evidence.answer_quality_issue(question, answer)

    def _remaining_questions_after_quality(
        self, run_id: str, request: ResearchRequest, ctx: RunContext
    ) -> list[str]:
        question = request.question
        draft = ctx.last_draft
        if draft is None:
            return ctx.remaining_questions
        if not draft.source_answers_question:
            if not draft.remaining_questions:
                draft.remaining_questions = [f"Find a source that directly answers: {question}"]
            return draft.remaining_questions
        quality_issue = self._answer_quality_issue(question, draft.answer)
        if not draft.remaining_questions and quality_issue:
            draft.remaining_questions = [quality_issue]
            if quality_issue not in ctx.limitations:
                ctx.limitations.append(quality_issue)
        return draft.remaining_questions

    def _evidence_from_sources(
        self, run_id: str, source_ids: list[str], question: str, ctx: RunContext, note: str
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for source_id in source_ids:
            if ctx.budget.evidence_remaining <= 0:
                break
            text = self.storage.get_extract_text(source_id)
            if not text:
                continue
            self.storage.link_run_source(run_id, source_id)
            item = self._select_evidence(run_id, source_id, text, question, [], ctx.budget, evidence, note)
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
        self, run_id: str, request: ResearchRequest, ctx: RunContext,
        reason: StopReason, timings: dict[str, list[float]],
    ) -> ResearchResult | ResearchErrorResult:
        draft = ctx.last_draft
        if draft and draft.answer and draft.cited_source_ids and not self._answer_quality_issue(request.question, draft.answer):
            ctx.limitations.append("Research stopped before all remaining questions were answered.")
            return self._finalize(run_id, request, ctx, reason, timings, status="partial")
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
        self, run_id: str, request: ResearchRequest, ctx: RunContext, reason: StopReason,
        timings: dict[str, list[float]], status: str = "ok",
    ) -> ResearchResult:
        draft = ctx.last_draft
        if draft is None:
            # _finalize is only reached after a synthesis produced a draft; a
            # missing draft here is a programming error rather than a run outcome.
            raise ResolverError("Cannot finalize a run with no synthesis draft.")
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
            draft.warnings + ctx.warnings, draft.limitations + ctx.limitations,
            reason, self._build_debug(run_id, request, timings),
        )

    def finalize_specialized_route(
        self,
        run_id: str,
        request: ResearchRequest,
        timings: dict[str, list[float]],
        warnings: list[str],
        result: SpecializedRouteResult,
    ) -> ResearchResult:
        """Finalize a specialized (weather/anime/news) route.

        This is the single persistence/event seam for specialized routes.
        Routes return a :class:`~magpie.models.SpecializedRouteResult` and the
        service owns: durable finalize, per-source fetch telemetry (amortized
        across references), the terminal ``research.run.completed`` event, the
        ``COMPLETED`` trace line, and debug payload construction. Keeps
        :mod:`magpie.routes` off of private helpers.
        """
        references = result.references
        self._finalize_storage(
            run_id, result.summary, result.answer, references,
            "completed", result.stop_reason,
        )
        # News references come from one batched fetch; amortize the batch
        # duration across references rather than overstating each. For the
        # single-reference weather/anime routes the divisor is 1.
        per_source_elapsed = (
            round(result.elapsed_ms / len(references), 2) if references else 0.0
        )
        for reference in references:
            self._record_specialized_source(run_id, reference, result.provider, per_source_elapsed)
        self._record_run_finished(
            run_id,
            "research.run.completed",
            "ok",
            result.stop_reason,
            timings,
            reference_ids=[item.source_id for item in references],
        )
        self._trace(run_id, "COMPLETED", [
            "status: ok",
            f"route: {result.route_name}",
            f"reference_count: {len(references)}",
        ])
        return ResearchResult(
            "ok", run_id, result.summary, result.answer, references,
            warnings=warnings, stop_reason=result.stop_reason,
            debug=self._build_debug(run_id, request, timings),
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
