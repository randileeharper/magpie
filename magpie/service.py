"""Bounded, state-aware research orchestration."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import perf_counter
from pathlib import Path

from .errors import FetchError, ResearchCancelled, ResolverError, WeatherError
from .config import Settings
from .models import (
    EvidenceItem, FreshnessClass, PlanningContext, ResearchErrorResult, ResearchRequest,
    ResearchResult, RequestRoute, ResponseDetail, RunBudget, SearchRequest, SearchResultRecord,
    SourceKind, StopReason, SynthesisDraft, WeatherKind, to_jsonable,
)
from .providers.base import Fetcher, ResolverClient, SearchClient, WeatherClient
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
    _resolver_semaphore: threading.BoundedSemaphore = field(default=_GLOBAL_RESOLVER_GATE)

    def cancel_run(self, run_id: str) -> None:
        self.storage.request_cancel(run_id)
        self.storage.update_run_status(run_id, "cancelled", StopReason.CANCELLED.value)
        self.storage.append_event(run_id, "run_cancelled", {"run_id": run_id})

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
            if self.weather_client is not None:
                weather_result = self._try_weather_route(run_id, request, timings, warnings)
                if weather_result is not None:
                    return weather_result
            cached_ids = self.storage.find_fresh_source_ids_for_exact_query(
                normalize_query(request.question), self._min_fetched_at(freshness)
            )
            if cached_ids:
                seen_urls.update(self.storage.get_canonical_urls(cached_ids))
                evidence.extend(self._evidence_from_sources(
                    run_id, cached_ids, request.question, budget, "Reused from exact-query cache"
                ))
                for item in evidence:
                    last_draft = self._synthesize(run_id, request.question, item, last_draft, timings)
                    if not last_draft.source_answers_question:
                        self.storage.reject_source_for_query(normalize_query(request.question), item.source_id)
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
                results, elapsed = self._search(proposal.query, freshness)
                self._trace(run_id, "SEARCH RESULTS", [f"query: {proposal.query}", f"result_count: {len(results)}"])
                self._record_timing(timings, "search", elapsed)
                result_ids = self.storage.add_search_results(query_id, [to_jsonable(result) for result in results])
                candidates: list[SearchResultRecord] = []
                for result in results:
                    canonical = canonicalize_url(result.url)
                    if canonical not in seen_urls:
                        seen_urls.add(canonical)
                        candidates.append(result)
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
            self.storage.update_run_status(run_id, "cancelled", StopReason.CANCELLED.value)
            return ResearchErrorResult("error", run_id, "research", str(exc), StopReason.CANCELLED)
        except Exception as exc:  # noqa: BLE001
            self.storage.update_run_status(run_id, "failed", StopReason.FAILED.value)
            self.storage.append_event(run_id, "run_failed", {"error": str(exc)})
            return ResearchErrorResult(
                "error", run_id, "research", str(exc), StopReason.FAILED,
                debug=self._build_debug(run_id, request, timings),
            )

    def _try_weather_route(
        self, run_id: str, request: ResearchRequest, timings: dict[str, list[float]], warnings: list[str]
    ) -> ResearchResult | None:
        try:
            decision, elapsed = self._call_resolver("route_request", request.question)
            self._record_timing(timings, "resolver.route_request", elapsed)
            self._trace(run_id, "REQUEST ROUTED", [
                f"route: {decision.route.value}",
                f"weather_kind: {decision.weather_kind.value if decision.weather_kind else ''}",
                f"zip_code: {decision.zip_code or ''}",
                f"elapsed_ms: {elapsed}",
            ])
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Request routing failed; used web research instead: {exc}")
            self._trace(run_id, "REQUEST ROUTING FALLBACK", [f"error: {exc}"])
            return None
        if decision.route != RequestRoute.WEATHER:
            return None
        if not decision.zip_code:
            warnings.append("Weather route could not determine a US ZIP code; used web research instead.")
            return None

        started = perf_counter()
        try:
            report = self.weather_client.get_weather(
                decision.zip_code, decision.weather_kind or WeatherKind.CONDITIONS
            )
        except WeatherError as exc:
            warnings.append(f"Specialized weather lookup failed; used web research instead: {exc}")
            self._trace(run_id, "WEATHER ROUTE FALLBACK", [f"error: {exc}"])
            return None
        self._record_timing(timings, "weather", round((perf_counter() - started) * 1000, 2))
        references = [report.reference][: max(0, request.max_references)]
        self.storage.save_final_answer(run_id, report.summary, report.answer, references)
        self.storage.update_run_status(run_id, "completed", StopReason.SPECIALIZED_ROUTE.value)
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

    def _call_resolver(self, method: str, *args: object) -> tuple[object, float]:
        started = perf_counter()
        with self._resolver_semaphore:
            result = getattr(self.resolver, method)(*args)
        return result, round((perf_counter() - started) * 1000, 2)

    def _search(self, query: str, freshness: FreshnessClass) -> tuple[list[SearchResultRecord], float]:
        started = perf_counter()
        results = self.search_client.search(SearchRequest(
            query, getattr(self.settings, "max_search_results_per_query"), freshness
        ))
        return results, round((perf_counter() - started) * 1000, 2)

    def _acquire(
        self, run_id: str, result: SearchResultRecord, search_result_id: str | None, freshness: FreshnessClass
    ) -> tuple[str, str, list[str], list[str], float]:
        cached = self.storage.get_cached_source_by_url(result.url, self._min_fetched_at(freshness))
        if cached:
            self.storage.link_run_source(run_id, cached["source_id"])
            return cached["source_id"], cached["text"], [], [], 0.0
        started = perf_counter()
        try:
            fetched = self.fetcher.fetch(result.url)
            elapsed = round((perf_counter() - started) * 1000, 2)
            source_id = self.storage.upsert_source(
                run_id, fetched.url, fetched.title, fetched.site_name, fetched.published_at, fetched.text,
                {"metadata": fetched.metadata, "markdown": fetched.markdown, "raw_html": fetched.raw_html,
                 "retrieved_via": fetched.retrieved_via},
                fetched.source_kind, search_result_id, fetched.fetch_error,
            )
            return source_id, fetched.text, [], [], elapsed
        except FetchError as exc:
            fallback = result.inline_text or "\n".join(result.highlights)
            if not fallback:
                raise
            source_id = self.storage.upsert_source(
                run_id, result.url, result.title, result.site_name, result.published_at, fallback,
                {"provider_result": result.raw_result, "provider": result.provider},
                SourceKind.SEARCH_RESULT_FALLBACK, search_result_id, str(exc),
            )
            return (
                source_id, fallback,
                [f"Used search-provider content for {result.url} because page fetch failed."],
                [f"Citation for {result.url} came from search-provider content after fetch failure."],
                round((perf_counter() - started) * 1000, 2),
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
        return draft

    def _finish_incomplete(
        self, run_id: str, request: ResearchRequest, draft: SynthesisDraft | None,
        warnings: list[str], limitations: list[str], reason: StopReason, timings: dict[str, list[float]],
    ) -> ResearchResult | ResearchErrorResult:
        if draft and draft.answer and draft.cited_source_ids and not self._answer_quality_issue(request.question, draft.answer):
            limitations.append("Research stopped before all remaining questions were answered.")
            return self._finalize(run_id, request, draft, reason, timings, warnings, limitations, status="partial")
        self.storage.update_run_status(run_id, "failed", StopReason.INSUFFICIENT_EVIDENCE.value)
        return ResearchErrorResult(
            "error", run_id, "research", "The service could not gather enough evidence to answer the question.",
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
