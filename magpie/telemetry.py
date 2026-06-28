"""Historian event emission and run telemetry, extracted from :class:`ResearchService`.

Owns the per-run :class:`RunTelemetry` counters, the terminal-event guard
(a run must emit exactly one terminal historian event), and secret
sanitization of event payloads. Wrapping all of this in one collaborator keeps
the service focused on orchestration and makes event emission independently
testable.

The emitter holds the historian sink and settings it needs to build and
deliver events; the service delegates its ``_emit`` / ``_record_*`` helpers
here.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .config import Settings
from .historian import HistorianSink, build_event
from .models import FreshnessClass, RequestRoute, SearchResultRecord, StopReason
from .storage import canonicalize_url
from .text import valid_unicode

# Historian delivery-failure warnings are emitted under the service's logger
# (not ``magpie.telemetry``) so they remain discoverable alongside the
# resolver/fetch diagnostics operators already monitor, and existing log
# assertions keep working. The emitter is the only place this fires from now.
LOGGER = logging.getLogger("magpie.service")


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


class TelemetryEmitter:
    """Builds and delivers historian events and tracks per-run counters.

    Holds the run telemetry registry, the terminal-event guard, and the lock
    guarding both. The service registers a run at start (``register_run``),
    records events through a run, and clears run state on completion
    (``clear_run``). The terminal guard ensures a run emits exactly one
    terminal event even under concurrent emit paths (e.g. a cancel racing the
    research loop).
    """

    _TERMINAL_EVENT_TYPES = frozenset({
        "research.run.completed",
        "research.run.partial",
        "research.run.failed",
        "research.run.canceled",
    })

    __slots__ = ("_historian_sink", "_settings", "_telemetry", "_terminal_emitted", "_telemetry_lock")

    def __init__(self, historian_sink: HistorianSink, settings: Settings) -> None:
        self._historian_sink = historian_sink
        self._settings = settings
        self._telemetry: dict[str, RunTelemetry] = {}
        self._terminal_emitted: set[str] = set()
        self._telemetry_lock = threading.Lock()

    # -- run lifecycle -------------------------------------------------

    def register_run(
        self, run_id: str, started_event_id: str, freshness: FreshnessClass
    ) -> None:
        """Register telemetry backing for a freshly started run."""
        with self._telemetry_lock:
            self._telemetry[run_id] = RunTelemetry(
                run_id, perf_counter(), started_event_id, freshness
            )

    def clear_run(self, run_id: str) -> None:
        """Drop in-memory run state (telemetry, terminal-event guard) after a run ends."""
        with self._telemetry_lock:
            self._telemetry.pop(run_id, None)
            self._terminal_emitted.discard(run_id)

    def get_telemetry(self, run_id: str) -> RunTelemetry | None:
        with self._telemetry_lock:
            return self._telemetry.get(run_id)

    # -- event emission -----------------------------------------------

    def emit(
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
            self.sanitize_event_data(data),
            source=source,
            subject=subject,
            correlation_id=correlation_id,
            causation_id=causation_id,
        )
        try:
            self._historian_sink.emit(event)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Historian delivery failed for event_id=%s type=%s: %s",
                event["id"],
                event_type,
                exc,
            )
        return str(event["id"])

    def emit_run_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        subject: str | None = None,
        source: str = "app://magpie/research",
    ) -> str:
        telemetry = self.get_telemetry(run_id)
        if self._is_terminal_event(event_type) and not self._mark_terminal_emitted(run_id):
            # A run must emit exactly one terminal historian event. If one has
            # already been emitted for this run (e.g. a cancel raced with the
            # research loop's own cancellation handler), drop the duplicate.
            return ""
        return self.emit(
            event_type,
            data,
            subject=subject or run_id,
            correlation_id=run_id,
            causation_id=telemetry.started_event_id if telemetry else None,
            source=source,
        )

    def _is_terminal_event(self, event_type: str) -> bool:
        return event_type in self._TERMINAL_EVENT_TYPES

    def _mark_terminal_emitted(self, run_id: str) -> bool:
        """Record that a terminal event is being emitted for a run.

        Returns True if this is the first terminal for the run (caller should
        emit), or False if a terminal was already emitted (caller should drop
        the duplicate). Guarded by the telemetry lock so concurrent emit paths
        for the same run cannot both observe "first".
        """
        with self._telemetry_lock:
            if run_id in self._terminal_emitted:
                return False
            self._terminal_emitted.add(run_id)
            return True

    def sanitize_event_data(self, value: Any) -> Any:
        secrets = [
            secret
            for secret in (
                self._settings.search_api_key,
                self._settings.resolver_api_key,
                self._settings.historian_token,
            )
            if secret
        ]
        if isinstance(value, str):
            sanitized = valid_unicode(value)
            for secret in secrets:
                sanitized = sanitized.replace(secret, "[REDACTED]")
            return sanitized
        if isinstance(value, dict):
            return {str(key): self.sanitize_event_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.sanitize_event_data(item) for item in value]
        return value

    # -- stage / route ------------------------------------------------

    def set_stage(self, run_id: str, stage: str) -> None:
        telemetry = self.get_telemetry(run_id)
        if telemetry:
            telemetry.stage = stage

    def stage(self, run_id: str) -> str:
        telemetry = self.get_telemetry(run_id)
        return telemetry.stage if telemetry else "research"

    def select_route(self, run_id: str, route: str, fallback_reason: str | None = None) -> None:
        telemetry = self.get_telemetry(run_id)
        if telemetry and telemetry.route == route and fallback_reason is None:
            return
        if telemetry:
            telemetry.route = route
            telemetry.stage = "route"
        self.emit_run_event(
            run_id,
            "research.route.selected",
            {
                "run_id": run_id,
                "route": route,
                "fallback_reason": fallback_reason,
            },
        )

    # -- recorders ----------------------------------------------------

    def record_operation_error(
        self, run_id: str, component: str, operation: str | None, exc: Exception
    ) -> None:
        telemetry = self.get_telemetry(run_id)
        if telemetry:
            telemetry.operation_error_recorded = True
        self.emit_run_event(
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

    def record_query_executed(
        self,
        run_id: str,
        query_id: str,
        query: str,
        freshness: FreshnessClass,
        result_count: int,
        elapsed_ms: float,
    ) -> None:
        telemetry = self.get_telemetry(run_id)
        if telemetry:
            telemetry.queries += 1
        self.emit_run_event(
            run_id,
            "research.query.executed",
            {
                "run_id": run_id,
                "query_id": query_id,
                "normalized_query": query,
                "provider": self._settings.search_provider,
                "freshness_class": freshness.value,
                "result_count": result_count,
                "duration_ms": elapsed_ms,
            },
            subject=query_id,
        )

    def record_source_discovered(
        self,
        run_id: str,
        search_result_id: str | None,
        result: SearchResultRecord,
        canonical_url: str,
    ) -> None:
        telemetry = self.get_telemetry(run_id)
        if telemetry:
            telemetry.sources_discovered += 1
        self.emit_run_event(
            run_id,
            "research.source.discovered",
            {
                "run_id": run_id,
                "search_result_id": search_result_id,
                "canonical_url": canonical_url,
                "title": result.title,
                "provider": result.provider or self._settings.search_provider,
                "published_at": result.published_at,
            },
            subject=search_result_id or canonical_url,
        )

    def record_source_fetched(
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
        telemetry = self.get_telemetry(run_id)
        if telemetry:
            telemetry.sources_fetched += 1
        self.emit_run_event(
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

    def record_specialized_source(
        self, run_id: str, reference: Any, provider: str, duration_ms: float
    ) -> None:
        self.record_source_fetched(
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

    def record_source_rejected(
        self, run_id: str, source_id: str | None, query: str | None, reason: str
    ) -> None:
        telemetry = self.get_telemetry(run_id)
        if telemetry:
            telemetry.sources_rejected += 1
        self.emit_run_event(
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

    def record_cache_hit(self, run_id: str, reference: Any, cache_kind: str) -> None:
        telemetry = self.get_telemetry(run_id)
        if telemetry:
            telemetry.cache_hits += 1
        self.emit_run_event(
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

    def record_run_finished(
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
        telemetry = self.get_telemetry(run_id)
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
        self.emit_run_event(run_id, event_type, data)
