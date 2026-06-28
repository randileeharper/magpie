"""Specialized request routes (weather, anime, news) for the research service.

These were extracted from :class:`magpie.service.ResearchService` for file
organization. They are standalone functions that receive a
:class:`~magpie.service.RouteContext` (a narrow interface over the service)
and return a :class:`~magpie.models.SpecializedRouteResult` on success or
``None`` to fall back to web research. They do not reach into
:class:`ResearchService` private helpers; the service owns all persistence and
event emission through :meth:`ResearchService.finalize_specialized_route`.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

import httpx

from .errors import AnimeError, NewsError, ResearchCancelled, ResolverError, WeatherError
from .models import (
    AnimeReport,
    AnimeRequestKind,
    NewsRequestKind,
    ResearchRequest,
    SpecializedRouteResult,
    RequestRoute,
    StopReason,
    WeatherKind,
)

if TYPE_CHECKING:
    from .service import RouteContext


def try_specialized_route(
    ctx: RouteContext,
    run_id: str,
    request: ResearchRequest,
    timings: dict[str, list[float]],
    warnings: list[str],
) -> SpecializedRouteResult | None:
    if ctx.weather_client is None and ctx.anime_client is None and ctx.news_client is None:
        return None
    try:
        ctx.set_stage(run_id, "route")
        decision, elapsed = ctx.call_resolver("route_request", request.question)
        ctx.record_timing(timings, "resolver.route_request", elapsed)
        ctx.trace(run_id, "REQUEST ROUTED", [
            f"route: {decision.route.value}",
            f"weather_kind: {decision.weather_kind.value if decision.weather_kind else ''}",
            f"zip_code: {decision.zip_code or ''}",
            f"elapsed_ms: {elapsed}",
        ])
    except (ResolverError, httpx.HTTPError) as exc:
        # Resolver failures (malformed responses or network errors) are an
        # expected, recoverable condition: fall back to web research. Other
        # exceptions (StorageError, programming bugs) propagate so they are not
        # silently swallowed.
        ctx.record_operation_error(run_id, "resolver", "route_request", exc)
        warnings.append(f"Request routing failed; used web research instead: {exc}")
        ctx.trace(run_id, "REQUEST ROUTING FALLBACK", [f"error: {exc}"])
        ctx.select_route(run_id, RequestRoute.WEB_RESEARCH.value, str(exc))
        return None
    ctx.select_route(run_id, decision.route.value)
    if decision.route == RequestRoute.ANIME and ctx.anime_client is not None:
        return try_anime_route(ctx, run_id, request, timings, warnings)
    if decision.route == RequestRoute.NEWS and ctx.news_client is not None:
        return try_news_route(ctx, run_id, request, timings, warnings)
    if decision.route != RequestRoute.WEATHER or ctx.weather_client is None:
        return None
    if not decision.zip_code:
        warnings.append("Weather route could not determine a US ZIP code; used web research instead.")
        ctx.select_route(
            run_id, RequestRoute.WEB_RESEARCH.value, "weather_zip_code_unavailable"
        )
        return None

    started = perf_counter()
    try:
        ctx.set_stage(run_id, "weather")
        report = ctx.weather_client.get_weather(
            decision.zip_code, decision.weather_kind or WeatherKind.CONDITIONS
        )
    except ResearchCancelled:
        raise
    except WeatherError as exc:
        ctx.record_operation_error(run_id, "weather", "get_weather", exc)
        warnings.append(f"Specialized weather lookup failed; used web research instead: {exc}")
        ctx.trace(run_id, "WEATHER ROUTE FALLBACK", [f"error: {exc}"])
        ctx.select_route(run_id, RequestRoute.WEB_RESEARCH.value, str(exc))
        return None
    elapsed = round((perf_counter() - started) * 1000, 2)
    ctx.record_timing(timings, "weather", elapsed)
    references = [report.reference][: max(0, request.max_references)]
    return SpecializedRouteResult(
        summary=report.summary,
        answer=report.answer,
        references=references,
        stop_reason=StopReason.SPECIALIZED_ROUTE,
        provider="neonhail",
        route_name="weather",
        elapsed_ms=elapsed,
    )


def try_anime_route(
    ctx: RouteContext,
    run_id: str,
    request: ResearchRequest,
    timings: dict[str, list[float]],
    warnings: list[str],
) -> SpecializedRouteResult | None:
    assert ctx.anime_client is not None
    try:
        ctx.set_stage(run_id, "anime")
        anime_request, elapsed = ctx.call_resolver("classify_anime_request", request.question)
        ctx.record_timing(timings, "resolver.classify_anime_request", elapsed)
        ctx.trace(run_id, "ANIME REQUEST CLASSIFIED", [
            f"kind: {anime_request.kind.value}",
            f"title_query: {anime_request.title_query or ''}",
            f"character_query: {anime_request.character_query or ''}",
            f"requested_fields: {', '.join(item.value for item in anime_request.requested_fields)}",
            f"elapsed_ms: {elapsed}",
        ])
        started = perf_counter()
        if anime_request.kind == AnimeRequestKind.SCHEDULE:
            report = ctx.anime_client.get_daily_schedule()
        else:
            if not anime_request.title_query:
                raise AnimeError("Anime title could not be determined.")
            candidates = ctx.anime_client.search_anime(anime_request.title_query)
            if not candidates:
                refined_queries, elapsed = ctx.call_resolver(
                    "refine_anime_title_queries", request.question, anime_request.title_query
                )
                ctx.record_timing(timings, "resolver.refine_anime_title_queries", elapsed)
                for refined_query in refined_queries:
                    if refined_query == anime_request.title_query:
                        continue
                    candidates = ctx.anime_client.search_anime(refined_query)
                    if candidates:
                        break
            if len(candidates) == 1:
                selected_id = candidates[0].anime_id
            else:
                selected_id, elapsed = ctx.call_resolver(
                    "select_anime_candidate", request.question, candidates
                )
                ctx.record_timing(timings, "resolver.select_anime_candidate", elapsed)
            if selected_id is None:
                raise AnimeError("No AniList title candidate matched the request.")
            if anime_request.kind == AnimeRequestKind.LOOKUP:
                report = ctx.anime_client.get_anime_info(selected_id, anime_request.requested_fields)
            else:
                title, credits, reference = ctx.anime_client.get_credits(selected_id)
                if anime_request.character_query:
                    character_name, elapsed = ctx.call_resolver(
                        "select_character", anime_request.character_query, credits
                    )
                    ctx.record_timing(timings, "resolver.select_anime_character", elapsed)
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
        elapsed = round((perf_counter() - started) * 1000, 2)
        ctx.record_timing(timings, "anime", elapsed)
    except ResearchCancelled:
        raise
    except (ResolverError, AnimeError, httpx.HTTPError) as exc:
        # Specialized-lookup failures (resolver, AniList, or network) fall back
        # to web research. Other exceptions (StorageError, programming bugs)
        # propagate so they are not silently swallowed.
        ctx.record_operation_error(run_id, "anime", "specialized_lookup", exc)
        warnings.append(f"Specialized anime lookup failed; used web research instead: {exc}")
        ctx.trace(run_id, "ANIME ROUTE FALLBACK", [f"error: {exc}"])
        ctx.select_route(run_id, RequestRoute.WEB_RESEARCH.value, str(exc))
        return None
    references = [report.reference][: max(0, request.max_references)]
    return SpecializedRouteResult(
        summary=report.summary,
        answer=report.answer,
        references=references,
        stop_reason=StopReason.SPECIALIZED_ROUTE,
        provider="anilist",
        route_name="anime",
        elapsed_ms=timings.get("anime", [0.0])[-1],
    )


def try_news_route(
    ctx: RouteContext,
    run_id: str,
    request: ResearchRequest,
    timings: dict[str, list[float]],
    warnings: list[str],
) -> SpecializedRouteResult | None:
    assert ctx.news_client is not None
    try:
        ctx.set_stage(run_id, "news")
        news_request, elapsed = ctx.call_resolver("classify_news_request", request.question)
        ctx.record_timing(timings, "resolver.classify_news_request", elapsed)
        ctx.trace(run_id, "NEWS REQUEST CLASSIFIED", [
            f"kind: {news_request.kind.value}",
            f"category: {news_request.category.value if news_request.category else ''}",
            f"time_scope: {news_request.time_scope.value}",
            f"elapsed_ms: {elapsed}",
        ])
        if news_request.kind == NewsRequestKind.UNSUPPORTED_TOPIC:
            ctx.trace(run_id, "NEWS ROUTE FALLBACK", ["reason: unsupported_topic"])
            ctx.select_route(run_id, RequestRoute.WEB_RESEARCH.value, "unsupported_news_topic")
            return None
        started = perf_counter()
        report = ctx.news_client.get_news(news_request, ctx.settings.news_digest_size)
        elapsed = round((perf_counter() - started) * 1000, 2)
        ctx.record_timing(timings, "news", elapsed)
    except ResearchCancelled:
        raise
    except (ResolverError, NewsError, httpx.HTTPError) as exc:
        # Specialized-news failures (resolver classify, RSS fetch, or network)
        # fall back to web research. Other exceptions (StorageError, programming
        # bugs) propagate so they are not silently swallowed.
        ctx.record_operation_error(run_id, "news", "get_news", exc)
        warnings.append(f"Specialized news lookup failed; used web research instead: {exc}")
        ctx.trace(run_id, "NEWS ROUTE FALLBACK", [f"error: {exc}"])
        ctx.select_route(run_id, RequestRoute.WEB_RESEARCH.value, str(exc))
        return None
    warnings.extend(report.warnings)
    references = report.references[: max(0, request.max_references)]
    return SpecializedRouteResult(
        summary=report.summary,
        answer=report.answer,
        references=references,
        stop_reason=StopReason.SPECIALIZED_ROUTE,
        provider="rss",
        route_name="news",
        elapsed_ms=timings.get("news", [0.0])[-1],
    )
