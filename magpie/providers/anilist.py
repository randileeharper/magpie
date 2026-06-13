"""Compact AniList GraphQL client for specialized anime requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from html.parser import HTMLParser
from typing import Any
import unicodedata

import httpx

from ..config import Settings
from ..errors import AnimeError
from ..models import (
    AnimeCandidate,
    AnimeField,
    AnimeReport,
    CharacterCredit,
    Reference,
    SourceKind,
    utc_now,
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.parts.append(text)


@dataclass(slots=True)
class AniListClient:
    settings: Settings
    transport: httpx.BaseTransport | None = None

    def search_anime(self, title_query: str) -> list[AnimeCandidate]:
        candidates = self._search_anime_once(title_query)
        normalized = unicodedata.normalize("NFKD", title_query).encode("ascii", "ignore").decode("ascii")
        if not candidates and normalized and normalized != title_query:
            candidates = self._search_anime_once(normalized)
        if not candidates and self.settings.anime_title_search_fallback_url:
            for fallback_title in self._fallback_titles(title_query):
                candidates = self._search_anime_once(fallback_title)
                if candidates:
                    break
        return candidates

    def _search_anime_once(self, title_query: str) -> list[AnimeCandidate]:
        data = self._query(
            """
            query ($search: String!, $limit: Int!) {
              Page(perPage: $limit) {
                media(search: $search, type: ANIME) {
                  id
                  title { english romaji native }
                  format
                  seasonYear
                }
              }
            }
            """,
            {"search": title_query, "limit": self.settings.anime_candidate_limit},
        )
        media = data.get("Page", {}).get("media", [])
        return [
            AnimeCandidate(
                anime_id=item["id"],
                english_title=item.get("title", {}).get("english"),
                romaji_title=item.get("title", {}).get("romaji"),
                native_title=item.get("title", {}).get("native"),
                format=item.get("format"),
                season_year=item.get("seasonYear"),
            )
            for item in media
            if isinstance(item, dict) and isinstance(item.get("id"), int)
        ]

    def get_anime_info(self, anime_id: int, requested_fields: list[AnimeField]) -> AnimeReport:
        fields = list(dict.fromkeys(requested_fields)) or [AnimeField.DESCRIPTION]
        selections = {
            AnimeField.DESCRIPTION: "description",
            AnimeField.EPISODES: "episodes",
            AnimeField.DURATION: "duration",
            AnimeField.STATUS: "status",
            AnimeField.FORMAT: "format",
            AnimeField.SEASON: "season",
            AnimeField.SEASON_YEAR: "seasonYear",
            AnimeField.START_DATE: "startDate { year month day }",
            AnimeField.END_DATE: "endDate { year month day }",
            AnimeField.GENRES: "genres",
            AnimeField.STUDIOS: "studios(isMain: true, perPage: 5) { nodes { name } }",
            AnimeField.SOURCE_MATERIAL: "source",
            AnimeField.AVERAGE_SCORE: "averageScore",
            AnimeField.NEXT_AIRING_EPISODE: "nextAiringEpisode { airingAt episode }",
        }
        requested_selection = "\n".join(selections[item] for item in fields)
        media = self._query(
            f"""
            query ($id: Int!) {{
              Media(id: $id, type: ANIME) {{
                id
                title {{ english romaji }}
                {requested_selection}
              }}
            }}
            """,
            {"id": anime_id},
        ).get("Media")
        if not isinstance(media, dict):
            raise AnimeError("AniList did not return the selected anime.")
        title = self._title(media)
        values = [self._format_field(item, media) for item in fields]
        usable = [value for value in values if value]
        if not usable:
            raise AnimeError(f"AniList has none of the requested information for {title}.")
        answer = f"{title}\n" + "\n".join(usable)
        return AnimeReport(
            summary=f"Anime information for {title}.",
            answer=answer,
            reference=self._reference(anime_id, title),
        )

    def _format_field(self, field: AnimeField, media: dict[str, Any]) -> str | None:
        if field == AnimeField.DESCRIPTION:
            value = self._plain_text(media.get("description"))
            return value or None
        if field == AnimeField.EPISODES:
            return f"Episodes: {media['episodes']}" if media.get("episodes") is not None else None
        if field == AnimeField.DURATION:
            return f"Episode duration: {media['duration']} minutes" if media.get("duration") else None
        if field == AnimeField.STATUS:
            return f"Status: {self._friendly(media.get('status'))}" if media.get("status") else None
        if field == AnimeField.FORMAT:
            return f"Format: {self._friendly(media.get('format'))}" if media.get("format") else None
        if field == AnimeField.SEASON:
            return f"Season: {self._friendly(media.get('season'))}" if media.get("season") else None
        if field == AnimeField.SEASON_YEAR:
            return f"Season year: {media['seasonYear']}" if media.get("seasonYear") else None
        if field in {AnimeField.START_DATE, AnimeField.END_DATE}:
            key = "startDate" if field == AnimeField.START_DATE else "endDate"
            value = self._format_date(media.get(key))
            label = "Start date" if field == AnimeField.START_DATE else "End date"
            return f"{label}: {value}" if value else None
        if field == AnimeField.GENRES:
            return f"Genres: {', '.join(media.get('genres', []))}" if media.get("genres") else None
        if field == AnimeField.STUDIOS:
            studios = [item.get("name") for item in media.get("studios", {}).get("nodes", []) if item.get("name")]
            return f"Studio: {', '.join(studios)}" if studios else None
        if field == AnimeField.SOURCE_MATERIAL:
            return f"Source material: {self._friendly(media.get('source'))}" if media.get("source") else None
        if field == AnimeField.AVERAGE_SCORE:
            return f"Average AniList score: {media['averageScore']}%" if media.get("averageScore") else None
        if field == AnimeField.NEXT_AIRING_EPISODE:
            item = media.get("nextAiringEpisode")
            if not isinstance(item, dict) or not isinstance(item.get("airingAt"), int):
                return None
            airing = datetime.fromtimestamp(item["airingAt"]).astimezone()
            return f"Next airing: episode {item.get('episode', '?')} on {airing.strftime('%A, %B %-d at %-I:%M %p %Z')}"
        return None

    def get_credits(self, anime_id: int) -> tuple[str, list[CharacterCredit], Reference]:
        media = self._query(
            """
            query ($id: Int!, $limit: Int!) {
              Media(id: $id, type: ANIME) {
                id
                title { english romaji }
                characters(perPage: $limit, sort: [ROLE, RELEVANCE, ID]) {
                  edges {
                    node { name { full } }
                    voiceActors(language: JAPANESE, sort: [RELEVANCE, ID]) { name { full } }
                  }
                }
              }
            }
            """,
            {"id": anime_id, "limit": self.settings.anime_character_limit},
        ).get("Media")
        if not isinstance(media, dict):
            raise AnimeError("AniList did not return the selected anime.")
        title = self._title(media)
        credits: list[CharacterCredit] = []
        for edge in media.get("characters", {}).get("edges", []):
            if not isinstance(edge, dict):
                continue
            name = edge.get("node", {}).get("name", {}).get("full")
            actors = [
                actor.get("name", {}).get("full")
                for actor in edge.get("voiceActors", [])
                if isinstance(actor, dict) and actor.get("name", {}).get("full")
            ]
            if name and actors:
                credits.append(CharacterCredit(str(name), [str(actor) for actor in actors]))
        return title, credits, self._reference(anime_id, title)

    def get_daily_schedule(self) -> AnimeReport:
        local_now = datetime.now().astimezone()
        local_tz = local_now.tzinfo
        start = datetime.combine(local_now.date(), time.min, local_tz)
        end = datetime.combine(local_now.date(), time.max, local_tz)
        data = self._query(
            """
            query ($start: Int!, $end: Int!, $limit: Int!) {
              Page(perPage: $limit) {
                airingSchedules(airingAt_greater: $start, airingAt_lesser: $end, sort: TIME) {
                  airingAt
                  episode
                  media { id title { english romaji } }
                }
              }
            }
            """,
            {
                "start": int(start.timestamp()),
                "end": int(end.timestamp()),
                "limit": self.settings.anime_schedule_limit,
            },
        )
        schedules = data.get("Page", {}).get("airingSchedules", [])
        lines: list[str] = []
        for item in schedules:
            if not isinstance(item, dict) or not isinstance(item.get("airingAt"), int):
                continue
            media = item.get("media", {})
            title = self._title(media)
            airing = datetime.fromtimestamp(item["airingAt"]).astimezone()
            lines.append(f"{airing.strftime('%-I:%M %p')} - {title}, episode {item.get('episode', '?')}")
        date_label = local_now.strftime("%A, %B %-d, %Y")
        zone_label = local_now.tzname() or "local time"
        answer = (
            f"Anime airing schedule for {date_label} ({zone_label}):\n" + "\n".join(lines)
            if lines
            else f"No anime episodes were found airing on {date_label} ({zone_label})."
        )
        return AnimeReport(
            summary=f"Anime airing schedule for {date_label}.",
            answer=answer,
            reference=Reference(
                "anilist:schedule:" + local_now.date().isoformat(),
                "AniList airing schedule",
                "https://anilist.co/search/anime",
                "AniList",
                None,
                utc_now(),
                SourceKind.ANILIST_API,
            ),
        )

    def _query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(
                timeout=self.settings.anime_timeout_seconds,
                verify=self.settings.verify_tls,
                transport=self.transport,
            ) as client:
                response = client.post(
                    self.settings.anime_base_url,
                    json={"query": query, "variables": variables},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AnimeError(f"AniList request failed: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise AnimeError("AniList returned an invalid response.")
        if payload.get("errors"):
            raise AnimeError("AniList returned a GraphQL error.")
        return payload["data"]

    def _fallback_titles(self, title_query: str) -> list[str]:
        try:
            with httpx.Client(
                timeout=self.settings.anime_timeout_seconds,
                verify=self.settings.verify_tls,
                transport=self.transport,
            ) as client:
                response = client.get(
                    self.settings.anime_title_search_fallback_url,
                    params={"q": title_query, "limit": self.settings.anime_candidate_limit},
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return []
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            return []
        titles: list[str] = []
        for item in payload["data"]:
            if not isinstance(item, dict):
                continue
            for value in (item.get("title"), item.get("title_english")):
                if isinstance(value, str) and value and value not in titles:
                    titles.append(value)
        return titles[: self.settings.anime_candidate_limit]

    def _reference(self, anime_id: int, title: str) -> Reference:
        return Reference(
            f"anilist:anime:{anime_id}",
            title,
            f"https://anilist.co/anime/{anime_id}",
            "AniList",
            None,
            utc_now(),
            SourceKind.ANILIST_API,
        )

    @staticmethod
    def _title(media: dict[str, Any]) -> str:
        titles = media.get("title", {}) if isinstance(media, dict) else {}
        return str(titles.get("english") or titles.get("romaji") or "Unknown anime")

    @staticmethod
    def _friendly(value: Any) -> str | None:
        return str(value).replace("_", " ").title() if value else None

    @staticmethod
    def _plain_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        parser = _TextExtractor()
        parser.feed(value)
        text = " ".join(parser.parts)
        for punctuation in (".", ",", "!", "?", ":", ";"):
            text = text.replace(f" {punctuation}", punctuation)
        return text

    @staticmethod
    def _format_date(value: Any) -> str | None:
        if not isinstance(value, dict) or not value.get("year"):
            return None
        parts = [str(value["year"])]
        if value.get("month"):
            parts.append(f"{value['month']:02d}")
        if value.get("day"):
            parts.append(f"{value['day']:02d}")
        return "-".join(parts)
