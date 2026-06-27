from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path

from jsonschema import Draft202012Validator

from magpie.config import Settings
from magpie.errors import FetchError, ResolverError, SearchError
from magpie.historian import FakeHistorianSink
from magpie.models import (
    AnimeCandidate,
    AnimeField,
    AnimeReport,
    AnimeRequest,
    AnimeRequestKind,
    CharacterCredit,
    FetchedSource,
    FreshnessClass,
    NewsCategory,
    NewsReport,
    NewsRequest,
    NewsRequestKind,
    NewsTimeScope,
    Reference,
    ResearchRequest,
    ResponseDetail,
    RequestRoute,
    RouteDecision,
    SearchResultRecord,
    SourceKind,
    StopReason,
    SynthesisDraft,
    WeatherKind,
    WeatherReport,
)
from magpie.providers.fake import FakeFetcher, FakeResolverClient, FakeSearchClient
from magpie.a2a import build_fastapi_app, build_sdk_server
from magpie.service import ResearchService, detect_freshness_class
from magpie.storage import SQLiteStorage


def _manifest_schemas() -> dict[tuple[str, int], dict]:
    manifest = json.loads(
        (Path(__file__).parents[1] / "historian.manifest.json").read_text(encoding="utf-8")
    )
    return {
        (item["event_type"], item["version"]): item["json_schema"]
        for item in manifest["schemas"]
    }


def _validate_manifest_events(events: list[dict]) -> None:
    schemas = _manifest_schemas()
    for event in events:
        if str(event["type"]).startswith("core."):
            continue
        Draft202012Validator(schemas[(event["type"], event["schemaversion"])]).validate(
            event["data"]
        )


class RogueResolver(FakeResolverClient):
    def synthesize(self, question, evidence, prior_draft=None):
        draft = super().synthesize(question, evidence, prior_draft)
        return SynthesisDraft(
            summary=draft.summary,
            answer=draft.answer,
            cited_source_ids=["not-a-real-source"],
            remaining_questions=draft.remaining_questions,
        )


class NeedsMoreInfoResolver(FakeResolverClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def synthesize(self, question, evidence, prior_draft=None):
        self.calls += 1
        if self.calls == 1:
            return SynthesisDraft(
                summary="",
                answer="",
                cited_source_ids=[],
                remaining_questions=["What additional information completes the answer?"],
                limitations=["Need more evidence."],
            )
        return super().synthesize(question, evidence, prior_draft)


class NeedsFivePagesResolver(FakeResolverClient):
    def __init__(self) -> None:
        super().__init__()
        self.seen_sources: list[str] = []

    def synthesize(self, question, evidence, prior_draft=None):
        self.seen_sources.extend(item.source_id for item in evidence)
        if len(self.seen_sources) < 5:
            return SynthesisDraft(
                summary="",
                answer="",
                cited_source_ids=[],
                remaining_questions=["What additional information completes the answer?"],
                limitations=["Need more evidence."],
            )
        return super().synthesize(question, evidence, prior_draft)


class DebugResolver(FakeResolverClient):
    def __init__(self) -> None:
        super().__init__()
        self.begin_calls: list[tuple[str, str]] = []

    def begin_request_debug_log(self, run_id: str, question: str) -> None:
        self.begin_calls.append((run_id, question))


class VagueThenUsefulRecipeResolver(FakeResolverClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def synthesize(self, question, evidence, prior_draft=None):
        self.calls += 1
        if self.calls == 1:
            return SynthesisDraft(
                summary="General bread stages.",
                answer="Mix the dough, ferment it, shape it, and bake it.",
                cited_source_ids=[evidence[0].source_id],
                remaining_questions=[],
            )
        return SynthesisDraft(
            summary="A complete sourdough recipe.",
            answer=(
                "Ingredients: 500g flour, 350g water, 100g starter, and 10g salt.\n"
                "1. Mix the ingredients and rest for 30 minutes.\n"
                "2. Fold the dough, then ferment for 4 hours.\n"
                "3. Shape and proof for 12 hours.\n"
                "4. Bake at 450°F for 40 minutes."
            ),
            cited_source_ids=[item.source_id for item in evidence],
            remaining_questions=[],
        )


class SerialResolver(FakeResolverClient):
    def __init__(self) -> None:
        super().__init__()
        self.source_counts: list[int] = []
        self.prior_drafts: list[SynthesisDraft | None] = []

    def synthesize(self, question, evidence, prior_draft=None):
        self.source_counts.append(len(evidence))
        self.prior_drafts.append(prior_draft)
        cited = [*(prior_draft.cited_source_ids if prior_draft else []), *[item.source_id for item in evidence]]
        if len(self.source_counts) == 1:
            return SynthesisDraft("partial", "partial answer", cited, ["Need another source"])
        return SynthesisDraft("complete", "complete answer", cited, [])


class RejectThenAnswerResolver(FakeResolverClient):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.priors: list[SynthesisDraft | None] = []

    def synthesize(self, question, evidence, prior_draft=None):
        self.calls += 1
        self.priors.append(prior_draft)
        if self.calls == 1:
            return SynthesisDraft(
                "The source does not contain weather information.",
                "I cannot provide the current weather because the source does not contain that information.",
                [],
                ["Find a source with current weather for the zip code"],
                source_answers_question=False,
            )
        return SynthesisDraft(
            "Current weather found.",
            "It is 62°F and cloudy.",
            [item.source_id for item in evidence],
            [],
        )


class CachedThenNewResolver(FakeResolverClient):
    def __init__(self) -> None:
        super().__init__()
        self.source_ids: list[str] = []

    def synthesize(self, question, evidence, prior_draft=None):
        self.source_ids.extend(item.source_id for item in evidence)
        cited = [*(prior_draft.cited_source_ids if prior_draft else []), *[item.source_id for item in evidence]]
        if prior_draft is None:
            return SynthesisDraft("partial", "partial answer", cited, ["Need a new source"])
        return SynthesisDraft("complete", "complete answer", cited, [])


class WeatherRoutingResolver(FakeResolverClient):
    def __init__(self, zip_code: str | None = "98230") -> None:
        super().__init__()
        self.zip_code = zip_code

    def route_request(self, question):
        return RouteDecision(RequestRoute.WEATHER, WeatherKind.CONDITIONS, self.zip_code)


class FakeWeatherClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, WeatherKind]] = []

    def get_weather(self, zip_code, kind):
        self.calls.append((zip_code, kind))
        return WeatherReport(
            "Current weather for 98230.",
            "Current conditions for 98230:\nTemperature: 71°F",
            Reference("weather", "Neon Hail", "https://weather.test/98230", "Neon Hail", None, None),
        )


class BlockingWeatherClient(FakeWeatherClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def get_weather(self, zip_code, kind):
        self.started.set()
        self.release.wait(2)
        return super().get_weather(zip_code, kind)


class AnimeRoutingResolver(FakeResolverClient):
    def __init__(self, kind=AnimeRequestKind.CREDITS) -> None:
        super().__init__()
        self.kind = kind

    def route_request(self, question):
        return RouteDecision(RequestRoute.ANIME)

    def classify_anime_request(self, question):
        if self.kind == AnimeRequestKind.SCHEDULE:
            return AnimeRequest(self.kind)
        return AnimeRequest(self.kind, "Yakuza Fiance", "Kirishima", [AnimeField.DESCRIPTION])

    def select_anime_candidate(self, question, candidates):
        return candidates[0].anime_id

    def select_character(self, query, credits):
        return "Kirishima Miyama"


class FakeAnimeClient:
    def search_anime(self, title_query):
        return [AnimeCandidate(170468, "Yakuza Fiancé", "Raise wa Tanin ga Ii", "来世は他人がいい", "TV", 2024)]

    def get_anime_info(self, anime_id, requested_fields):
        raise AssertionError("wrong anime operation")

    def get_credits(self, anime_id):
        return (
            "Yakuza Fiancé",
            [
                CharacterCredit("Kirishima Miyama", ["Akira Ishida"]),
                CharacterCredit("Yoshino Somei", ["Hitomi Ueda"]),
            ],
            Reference("anime", "Yakuza Fiancé", "https://anilist.co/anime/170468", "AniList", None, None),
        )

    def get_daily_schedule(self):
        return AnimeReport(
            "Today's anime schedule.",
            "Anime airing schedule for Saturday (PDT):\n5:00 PM - Example Anime, episode 3",
            Reference("schedule", "AniList schedule", "https://anilist.co", "AniList", None, None),
        )


class NewsRoutingResolver(FakeResolverClient):
    def __init__(self, news_request: NewsRequest | None = None) -> None:
        super().__init__()
        self.news_request = news_request or NewsRequest(
            NewsRequestKind.CATEGORY, NewsCategory.AI, NewsTimeScope.LAST_24_HOURS
        )

    def route_request(self, question):
        return RouteDecision(RequestRoute.NEWS)

    def classify_news_request(self, question):
        return self.news_request


class FakeNewsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[NewsRequest, int]] = []

    def get_news(self, request: NewsRequest, max_items: int) -> NewsReport:
        self.calls.append((request, max_items))
        references = [
            Reference(
                f"rss:{index}",
                f"Story {index}",
                f"https://example.com/story-{index}",
                "Example Feed",
                f"2026-06-15T0{index}:00:00-07:00",
                None,
                SourceKind.RSS_FEED,
            )
            for index in range(1, max_items + 1)
        ]
        answer = "\n".join(
            f"{index}. 2026-06-15 0{index}:00 PDT | Story {index} | Summary {index}. | Example Feed | https://example.com/story-{index}"
            for index in range(1, max_items + 1)
        )
        return NewsReport("Latest AI news.", answer, references)

    def doctor_check(self, live: bool = False) -> dict[str, object]:
        return {"status": "ok", "live": live}


class ServiceTests(unittest.TestCase):
    def _service(
        self, tmpdir: str, resolver=None, search_client=None, fetcher=None, weather_client=None,
        anime_client=None, news_client=None, historian_sink=None,
    ) -> ResearchService:
        storage = SQLiteStorage(Path(tmpdir) / "magpie.db")
        storage.initialize()
        settings = Settings(database_path=str(Path(tmpdir) / "magpie.db"))
        return ResearchService(
            storage=storage,
            resolver=resolver or FakeResolverClient(),
            search_client=search_client or FakeSearchClient(),
            fetcher=fetcher or FakeFetcher(),
            settings=settings,
            weather_client=weather_client,
            anime_client=anime_client,
            news_client=news_client,
            historian_sink=historian_sink,
        )

    def test_anime_credit_route_returns_only_selected_credit(self) -> None:
        class ExplodingSearch(FakeSearchClient):
            def search(self, request):
                raise AssertionError("anime route must not search")

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=AnimeRoutingResolver(),
                search_client=ExplodingSearch(),
                anime_client=FakeAnimeClient(),
            )
            result = service.research(
                ResearchRequest(question="voice actor for Kirishima in Yakuza Fiance")
            )

        self.assertEqual(
            result.answer,
            "Kirishima Miyama in Yakuza Fiancé is voiced in Japanese by Akira Ishida.",
        )
        self.assertNotIn("Yoshino", result.answer)

    def test_anime_schedule_route_bypasses_title_disambiguation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=AnimeRoutingResolver(AnimeRequestKind.SCHEDULE),
                anime_client=FakeAnimeClient(),
            )
            result = service.research(ResearchRequest(question="anime schedule for today"))

        self.assertIn("Example Anime, episode 3", result.answer)

    def test_weather_route_bypasses_web_research(self) -> None:
        class ExplodingSearch(FakeSearchClient):
            def search(self, request):
                raise AssertionError("weather route must not search")

        weather = FakeWeatherClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=WeatherRoutingResolver(),
                search_client=ExplodingSearch(),
                weather_client=weather,
            )
            result = service.research(ResearchRequest(question="what's the weather in 98230?"))
            with service.storage._connect() as connection:
                stored = connection.execute(
                    "SELECT answer FROM final_answers WHERE run_id=?", (result.run_id,)
                ).fetchone()

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.answer, "Current conditions for 98230:\nTemperature: 71°F")
        self.assertEqual(weather.calls, [("98230", WeatherKind.CONDITIONS)])
        self.assertEqual(result.stop_reason.value, "specialized_route")
        self.assertEqual(stored["answer"], result.answer)

    def test_weather_route_without_zip_falls_back_to_web_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=WeatherRoutingResolver(zip_code=None),
                weather_client=FakeWeatherClient(),
            )
            result = service.research(ResearchRequest(question="Who is the mayor of New York?"))

        self.assertEqual(result.status, "ok")
        self.assertTrue(any("could not determine a US ZIP code" in item for item in result.warnings))

    def test_cancellation_wins_race_with_specialized_completion(self) -> None:
        weather = BlockingWeatherClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=WeatherRoutingResolver(),
                weather_client=weather,
            )
            holder = {}

            thread = threading.Thread(
                target=lambda: holder.setdefault(
                    "result",
                    service.research(
                        ResearchRequest(question="what's the weather in 98230?"),
                        run_id="weather-task",
                    ),
                )
            )
            thread.start()
            self.assertTrue(weather.started.wait(1))
            service.cancel_run("weather-task")
            weather.release.set()
            thread.join(2)

            run = service.storage.get_run("weather-task")
            with service.storage._connect() as connection:
                final_answer = connection.execute(
                    "SELECT 1 FROM final_answers WHERE run_id='weather-task'"
                ).fetchone()

        self.assertFalse(thread.is_alive())
        self.assertEqual(run["status"], "cancelled")
        self.assertEqual(holder["result"].stop_reason.value, "cancelled")
        self.assertIsNone(final_answer)

    def test_news_route_bypasses_web_research(self) -> None:
        class ExplodingSearch(FakeSearchClient):
            def search(self, request):
                raise AssertionError("news route must not search")

        news = FakeNewsClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=NewsRoutingResolver(),
                search_client=ExplodingSearch(),
                news_client=news,
            )
            result = service.research(ResearchRequest(question="what's the latest AI news?", max_references=3))

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(result.references), 3)
        self.assertEqual(news.calls[0][1], 5)
        self.assertEqual(result.stop_reason.value, "specialized_route")

    def test_news_answer_is_populated_when_references_are_disabled(self) -> None:
        news = FakeNewsClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=NewsRoutingResolver(),
                news_client=news,
            )
            result = service.research(
                ResearchRequest(question="what's the latest AI news?", max_references=0)
            )

        self.assertEqual(result.status, "ok")
        self.assertIn("Story 1", result.answer)
        self.assertEqual(result.references, [])
        self.assertEqual(news.calls[0][1], 5)

    def test_news_unsupported_topic_falls_back_to_web_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            search_client = FakeSearchClient(index={
                "latest openai news": [
                    SearchResultRecord(
                        title="OpenAI launches example feature",
                        url="https://example.com/openai-feature",
                        snippet="OpenAI launched an example feature.",
                        site_name="Example News",
                        published_at="2026-06-15",
                        provider="fake",
                        inline_text="OpenAI launched an example feature on June 15, 2026.",
                    )
                ]
            })
            fetcher = FakeFetcher(pages={
                "https://example.com/openai-feature": FetchedSource(
                    url="https://example.com/openai-feature",
                    title="OpenAI launches example feature",
                    site_name="Example News",
                    text="OpenAI launched an example feature on June 15, 2026.",
                    published_at="2026-06-15",
                    retrieved_via="fake",
                    source_kind=SourceKind.PAGE_FETCH,
                )
            })
            service = self._service(
                tmpdir,
                resolver=NewsRoutingResolver(NewsRequest(NewsRequestKind.UNSUPPORTED_TOPIC, None, NewsTimeScope.LAST_24_HOURS)),
                news_client=FakeNewsClient(),
                search_client=search_client,
                fetcher=fetcher,
            )
            result = service.research(ResearchRequest(question="latest OpenAI news"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.stop_reason.value, "needed_new_search")

    def test_exact_query_match_answers_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            first = service.research(ResearchRequest(question="Who is the mayor of New York?"))
            self.assertEqual(first.status, "ok")
            second = service.research(ResearchRequest(question="Who is the mayor of New York?"))
            self.assertEqual(second.status, "ok")
            self.assertEqual(second.stop_reason.value, "answered_from_cache")
            service.storage.close()

    def test_different_query_reuses_cached_source_after_search(self) -> None:
        class CountingFetcher(FakeFetcher):
            def __init__(self) -> None:
                super().__init__()
                self.fetch_count = 0

            def fetch(self, url: str):
                self.fetch_count += 1
                return super().fetch(url)

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = CountingFetcher()
            service = self._service(tmpdir, fetcher=fetcher)
            first = service.research(ResearchRequest(question="Who is the mayor of New York?"))
            self.assertEqual(first.status, "ok")
            self.assertEqual(fetcher.fetch_count, 1)
            second = service.research(ResearchRequest(question="When was Zohran Mamdani elected mayor?"))
            self.assertEqual(second.status, "ok")
            self.assertEqual(second.stop_reason.value, "needed_new_search")
            self.assertEqual(fetcher.fetch_count, 1)
            service.storage.close()

    def test_fetch_failure_uses_search_result_fallback(self) -> None:
        class FailingFetcher(FakeFetcher):
            def fetch(self, url: str):
                raise FetchError(f"boom for {url}")

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, fetcher=FailingFetcher())
            result = service.research(ResearchRequest(question="Who is the mayor of New York?"))
            self.assertEqual(result.status, "ok")
            self.assertTrue(any("search-provider content" in warning for warning in result.warnings))
            self.assertEqual(result.references[0].source_kind.value, "search_result_fallback")
            service.storage.close()

    def test_synthesis_can_request_more_information_before_completing(self) -> None:
        call_count = 0

        class TwoPageSearch(FakeSearchClient):
            def search(self, request):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return [
                        SearchResultRecord(
                            title="First page",
                            url="https://example.com/first",
                            snippet="first snippet",
                            provider="fake",
                        ),
                    ]
                return [
                    SearchResultRecord(
                        title="Second page",
                        url="https://example.com/second",
                        snippet="second snippet",
                        provider="fake",
                    ),
                ]

        class TwoPageFetcher(FakeFetcher):
            def __post_init__(self) -> None:
                self.pages = {
                    "https://example.com/first": FetchedSource(
                        url="https://example.com/first",
                        title="First page",
                        site_name="Example",
                        text="Partial page content only.",
                        markdown="Partial page content only.",
                        retrieved_via="fake",
                        source_kind=SourceKind.PAGE_FETCH,
                    ),
                    "https://example.com/second": FetchedSource(
                        url="https://example.com/second",
                        title="Second page",
                        site_name="Example",
                        text="Complete answer content on second page.",
                        markdown="Complete answer content on second page.",
                        retrieved_via="fake",
                        source_kind=SourceKind.PAGE_FETCH,
                    ),
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=NeedsMoreInfoResolver(),
                search_client=TwoPageSearch(),
                fetcher=TwoPageFetcher(),
            )
            result = service.research(ResearchRequest(question="Who is the mayor of New York?"))
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.answer, "Complete answer content on second page.")
            service.storage.close()

    def test_synthesis_cannot_cite_unknown_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, resolver=RogueResolver())
            result = service.research(ResearchRequest(question="Who is the mayor of New York?"))
            self.assertEqual(result.status, "error")
            self.assertEqual(result.message.startswith("Synthesis cited unknown source IDs"), True)
            service.storage.close()

    def test_vague_recipe_answer_triggers_more_research(self) -> None:
        class RecipeSearch(FakeSearchClient):
            def search(self, request):
                suffix = len(request.query)
                return [
                    SearchResultRecord(
                        title="Complete sourdough recipe",
                        url=f"https://example.com/recipe-{suffix}",
                        snippet="Complete recipe",
                        provider="fake",
                    )
                ]

        class RecipeFetcher(FakeFetcher):
            def fetch(self, url):
                return FetchedSource(
                    url=url,
                    title="Complete sourdough recipe",
                    site_name="Example",
                    text=(
                        "Ingredients: 500g flour, 350g water, 100g starter, 10g salt. "
                        "Mix and rest 30 minutes. Fold and ferment 4 hours. "
                        "Shape and proof 12 hours. Bake at 450°F for 40 minutes."
                    ),
                    retrieved_via="fake",
                    source_kind=SourceKind.PAGE_FETCH,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = VagueThenUsefulRecipeResolver()
            service = self._service(
                tmpdir,
                resolver=resolver,
                search_client=RecipeSearch(),
                fetcher=RecipeFetcher(),
            )
            result = service.research(ResearchRequest(question="how do i make sourdough bread"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(resolver.calls, 2)
        self.assertIn("500g flour", result.answer)

    def test_vague_procedural_answer_triggers_more_research(self) -> None:
        class ProceduralSearch(FakeSearchClient):
            def search(self, request):
                return [
                    SearchResultRecord(
                        title="Complete setup guide",
                        url=f"https://example.com/guide-{len(request.query)}",
                        snippet="Complete guide",
                        provider="fake",
                    )
                ]

        class ProceduralFetcher(FakeFetcher):
            def fetch(self, url):
                return FetchedSource(
                    url=url,
                    title="Complete setup guide",
                    site_name="Example",
                    text=(
                        "1. Install the package using apt.\n"
                        "2. Create the configuration file at /etc/app.conf.\n"
                        "3. Start the service with systemctl start app.\n"
                        "4. Verify the service is running on port 8080."
                    ),
                    retrieved_via="fake",
                    source_kind=SourceKind.PAGE_FETCH,
                )

        class VagueThenUsefulProceduralResolver(FakeResolverClient):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def synthesize(self, question, evidence, prior_draft=None):
                self.calls += 1
                if self.calls == 1:
                    return SynthesisDraft(
                        summary="General setup steps.",
                        answer="Install it, configure it, and start it.",
                        cited_source_ids=[evidence[0].source_id],
                        remaining_questions=[],
                    )
                return SynthesisDraft(
                    summary="A complete setup guide.",
                    answer=(
                        "1. Install the package using apt.\n"
                        "2. Create the configuration file at /etc/app.conf.\n"
                        "3. Start the service with systemctl start app.\n"
                        "4. Verify the service is running on port 8080."
                    ),
                    cited_source_ids=[item.source_id for item in evidence],
                    remaining_questions=[],
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = VagueThenUsefulProceduralResolver()
            service = self._service(
                tmpdir,
                resolver=resolver,
                search_client=ProceduralSearch(),
                fetcher=ProceduralFetcher(),
            )
            result = service.research(ResearchRequest(question="how do i set up the app"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(resolver.calls, 2)
        self.assertIn("systemctl start app", result.answer)

    def test_synthesis_receives_all_evidence_per_round_with_prior_draft(self) -> None:
        class TwoResultSearch(FakeSearchClient):
            def search(self, request):
                return [
                    SearchResultRecord("One", "https://example.com/one", "one", provider="fake"),
                    SearchResultRecord("Two", "https://example.com/two", "two", provider="fake"),
                ]

        resolver = SerialResolver()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, resolver=resolver, search_client=TwoResultSearch())
            result = service.research(ResearchRequest(question="question"))

        self.assertEqual(result.status, "partial")
        # batch: first round gets both sources at once
        self.assertEqual(resolver.source_counts[0], 2)
        self.assertIsNone(resolver.prior_drafts[0])

    def test_non_answer_round_is_discarded_and_next_round_answers(self) -> None:
        call_count = 0

        class WeatherSearch(FakeSearchClient):
            def search(self, request):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return [
                        SearchResultRecord("Bad weather page", "https://example.com/bad", "bad", provider="fake"),
                    ]
                return [
                    SearchResultRecord("Useful weather page", "https://example.com/good", "good", provider="fake"),
                ]

        resolver = RejectThenAnswerResolver()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, resolver=resolver, search_client=WeatherSearch())
            result = service.research(ResearchRequest(question="what's the weather in 98230?"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(resolver.calls, 2)
        self.assertEqual(result.answer, "It is 62°F and cloudy.")
        self.assertEqual(len(result.references), 1)

    def test_cached_url_is_not_processed_again_from_search_results(self) -> None:
        class DuplicateSearch(FakeSearchClient):
            def search(self, request):
                return [
                    SearchResultRecord(
                        "Cached duplicate", "https://example.com/cached?utm_source=search", "duplicate", provider="fake"
                    ),
                    SearchResultRecord("New source", "https://example.com/new", "new", provider="fake"),
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = CachedThenNewResolver()
            service = self._service(tmpdir, resolver=resolver, search_client=DuplicateSearch())
            prior_run = service.storage.create_run(
                "question", None, FreshnessClass.EVERGREEN, "compact"
            )
            cached_id = service.storage.upsert_source(
                prior_run, "https://example.com/cached", "Cached", None, None, "partial evidence", {}
            )
            service.storage.save_final_answer(
                prior_run,
                "partial",
                "partial answer",
                [Reference(cached_id, "Cached", "https://example.com/cached", None, None, None)],
            )
            service.storage.update_run_status(prior_run, "partial")

            result = service.research(ResearchRequest(question="question"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(len(resolver.source_ids), 2)
        self.assertEqual(resolver.source_ids.count(cached_id), 1)

    def test_actionable_recipe_chunks_outrank_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            run_id = service.storage.create_run("q", None, FreshnessClass.EVERGREEN, "compact")
            budget = type("Budget", (), {"evidence_remaining": 1})()
            text = (
                "[Recipes](https://example.com/recipes) [Bread](https://example.com/bread) "
                "[Shop](https://example.com/shop) [About](https://example.com/about)\n\n"
                "Ingredients: 500g flour, 350g water, 100g starter, and 10g salt.\n\n"
                "Mix the ingredients and rest for 30 minutes. Fold the dough and ferment for 4 hours. "
                "Shape it, proof for 12 hours, then bake at 450°F for 40 minutes."
            )
            source_id = service.storage.upsert_source(
                run_id, "https://example.com/recipe", "Recipe", None, None, text, {}
            )
            item = service._select_evidence(
                run_id, source_id, text, "how do i make sourdough bread", [], budget, []
            )

        self.assertIn("500g flour", item.excerpt)
        self.assertNotIn("[Shop]", item.excerpt)

    def test_non_english_evidence_is_scored_by_unicode_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            run_id = service.storage.create_run("q", None, FreshnessClass.EVERGREEN, "compact")
            budget = type("Budget", (), {"evidence_remaining": 1})()
            text = (
                "[Menu](https://example.com/menu) [Shop](https://example.com/shop)\n\n"
                "来世は他人がいい。キリシマはある意味で最も誠実なキャラクターである。"
            )
            source_id = service.storage.upsert_source(
                run_id, "https://example.com/anime", "Anime", None, None, text, {}
            )
            item = service._select_evidence(
                run_id, source_id, text, "キリシマの正体は何か", [], budget, []
            )

        self.assertIsNotNone(item)
        self.assertIn("キリシマ", item.excerpt)
        self.assertNotIn("[Shop]", item.excerpt)

    def test_debug_response_includes_reasoning_request_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            result = service.research(
                ResearchRequest(
                    question="Who is the mayor of New York?",
                    response_detail=ResponseDetail.DEBUG,
                )
            )
            self.assertEqual(result.status, "ok")
            self.assertIn("reasoning_options", result.debug)
            service.storage.close()

    def test_timing_debug_and_resolver_log_hook_are_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resolver = DebugResolver()
            settings = Settings(
                database_path=str(Path(tmpdir) / "magpie.db"),
                include_timing_debug=True,
                fetch_debug_log_path=str(Path(tmpdir) / "fetch.log"),
            )
            storage = SQLiteStorage(Path(tmpdir) / "magpie.db")
            storage.initialize()
            service = ResearchService(
                storage=storage,
                resolver=resolver,
                search_client=FakeSearchClient(),
                fetcher=FakeFetcher(),
                settings=settings,
            )
            result = service.research(ResearchRequest(question="Who is the mayor of New York?"))
            self.assertEqual(result.status, "ok")
            self.assertIn("timings", result.debug)
            self.assertTrue(resolver.begin_calls)
            fetch_log = (Path(tmpdir) / "fetch.log").read_text(encoding="utf-8")
            self.assertIn("=== QUERY PROPOSED ===", fetch_log)
            self.assertIn("=== SEARCH RESULTS ===", fetch_log)
            self.assertIn("=== EVIDENCE SELECTED ===", fetch_log)
            service.storage.close()

    def test_cancel_run_updates_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            run_id = service.storage.create_run("q", None, FreshnessClass.EVERGREEN, "compact")
            service.cancel_run(run_id)
            run = service.storage.get_run(run_id)
            self.assertEqual(run["status"], "cancelled")
            self.assertEqual(run["stop_reason"], "cancelled")
            service.storage.close()

    def test_cancel_run_does_not_rewrite_completed_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            result = service.research(ResearchRequest(question="Who is the mayor of New York?"))
            service.cancel_run(result.run_id)
            run = service.storage.get_run(result.run_id)

        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["cancel_requested"], 0)

    def test_fetches_top_five_results_at_most(self) -> None:
        class ManyResultsSearch(FakeSearchClient):
            def search(self, request):
                return [
                    *[
                        type(self.index["who is the mayor of new york"][0])(
                            title=f"Irrelevant {i}",
                            url=f"https://example.com/irrelevant-{i}",
                            snippet="not relevant",
                            provider="fake",
                        )
                        for i in range(5)
                    ],
                    *self.index["who is the mayor of new york"],
                ]

        class CountingFetcher(FakeFetcher):
            def __init__(self) -> None:
                super().__init__()
                self.urls: list[str] = []

            def fetch(self, url: str):
                self.urls.append(url)
                return super().fetch(url)

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = CountingFetcher()
            service = self._service(
                tmpdir,
                resolver=NeedsFivePagesResolver(),
                search_client=ManyResultsSearch(),
                fetcher=fetcher,
            )
            result = service.research(ResearchRequest(question="Who is the mayor of New York?"))
            self.assertEqual(result.status, "ok")
            self.assertEqual(len(fetcher.urls), 5)
            service.storage.close()

    def test_a2a_sdk_server_scaffold_builds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            server_bits = build_sdk_server(service, "http://127.0.0.1:8766")
            app = build_fastapi_app(service, "http://127.0.0.1:8766")
            self.assertIn("request_handler", server_bits)
            self.assertGreater(len(app.routes), 0)
            service.storage.close()

    def test_failed_round_rolls_back_while_prior_round_persists(self) -> None:
        """A failure during a research round must roll back that round's writes
        (query, search results, sources) while leaving the run record intact.
        """
        class ExplodingSynthesisResolver(FakeResolverClient):
            def synthesize(self, question, evidence, prior_draft=None):
                raise RuntimeError("simulated synthesis failure")

        class TwoResultSearch(FakeSearchClient):
            def search(self, request):
                return [
                    SearchResultRecord("One", "https://example.com/one", "one", provider="fake"),
                    SearchResultRecord("Two", "https://example.com/two", "two", provider="fake"),
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=ExplodingSynthesisResolver(),
                search_client=TwoResultSearch(),
            )
            result = service.research(ResearchRequest(question="question"))

            self.assertEqual(result.status, "error")
            # The run row was written outside the transaction and survives.
            run = service.storage.get_run(result.run_id)
            self.assertEqual(run["status"], "failed")
            # The round's query was rolled back by the transaction.
            with service.storage._connect() as connection:
                queries = connection.execute(
                    "SELECT COUNT(*) FROM research_queries WHERE run_id=?", (result.run_id,)
                ).fetchone()[0]
                sources = connection.execute(
                    "SELECT COUNT(*) FROM run_source_links WHERE run_id=?", (result.run_id,)
                ).fetchone()[0]
            self.assertEqual(queries, 0)
            self.assertEqual(sources, 0)

    def test_unexpected_exception_yields_internal_error_terminal(self) -> None:
        """A non-domain exception (logic bug) must surface as a distinct
        INTERNAL_ERROR terminal rather than being indistinguishable from a
        provider failure (issue #41).
        """

        class ExplodingSynthesisResolver(FakeResolverClient):
            def synthesize(self, question, evidence, prior_draft=None):
                raise RuntimeError("simulated logic bug")

        class TwoResultSearch(FakeSearchClient):
            def search(self, request):
                return [
                    SearchResultRecord("One", "https://example.com/one", "one", provider="fake"),
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=ExplodingSynthesisResolver(),
                search_client=TwoResultSearch(),
            )
            result = service.research(ResearchRequest(question="question"))

            self.assertEqual(result.status, "error")
            self.assertEqual(result.stop_reason, StopReason.INTERNAL_ERROR)
            self.assertIn("simulated logic bug", result.message)
            run = service.storage.get_run(result.run_id)
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["stop_reason"], StopReason.INTERNAL_ERROR.value)
            service.storage.close()

    def test_domain_error_yields_failed_terminal_not_internal_error(self) -> None:
        """A provider/domain failure must finalize as FAILED, not INTERNAL_ERROR,
        so the two remain distinguishable (issue #41).
        """

        class FailingSearch(FakeSearchClient):
            def search(self, request):
                raise SearchError("provider down")

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                search_client=FailingSearch(),
            )
            result = service.research(ResearchRequest(question="question"))

            self.assertEqual(result.status, "error")
            self.assertEqual(result.stop_reason, StopReason.FAILED)
            self.assertIn("provider down", result.message)
            run = service.storage.get_run(result.run_id)
            self.assertEqual(run["stop_reason"], StopReason.FAILED.value)
            service.storage.close()

    def test_routing_non_domain_error_propagates_instead_of_fallback(self) -> None:
        """A non-domain error inside a specialized route must not be silently
        swallowed into a web-research fallback (issue #41). It propagates out
        of the routing layer; the research outer handler then finalizes the run
        as a normal FAILED terminal (StorageError is a domain error there), not
        as a silent fallback.
        """
        from magpie.errors import StorageError

        class StorageErrorResolver(FakeResolverClient):
            def route_request(self, question):
                raise StorageError("storage unavailable")

        class FailingSearch(FakeSearchClient):
            def search(self, request):
                raise AssertionError("routing storage error must not fall back to web research")

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=StorageErrorResolver(),
                search_client=FailingSearch(),
                weather_client=FakeWeatherClient(),
            )
            result = service.research(ResearchRequest(question="weather for 98230"))

            # Not silently swallowed into web research: the failing search was
            # never reached because the storage error propagated out of routing.
            self.assertEqual(result.status, "error")
            self.assertEqual(result.stop_reason, StopReason.FAILED)
            self.assertIn("storage unavailable", result.message)
            service.storage.close()

    def test_routing_resolver_error_falls_back_to_web_research(self) -> None:
        """A resolver (domain) error during routing still falls back to web
        research; only the catch set changed, not the fallback behavior.
        """
        from magpie.errors import ResolverError

        class ResolverErrorRouting(FakeResolverClient):
            def route_request(self, question):
                raise ResolverError("resolver returned junk")

        class OneResultSearch(FakeSearchClient):
            def search(self, request):
                return [SearchResultRecord("One", "https://example.com/one", "one", provider="fake")]

        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(
                tmpdir,
                resolver=ResolverErrorRouting(),
                search_client=OneResultSearch(),
                weather_client=FakeWeatherClient(),
            )
            result = service.research(ResearchRequest(question="weather for 98230"))

            # Fell back to web research rather than propagating.
            self.assertNotEqual(result.stop_reason, StopReason.INTERNAL_ERROR)
            run = service.storage.get_run(result.run_id)
            self.assertNotEqual(run["stop_reason"], StopReason.INTERNAL_ERROR.value)
            service.storage.close()

    def test_fetch_by_url_finalizes_run_as_completed(self) -> None:
        sink = FakeHistorianSink()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, historian_sink=sink)
            result = service.fetch(url="https://example.com/mayor-election")
            try:
                run = service.storage.get_run(result.run_id)
                self.assertEqual(run["status"], "completed")
                self.assertEqual(run["stop_reason"], StopReason.NEEDED_NEW_SEARCH.value)
                event_types = [event["type"] for event in sink.events]
                self.assertEqual(event_types.count("research.run.started"), 1)
                self.assertEqual(event_types.count("research.run.completed"), 1)
                self.assertNotIn("research.run.failed", event_types)
                completed = next(
                    event for event in sink.events if event["type"] == "research.run.completed"
                )
                self.assertEqual(completed["data"]["status"], "ok")
                self.assertEqual(
                    completed["data"]["stop_reason"], StopReason.NEEDED_NEW_SEARCH.value
                )
                _validate_manifest_events(sink.events)
            finally:
                service.storage.close()

    def test_fetch_by_url_failure_finalizes_run_as_failed(self) -> None:
        class FailingFetcher(FakeFetcher):
            def fetch(self, url: str):
                raise FetchError(f"boom for {url}")

        sink = FakeHistorianSink()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, fetcher=FailingFetcher(), historian_sink=sink)
            with self.assertRaises(ResolverError):
                service.fetch(url="https://example.com/mayor-election")
            try:
                with service.storage._connect() as connection:
                    running = connection.execute(
                        "SELECT run_id FROM research_runs WHERE status='running'"
                    ).fetchall()
                    failed_rows = connection.execute(
                        "SELECT run_id, status, stop_reason FROM research_runs WHERE status='failed'"
                    ).fetchall()
                self.assertEqual(running, [])
                self.assertEqual(len(failed_rows), 1)
                self.assertEqual(failed_rows[0]["stop_reason"], StopReason.FAILED.value)
                event_types = [event["type"] for event in sink.events]
                self.assertEqual(event_types.count("research.run.started"), 1)
                self.assertEqual(event_types.count("research.run.failed"), 1)
                self.assertNotIn("research.run.completed", event_types)
                failed = next(
                    event for event in sink.events if event["type"] == "research.run.failed"
                )
                self.assertEqual(failed["data"]["status"], "error")
                self.assertEqual(failed["data"]["stop_reason"], StopReason.FAILED.value)
                self.assertEqual(failed["data"]["error_type"], "FetchError")
                _validate_manifest_events(sink.events)
            finally:
                service.storage.close()

    def test_fetch_by_index_does_not_finalize_owning_search_run(self) -> None:
        # fetch() reuses an existing search run when given run_id + index, and
        # must not transition that run's status (the search owns its terminal).
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir)
            search = service.search("mayor of New York")
            self.assertEqual(service.storage.get_run(search.run_id)["status"], "completed")
            # Index fetch against the completed search run returns stored content
            # without rewriting the run's terminal status.
            result = service.fetch(run_id=search.run_id, index=0)
            self.assertEqual(result.run_id, search.run_id)
            run = service.storage.get_run(search.run_id)
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["stop_reason"], StopReason.NEEDED_NEW_SEARCH.value)
            service.storage.close()


class FreshnessDetectionTests(unittest.TestCase):
    def test_future_year_is_not_recent(self) -> None:
        self.assertEqual(detect_freshness_class("events in 2099"), FreshnessClass.EVERGREEN)

    def test_current_year_is_recent(self) -> None:
        current_year = datetime.now(UTC).year
        self.assertEqual(
            detect_freshness_class(f"news from {current_year}"), FreshnessClass.RECENT
        )

    def test_last_year_is_recent(self) -> None:
        last_year = datetime.now(UTC).year - 1
        self.assertEqual(
            detect_freshness_class(f"summary of {last_year}"), FreshnessClass.RECENT
        )

    def test_recent_signal_words_are_recent(self) -> None:
        self.assertEqual(detect_freshness_class("what happened today"), FreshnessClass.RECENT)

    def test_evergreen_questions_are_evergreen(self) -> None:
        self.assertEqual(detect_freshness_class("how to bake sourdough bread"), FreshnessClass.EVERGREEN)


if __name__ == "__main__":
    unittest.main()
