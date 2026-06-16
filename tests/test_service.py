from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from magpie.config import Settings
from magpie.errors import FetchError
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
    SynthesisDraft,
    WeatherKind,
    WeatherReport,
)
from magpie.providers.fake import FakeFetcher, FakeResolverClient, FakeSearchClient
from magpie.a2a import build_fastapi_app, build_sdk_server
from magpie.service import ResearchService
from magpie.storage import SQLiteStorage


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
    def synthesize(self, question, evidence, prior_draft=None):
        if prior_draft is None:
            return SynthesisDraft(
                summary="",
                answer="",
                cited_source_ids=[],
                remaining_questions=["What additional information completes the answer?"],
                limitations=["Need more evidence."],
            )
        return super().synthesize(question, evidence, prior_draft)


class NeedsFivePagesResolver(FakeResolverClient):
    def synthesize(self, question, evidence, prior_draft=None):
        prior_count = len(prior_draft.cited_source_ids) if prior_draft else 0
        if prior_count < 4:
            return SynthesisDraft(
                summary="",
                answer="",
                cited_source_ids=[*(prior_draft.cited_source_ids if prior_draft else []), evidence[0].source_id],
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
        cited = [*(prior_draft.cited_source_ids if prior_draft else []), evidence[0].source_id]
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
                [evidence[0].source_id],
                [],
                source_answers_question=False,
            )
        return SynthesisDraft(
            "Current weather found.",
            "It is 62°F and cloudy.",
            [evidence[0].source_id],
            [],
        )


class CachedThenNewResolver(FakeResolverClient):
    def __init__(self) -> None:
        super().__init__()
        self.source_ids: list[str] = []

    def synthesize(self, question, evidence, prior_draft=None):
        self.source_ids.append(evidence[0].source_id)
        cited = [*(prior_draft.cited_source_ids if prior_draft else []), evidence[0].source_id]
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
        anime_client=None, news_client=None,
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
        self.assertEqual(news.calls[0][1], 3)
        self.assertEqual(result.stop_reason.value, "specialized_route")

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
        class TwoPageSearch(FakeSearchClient):
            def __post_init__(self) -> None:
                self.index = {
                    "who is the mayor of new york": [
                        SearchResultRecord(
                            title="First page",
                            url="https://example.com/first",
                            snippet="first snippet",
                            provider="fake",
                        ),
                        SearchResultRecord(
                            title="Second page",
                            url="https://example.com/second",
                            snippet="second snippet",
                            provider="fake",
                        ),
                    ]
                }

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

    def test_synthesis_receives_one_source_at_a_time_with_prior_draft(self) -> None:
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

        self.assertEqual(result.status, "ok")
        self.assertEqual(resolver.source_counts, [1, 1])
        self.assertIsNone(resolver.prior_drafts[0])
        self.assertEqual(resolver.prior_drafts[1].answer, "partial answer")

    def test_non_answer_draft_is_discarded_and_next_source_is_checked(self) -> None:
        class WeatherSearch(FakeSearchClient):
            def search(self, request):
                return [
                    SearchResultRecord("Bad weather page", "https://example.com/bad", "bad", provider="fake"),
                    SearchResultRecord("Useful weather page", "https://example.com/good", "good", provider="fake"),
                ]

        resolver = RejectThenAnswerResolver()
        with tempfile.TemporaryDirectory() as tmpdir:
            service = self._service(tmpdir, resolver=resolver, search_client=WeatherSearch())
            result = service.research(ResearchRequest(question="what's the weather in 98230?"))

        self.assertEqual(result.status, "ok")
        self.assertEqual(resolver.calls, 2)
        self.assertEqual(resolver.priors[1].answer, "")
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
            self.assertIn("=== SYNTHESIS CHECK ===", fetch_log)
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


if __name__ == "__main__":
    unittest.main()
