from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from magpie.config import Settings
from magpie.models import (
    AnimeField,
    AnimeRequestKind,
    EvidenceItem,
    NewsCategory,
    NewsRequestKind,
    NewsTimeScope,
    PlanningContext,
    RequestRoute,
    RunBudget,
    WeatherKind,
)
from magpie.providers.openai_compatible import OpenAICompatibleResolverClient
from magpie.providers.base import reasoning_request_options


class OpenAICompatibleResolverTests(unittest.TestCase):
    def _settings(self, tmpdir: str, **overrides: object) -> Settings:
        data: dict[str, object] = {
            "database_path": str(Path(tmpdir) / "magpie.db"),
            "search_provider": "fake",
            "fetch_provider": "fake",
            "resolver_backend": "openai_compatible",
            "resolver_base_url": "http://resolver.test/v1",
            "resolver_model": "test-model",
            "resolver_api_key": "",
            "resolver_debug_log_path": str(Path(tmpdir) / "resolver.log"),
        }
        data.update(overrides)
        path = Path(tmpdir) / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return Settings.load(str(path))

    def test_reasoning_request_options_toggle(self) -> None:
        self.assertEqual(
            reasoning_request_options(False),
            {"think": False, "reasoning_effort": "none", "reasoning": {"effort": "none"}},
        )
        self.assertEqual(
            reasoning_request_options(True),
            {"think": True, "reasoning_effort": "medium", "reasoning": {"effort": "medium"}},
        )

    def test_synthesize_uses_structured_json_response_format(self) -> None:
        captured_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"short","answer":"fact","cited_source_ids":["source-1"],'
                                    '"remaining_questions":[],"source_answers_question":true}'
                                )
                            }
                        }
                    ]
                },
            )

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir, resolver_include_reasoning=True),
                transport=httpx.MockTransport(handler),
            )
            draft = client.synthesize("question", evidence)

        self.assertEqual(draft.answer, "fact")
        self.assertEqual(draft.cited_source_ids, ["source-1"])
        self.assertEqual(captured_payloads[0]["think"], True)
        self.assertEqual(captured_payloads[0]["reasoning_effort"], "medium")
        self.assertEqual(captured_payloads[0]["response_format"]["type"], "json_schema")
        self.assertEqual(captured_payloads[0]["response_format"]["json_schema"]["name"], "magpie_synthesis")
        self.assertTrue(captured_payloads[0]["response_format"]["json_schema"]["strict"])
        schema = captured_payloads[0]["response_format"]["json_schema"]["schema"]
        self.assertIn("source_answers_question", schema["required"])
        user_payload = json.loads(captured_payloads[0]["messages"][1]["content"])
        self.assertIn("sources", user_payload)
        self.assertNotIn("new_source", user_payload)

    def test_synthesize_accepts_multiple_sources(self) -> None:
        captured_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"short","answer":"combined fact","cited_source_ids":["source-1","source-2"],'
                                    '"remaining_questions":[],"source_answers_question":true}'
                                )
                            }
                        }
                    ]
                },
            )

        evidence = [
            EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="one", note="note"),
            EvidenceItem(evidence_id="e-2", source_id="source-2", excerpt="two", note="note"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            draft = client.synthesize("question", evidence)

        self.assertEqual(draft.answer, "combined fact")
        self.assertEqual(draft.cited_source_ids, ["source-1", "source-2"])
        user_payload = json.loads(captured_payloads[0]["messages"][1]["content"])
        self.assertIn("sources", user_payload)
        self.assertNotIn("new_source", user_payload)
        self.assertEqual(len(user_payload["sources"]), 2)

    def test_synthesize_repairs_invalid_citation_before_non_answer_filtering(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": (
                    '{"summary":"missing","answer":"I cannot provide the answer because the source does not contain it.",'
                    '"cited_source_ids":["malformed-source"],"remaining_questions":["What is the answer?"],'
                    '"source_answers_question":true}'
                )}}]},
            )

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="irrelevant", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            draft = client.synthesize("question", evidence)

        self.assertEqual(draft.cited_source_ids, ["source-1"])
        self.assertEqual(draft.remaining_questions, ["What is the answer?"])
        self.assertTrue(draft.source_answers_question)

    def test_structured_rejection_wins_over_useful_sounding_prose(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": (
                    '{"summary":"looks useful","answer":"It is 72 degrees.",'
                    '"cited_source_ids":["source-1"],"remaining_questions":["Need actual weather"],'
                    '"source_answers_question":false}'
                )}}]},
            )

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="irrelevant", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            draft = client.synthesize("question", evidence)

        self.assertFalse(draft.source_answers_question)
        self.assertEqual(draft.answer, "")
        self.assertEqual(draft.cited_source_ids, [])

    def test_synthesize_accepts_markdown_wrapped_json(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "```json\n"
                                    '{"summary":"short","answer":"fact","cited_source_ids":["source-1"],'
                                    '"remaining_questions":[],"source_answers_question":true}\n'
                                    "```"
                                )
                            }
                        }
                    ]
                },
            )

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            draft = client.synthesize("question", evidence)

        self.assertEqual(draft.answer, "fact")
        self.assertEqual(draft.cited_source_ids, ["source-1"])

    def test_synthesize_accepts_plain_code_fence_with_surrounding_whitespace(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "  \n```\n"
                                    '{"summary":"short","answer":"fact","cited_source_ids":["source-1"],'
                                    '"remaining_questions":[],"source_answers_question":true}\n'
                                    "```  \n"
                                )
                            }
                        }
                    ]
                },
            )

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            draft = client.synthesize("question", evidence)

        self.assertEqual(draft.answer, "fact")

    def test_synthesize_accepts_valid_json_with_trailing_tool_marker(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"short","answer":"fact","cited_source_ids":["source-1"],'
                                    '"remaining_questions":[],"source_answers_question":true}'
                                    '<|tool_response>'
                                )
                            }
                        }
                    ]
                },
            )

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            draft = client.synthesize("question", evidence)

        self.assertEqual(draft.answer, "fact")
        self.assertEqual(draft.cited_source_ids, ["source-1"])

    def test_synthesize_retries_once_after_incomplete_json_output(self) -> None:
        call_count = 0
        captured_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            captured_payloads.append(json.loads(request.content.decode("utf-8")))
            if call_count == 1:
                content = '{"summary":"'
            else:
                content = (
                    '{"summary":"short","answer":"fact","cited_source_ids":["source-1"],'
                    '"remaining_questions":[],"source_answers_question":true}'
                )
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            draft = client.synthesize("question", evidence)

        self.assertEqual(call_count, 2)
        self.assertEqual(draft.answer, "fact")
        self.assertIn(
            "Your previous response was malformed or incomplete.",
            captured_payloads[1]["messages"][0]["content"],
        )

    def test_synthesize_retries_once_when_answer_contains_control_artifacts(self) -> None:
        call_count = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                content = (
                    '{"summary":"short","answer":"Lead in <channel|>```json`{","cited_source_ids":["source-1"],'
                    '"remaining_questions":[],"source_answers_question":true}'
                )
            else:
                content = (
                    '{"summary":"short","answer":"Clean answer","cited_source_ids":["source-1"],'
                    '"remaining_questions":[],"source_answers_question":true}'
                )
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            with self.assertLogs("magpie.providers.openai_compatible", level="WARNING") as log_records:
                draft = client.synthesize("question", evidence)

        self.assertEqual(call_count, 2)
        self.assertEqual(draft.answer, "Clean answer")
        self.assertTrue(
            any("control artifacts" in record for record in log_records.output),
            f"Expected a warning about control artifacts, got: {log_records.output}",
        )

    def test_synthesize_does_not_retry_for_inline_markdown_fence(self) -> None:
        # Regression test for #60: a legitimate synthesis answer that contains
        # an inline Markdown code block must NOT be treated as a control
        # artifact and must NOT trigger a retry. Whole-payload fences are
        # already stripped by _strip_code_fence(); fences inside parsed
        # values are legitimate answer content.
        call_count = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            content = (
                '{"summary":"short","answer":"Here is an example:\\n```python\\n'
                'print(\\"hello\\")\\n```\\nDone.",'
                '"cited_source_ids":["source-1"],'
                '"remaining_questions":[],"source_answers_question":true}'
            )
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            draft = client.synthesize("question", evidence)

        # Only one call should occur: the inline fences are not control artifacts.
        self.assertEqual(call_count, 1)
        self.assertIn("```python", draft.answer)

    def test_propose_query_uses_state_aware_user_message(self) -> None:
        captured_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": '{"query":"current mayor of Seattle"}'
                            }
                        }
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            proposal = client.propose_query(
                "who is the mayor of seattle",
                PlanningContext(["old query"], [], [], RunBudget(1, 1, 1)),
            )

        self.assertEqual(proposal.query, "current mayor of Seattle")
        user_payload = json.loads(captured_payloads[0]["messages"][1]["content"])
        self.assertEqual(user_payload["question"], "who is the mayor of seattle")
        self.assertEqual(user_payload["prior_queries"], ["old query"])
        self.assertEqual(captured_payloads[0]["response_format"]["type"], "json_schema")

    def test_route_request_supports_news_route(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"route":"news","weather_kind":null,"zip_code":null}'}}]},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            decision = client.route_request("latest AI news")

        self.assertEqual(decision.route, RequestRoute.NEWS)
        self.assertIsNone(decision.zip_code)

    def test_classify_news_request_returns_category_and_time_scope(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": (
                    '{"kind":"category","category":"ai","time_scope":"today"}'
                )}}]},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            news_request = client.classify_news_request("AI news today")

        self.assertEqual(news_request.kind, NewsRequestKind.CATEGORY)
        self.assertEqual(news_request.category, NewsCategory.AI)
        self.assertEqual(news_request.time_scope, NewsTimeScope.TODAY)

    def test_classify_news_request_preserves_unsupported_topic(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": (
                    '{"kind":"unsupported_topic","category":null,"time_scope":"last_24_hours"}'
                )}}]},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            news_request = client.classify_news_request("latest OpenAI news")

        self.assertEqual(news_request.kind, NewsRequestKind.UNSUPPORTED_TOPIC)
        self.assertIsNone(news_request.category)
        self.assertEqual(news_request.time_scope, NewsTimeScope.LAST_24_HOURS)

    def test_route_request_combines_classification_and_zip_normalization(self) -> None:
        captured_payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": (
                    '{"route":"weather","weather_kind":"forecast","zip_code":"98230"}'
                )}}]},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            decision = client.route_request("Will it rain tomorrow in Blaine?")

        self.assertEqual(decision.route, RequestRoute.WEATHER)
        self.assertEqual(decision.weather_kind, WeatherKind.FORECAST)
        self.assertEqual(decision.zip_code, "98230")
        self.assertEqual(
            captured_payloads[0]["response_format"]["json_schema"]["name"],
            "magpie_route_request",
        )

    def test_weather_route_with_invalid_zip_preserves_route_for_web_fallback(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": (
                    '{"route":"weather","weather_kind":"conditions","zip_code":"unknown"}'
                )}}]},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            decision = client.route_request("what is the weather in a mystery place?")

        self.assertEqual(decision.route, RequestRoute.WEATHER)
        self.assertIsNone(decision.zip_code)

    def test_anime_request_selects_only_needed_lookup_fields(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": (
                    '{"kind":"lookup","title_query":"Ascendance of a Bookworm season 2",'
                    '"character_query":null,"requested_fields":["episodes"]}'
                )}}]},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            request = client.classify_anime_request(
                "how many episodes are in season 2 of ascendence of a bookworm"
            )

        self.assertEqual(request.kind, AnimeRequestKind.LOOKUP)
        self.assertEqual(request.requested_fields, [AnimeField.EPISODES])

    def test_lookup_fields_override_contradictory_anime_kind(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": (
                    '{"kind":"credits","title_query":"Bookworm","character_query":null,'
                    '"requested_fields":["episodes"]}'
                )}}]},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            request = client.classify_anime_request("how many episodes are in Bookworm?")

        self.assertEqual(request.kind, AnimeRequestKind.LOOKUP)

    def test_character_credit_intent_overrides_spurious_lookup_fields(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": (
                    '{"kind":"credits","title_query":"Yakuza Fiance","character_query":"Kirishima",'
                    '"requested_fields":["episodes","end_date"]}'
                )}}]},
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir),
                transport=httpx.MockTransport(handler),
            )
            request = client.classify_anime_request("voice actor for Kirishima in Yakuza Fiance")

        self.assertEqual(request.kind, AnimeRequestKind.CREDITS)
        self.assertEqual(request.requested_fields, [])

    def test_resolver_debug_log_is_cleared_and_written(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"short","answer":"fact","cited_source_ids":["source-1"],'
                                    '"remaining_questions":[],"source_answers_question":true}'
                                )
                            }
                        }
                    ]
                },
            )

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "resolver.log"
            log_path.write_text("old junk", encoding="utf-8")
            client = OpenAICompatibleResolverClient(
                settings=self._settings(
                    tmpdir,
                    resolver_debug_log_path=str(log_path),
                    resolver_include_raw_output=True,
                ),
                transport=httpx.MockTransport(handler),
            )
            client.begin_request_debug_log("run-123", "question")
            client.synthesize("question", evidence)

            content = log_path.read_text(encoding="utf-8")

        self.assertIn("old junk", content)
        self.assertIn("=== Step 1: synthesize ===", content)
        self.assertIn(f"config_path: {str((Path(tmpdir) / 'config.json').resolve())}", content)
        self.assertIn("resolver_include_raw_output: true", content)
        self.assertIn("SYSTEM PROMPT", content)
        self.assertIn("USER PAYLOAD", content)
        self.assertIn("MODEL OUTPUT", content)
        self.assertEqual(content.count("=== Step 1: synthesize ==="), 1)

    def test_resolver_debug_log_records_http_errors(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow slow slow")

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "resolver.log"
            client = OpenAICompatibleResolverClient(
                settings=self._settings(tmpdir, resolver_debug_log_path=str(log_path)),
                transport=httpx.MockTransport(handler),
            )
            client.begin_request_debug_log("run-123", "question")
            with self.assertRaises(httpx.ReadTimeout):
                client.synthesize("question", evidence)
            content = log_path.read_text(encoding="utf-8")

        self.assertIn("ERROR", content)
        self.assertIn("ReadTimeout", content)

    def test_resolver_debug_log_omits_model_output_when_disabled(self) -> None:
        raw_output = '{"summary":"secret","answer":"secret","cited_source_ids":[],"remaining_questions":[]}'

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": raw_output}}]})

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "resolver.log"
            client = OpenAICompatibleResolverClient(
                settings=self._settings(
                    tmpdir,
                    resolver_debug_log_path=str(log_path),
                    resolver_include_raw_output=False,
                ),
                transport=httpx.MockTransport(handler),
            )
            client.begin_request_debug_log("run-123", "question")
            client.propose_query("question", PlanningContext([], [], [], RunBudget(1, 1, 1)))
            content = log_path.read_text(encoding="utf-8")

        self.assertIn("http_status: 200", content)
        self.assertNotIn("MODEL OUTPUT", content)
        self.assertNotIn(raw_output, content)
        self.assertNotIn("hidden; set resolver_include_raw_output", content)

    def test_raw_log_and_result_tolerate_unpaired_surrogate(self) -> None:
        raw_output = (
            '{"summary":"short","answer":"bad \\ud8a2 text","cited_source_ids":["source-1"],'
            '"remaining_questions":[],"source_answers_question":true}'
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": raw_output}}]})

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "resolver.log"
            client = OpenAICompatibleResolverClient(
                settings=self._settings(
                    tmpdir,
                    resolver_debug_log_path=str(log_path),
                    resolver_include_raw_output=True,
                ),
                transport=httpx.MockTransport(handler),
            )
            client.begin_request_debug_log("run-123", "question")
            draft = client.synthesize("question", evidence)
            content = log_path.read_text(encoding="utf-8")

        self.assertEqual(draft.answer, "bad ? text")
        self.assertIn("\\ud8a2", content)

    def test_resolver_debug_log_rotates_when_size_cap_exceeded(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"s","answer":"a","cited_source_ids":[],'
                                    '"remaining_questions":[],"source_answers_question":true}'
                                )
                            }
                        }
                    ]
                },
            )

        evidence = [EvidenceItem(evidence_id="e-1", source_id="source-1", excerpt="fact", note="note")]
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "resolver.log"
            # Pre-seed the log so it already exceeds the small cap.
            log_path.write_text("x" * 100, encoding="utf-8")
            client = OpenAICompatibleResolverClient(
                settings=self._settings(
                    tmpdir,
                    resolver_debug_log_path=str(log_path),
                    resolver_debug_log_max_bytes=50,
                ),
                transport=httpx.MockTransport(handler),
            )
            client.begin_request_debug_log("run-123", "question")
            client.synthesize("question", evidence)

            self.assertTrue(log_path.exists())
            rotated = log_path.with_suffix(".log.1")
            self.assertTrue(rotated.exists())
            old = rotated.read_text(encoding="utf-8")
            new = log_path.read_text(encoding="utf-8")
            # The oversized pre-seed must have moved to the rotation sidecar,
            # and the new run must start fresh in the primary log.
            self.assertEqual(old, "x" * 100)
            self.assertNotIn("x" * 100, new)
            self.assertIn("run_id: run-123", new)

    def test_resolver_debug_log_rotation_disabled_when_cap_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "resolver.log"
            log_path.write_text("x" * 100, encoding="utf-8")
            client = OpenAICompatibleResolverClient(
                settings=self._settings(
                    tmpdir,
                    resolver_debug_log_path=str(log_path),
                    resolver_debug_log_max_bytes=0,
                ),
                transport=httpx.MockTransport(handler=lambda _r: httpx.Response(200, json={})),
            )
            client.begin_request_debug_log("run-123", "question")
            # No rotation sidecar should appear when the cap is disabled.
            self.assertFalse(log_path.with_suffix(".log.1").exists())
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("x" * 100, content)
            self.assertIn("run_id: run-123", content)


if __name__ == "__main__":
    unittest.main()
