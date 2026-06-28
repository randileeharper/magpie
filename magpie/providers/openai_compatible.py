"""OpenAI-compatible resolver client."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import httpx

from ..config import Settings
from ..errors import ResolverError
from ..models import (
    AnimeCandidate,
    AnimeField,
    AnimeRequest,
    AnimeRequestKind,
    CharacterCredit,
    EvidenceItem,
    NewsCategory,
    NewsRequest,
    NewsRequestKind,
    NewsTimeScope,
    PlanningContext,
    QueryProposal,
    RequestRoute,
    RouteDecision,
    SynthesisDraft,
    WeatherKind,
)
from ..text import valid_unicode, valid_unicode_tree
from .base import reasoning_request_options
from ..prompts.loader import load_prompt

_DEBUG_LOG_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenAICompatibleResolverClient:
    """Resolver client for local OpenAI-compatible chat APIs."""

    settings: Settings
    transport: httpx.BaseTransport | None = None
    _local: threading.local = field(default_factory=threading.local)

    def route_request(self, question: str) -> RouteDecision:
        payload = self._ask_json(
            "route_request",
            system=load_prompt("route_request"),
            user={"question": question},
            schema_name="magpie_route_request",
            schema={
                "type": "object",
                "properties": {
                    "route": {"type": "string", "enum": ["web_research", "weather", "anime", "news"]},
                    "weather_kind": {"type": ["string", "null"], "enum": ["conditions", "forecast", None]},
                    "zip_code": {"type": ["string", "null"]},
                },
                "required": ["route", "weather_kind", "zip_code"],
                "additionalProperties": False,
            },
        )
        try:
            route = RequestRoute(str(payload.get("route")))
        except ValueError:
            route = RequestRoute.WEB_RESEARCH
        zip_code = str(payload["zip_code"]).strip() if payload.get("zip_code") is not None else None
        if not zip_code or len(zip_code) != 5 or not zip_code.isdigit():
            zip_code = None
        if route != RequestRoute.WEATHER:
            return RouteDecision(route=route)
        try:
            weather_kind = WeatherKind(str(payload.get("weather_kind")))
        except ValueError:
            weather_kind = WeatherKind.CONDITIONS
        return RouteDecision(route=route, weather_kind=weather_kind, zip_code=zip_code)

    def classify_anime_request(self, question: str) -> AnimeRequest:
        payload = self._ask_json(
            "classify_anime_request",
            system=load_prompt("classify_anime_request"),
            user={"question": question},
            schema_name="magpie_anime_request",
            schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["lookup", "credits", "schedule"]},
                    "title_query": {"type": ["string", "null"]},
                    "character_query": {"type": ["string", "null"]},
                    "requested_fields": {
                        "type": "array",
                        "items": {"type": "string", "enum": [item.value for item in AnimeField]},
                        "maxItems": 6,
                        "uniqueItems": True,
                    },
                },
                "required": ["kind", "title_query", "character_query", "requested_fields"],
                "additionalProperties": False,
            },
        )
        try:
            kind = AnimeRequestKind(str(payload.get("kind")))
        except ValueError:
            kind = AnimeRequestKind.LOOKUP
        title = str(payload["title_query"]).strip() if payload.get("title_query") else None
        character = str(payload["character_query"]).strip() if payload.get("character_query") else None
        fields: list[AnimeField] = []
        for value in payload.get("requested_fields", []):
            try:
                field = AnimeField(str(value))
            except ValueError:
                continue
            if field not in fields:
                fields.append(field)
        if kind == AnimeRequestKind.SCHEDULE:
            fields = []
        elif kind == AnimeRequestKind.CREDITS and character:
            fields = []
        elif fields:
            kind = AnimeRequestKind.LOOKUP
        if kind == AnimeRequestKind.LOOKUP and not fields:
            fields = [AnimeField.DESCRIPTION]
        return AnimeRequest(kind, title, character, fields)

    def classify_news_request(self, question: str) -> NewsRequest:
        payload = self._ask_json(
            "classify_news_request",
            system=load_prompt("classify_news_request"),
            user={"question": question},
            schema_name="magpie_news_request",
            schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["category", "unsupported_topic"]},
                    "category": {
                        "type": ["string", "null"],
                        "enum": [
                            "general", "world", "us", "politics", "business", "technology", "ai",
                            "science", "health", "entertainment", "sports", None,
                        ],
                    },
                    "time_scope": {
                        "type": "string",
                        "enum": ["last_24_hours", "today", "yesterday", "last_7_days"],
                    },
                },
                "required": ["kind", "category", "time_scope"],
                "additionalProperties": False,
            },
        )
        try:
            kind = NewsRequestKind(str(payload.get("kind")))
        except ValueError:
            kind = NewsRequestKind.UNSUPPORTED_TOPIC
        try:
            category = NewsCategory(str(payload.get("category"))) if payload.get("category") is not None else None
        except ValueError:
            category = None
        try:
            time_scope = NewsTimeScope(str(payload.get("time_scope")))
        except ValueError:
            time_scope = NewsTimeScope.LAST_24_HOURS
        if kind == NewsRequestKind.CATEGORY and category is None:
            kind = NewsRequestKind.UNSUPPORTED_TOPIC
        return NewsRequest(kind=kind, category=category, time_scope=time_scope)

    def refine_anime_title_queries(self, question: str, attempted_query: str) -> list[str]:
        payload = self._ask_json(
            "refine_anime_title_queries",
            system=load_prompt("refine_anime_title_queries"),
            user={"question": question, "attempted_query": attempted_query},
            schema_name="magpie_anime_title_queries",
            schema={
                "type": "object",
                "properties": {
                    "title_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                        "uniqueItems": True,
                    }
                },
                "required": ["title_queries"],
                "additionalProperties": False,
            },
        )
        queries = self._string_list(payload.get("title_queries"))
        return queries[:3] or [attempted_query]

    def select_anime_candidate(self, question: str, candidates: list[AnimeCandidate]) -> int | None:
        if not candidates:
            return None
        payload = self._ask_json(
            "select_anime_candidate",
            system=load_prompt("select_anime_candidate"),
            user={"question": question, "candidates": [
                {
                    "anime_id": item.anime_id,
                    "english": item.english_title,
                    "romaji": item.romaji_title,
                    "native": item.native_title,
                    "format": item.format,
                    "year": item.season_year,
                }
                for item in candidates
            ]},
            schema_name="magpie_anime_candidate",
            schema={
                "type": "object",
                "properties": {"anime_id": {"type": ["integer", "null"]}},
                "required": ["anime_id"],
                "additionalProperties": False,
            },
        )
        selected = payload.get("anime_id")
        allowed = {item.anime_id for item in candidates}
        return selected if isinstance(selected, int) and selected in allowed else None

    def select_character(self, query: str, credits: list[CharacterCredit]) -> str | None:
        if not credits:
            return None
        payload = self._ask_json(
            "select_anime_character",
            system=load_prompt("select_character"),
            user={"character_query": query, "characters": [item.character_name for item in credits]},
            schema_name="magpie_anime_character",
            schema={
                "type": "object",
                "properties": {"character_name": {"type": ["string", "null"]}},
                "required": ["character_name"],
                "additionalProperties": False,
            },
        )
        selected = payload.get("character_name")
        allowed = {item.character_name for item in credits}
        return selected if isinstance(selected, str) and selected in allowed else None

    def propose_query(self, question: str, context: PlanningContext) -> QueryProposal:
        payload = self._ask_json(
            "propose_query",
            system=load_prompt("propose_query"),
            user={
                "question": question,
                "prior_queries": context.prior_queries,
                "seen_urls": context.seen_urls[-20:],
                "remaining_questions": context.remaining_questions,
                "budget": {
                    "queries_remaining": context.budget.queries_remaining,
                    "sources_remaining": context.budget.sources_remaining,
                    "evidence_remaining": context.budget.evidence_remaining,
                },
            },
            schema_name="magpie_propose_query",
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )
        query = str(payload.get("query", question)).strip() or question
        return QueryProposal(query=query)

    def synthesize(
        self,
        question: str,
        evidence: list[EvidenceItem],
        prior_draft: SynthesisDraft | None = None,
    ) -> SynthesisDraft:
        if not evidence:
            raise ResolverError("Synthesis requires at least one evidence item.")
        max_chars = self.settings.max_synthesis_input_characters
        per_item = max(500, max_chars // max(len(evidence), 1))
        bounded = [
            EvidenceItem(item.evidence_id, item.source_id, item.excerpt[:per_item], item.note)
            for item in evidence
        ]
        allowed_source_ids = list(dict.fromkeys(
            [*(prior_draft.cited_source_ids if prior_draft else []), *[item.source_id for item in bounded]]
        ))
        procedural = question.lower().strip().startswith(("how do i ", "how to ", "steps to ", "guide to "))
        explanatory = question.lower().strip().startswith(
            ("explain ", "what is ", "what are ", "describe ", "overview of ", "introduction to ")
        )
        system = load_prompt("synthesize")
        if explanatory:
            system += load_prompt("synthesize_explanatory")
        if procedural:
            system += load_prompt("synthesize_procedural")
        user = {
            "question": question,
            "prior_draft": {
                "summary": prior_draft.summary,
                "answer": self._bounded_prior_answer(prior_draft.answer),
                "cited_source_ids": prior_draft.cited_source_ids,
                "remaining_questions": prior_draft.remaining_questions,
            } if prior_draft else None,
            "sources": [
                {"source_id": item.source_id, "content": item.excerpt}
                for item in bounded
            ],
            "allowed_source_ids": allowed_source_ids,
        }
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "answer": {"type": "string"},
                "cited_source_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "remaining_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "source_answers_question": {"type": "boolean"},
            },
            "required": [
                "summary",
                "answer",
                "cited_source_ids",
                "remaining_questions",
                "source_answers_question",
            ],
            "additionalProperties": False,
        }
        payload = self._ask_json(
            "synthesize",
            system=system,
            user=user,
            schema_name="magpie_synthesis",
            schema=schema,
        )
        if self._payload_contains_control_artifacts(payload):
            LOGGER.warning(
                "Synthesis response contained control artifacts; retrying with hardened prompt."
            )
            payload = self._ask_json(
                "synthesize_retry",
                system=(
                    system
                    + load_prompt("synthesize_retry")
                ),
                user=user,
                schema_name="magpie_synthesis_retry",
                schema=schema,
            )
        evidence_ids = set(allowed_source_ids)
        remaining_questions = self._string_list(payload.get("remaining_questions"))
        source_answers_question = payload.get("source_answers_question") is True
        cited = [source_id for source_id in self._string_list(payload.get("cited_source_ids")) if source_id in evidence_ids]
        answer = str(payload.get("answer", "")).strip()
        if source_answers_question and answer and not cited:
            cited = list(prior_draft.cited_source_ids) if prior_draft and prior_draft.cited_source_ids else [
                bounded[0].source_id
            ]
        if not source_answers_question:
            return SynthesisDraft(
                summary=prior_draft.summary if prior_draft else "",
                answer=prior_draft.answer if prior_draft else "",
                cited_source_ids=list(prior_draft.cited_source_ids) if prior_draft else [],
                remaining_questions=remaining_questions or (
                    list(prior_draft.remaining_questions) if prior_draft else [question]
                ),
                source_answers_question=False,
            )
        return SynthesisDraft(
            summary=str(payload.get("summary", f"Grounded answer for: {question}")).strip(),
            answer=answer,
            cited_source_ids=cited,
            remaining_questions=remaining_questions,
            source_answers_question=True,
        )

    def reasoning_request_options(self) -> dict[str, object]:
        return reasoning_request_options(self.settings.resolver_include_reasoning)

    def begin_request_debug_log(self, run_id: str, question: str) -> None:
        self._local.run_id = run_id
        self._local.call_index = 0
        path = self.settings.expanded_resolver_debug_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with _DEBUG_LOG_LOCK, path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(
                f"run_id: {run_id}\n"
                f"started_at: {timestamp}\n"
                f"config_path: {self.settings.loaded_config_path or '[defaults; no config file loaded]'}\n"
                f"resolver_include_raw_output: {str(self.settings.resolver_include_raw_output).lower()}\n"
                f"question: {question}\n"
                "\n"
            )

    def _ask_json(
        self,
        step: str,
        system: str,
        user: Any,
        *,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        user_content = user if isinstance(user, str) else json.dumps(user, ensure_ascii=True)
        headers = {"Content-Type": "application/json"}
        if self.settings.resolver_api_key:
            headers["Authorization"] = f"Bearer {self.settings.resolver_api_key}"
        last_content = ""
        for attempt in range(2):
            attempt_step = step if attempt == 0 else f"{step}_json_retry"
            system_prompt = system
            if attempt == 1:
                system_prompt += load_prompt("json_retry")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            payload = {
                "model": self.settings.resolver_model,
                "messages": messages,
                "max_tokens": self.settings.resolver_max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
                **self.reasoning_request_options(),
            }
            started = perf_counter()
            self._log_request_start(
                step=attempt_step, system_prompt=system_prompt, user_payload=user,
            )
            response_json: dict[str, Any] | None = None
            try:
                with httpx.Client(
                    timeout=self.settings.request_timeout_seconds,
                    verify=self.settings.verify_tls,
                    transport=self.transport,
                ) as client:
                    response = client.post(
                        self.settings.resolver_base_url.rstrip("/") + "/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                elapsed_ms = round((perf_counter() - started) * 1000, 2)
                response_text = response.text
                try:
                    response_json = response.json()
                except ValueError:
                    response_json = None
                self._log_request_response(
                    elapsed_ms=elapsed_ms,
                    response_status_code=response.status_code,
                    response_text=self._extract_response_content(response_json) or response_text,
                )
            except httpx.HTTPError as exc:
                elapsed_ms = round((perf_counter() - started) * 1000, 2)
                self._log_request_error(
                    elapsed_ms=elapsed_ms, error_text=f"{type(exc).__name__}: {exc}",
                )
                raise
            if response.status_code >= 400:
                raise ResolverError(f"Resolver HTTP error {response.status_code}: {response.text[:300]}")
            if response_json is None:
                raise ResolverError("Resolver returned a non-JSON HTTP response.")
            data = response_json
            try:
                content = data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001
                raise ResolverError("Resolver response did not include a chat completion message.") from exc
            if not isinstance(content, str) or not content.strip():
                raise ResolverError("Resolver returned empty content.")
            last_content = content
            try:
                parsed = self._load_json_object_prefix(content)
            except json.JSONDecodeError:
                if attempt == 0:
                    continue
                raise ResolverError(f"Resolver returned malformed JSON: {content[:300]}") from None
            if not isinstance(parsed, dict):
                raise ResolverError("Resolver returned JSON that was not an object.")
            return valid_unicode_tree(parsed)
        raise ResolverError(f"Resolver returned malformed JSON: {last_content[:300]}")

    def _redact(self, text: str) -> str:
        """Strip configured secrets from text before writing debug logs."""
        redacted = valid_unicode(text)
        for secret in (
            self.settings.search_api_key,
            self.settings.resolver_api_key,
            self.settings.historian_token,
        ):
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    def _log_request_start(
        self, *, step: str, system_prompt: str, user_payload: Any
    ) -> None:
        path = self.settings.expanded_resolver_debug_log_path
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_LOCK, path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            self._local.call_index = getattr(self._local, "call_index", 0) + 1
            formatted_user_payload = self._format_user_payload(user_payload)
            handle.write(f"=== Step {self._local.call_index}: {step} ===\n")
            handle.write(f"run_id: {getattr(self._local, 'run_id', '')}\n")
            handle.write(f"input_characters: {len(system_prompt) + len(formatted_user_payload)}\n")
            handle.write("\nSYSTEM PROMPT\n")
            handle.write(self._redact(system_prompt.strip()))
            handle.write("\n\nUSER PAYLOAD\n")
            handle.write(self._redact(formatted_user_payload))
            handle.write("\n\n")

    def _log_request_response(
        self, *, elapsed_ms: float, response_status_code: int, response_text: str | None
    ) -> None:
        path = self.settings.expanded_resolver_debug_log_path
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_LOCK, path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(f"elapsed_ms: {elapsed_ms}\n")
            handle.write(f"http_status: {response_status_code}\n")
            if response_text is not None and self.settings.resolver_include_raw_output:
                handle.write("\nMODEL OUTPUT\n")
                handle.write(self._redact(response_text.strip()))
                handle.write("\n\n")

    def _log_request_error(self, *, elapsed_ms: float, error_text: str) -> None:
        path = self.settings.expanded_resolver_debug_log_path
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_LOCK, path.open("a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(f"elapsed_ms: {elapsed_ms}\n")
            handle.write("\nERROR\n")
            handle.write(self._redact(error_text.strip()))
            handle.write("\n\n")

    def _format_user_payload(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)

    def _bounded_prior_answer(self, answer: str) -> str:
        limit = self.settings.max_incremental_answer_characters
        if len(answer) <= limit:
            return answer
        half = (limit - 32) // 2
        return answer[:half] + "\n[prior answer truncated]\n" + answer[-half:]

    def _extract_response_content(self, response_json: dict[str, Any] | None) -> str | None:
        if not isinstance(response_json, dict):
            return None
        try:
            content = response_json["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            return None
        return valid_unicode(content) if isinstance(content, str) else None

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def _load_json_object_prefix(self, content: str) -> dict[str, Any]:
        content = self._strip_code_fence(content)
        decoder = json.JSONDecoder()
        parsed, end = decoder.raw_decode(content)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("Expected JSON object", content, 0)
        trailing = content[end:].strip()
        if trailing and not self._is_ignorable_trailing_output(trailing):
            raise json.JSONDecodeError("Unexpected trailing content", content, end)
        return parsed

    _CODE_FENCE_RE = re.compile(
        r"\A\s*```(?:json)?\s*\n(?P<body>.*?)\n\s*```\s*\Z",
        re.DOTALL,
    )

    def _strip_code_fence(self, content: str) -> str:
        """Strip a single markdown code-fence wrapper around the whole payload.

        Local models frequently emit JSON wrapped in ```` ```json ```` fences.
        Only a fence enclosing the *entire* payload is stripped here; fences
        appearing *inside* parsed values are legitimate answer content and are
        not treated as control artifacts by ``_payload_contains_control_artifacts``.
        """
        match = self._CODE_FENCE_RE.match(content)
        if match:
            return match.group("body")
        return content

    def _is_ignorable_trailing_output(self, trailing: str) -> bool:
        ignorable_prefixes = (
            "<|tool_response>",
            "<channel|>",
        )
        return any(trailing.startswith(prefix) for prefix in ignorable_prefixes)

    def _payload_contains_control_artifacts(self, value: Any) -> bool:
        # Markdown code fences (```json / ```) are intentionally NOT treated as
        # control artifacts: a fence wrapping the whole payload is already
        # stripped by _strip_code_fence() before parsing, and fences appearing
        # inside parsed string values are legitimate answer content (e.g. a
        # synthesis answer that quotes a code block). Only model-specific
        # control tokens that indicate leaked control output are detected.
        markers = ("<channel|>", "<|tool_response>")
        if isinstance(value, str):
            return any(marker in value for marker in markers)
        if isinstance(value, list):
            return any(self._payload_contains_control_artifacts(item) for item in value)
        if isinstance(value, dict):
            return any(self._payload_contains_control_artifacts(item) for item in value.values())
        return False
