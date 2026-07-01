"""Exa-backed search provider implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from ..config import Settings
from ..errors import SearchError
from ..models import FreshnessClass, SearchRequest, SearchResultRecord
from ._http import request_with_retry


@dataclass(slots=True)
class ExaSearchClient:
    """Search provider using Exa MCP first, with optional HTTP API fallback."""

    settings: Settings
    transport: httpx.BaseTransport | None = None
    _http_client_store: httpx.Client | None = field(default=None, repr=False)

    def search(self, request: SearchRequest) -> list[SearchResultRecord]:
        errors: list[str] = []
        if self.settings.search_transport in {"mcp_first", "mcp_only"}:
            try:
                return self._search_via_mcp(request)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"mcp: {exc}")
                if self.settings.search_transport == "mcp_only":
                    raise SearchError("; ".join(errors)) from exc

        if self.settings.search_transport in {"mcp_first", "api_only"} and self.settings.search_api_key:
            try:
                return self._search_via_api(request)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"api: {exc}")
                raise SearchError("; ".join(errors)) from exc

        if errors:
            raise SearchError("; ".join(errors))
        raise SearchError("Exa search is not configured: MCP disabled/unavailable and no API key is set.")

    def doctor_check(self, live: bool = False) -> dict[str, object]:
        report: dict[str, object] = {
            "status": "ok",
            "provider": "exa",
            "transport": self.settings.search_transport,
            "mcp_url": self.settings.search_mcp_url,
            "api_key_configured": bool(self.settings.search_api_key),
            "live": live,
        }
        if not live:
            return report
        try:
            if self.settings.search_transport in {"mcp_first", "mcp_only"}:
                response = request_with_retry(
                    self._client(),
                    "POST",
                    self.settings.search_mcp_url,
                    headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                    json={
                        "jsonrpc": "2.0",
                        "id": "doctor",
                        "method": "tools/call",
                        "params": {"name": self.settings.search_mcp_tool_name, "arguments": {"query": "test", "numResults": 1}},
                    },
                    max_attempts=self.settings.http_retry_max_attempts,
                    backoff_seconds=self.settings.http_retry_backoff_seconds,
                )
                report["mcp_status_code"] = response.status_code
            elif self.settings.search_transport == "api_only" and self.settings.search_api_key:
                response = request_with_retry(
                    self._client(),
                    "POST",
                    self.settings.search_base_url.rstrip("/") + "/search",
                    headers={
                        "x-api-key": self.settings.search_api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": "test",
                        "type": "auto",
                        "numResults": 1,
                        "contents": {"text": {"maxCharacters": 500}, "highlights": True},
                    },
                    max_attempts=self.settings.http_retry_max_attempts,
                    backoff_seconds=self.settings.http_retry_backoff_seconds,
                )
                report["api_status_code"] = response.status_code
            else:
                return report
            if response.status_code >= 400:
                report["status"] = "error"
                report["message"] = response.text[:300]
        except httpx.RequestError as exc:
            report["status"] = "error"
            report["message"] = f"Exa live check failed: {exc}"
        return report

    def _search_via_mcp(self, request: SearchRequest) -> list[SearchResultRecord]:
        response = request_with_retry(
            self._client(),
            "POST",
            self.settings.search_mcp_url,
            headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": self.settings.search_mcp_tool_name,
                    "arguments": {
                        "query": request.query,
                        "numResults": request.limit,
                        "livecrawl": "fallback",
                        "type": "auto",
                        "contextMaxCharacters": self.settings.search_inline_content_max_characters,
                    },
                },
            },
            max_attempts=self.settings.http_retry_max_attempts,
            backoff_seconds=self.settings.http_retry_backoff_seconds,
        )
        if response.status_code >= 400:
            raise SearchError(f"Exa MCP error {response.status_code}: {response.text[:300]}")

        text = self._extract_mcp_text(response.text)
        return self._parse_mcp_blocks(text)

    def _search_via_api(self, request: SearchRequest) -> list[SearchResultRecord]:
        payload: dict[str, Any] = {
            "query": request.query,
            "type": "auto",
            "numResults": request.limit,
            "contents": {
                "text": {"maxCharacters": self.settings.search_inline_content_max_characters},
                "highlights": True,
            },
        }
        if request.freshness_class == FreshnessClass.RECENT:
            payload["startPublishedDate"] = (
                datetime.now(UTC) - timedelta(days=7)
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        response = request_with_retry(
            self._client(),
            "POST",
            self.settings.search_base_url.rstrip("/") + "/search",
            headers={
                "x-api-key": self.settings.search_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            max_attempts=self.settings.http_retry_max_attempts,
            backoff_seconds=self.settings.http_retry_backoff_seconds,
        )
        if response.status_code >= 400:
            raise SearchError(f"Exa API error {response.status_code}: {response.text[:300]}")
        data = response.json()
        return [self._map_api_result(item) for item in data.get("results", []) if item.get("url")]

    def _client(self) -> httpx.Client:
        # Reuse a long-lived client across calls to avoid repeated TLS
        # handshakes; honor the injected transport for tests.
        if self._http_client_store is None:
            self._http_client_store = httpx.Client(
                timeout=self.settings.search_timeout_seconds,
                verify=self.settings.verify_tls,
                transport=self.transport,
            )
        return self._http_client_store

    def _extract_mcp_text(self, body: str) -> str:
        """Extract the text content from an Exa MCP JSON-RPC/SSE response.

        MCP parsing is best-effort: Exa returns results as a text blob inside
        a JSON-RPC ``content`` array, and this method locates that text. If
        Exa changes the envelope shape, a ``SearchError`` is raised. For
        production use, ``api_only`` is recommended when an API key is
        available; the REST API returns structured JSON and is unaffected by
        text-format changes.
        """
        parsed: dict[str, Any] | None = None
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            try:
                candidate = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if candidate.get("result") or candidate.get("error"):
                parsed = candidate
                break
        if parsed is None:
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise SearchError("Exa MCP returned an unreadable response.") from exc
        if parsed.get("error"):
            raise SearchError(parsed["error"].get("message", "Exa MCP returned an error."))
        if parsed.get("result", {}).get("isError"):
            content = parsed["result"].get("content", [])
            message = next((item.get("text", "").strip() for item in content if item.get("type") == "text"), "")
            raise SearchError(message or "Exa MCP returned an error.")
        for item in parsed.get("result", {}).get("content", []):
            if item.get("type") == "text" and item.get("text", "").strip():
                return item["text"]
        raise SearchError("Exa MCP returned empty content.")

    def _parse_mcp_blocks(self, text: str) -> list[SearchResultRecord]:
        """Parse Exa MCP text content into search-result records.

        MCP parsing is best-effort: the text is split heuristically on the
        ``Title:`` separator and each block is scanned for ``URL:``,
        ``Text:``, and ``Highlights:`` fields. If Exa changes this text
        format, ``SearchError`` is raised instead of silently returning an
        empty list. In ``mcp_first`` mode this triggers API fallback; in
        ``mcp_only`` mode it surfaces the error. For production use,
        ``api_only`` is recommended when an API key is available.
        """
        results: list[SearchResultRecord] = []
        blocks = [block for block in text.split("Title: ") if block.strip()]
        for block in blocks:
            cleaned = "Title: " + block if not block.startswith("Title: ") else block
            title = self._match_line(cleaned, "Title") or "Untitled result"
            url = self._match_line(cleaned, "URL")
            if not url:
                continue
            content = ""
            text_start = cleaned.find("\nText: ")
            if text_start >= 0:
                content = cleaned[text_start + 7 :].strip()
            else:
                highlights_start = cleaned.find("\nHighlights:\n")
                if highlights_start >= 0:
                    content = cleaned[highlights_start + len("\nHighlights:\n") :].strip()
            content = content.removesuffix("\n---").strip()
            snippet = " ".join(content.split())[:500]
            results.append(
                SearchResultRecord(
                    title=title,
                    url=url,
                    snippet=snippet,
                    provider="exa_mcp",
                    inline_text=content or None,
                    raw_result={"text_block": cleaned},
                )
            )
        if not results:
            raise SearchError(
                "Exa MCP response did not contain any parseable results; "
                "the MCP text format may have changed. Consider using "
                "search_transport='api_only'."
            )
        return results

    def _map_api_result(self, item: dict[str, Any]) -> SearchResultRecord:
        highlights = [value for value in item.get("highlights", []) if isinstance(value, str) and value.strip()]
        text_value = item.get("text")
        inline_text = text_value if isinstance(text_value, str) and text_value.strip() else None
        snippet = highlights[0] if highlights else (inline_text or "")
        return SearchResultRecord(
            title=item.get("title") or "Untitled result",
            url=item["url"],
            snippet=snippet[:500],
            site_name=item.get("author"),
            published_at=item.get("publishedDate"),
            provider="exa_api",
            author=item.get("author"),
            inline_text=inline_text,
            highlights=highlights,
            raw_result=item,
        )

    def _match_line(self, text: str, label: str) -> str | None:
        prefix = f"{label}: "
        for line in text.splitlines():
            if line.startswith(prefix):
                return line[len(prefix) :].strip()
        return None
