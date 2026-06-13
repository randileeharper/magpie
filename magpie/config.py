"""Configuration loading and validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .models import ResponseDetail


DEFAULT_CONFIG_CANDIDATES = (
    Path("config.json"),
    Path("~/.config/magpie/config.json"),
)


def _env_name(field_name: str) -> str:
    return f"MAGPIE_{field_name.upper()}"


def _coerce_value(raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass(slots=True)
class Settings:
    loaded_config_path: str | None = field(default=None, init=False)
    http_host: str = "127.0.0.1"
    http_port: int = 8766
    search_provider: str = "exa"
    search_transport: str = "mcp_first"
    search_mcp_url: str = "https://mcp.exa.ai/mcp"
    search_mcp_tool_name: str = "web_search_exa"
    search_base_url: str = "https://api.exa.ai"
    search_api_key: str = ""
    search_timeout_seconds: float = 60.0
    search_inline_content_max_characters: int = 50000
    fetch_provider: str = "crawl4ai"
    crawl4ai_setup_required: bool = True
    weather_enabled: bool = True
    weather_base_url: str = "https://api.neonhail.cloud/v0"
    weather_timeout_seconds: float = 30.0
    resolver_backend: str = "openai_compatible"
    resolver_base_url: str = "http://localhost:11434/v1"
    resolver_model: str = "qwen3:8b"
    resolver_api_key: str = "your-openai-compatible-key"
    resolver_include_reasoning: bool = False
    resolver_include_raw_output: bool = False
    resolver_debug_log_path: str = "/tmp/magpie-resolver.log"
    fetch_debug_log_path: str = "/tmp/magpie-fetch.log"
    include_timing_debug: bool = False
    response_detail: ResponseDetail = ResponseDetail.COMPACT
    max_search_queries_per_run: int = 8
    max_search_results_per_query: int = 10
    max_sources_per_query: int = 5
    max_sources_per_run: int = 12
    max_evidence_items_per_run: int = 24
    max_evidence_characters_per_item: int = 4000
    max_synthesis_input_characters: int = 32000
    max_incremental_answer_characters: int = 6000
    request_timeout_seconds: float = 60.0
    verify_tls: bool = True
    log_level: str = "INFO"
    database_path: str = "~/.local/share/magpie/magpie.db"
    cache_recent_ttl_seconds: int = 86400
    cache_evergreen_ttl_seconds: int = 2592000
    a2a_base_url: str = "http://127.0.0.1:8766"

    @classmethod
    def load(cls, path: str | None = None) -> "Settings":
        data: dict[str, Any] = {}
        config_path = cls.resolve_config_path(path)
        if config_path is not None:
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise ConfigError(f"Config file not found: {config_path}") from exc
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Config file is not valid JSON: {config_path}") from exc
        configurable_fields = {item.name for item in fields(cls) if item.init}
        unknown = sorted(set(data) - configurable_fields)
        if unknown:
            raise ConfigError(f"Unknown config fields: {', '.join(unknown)}")

        values: dict[str, Any] = {}
        for field_info in fields(cls):
            if not field_info.init:
                continue
            field_name = field_info.name
            default = field_info.default
            value = data.get(field_name, default)
            env_name = _env_name(field_name)
            if env_name in os.environ:
                value = _coerce_value(os.environ[env_name], default)
            if field_name == "response_detail" and not isinstance(value, ResponseDetail):
                value = ResponseDetail(str(value))
            values[field_name] = value

        settings = cls(**values)
        settings.loaded_config_path = str(config_path.resolve()) if config_path is not None else None
        settings.validate()
        return settings

    @classmethod
    def resolve_config_path(cls, path: str | None = None) -> Path | None:
        if path:
            return Path(path).expanduser()
        for candidate in DEFAULT_CONFIG_CANDIDATES:
            expanded = candidate.expanduser()
            if expanded.is_file():
                return expanded
        return None

    def validate(self) -> None:
        if self.max_search_queries_per_run < 1:
            raise ConfigError("max_search_queries_per_run must be positive.")
        if self.max_sources_per_run < 1:
            raise ConfigError("max_sources_per_run must be positive.")
        if self.max_sources_per_query < 1:
            raise ConfigError("max_sources_per_query must be positive.")
        if self.max_evidence_items_per_run < 1:
            raise ConfigError("max_evidence_items_per_run must be positive.")
        if self.max_evidence_characters_per_item < 100:
            raise ConfigError("max_evidence_characters_per_item must be at least 100.")
        if self.max_synthesis_input_characters < self.max_evidence_characters_per_item:
            raise ConfigError("max_synthesis_input_characters must fit at least one evidence item.")
        if self.max_incremental_answer_characters < 500:
            raise ConfigError("max_incremental_answer_characters must be at least 500.")
        if self.search_provider not in {"exa", "fake"}:
            raise ConfigError("search_provider must be 'exa' or 'fake'.")
        if self.fetch_provider not in {"crawl4ai", "fake"}:
            raise ConfigError("fetch_provider must be 'crawl4ai' or 'fake'.")
        if self.resolver_backend not in {"openai_compatible", "fake"}:
            raise ConfigError("resolver_backend must be 'openai_compatible' or 'fake'.")
        if self.search_transport not in {"mcp_first", "mcp_only", "api_only"}:
            raise ConfigError("search_transport must be mcp_first, mcp_only, or api_only.")
        if self.search_inline_content_max_characters < 1000:
            raise ConfigError("search_inline_content_max_characters must be at least 1000.")
        if self.weather_timeout_seconds <= 0:
            raise ConfigError("weather_timeout_seconds must be positive.")

    @property
    def expanded_database_path(self) -> Path:
        return Path(self.database_path).expanduser()

    def sanitized_diagnostics(self) -> dict[str, Any]:
        return {
            "loaded_config_path": self.loaded_config_path,
            "http_host": self.http_host,
            "http_port": self.http_port,
            "search_provider": self.search_provider,
            "search_transport": self.search_transport,
            "search_mcp_url": self.search_mcp_url,
            "search_mcp_tool_name": self.search_mcp_tool_name,
            "search_base_url": self.search_base_url,
            "search_timeout_seconds": self.search_timeout_seconds,
            "search_inline_content_max_characters": self.search_inline_content_max_characters,
            "fetch_provider": self.fetch_provider,
            "crawl4ai_setup_required": self.crawl4ai_setup_required,
            "weather_enabled": self.weather_enabled,
            "weather_base_url": self.weather_base_url,
            "weather_timeout_seconds": self.weather_timeout_seconds,
            "resolver_backend": self.resolver_backend,
            "resolver_base_url": self.resolver_base_url,
            "resolver_model": self.resolver_model,
            "resolver_include_reasoning": self.resolver_include_reasoning,
            "resolver_include_raw_output": self.resolver_include_raw_output,
            "resolver_debug_log_path": self.resolver_debug_log_path,
            "fetch_debug_log_path": self.fetch_debug_log_path,
            "include_timing_debug": self.include_timing_debug,
            "response_detail": self.response_detail.value,
            "max_search_queries_per_run": self.max_search_queries_per_run,
            "max_search_results_per_query": self.max_search_results_per_query,
            "max_sources_per_query": self.max_sources_per_query,
            "max_sources_per_run": self.max_sources_per_run,
            "max_evidence_items_per_run": self.max_evidence_items_per_run,
            "max_evidence_characters_per_item": self.max_evidence_characters_per_item,
            "max_synthesis_input_characters": self.max_synthesis_input_characters,
            "max_incremental_answer_characters": self.max_incremental_answer_characters,
            "database_path": str(self.expanded_database_path),
            "cache_recent_ttl_seconds": self.cache_recent_ttl_seconds,
            "cache_evergreen_ttl_seconds": self.cache_evergreen_ttl_seconds,
            "a2a_base_url": self.a2a_base_url,
        }
