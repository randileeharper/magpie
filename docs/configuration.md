# Configuration

Magpie is configured through a JSON config file, with every setting also
supporting an environment-variable override using the `MAGPIE_` prefix. The
example schema lives in `magpie/config.example.json` in the repository, and
can be written to the standard config location with `magpie config init`.

## Config Resolution Order

1. The path passed with `--config`
2. `./config.json`
3. `~/.config/magpie/config.json`
4. Built-in defaults

If no file is found, the built-in defaults in `Settings` are used. Unknown
fields in a config file raise a `ConfigError` at load time.

## Environment-Variable Overrides

Every setting supports a `MAGPIE_`-prefixed override, uppercased. Values are
coerced to match the default's type:

- Booleans accept `1`, `true`, `yes`, `on` (case-insensitive) as true.
- Integers and floats are parsed numerically.
- Everything else is treated as a string.

Common settings:

- `MAGPIE_RESOLVER_BASE_URL`
- `MAGPIE_RESOLVER_MODEL`
- `MAGPIE_RESOLVER_API_KEY`
- `MAGPIE_SEARCH_API_KEY`
- `MAGPIE_DATABASE_PATH`
- `MAGPIE_A2A_BASE_URL`
- `MAGPIE_WEATHER_ENABLED`
- `MAGPIE_NEWS_ENABLED`
- `MAGPIE_HISTORIAN_ENABLED`
- `MAGPIE_HISTORIAN_BASE_URL`
- `MAGPIE_HISTORIAN_TOKEN`

## Setting Reference

The groups below mirror `magpie/config.example.json`. Validation constraints noted
where relevant.

### HTTP Server

| Setting       | Default       | Notes                                  |
| ------------- | ------------- | -------------------------------------- |
| `http_host`   | `127.0.0.1`   | Bind address for the A2A server        |
| `http_port`   | `8766`        | Port for the A2A server                |
| `a2a_base_url`| `http://127.0.0.1:8766` | Base URL published on the agent card |

### Search

| Setting                              | Default                  | Notes                                       |
| ------------------------------------ | ------------------------ | ------------------------------------------- |
| `search_provider`                    | `exa`                    | `exa` or `fake`                             |
| `search_transport`                   | `mcp_first`              | `mcp_first`, `mcp_only`, or `api_only`      |
| `search_mcp_url`                      | `https://mcp.exa.ai/mcp` | Exa MCP endpoint                            |
| `search_mcp_tool_name`                | `web_search_exa`         | MCP tool name to call                       |
| `search_base_url`                     | `https://api.exa.ai`     | Exa REST API base URL                       |
| `search_api_key`                      | _(empty)_                | Exa API key                                 |
| `search_timeout_seconds`              | `60.0`                   | Must be positive                            |
| `search_inline_content_max_characters`| `50000`                  | Min 1000                                    |

`search_transport` controls how the Exa client reaches the search provider:
MCP first with API fallback (`mcp_first`), MCP only (`mcp_only`), or REST API
only (`api_only`).

### Fetch

| Setting                  | Default     | Notes                       |
| ------------------------ | ----------- | --------------------------- |
| `fetch_provider`         | `crawl4ai`  | `crawl4ai` or `fake`        |
| `crawl4ai_setup_required`| `true`      | Whether `doctor` enforces browser asset setup |

### Resolver

| Setting                      | Default                                   | Notes                                              |
| ---------------------------- | ----------------------------------------- | ------------------------------------------------- |
| `resolver_backend`           | `openai_compatible`                       | `openai_compatible` or `fake`                     |
| `resolver_base_url`          | `http://localhost:11434/v1`               | OpenAI-compatible endpoint                        |
| `resolver_model`             | `qwen3:8b`                                | Model name                                        |
| `resolver_api_key`           | `your-openai-compatible-key`             | API key                                           |
| `resolver_include_reasoning` | `false`                                   | Send reasoning-effort hints to the resolver      |
| `resolver_include_raw_output`| `false`                                   | Write raw model output to the resolver log        |
| `resolver_max_tokens`        | `8192`                                    | Max generation tokens                             |
| `resolver_debug_log_path`    | `~/.local/share/magpie/magpie-resolver.log`| Resolver log file                                |
| `fetch_debug_log_path`       | `~/.local/share/magpie/magpie-fetch.log`  | Fetch log file                                    |

### Weather

| Setting                | Default                         | Notes                       |
| ---------------------- | ------------------------------- | --------------------------- |
| `weather_enabled`      | `true`                          | Enables the weather route   |
| `weather_base_url`     | `https://api.neonhail.cloud/v0` | Neon Hail API base URL      |
| `weather_timeout_seconds`| `30.0`                        | Must be positive             |

### Anime

| Setting                    | Default                          | Notes                                |
| -------------------------- | -------------------------------- | ------------------------------------ |
| `anime_enabled`            | `true`                           | Enables the anime route              |
| `anime_base_url`           | `https://graphql.anilist.co`     | AniList GraphQL endpoint             |
| `anime_title_search_fallback_url` | `https://api.jikan.moe/v4/anime`| Jikan fallback for title discovery   |
| `anime_timeout_seconds`    | `30.0`                           | Must be positive                     |
| `anime_candidate_limit`    | `5`                              | 1–10                                 |
| `anime_character_limit`    | `50`                             | 1–50                                 |
| `anime_schedule_limit`     | `50`                             | 1–50                                 |

### News

| Setting                    | Default  | Notes                                              |
| -------------------------- | -------- | -------------------------------------------------- |
| `news_enabled`             | `true`   | Enables the news route                             |
| `news_feed_registry_path` | _(empty)_| Path to a custom feed registry JSON; bundled `magpie/news_feeds.json` used when empty |
| `news_timeout_seconds`     | `15.0`   | Must be positive                                    |
| `news_cache_ttl_seconds`   | `300`    | Non-negative                                        |
| `news_max_feed_bytes`      | `1048576`| Min 1024                                           |
| `news_fetch_concurrency`   | `4`      | 1–16                                               |
| `news_digest_size`         | `5`      | 1–10                                               |
| `news_per_source_limit`    | `2`      | 1–5                                                |
| `news_summary_max_characters`| `280`  | 40–1000                                            |

### Database and Cache

| Setting                    | Default                                   | Notes                              |
| -------------------------- | ----------------------------------------- | ---------------------------------- |
| `database_path`            | `~/.local/share/magpie/magpie.db`         | SQLite database path               |
| `cache_recent_ttl_seconds` | `86400`                                   | Cache lifetime for recent topics   |
| `cache_evergreen_ttl_seconds`| `2592000`                               | Cache lifetime for evergreen topics|

See [storage.md](storage.md) for how the cache is used and freshness is
detected.

### Budgets and Limits

These bound a single research run. See [architecture.md](architecture.md) for
how they interact.

| Setting                              | Default  | Notes                                |
| ------------------------------------ | -------- | ------------------------------------ |
| `max_search_queries_per_run`         | `8`      | Min 1                                |
| `max_search_results_per_query`       | `10`     |                                      |
| `max_sources_per_query`              | `5`      | Min 1                                |
| `max_sources_per_run`                | `12`     | Min 1                                |
| `max_evidence_items_per_run`         | `24`     | Min 1                                |
| `max_evidence_characters_per_item`   | `4000`   | Min 100                              |
| `max_synthesis_input_characters`     | `32000`  | Must fit at least one evidence item  |
| `max_incremental_answer_characters` | `6000`   | Min 500                              |
| `request_timeout_seconds`            | `60.0`   |                                      |

### Diagnostics

| Setting                | Default  | Notes                                                       |
| ---------------------- | -------- | ----------------------------------------------------------- |
| `include_timing_debug` | `false`  | Include timings and run events in results                   |
| `response_detail`      | `compact`| `compact` or `debug`                                        |
| `verify_tls`           | `true`   | Verify TLS for outbound HTTP                                |
| `log_level`            | `INFO`   | Python logging level                                        |

### Historian

Historian event production is optional and disabled by default. See
[historian.md](historian.md) for setup.

| Setting                    | Default                   | Notes                              |
| -------------------------- | ------------------------- | ---------------------------------- |
| `historian_enabled`        | `false`                   | Requires `historian_token` when true|
| `historian_base_url`       | `http://127.0.0.1:8768`   | Non-empty                          |
| `historian_token`          | _(null)_                  | May also be set via `historian_token` config key |
| `historian_timeout_seconds`| `5.0`                     | Must be positive                   |
| `historian_verify_tls`     | `true`                    |                                    |
| `historian_retry_count`    | `2`                       | Non-negative                       |

## Verifying Your Setup

```bash
magpie doctor --live
```

`doctor --live` runs live connectivity checks against configured providers and
the database. Without `--live` it checks configuration and the local database
only.
