# Development Notes

Guidance for anyone working on the Magpie codebase. For architecture and data
flow, see [architecture.md](architecture.md); for config, see
[configuration.md](configuration.md).

## Design Principles

- **Keep resolver prompts and decision surfaces small.** More choices can make
  a local model slower and less reliable even when prompt ingestion is fast.
- **Prefer structured model decisions and deterministic validation** over prose
  interpretation. The resolver returns structured fields (route, accept/reject,
  query proposals); code validates and acts on them.
- **Specialized API routes should bypass general research** when they can
  produce a better grounded result.
- **API clients should request and retain only fields needed for the final
  answer.** Provider metadata must not leak into resolver prompts or
  user-facing output.
- **Do not cache rejected sources** as answer candidates. Rejected sources are
  recorded per-query and excluded from future cache reuse.
- **Do not silently retry requests that may already have been accepted.**
- **The SQLite schema is versioned.** Incompatible pre-release databases may be
  replaced during initialization.

## RSS Feed Registry

The built-in RSS registry (`magpie/news_feeds.json`) is intended for local or
personal aggregation. Check publisher terms before redistributing feed content.
A custom registry can be supplied via `news_feed_registry_path`.

## Testing

The test suite uses Python's `unittest` and is also runnable under `pytest`.
Tests use the `fake` provider backends to avoid network calls.

```bash
uv run python -m unittest discover -s tests -v
# or
uv run pytest
```

Test modules:

| Module                | Covers                                              |
| --------------------- | --------------------------------------------------- |
| `test_config.py`      | Config resolution, env overrides, validation        |
| `test_storage.py`     | SQLite storage, schema, cache reuse, rejections    |
| `test_service.py`     | Research loop, routing, synthesis, cancellation   |
| `test_resolver.py`    | OpenAI-compatible resolver client                  |
| `test_providers.py`   | Exa search, Crawl4AI fetch, Neon Hail, AniList, RSS |
| `test_cli.py`         | CLI commands and output formatting                 |
| `test_historian.py`   | Historian sink emission and sanitization          |
| `test_invariants.py`  | Cross-cutting invariants (A2A app, edge cases)     |

## Provider Protocol Interfaces

All external integrations are defined as `Protocol` classes in
`providers/base.py`:

- `ResolverClient` — routing, classification, query proposal, synthesis
- `SearchClient` — web search
- `Fetcher` — page content fetching
- `WeatherClient` — Neon Hail weather
- `AnimeClient` — AniList anime
- `NewsClient` — RSS/Atom news

Any object satisfying a protocol can be substituted via the matching `*_backend`
or `*_provider` setting. The `fake` backends (`providers/fake.py`) are used by
tests and provide in-memory implementations of all protocols.

## Resolver Prompts

Resolver prompt text lives in `magpie/prompts/*.txt` and is loaded by
`prompts/loader.py` via `importlib.resources`, with an `lru_cache` wrapper.
Prompts are kept deliberately small to suit smaller local models.
