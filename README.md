# Magpie

Magpie is a natural-language information retrieval agent built for delegation
from another agent. It synthesizes short grounded answers from web search and
routes requests such as weather lookups to more appropriate APIs when possible.

It exposes an A2A interface for agent-to-agent use and a local CLI for direct
queries. Results are compact, structured, grounded in references, and suitable
for a conversational agent to present in its own voice.

## What It Does

General Q&A and the specialized routes below are all invoked through
`magpie ask`; `search` and `fetch` are separate commands (and A2A skills).

- **ask** — synthesizes search results into a short grounded answer (a few
  paragraphs) with references. A bounded lookup, not open-ended research.
- **search** — returns indexed search results with summaries and source URLs.
  No synthesis.
- **fetch** — retrieves full web page content by index or by URL.
- **weather** — routes current-condition and forecast requests to the Neon Hail
  API.
- **anime** — pulls anime information from AniList.
- **news** — returns compact category digests from publisher RSS and Atom feeds.

Across all query types, Magpie caches source snapshots and completed answers in
SQLite, exposes durable cancellable runs through A2A, and records resolver,
fetch, timing, and run diagnostics for debugging.

## Requirements

- Python 3.12+
- an OpenAI-compatible local or remote model endpoint
- Crawl4AI and its browser assets for page fetching
- network access to the configured search provider

The default configuration uses Exa MCP for search, Crawl4AI for page fetching,
and an OpenAI-compatible resolver at `http://localhost:11434/v1`.

## Install

```bash
uv sync
uv run crawl4ai-setup
cp config.example.json config.json
```

Edit `config.json` to select the resolver model and any provider credentials,
then check the environment:

```bash
uv run magpie doctor --live
```

## Usage

Start the A2A server:

```bash
uv run magpie serve
```

### Ask

Synthesizes a short grounded answer from search. Routes to specialized APIs
when the question matches weather, anime, or news.

```bash
uv run magpie ask "Who is the mayor of New York?"
uv run magpie ask "How do I make homemade sourdough bread?"
uv run magpie ask "What's the weather in 98230?"
uv run magpie ask "Give me the forecast for 98230" --json
uv run magpie ask "Who voices Kirishima in Yakuza Fiancé?"
uv run magpie ask "anime schedule for today"
uv run magpie ask "What's the latest AI news?"
uv run magpie ask "world news from yesterday" --json
uv run magpie ask "Compare the latest policies" --json --debug
```

### Search

Indexed search results with summaries and source URLs. No synthesis.

```bash
uv run magpie search "a2a protocol"
uv run magpie search "rust borrow checker" --max-results 3 --json
```

### Fetch

Full page content by index (from a prior search) or by URL.

```bash
uv run magpie fetch 0 --run-id <run_id_from_search>
uv run magpie fetch "https://example.com/article"
uv run magpie fetch 2 --run-id <run_id> --full
```

### Other commands

```bash
uv run magpie doctor --live
uv run magpie clear-cache
```

`magpie ask` first tries the configured local A2A server. If initial A2A
discovery or connection fails, it runs the same service directly. It does not
silently retry a request after the A2A server has accepted it.

## A2A Usage

Magpie exposes three skills on its agent card:

### magpie_ask

Synthesizes a short grounded answer from search results. Sends
`skill: "magpie_ask"` (or omit the skill — it is the default). Returns:

- `summary`: a compact tool-friendly description
- `answer`: the grounded answer (a few paragraphs, with references)
- `references`: sources used by the answer
- `warnings` and `limitations`: relevant caveats
- `status`, `stop_reason`, and `run_id`: execution state

### magpie_search

Returns indexed search results with short summaries and source URLs. No
model synthesis. Sends `skill: "magpie_search"`. Returns:

- `run_id`: used for follow-up fetch calls
- `query`: the refined search query
- `results`: array of `{ index, title, url, site_name, published_at, summary }`

### magpie_fetch

Retrieves full web page content by index (from a prior search) or by URL.
Sends `skill: "magpie_fetch"` with either an index + `run_id` in metadata,
or a URL. By default returns stored Exa content (instant); set `full: true`
in metadata to force a fresh crawl4ai fetch. Returns:

- `run_id`, `index`, `url`, `title`, `content`, `fetched_via`, `warnings`

Published endpoints include:

- `POST /a2a`
- `GET /.well-known/agent-card.json`
- standard A2A REST task routes
- `GET /healthz`

A2A task IDs are also durable Magpie run IDs, so task cancellation targets the
same run recorded in SQLite.

## Diagnostics

Use `magpie ask ... --debug` or enable `include_timing_debug` to include
timings and run events in results. Shared resolver and fetch logs are tagged
with run IDs:

- `~/.local/share/magpie/magpie-resolver.log`
- `~/.local/share/magpie/magpie-fetch.log`

Raw model output is written to the resolver log only when
`resolver_include_raw_output` is enabled. Logs may contain full prompts, source
content, and model output; do not publish them without reviewing their contents.

## Further Documentation

For contributors and operators working with the code:

- [docs/architecture.md](docs/architecture.md) — design philosophy, routing,
  the research loop, synthesis, stop reasons, concurrency
- [docs/configuration.md](docs/configuration.md) — full settings reference,
  config resolution, environment-variable overrides
- [docs/storage.md](docs/storage.md) — SQLite schema, cache reuse, freshness
  detection, source rejection
- [docs/historian.md](docs/historian.md) — Historian event integration setup
  and emitted event types
- [docs/development.md](docs/development.md) — design principles, testing,
  provider protocols
