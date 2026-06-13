# Magpie

Magpie is a natural-language information retrieval agent built for delegation
from another agent. It answers general questions with bounded web research and
routes requests such as weather lookups to more appropriate APIs when possible.

It exposes an A2A interface for agent-to-agent use and a local CLI for direct
queries. Results are compact, structured, grounded in references, and suitable
for a conversational agent to present in its own voice.

## Design Philosophy

General-purpose conversational agents should not need to carry every tool,
provider, and research workflow in their prompt. That becomes especially slow
and unreliable when the system is backed by a smaller local model.

Magpie keeps that work inside a dedicated information-retrieval agent:

- The upstream agent delegates a plain-language question.
- A small routing decision chooses general web research or a specialized path.
- Deterministic code handles provider calls, budgets, caching, and validation.
- The resolver receives narrow decisions and one source at a time rather than a
  pile of competing excerpts.
- The upstream agent receives a grounded answer and references, then applies
  personality or continues the conversation.

The goal is not to make an LLM imitate a search engine. The goal is to give a
smaller model a constrained workflow in which it can make useful decisions
without drowning in context.

## What It Does

- researches natural-language questions using web search and fetched pages
- returns compact answers with grounded references
- checks sources incrementally until the answer is usable or a run budget ends
- routes current-condition and forecast requests to the Neon Hail weather API
- caches useful source snapshots and completed answers in SQLite
- exposes durable, cancellable research runs through A2A
- records resolver, fetch, timing, and run diagnostics for debugging

## Requirements

- Python 3.12+
- an OpenAI-compatible local or remote model endpoint
- Crawl4AI and its browser assets for page fetching
- network access to the configured search provider

The default configuration uses Exa MCP for search, Crawl4AI for page fetching,
and an OpenAI-compatible resolver at `http://localhost:11434/v1`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
crawl4ai-setup
cp config.example.json config.json
```

Edit `config.json` to select the resolver model and any provider credentials,
then check the environment:

```bash
magpie doctor --live
```

## Configuration

Config resolution order:

1. the path passed with `--config`
2. `./config.json`
3. `~/.config/magpie/config.json`
4. built-in defaults

See `config.example.json` for the full schema. Every setting also supports an
environment-variable override using the `MAGPIE_` prefix. Common settings
include:

- `MAGPIE_RESOLVER_BASE_URL`
- `MAGPIE_RESOLVER_MODEL`
- `MAGPIE_RESOLVER_API_KEY`
- `MAGPIE_SEARCH_API_KEY`
- `MAGPIE_DATABASE_PATH`
- `MAGPIE_A2A_BASE_URL`
- `MAGPIE_WEATHER_ENABLED`

## Run

Start the A2A server:

```bash
magpie serve
```

Ask questions from the CLI:

```bash
magpie research "Who is the mayor of New York?"
magpie research "How do I make homemade sourdough bread?"
magpie research "What's the weather in 98230?"
magpie research "Give me the forecast for 98230" --json
magpie research "Compare the latest policies" --json --debug
```

Other useful commands:

```bash
magpie doctor --live
magpie clear-cache
```

`magpie research` first tries the configured local A2A server. If initial A2A
discovery or connection fails, it runs the same service directly. It does not
silently retry a request after the A2A server has accepted it.

## A2A Usage

The intended integration path is plain-language delegation. An upstream agent
only needs to know that Magpie accepts information requests and returns a
structured result containing:

- `summary`: a compact tool-friendly description
- `answer`: the grounded answer
- `references`: sources used by the answer
- `warnings` and `limitations`: relevant caveats
- `status`, `stop_reason`, and `run_id`: execution state

Published endpoints include:

- `POST /a2a`
- `GET /.well-known/agent-card.json`
- standard A2A REST task routes
- `GET /healthz`

A2A task IDs are also durable Magpie run IDs, so task cancellation targets the
same run recorded in SQLite.

## How It Works

Every request begins with a compact resolver routing call. Weather requests
with a confident five-digit US ZIP code go directly to Neon Hail and bypass web
search and synthesis. If routing, ZIP normalization, or the weather API fails,
Magpie falls back to general web research.

General research follows a bounded incremental loop:

1. Reuse fresh, previously accepted sources for the exact question when available.
2. Ask the resolver for one focused search query.
3. Search, deduplicate canonical URLs, and fetch a limited source set.
4. Present one bounded source extract to the resolver.
5. Keep the source only if the resolver says it contributes to the answer.
6. Continue with the next source or query only when questions remain.

The resolver never receives every result at once. Each synthesis call sees one
new source plus a bounded prior draft. Resolver calls are also serialized across
concurrent runs because the expected deployment target is a smaller local model,
not a high-throughput frontier API.

## Grounding And Cache Behavior

Search results, fetched snapshots, evidence, run events, and final answers are
stored in SQLite. URL-specific snapshots remain distinct even when their text is
identical.

Exact-question cache reuse is intentionally conservative:

- only sources cited by completed or partial answers are reusable candidates
- sources rejected for a question remain excluded from that question
- cached canonical URLs are not processed again when search returns duplicates
- recent and evergreen questions use separate configurable cache lifetimes

Source acceptance is a structured resolver decision. Magpie does not attempt to
infer whether an answer is a refusal by matching generated prose with regex.

## Bounded Research

Runs are limited across queries, sources, evidence items, source characters, and
incremental answer size. The principal settings are:

- `max_search_queries_per_run`
- `max_search_results_per_query`
- `max_sources_per_query`
- `max_sources_per_run`
- `max_evidence_items_per_run`
- `max_evidence_characters_per_item`
- `max_synthesis_input_characters`
- `max_incremental_answer_characters`

Completed answers return `status: "ok"`. Grounded answers that stop before all
remaining questions are resolved return `status: "partial"`. Runs without a
usable grounded answer return `status: "error"`.

## Diagnostics

Use `magpie research ... --debug` or enable `include_timing_debug` to include
timings and run events in results. Shared resolver and fetch logs are tagged
with run IDs:

- `/tmp/magpie-resolver.log`
- `/tmp/magpie-fetch.log`

Raw model output is written to the resolver log only when
`resolver_include_raw_output` is enabled. Logs may contain full prompts, source
content, and model output; do not publish them without reviewing their contents.

## Development Notes

- Keep resolver prompts and decision surfaces small. More choices can make a
  local model slower and less reliable even when prompt ingestion is fast.
- Prefer structured model decisions and deterministic validation over prose
  interpretation.
- Specialized API routes should bypass general research when they can produce a
  better grounded result.
- Do not cache rejected sources as answer candidates.
- Do not silently retry requests that may already have been accepted.
- The SQLite schema is versioned. Incompatible pre-release databases may be
  replaced during initialization.

Run the test suite with:

```bash
python -m unittest discover -s tests -v
# or, when pytest is installed
pytest
```
