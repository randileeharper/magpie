# Architecture

Magpie is a natural-language information retrieval agent built for delegation from
another agent. It answers general questions with bounded web research and routes
specific request types (weather, anime, news) to specialized APIs when possible.

This document covers the design philosophy, request routing, the research loop,
synthesis, and the invariants the code is built around. Operators configuring the
system should read [configuration.md](configuration.md); the storage layer is
documented in [storage.md](storage.md).

## Design Philosophy

General-purpose conversational agents should not need to carry every tool,
provider, and lookup workflow in their prompt. That becomes especially slow
and unreliable when the system is backed by a smaller local model.

Magpie keeps that work inside a dedicated information-retrieval agent:

- The upstream agent delegates a plain-language question.
- A small routing decision chooses general web research or a specialized path.
- Deterministic code handles provider calls, budgets, caching, and validation.
- The resolver receives all gathered evidence at once per round, not one source
  at a time.
- The upstream agent receives a grounded answer and references, then applies
  personality or continues the conversation.

The goal is not to make an LLM imitate a search engine. The goal is to give a
smaller model a constrained workflow in which it can make useful decisions
without drowning in context.

## Module Layout

```
magpie/
  app.py            Wires services together for CLI and A2A entrypoints
  cli.py            `magpie` command-line interface
  a2a.py            A2A server, agent card, executor, local A2A client
  service.py        ResearchService: run orchestration (research loop, search,
                    fetch) coordinating evidence selection, telemetry, and
                    run state. Thin shims delegate to the collaborators below.
  evidence.py       EvidenceSelector: regex scoring and chunking that turns
                    fetched text into bounded evidence extracts
  telemetry.py      TelemetryEmitter + RunTelemetry: historian event emission,
                    per-run counters, terminal-event guard, secret sanitization
  runcontext.py     RunContext: mutable loop state (budget, warnings,
                    limitations, seen URLs, remaining questions, draft)
  routes.py         Specialized route handlers (weather, anime, news)
  storage.py        SQLiteStorage: schema, runs, sources, evidence, events
  models.py         Dataclasses and enums shared across modules
  config.py         Settings dataclass, config resolution, validation
  doctor.py         `magpie doctor` environment checks
  historian.py      Historian event sink (HTTP, null, fake)
  text.py           Unicode validation helpers
  errors.py          Exception hierarchy
  prompts/          Resolver prompt text files, loaded by prompts/loader.py
  providers/        Provider implementations and Protocol interfaces
    base.py         ResolverClient, SearchClient, Fetcher, WeatherClient,
                    AnimeClient, NewsClient protocols
    exa.py          Exa search (MCP-first, API fallback)
    crawl4ai_fetcher.py  Crawl4AI page fetcher
    openai_compatible.py  OpenAI-compatible resolver client
    neonhail.py     Neon Hail weather API client
    anilist.py      AniList GraphQL anime client (Jikan title fallback)
    news_rss.py     RSS/Atom news aggregation client
    fake.py         In-memory fakes for tests
```

The provider interfaces in `providers/base.py` are `Protocol` classes. Any
implementation that satisfies the protocol can be substituted via configuration
(`resolver_backend`, `search_provider`, `fetch_provider`). The `fake` backends
are used by the test suite.

## Request Routing

Every request begins with a compact resolver routing call
(`route_request`). The resolver returns a `RouteDecision` naming one of four
routes:

| Route         | Handler                  | When                                                       |
| ------------- | ------------------------ | ---------------------------------------------------------- |
| `weather`     | inline in `routes.py`    | Current conditions or forecast for a US ZIP code           |
| `anime`       | `try_anime_route`        | Anime facts, voice cast, or airing schedule                |
| `news`        | `try_news_route`         | Broad category news within a strict local-time window     |
| `web_research`| `ResearchService.research` | Everything else, or fallback when a route fails           |

Specialized routes bypass web search and synthesis entirely and answer directly
from their provider. If a specialized route fails or cannot produce a result,
Magpie falls back to general web research rather than returning an error.

### Weather route

Weather requests with a confident five-digit US ZIP code go directly to Neon
Hail and bypass web search and synthesis. If no ZIP code can be determined,
Magpie falls back to web research.

### Anime route

Anime requests are classified a second time (`classify_anime_request`) into
factual lookup, credits, or schedule operations and then sent to AniList. For
factual lookups, the resolver selects from an allowlist of `AnimeField` values;
deterministic code builds the GraphQL query and returns only the requested
values. Daily anime schedules use Japanese broadcast times converted to the
system timezone.

Jikan is used only as a fallback title-discovery index when AniList cannot
resolve a spelling variant; final anime data and references still come from
AniList.

### News route

Broad category news requests are classified a second time
(`classify_news_request`) into a `NewsCategory` and strict local-time window
(`NewsTimeScope`), then answered directly from configured RSS or Atom feeds
without article fetching or synthesis. Arbitrary topics such as
company-specific news fall back to general web research.

## The Research Loop

General web lookup follows a bounded batch loop implemented in
`ResearchService.research`:

1. **Reuse fresh cache.** Reuse previously accepted sources for the exact
   question when they are still within their cache lifetime.
2. **Propose a query.** Ask the resolver for one focused search query
   (`propose_query`), given prior queries, seen URLs, and remaining questions.
3. **Search and deduplicate.** Search, canonicalize URLs (stripping tracking
   parameters), and gather a limited source set per query.
4. **Acquire sources.** Use Exa inline content directly when available; fall
   back to Crawl4AI only when inline content is absent or too short.
5. **Synthesize per round.** Pass all evidence from the round to the resolver
   in one synthesis call (`synthesize`), along with any prior draft.
6. **Continue or finish.** Continue with the next query only when questions
   remain; otherwise finalize the answer.

A run keeps a `RunBudget` tracking remaining queries, sources, and evidence
items, carried alongside the accumulated warnings, limitations, seen URLs,
remaining questions, and last synthesis draft in a `RunContext`. The loop
exits early when the budget is exhausted, the planner cannot produce a new
useful query, or the answer is complete. Per-run counters and historian events
are tracked by `TelemetryEmitter`; evidence scoring and chunking are owned by
`EvidenceSelector`. `ResearchService` coordinates the three.

### Source acceptance

Source acceptance is a structured resolver decision, not prose matching. Each
synthesis call returns a `SynthesisDraft` that includes a boolean
`source_answers_question`. When a round's evidence does not answer the
question, those sources are recorded as rejected for that query in
`source_rejections` and excluded from future cache reuse for that question.
Magpie does not attempt to infer whether an answer is a refusal by matching
generated prose with regex.

### Synthesis behavior

The synthesis prompt prefers several substantive paragraphs over a single terse
paragraph. When sources present competing complete options, the resolver commits
to the single best one rather than surveying alternatives. Specialized routes
(weather, anime, news) bypass synthesis entirely and answer directly.

Resolver calls are serialized across concurrent runs because the expected
deployment target is a smaller local model, not a high-throughput frontier API.

## Bounded Lookup

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
- `resolver_max_tokens`

Completed answers return `status: "ok"`. Grounded answers that stop before all
remaining questions are resolved return `status: "partial"`. Runs without a
usable grounded answer return `status: "error"`.

## Stop Reasons

A run ends with a `StopReason` describing why it stopped:

| Stop reason              | Meaning                                                        |
| ------------------------ | ------------------------------------------------------------- |
| `answered_from_cache`    | Answer finalized from reused exact-query cache                |
| `needed_new_search`      | Answer finalized after a fresh search round                    |
| `specialized_route`      | A specialized route produced the answer directly              |
| `budget_exhausted`       | Run ended because a budget limit was reached                  |
| `insufficient_evidence`  | Run ended with no usable grounded answer                      |
| `no_progress`           | Planner could not produce a new useful query                  |
| `cancelled`              | Run was cancelled via A2A task cancellation                   |
| `failed`                 | Run ended due to an exception                                 |

## Concurrency and Cancellation

Resolver calls are serialized across concurrent runs. A run can be cancelled
through A2A task cancellation; the service checks `is_cancel_requested` between
stages and raises `ResearchCancelled`, which marks the run cancelled and emits a
`research.run.canceled` event. A2A task IDs are also durable Magpie run IDs, so
task cancellation targets the same run recorded in SQLite.

## Run Ownership Across Entry Points

`research`, `search`, and `fetch` all create `research_runs` rows, but only one
entry point owns a given run's terminal transition:

- `research` — creates and finalizes the run (completed/partial/failed).
- `search` — creates and finalizes the run (completed/failed).
- `fetch` by URL — creates and finalizes the run, ending with
  `completed` + `needed_new_search` on success or `failed` on a fetch error,
  and emits `research.run.started` / `research.run.completed` /
  `research.run.failed`.
- `fetch` by index — reuses an existing search run (`run_id` + `index`) and
  does **not** transition that run's status; the owning `search`/`research`
  call is responsible for its terminal.
