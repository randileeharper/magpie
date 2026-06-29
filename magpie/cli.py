"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .a2a import LocalA2AClient, build_fastapi_app
from .app import AppContext, build_app
from .config import Settings
from .doctor import run_doctor
from .errors import (
    A2ARequestError,
    A2AUnavailableError,
    ConfigError,
    FetchError,
    ResolverError,
    SearchError,
    StorageError,
)
from .models import ResearchRequest, ResponseDetail, to_jsonable
from .text import valid_unicode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="magpie", description="Magpie information lookup CLI.")
    parser.add_argument("--config", dest="config_path", default=None, help="Path to a JSON config file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Ask a natural-language question.")
    ask.add_argument("question", help="Question to answer.")
    ask.add_argument("--max-references", type=int, default=5)
    ask.add_argument("--json", action="store_true", dest="as_json")
    ask.add_argument("--debug", action="store_true")

    search = subparsers.add_parser("search", help="Search the web and return indexed results.")
    search.add_argument("query", help="Search query.")
    search.add_argument("--max-results", type=int, default=5)
    search.add_argument("--json", action="store_true", dest="as_json")

    fetch = subparsers.add_parser("fetch", help="Fetch web page content by index or URL.")
    fetch.add_argument("target", help="Index number (from a prior search) or URL.")
    fetch.add_argument("--run-id", default=None, help="Run ID from a prior magpie search.")
    fetch.add_argument("--full", action="store_true", help="Fetch the full page via crawl4ai instead of stored content.")
    fetch.add_argument("--json", action="store_true", dest="as_json")

    serve = subparsers.add_parser("serve", help="Run the local A2A server.")
    serve.add_argument("--host", default=None, help="Override the configured bind host.")
    serve.add_argument("--port", type=int, default=None, help="Override the configured bind port.")

    doctor = subparsers.add_parser("doctor", help="Check provider and environment readiness.")
    doctor.add_argument("--live", action="store_true", help="Run live network checks where supported.")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    clear_cache = subparsers.add_parser("clear-cache", help="Delete the configured cache database.")
    clear_cache.add_argument("--json", action="store_true", dest="as_json")

    config = subparsers.add_parser("config", help="Manage Magpie configuration.")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)

    config_init = config_subparsers.add_parser("init", help="Write the default config file.")
    config_init.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Config file path (default: ~/.config/magpie/config.json)",
    )
    config_init.add_argument("--force", action="store_true", help="Overwrite an existing config file.")
    config_init.add_argument(
        "--print",
        action="store_true",
        help="Print the template to stdout instead of writing a file.",
    )

    config_subparsers.add_parser("path", help="Print the path Magpie loads config from.")
    return parser


def _human_output(payload: dict[str, Any]) -> str:
    if payload.get("status") == "error":
        return f"error: {payload.get('message')}\nrun_id: {payload.get('run_id')}"
    summary = str(payload.get("summary", ""))
    answer = str(payload.get("answer", ""))
    lines = [
        f"run_id: {payload.get('run_id')}",
        f"summary: {summary}",
        "answer:",
        answer,
    ]
    references = payload.get("references", [])
    if references:
        lines.append("references:")
        for reference in references:
            lines.append(f"- {reference['title']} ({reference['url']})")
    return "\n".join(lines)


def _search_output(payload: dict[str, Any]) -> str:
    lines = [
        f"run_id: {payload.get('run_id')}",
        f"query: {payload.get('query')}",
        f"results: {len(payload.get('results', []))}",
        "",
    ]
    for item in payload.get("results", []):
        lines.append(f"[{item['index']}] {item['title']}")
        lines.append(f"    url: {item['url']}")
        if item.get("site_name"):
            lines.append(f"    site: {item['site_name']}")
        if item.get("published_at"):
            lines.append(f"    published: {item['published_at']}")
        summary = str(item.get("summary", ""))
        if summary:
            lines.append(f"    summary: {summary}")
        lines.append("")
    warnings = payload.get("warnings", [])
    if warnings:
        lines.append("warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)


def _fetch_output(payload: dict[str, Any]) -> str:
    lines = [
        f"run_id: {payload.get('run_id')}",
        f"url: {payload.get('url')}",
        f"title: {payload.get('title')}",
        f"fetched_via: {payload.get('fetched_via')}",
        f"content_length: {len(str(payload.get('content', '')))}",
        "",
        "content:",
        str(payload.get("content", "")),
    ]
    return "\n".join(lines)


def _clear_cache_payload(settings: Settings) -> tuple[dict[str, Any], int]:
    database_path = settings.expanded_database_path
    existing = [path for suffix in ("", "-wal", "-shm") if (path := Path(str(database_path) + suffix)).exists()]
    if existing:
        for path in existing:
            path.unlink()
        return (
            {
                "status": "ok",
                "database_path": str(database_path),
                "deleted": True,
            },
            0,
        )
    return (
        {
            "status": "ok",
            "database_path": str(database_path),
            "deleted": False,
            "message": "Cache database did not exist.",
        },
        0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "clear-cache":
        try:
            settings = Settings.load(args.config_path)
            payload, exit_code = _clear_cache_payload(settings)
        except (ConfigError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2, sort_keys=True))
        return exit_code

    if args.command == "config" and args.config_command == "init":
        from .config import read_config_template, write_default_config

        if args.print:
            try:
                template = read_config_template()
            except ConfigError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print(template, end="")
            return 0

        try:
            path = write_default_config(args.path, force=args.force)
        except (ConfigError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"Wrote config to {path}")
        return 0

    if args.command == "config" and args.config_command == "path":
        try:
            settings = Settings.load(args.config_path)
        except (ConfigError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(settings.loaded_config_path or "(none — using built-in defaults)")
        return 0

    app: AppContext | None = None
    try:
        if args.command in {"serve", "doctor"}:
            app = build_app(args.config_path, truncate_debug_logs=(args.command == "serve"))
        if args.command == "serve":
            import uvicorn

            assert app is not None
            host = args.host or app.settings.http_host
            port = args.port or app.settings.http_port
            server_app = build_fastapi_app(app.service, app.settings.a2a_base_url)
            uvicorn.run(server_app, host=host, port=port)
            return 0

        if args.command == "doctor":
            assert app is not None
            payload = run_doctor(app.settings, app.search_client, app.fetcher, app.news_client, live=args.live)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get("status") == "ok" else 2

       if args.command == "search":
            settings = Settings.load(args.config_path)
            try:
                from .app import build_app as _build_app
                _app = _build_app(args.config_path)
                result = _app.service.search(args.query, max_results=args.max_results)
                payload = to_jsonable(result)
                _app.service.close()
                _app.storage.close()
            except (SearchError, ResolverError, ConfigError, StorageError, OSError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
            except Exception as exc:
                import traceback
                traceback.print_exc(file=sys.stderr)
                print(f"Unexpected error: {exc}", file=sys.stderr)
                return 3
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(valid_unicode(_search_output(payload)))
            return 0

       if args.command == "fetch":
            try:
                from .app import build_app as _build_app
                _app = _build_app(args.config_path)
                target = args.target
                is_url = target.startswith("http://") or target.startswith("https://")
                fetch_kwargs: dict[str, Any] = {}
                if is_url:
                    fetch_kwargs["url"] = target
                else:
                    fetch_kwargs["index"] = int(target)
                    if args.run_id:
                        fetch_kwargs["run_id"] = args.run_id
                if args.full:
                    fetch_kwargs["full"] = True
                payload = to_jsonable(_app.service.fetch(**fetch_kwargs))
                _app.service.close()
                _app.storage.close()
            except (FetchError, ResolverError, ConfigError, StorageError, OSError) as exc:
                print(str(exc), file=sys.stderr)
                return 2
            except Exception as exc:
                import traceback
                traceback.print_exc(file=sys.stderr)
                print(f"Unexpected error: {exc}", file=sys.stderr)
                return 3
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(valid_unicode(_fetch_output(payload)))
            return 0

        settings = Settings.load(args.config_path)
        request = ResearchRequest(
            question=args.question,
            max_references=args.max_references,
            response_detail=ResponseDetail.DEBUG if args.debug else ResponseDetail.COMPACT,
        )
        try:
            payload = LocalA2AClient(
                settings.a2a_base_url,
                timeout_seconds=settings.request_timeout_seconds,
                verify_tls=settings.verify_tls,
            ).send(request)
        except A2AUnavailableError:
            app = build_app(args.config_path)
            payload = to_jsonable(app.service.research(request))
    except A2ARequestError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (A2AUnavailableError, ConfigError, StorageError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        if app is not None:
            app.service.close()
            app.storage.close()

    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(valid_unicode(_human_output(payload)))
    return 1 if payload.get("status") == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
