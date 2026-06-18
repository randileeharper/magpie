"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .a2a import LocalA2AClient, build_fastapi_app
from .app import build_app
from .config import Settings
from .doctor import run_doctor
from .errors import A2ARequestError, A2AUnavailableError, ConfigError, StorageError
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

    serve = subparsers.add_parser("serve", help="Run the local A2A server.")
    serve.add_argument("--host", default=None, help="Override the configured bind host.")
    serve.add_argument("--port", type=int, default=None, help="Override the configured bind port.")

    doctor = subparsers.add_parser("doctor", help="Check provider and environment readiness.")
    doctor.add_argument("--live", action="store_true", help="Run live network checks where supported.")
    doctor.add_argument("--json", action="store_true", dest="as_json")

    clear_cache = subparsers.add_parser("clear-cache", help="Delete the configured cache database.")
    clear_cache.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _human_output(payload: dict[str, object]) -> str:
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


def _clear_cache_payload(settings: Settings) -> tuple[dict[str, object], int]:
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
        if args.as_json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
        return exit_code

    app = None
    try:
        if args.command in {"serve", "doctor"}:
            app = build_app(args.config_path)
        if args.command == "serve":
            import uvicorn

            host = args.host or app.settings.http_host
            port = args.port or app.settings.http_port
            server_app = build_fastapi_app(app.service, app.settings.a2a_base_url)
            uvicorn.run(server_app, host=host, port=port)
            return 0

        if args.command == "doctor":
            payload = run_doctor(app.settings, app.search_client, app.fetcher, app.news_client, live=args.live)
            if args.as_json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
            return 0 if payload.get("status") == "ok" else 2

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
