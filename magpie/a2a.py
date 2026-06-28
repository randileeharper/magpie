"""A2A SDK 1.1 transport with durable Magpie ask run identity."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import httpx
from a2a.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.client.errors import AgentCardResolutionError
from a2a.helpers import new_data_part, new_task, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import (
    add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes, create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Message, Role, SendMessageRequest,
    Task, TaskState,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_1_0, TransportProtocol
from a2a.utils.errors import InternalError, InvalidParamsError
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.protobuf import json_format

from . import __version__
from .errors import A2ARequestError, A2AUnavailableError
from .models import (
    FetchResult, IndexedSearchResult, ResearchErrorResult, ResearchRequest, ResearchResult,
    ResponseDetail, StopReason, to_jsonable,
)
from .service import ResearchService


def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="Magpie",
        description="Natural-language information lookup with bounded web lookup and specialized API routes.",
        version=__version__,
        supported_interfaces=[
            AgentInterface(url=f"{base_url.rstrip('/')}/a2a", protocol_binding=TransportProtocol.JSONRPC.value,
                           protocol_version=PROTOCOL_VERSION_1_0),
            AgentInterface(url=base_url.rstrip("/"), protocol_binding=TransportProtocol.HTTP_JSON.value,
                           protocol_version=PROTOCOL_VERSION_1_0),
        ],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False, extended_agent_card=False),
        default_input_modes=["text/plain"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="magpie_ask", name="Natural-language question answering",
                description="Answer a question using bounded web lookup or a specialized information API.",
                tags=["ask", "search", "weather", "anime", "news"],
                examples=["Who is the mayor of New York?", "What anime airs today?", "What's the latest AI news?"],
                input_modes=["text/plain"], output_modes=["application/json"],
            ),
            AgentSkill(
                id="magpie_search", name="Indexed web search",
                description="Search the web and return indexed results with summaries and source URLs. No synthesis.",
                tags=["search", "index"],
                examples=["rust borrow checker", "a2a protocol", "latest AI news"],
                input_modes=["text/plain"], output_modes=["application/json"],
            ),
            AgentSkill(
                id="magpie_fetch", name="Fetch web page content",
                description="Fetch a web page by index from a previous search or by URL. Returns markdownified content.",
                tags=["fetch", "content", "url"],
                examples=["index 2", "https://example.com/article"],
                input_modes=["text/plain"], output_modes=["application/json"],
            ),
        ],
    )


def _request_metadata(context: RequestContext) -> dict[str, Any]:
    """Resolve request metadata from both request-level and message-level fields.

    The A2A SDK's ``RequestContext.metadata`` reads only the request-level
    ``SendMessageRequest.metadata`` field; clients set metadata on the message
    (``Message.metadata``), which is a separate protobuf field. Merge both so
    skill selection and other metadata-driven dispatch work over the transport.
    Message-level metadata takes precedence over request-level when both set
    the same key.
    """
    merged: dict[str, Any] = dict(context.metadata)
    if context.message is not None:
        merged.update(json_format.MessageToDict(context.message.metadata))
    return merged


class SDKResearchAgentExecutor(AgentExecutor):
    def __init__(self, service: ResearchService):
        self._service = service

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.message:
            raise InvalidParamsError("SendMessageRequest.message is required.")
        if not context.task_id or not context.context_id:
            raise InternalError("Request context did not include task identifiers.")
        metadata = _request_metadata(context)
        question = context.get_user_input()
        skill = metadata.get("skill", "magpie_ask")
        task = new_task(
            task_id=context.task_id, context_id=context.context_id,
            state=TaskState.TASK_STATE_SUBMITTED, history=[context.message],
        )
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=context.task_id, context_id=context.context_id)
        await updater.start_work()
        try:
            result: IndexedSearchResult | FetchResult | ResearchResult | ResearchErrorResult
            if skill == "magpie_search":
                result = await asyncio.to_thread(
                    self._service.search, question,
                    max_results=int(metadata.get("max_references", 5)),
                    run_id=context.task_id,
                )
            elif skill == "magpie_fetch":
                fetch_params = self._parse_fetch_request(question, metadata)
                result = await asyncio.to_thread(self._service.fetch, **fetch_params)
            else:
                request = ResearchRequest(
                    question=question,
                    max_references=int(metadata.get("max_references", 5)),
                    response_detail=ResponseDetail(metadata.get("response_detail", "compact")),
                    run_label=metadata.get("run_label"),
                )
                result = await asyncio.to_thread(self._service.research, request, run_id=context.task_id)
        except asyncio.CancelledError:
            self._service.cancel_run(context.task_id)
            raise
        except Exception as exc:  # noqa: BLE001
            await updater.failed(self._failure_message(context, exc))
            return
        try:
            payload = to_jsonable(result)
            artifact_name = (
                "magpie-search-result" if skill == "magpie_search"
                else "magpie-fetch-result" if skill == "magpie_fetch"
                else "magpie-ask-result"
            )
            await updater.add_artifact(
                parts=[new_data_part(payload, media_type="application/json")],
                name=artifact_name,
            )
            if skill == "magpie_search":
                search_result = cast(IndexedSearchResult, result)
                text_response = f"Found {len(search_result.results)} results for: {search_result.query}"
            elif skill == "magpie_fetch":
                fetch_result = cast(FetchResult, result)
                text_response = f"Fetched {fetch_result.url} ({len(fetch_result.content)} chars via {fetch_result.fetched_via})"
            else:
                text_response = (
                    getattr(result, "answer", "")
                    or getattr(result, "summary", "")
                    or getattr(result, "message", "")
                )
            message = Message(
                role=Role.ROLE_AGENT, message_id=str(uuid4()), task_id=context.task_id,
                context_id=context.context_id,
                parts=[new_text_part(text_response),
                       new_data_part(payload, media_type="application/json")],
            )
            if skill in {"magpie_search", "magpie_fetch"}:
                await updater.complete(message)
            else:
                research_result = cast(ResearchResult | ResearchErrorResult, result)
                if research_result.stop_reason == StopReason.CANCELLED:
                    await updater.update_status(TaskState.TASK_STATE_CANCELED, message)
                elif research_result.status in {"ok", "partial"}:
                    await updater.complete(message)
                else:
                    await updater.failed(message)
        except Exception as exc:  # noqa: BLE001
            await updater.failed(self._failure_message(context, exc))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            raise InternalError("Cancellation request did not include task identifiers.")
        self._service.cancel_run(context.task_id)
        updater = TaskUpdater(event_queue=event_queue, task_id=context.task_id, context_id=context.context_id)
        await updater.update_status(TaskState.TASK_STATE_CANCELED)

    @staticmethod
    def _parse_fetch_request(question: str, metadata: dict[str, Any]) -> dict[str, Any]:
        import re
        params: dict[str, Any] = {}
        index_match = re.search(r"(?:index\s*)?(\d+)", question, re.IGNORECASE)
        url_match = re.search(r"https?://\S+", question)
        if "index" in metadata:
            params["index"] = int(metadata["index"])
        elif index_match and not url_match:
            params["index"] = int(index_match.group(1))
        if "url" in metadata:
            params["url"] = metadata["url"]
        elif url_match:
            params["url"] = url_match.group(0).rstrip(".,;)")
        if "full" in metadata:
            params["full"] = bool(metadata["full"])
        if "run_id" in metadata:
            params["run_id"] = metadata["run_id"]
        elif params.get("index") is not None:
            raise InvalidParamsError("fetch by index requires run_id in metadata or prior search context.")
        if not params.get("index") and not params.get("url"):
            raise InvalidParamsError("fetch requires an index number or a URL.")
        return params

    @staticmethod
    def _failure_message(context: RequestContext, exc: BaseException) -> Message:
        return Message(
            role=Role.ROLE_AGENT, message_id=str(uuid4()), task_id=context.task_id,
            context_id=context.context_id,
            parts=[new_text_part(str(exc)), new_data_part({"status": "error", "message": str(exc)})],
        )


def build_sdk_server(service: ResearchService, base_url: str) -> dict[str, Any]:
    card = build_agent_card(base_url)
    handler = DefaultRequestHandlerV2(
        agent_executor=SDKResearchAgentExecutor(service), task_store=InMemoryTaskStore(), agent_card=card,
    )
    return {"agent_card": card, "request_handler": handler}


def build_fastapi_app(service: ResearchService, base_url: str) -> FastAPI:
    app = FastAPI(title="Magpie A2A Server", version=__version__)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    bits = build_sdk_server(service, base_url)
    card = bits["agent_card"]
    handler = bits["request_handler"]

    @app.get("/.well-known/agent-card")
    async def agent_card_alias() -> JSONResponse:
        return JSONResponse(json_format.MessageToDict(card))

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card, card_url=AGENT_CARD_WELL_KNOWN_PATH),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/a2a"),
        rest_routes=create_rest_routes(handler),
    )

    return app


@dataclass(slots=True)
class LocalA2AClient:
    base_url: str
    timeout_seconds: float = 30.0
    verify_tls: bool = True

    def send(self, request: ResearchRequest) -> dict[str, Any]:
        return asyncio.run(self._send(request))

    async def _send(self, request: ResearchRequest) -> dict[str, Any]:
        message = Message(
            role=Role.ROLE_USER, message_id=str(uuid4()),
            parts=[new_text_part(request.question, media_type="text/plain")],
            metadata={"max_references": request.max_references, "response_detail": request.response_detail.value,
                      "run_label": request.run_label},
        )
        http = httpx.AsyncClient(timeout=self.timeout_seconds, verify=self.verify_tls)
        try:
            try:
                client = await ClientFactory(
                    ClientConfig(streaming=False, polling=False, httpx_client=http)
                ).create_from_url(self.base_url)
            except (httpx.HTTPError, AgentCardResolutionError) as exc:
                await http.aclose()
                raise A2AUnavailableError(f"Local A2A server is unavailable at {self.base_url}: {exc}") from exc
            try:
                try:
                    async for response in client.send_message(SendMessageRequest(message=message)):
                        if response.HasField("task"):
                            return self._task_to_payload(response.task)
                        if response.HasField("message"):
                            payload = self._parts_to_payload(response.message.parts)
                            if payload:
                                return payload
                    raise A2ARequestError("Local A2A server returned an empty response after accepting the request.")
                except httpx.HTTPError as exc:
                    raise A2ARequestError(f"Local A2A request failed after submission: {exc}") from exc
            finally:
                await client.close()
        finally:
            if not http.is_closed:
                await http.aclose()

    def _task_to_payload(self, task: Task, task_state: Any = TaskState) -> dict[str, Any]:
        terminal = {
            task_state.TASK_STATE_COMPLETED, task_state.TASK_STATE_FAILED,
            task_state.TASK_STATE_REJECTED, task_state.TASK_STATE_CANCELED,
        }
        if task.status.state not in terminal:
            raise A2ARequestError("Local A2A server returned a non-terminal task.")
        for artifact in task.artifacts:
            if payload := self._parts_to_payload(artifact.parts):
                return payload
        if task.status.HasField("message"):
            if payload := self._parts_to_payload(task.status.message.parts):
                return payload
        raise A2ARequestError("Local A2A response did not include a result payload.")

    def _parts_to_payload(self, parts: Any) -> dict[str, Any] | None:
        for part in parts:
            if part.HasField("data"):
                payload = json_format.MessageToDict(part.data)
                if isinstance(payload, dict):
                    return payload
            if part.HasField("text") and part.text.strip().startswith("{"):
                return json.loads(part.text)
        return None
