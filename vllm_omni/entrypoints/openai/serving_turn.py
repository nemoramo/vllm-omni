from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, Protocol

import vllm.envs as envs
from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from vllm.entrypoints.launcher import serve_http
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.serving_soulx_duplug_turn import (
    SOULX_DUPLUG_TURN_MODEL,
)

logger = init_logger(__name__)

TURN_TASK = "turn"


class TurnSessionHandler(Protocol):
    async def session_gc_loop(self) -> None: ...

    async def handle_session(self, ws: WebSocket) -> None: ...


class TurnModelDefinition(Protocol):
    name: str
    supported_tasks: tuple[str, ...]

    def matches(self, model: str) -> bool: ...

    def create_handler(self, model: str) -> TurnSessionHandler: ...


TURN_MODEL_DEFINITIONS: tuple[TurnModelDefinition, ...] = (
    SOULX_DUPLUG_TURN_MODEL,
)


class _TurnEngineClient:
    def __init__(self, supported_tasks: tuple[str, ...]) -> None:
        self.errored = False
        self.is_running = True
        self.vllm_config = SimpleNamespace(shutdown_timeout=0)
        self._supported_tasks = set(supported_tasks)

    async def get_supported_tasks(self) -> set[str]:
        return set(self._supported_tasks)

    def shutdown(self, timeout: int = 0) -> None:
        self.is_running = False


def resolve_turn_model_definition(model: str) -> TurnModelDefinition | None:
    for definition in TURN_MODEL_DEFINITIONS:
        if definition.matches(model):
            return definition
    return None


def is_turn_model(model: str) -> bool:
    return resolve_turn_model_definition(model) is not None


def get_turn_supported_tasks(model: str) -> set[str]:
    definition = resolve_turn_model_definition(model)
    if definition is None:
        return set()
    return set(definition.supported_tasks)


def build_turn_app(args: Any, definition: TurnModelDefinition) -> FastAPI:
    served_model_names = args.served_model_name or [args.model]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        handler = definition.create_handler(args.model)
        app.state.turn_handler = handler
        gc_task = asyncio.create_task(handler.session_gc_loop())
        yield
        gc_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await gc_task

    app = FastAPI(lifespan=lifespan)
    app.state.engine_client = _TurnEngineClient(definition.supported_tasks)
    app.state.supported_tasks = set(definition.supported_tasks)
    app.state.turn_model_definition = definition

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "healthy",
                "mode": TURN_TASK,
                "turn_model": definition.name,
            }
        )

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return JSONResponse({"data": [{"id": served_model_names[0], "object": "model"}]})

    @app.websocket("/turn")
    async def turn_ws(ws: WebSocket) -> None:
        await ws.app.state.turn_handler.handle_session(ws)

    return app


async def maybe_run_turn_server(
    *,
    listen_address: str,
    sock: Any,
    args: Any,
    uvicorn_kwargs: dict[str, Any],
) -> bool:
    definition = resolve_turn_model_definition(args.model)
    if definition is None:
        return False

    app = build_turn_app(args, definition)
    logger.info("Starting turn server for %s on %s", definition.name, listen_address)

    local_uvicorn_kwargs = dict(uvicorn_kwargs)
    shutdown_task = await serve_http(
        app,
        sock=sock,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
        h11_max_header_count=args.h11_max_header_count,
        **local_uvicorn_kwargs,
    )
    try:
        await shutdown_task
    finally:
        sock.close()

    return True
