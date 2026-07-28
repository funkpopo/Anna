from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Iterable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from anna import __version__
from anna.api.routes import router
from anna.core.logging import clear_trace_id, set_trace_id
from anna.runtime.qwen3_5_text_engine import AnnaEngineError

_HTTP_METHOD_ORDER = {
    "HEAD": 0,
    "GET": 1,
    "POST": 2,
    "PUT": 3,
    "PATCH": 4,
    "DELETE": 5,
    "OPTIONS": 6,
}

_TRACE_HEADER = "x-request-id"


class RequestTraceMiddleware(BaseHTTPMiddleware):
    """Bind a request trace id for the full request lifetime (including streams)."""

    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get(_TRACE_HEADER) or request.headers.get("x-correlation-id")
        trace_id = (incoming or "").strip() or f"req_{uuid.uuid4().hex}"
        set_trace_id(trace_id)
        request.state.trace_id = trace_id
        try:
            response = await call_next(request)
        finally:
            clear_trace_id()
        response.headers[_TRACE_HEADER] = trace_id
        return response


def list_app_routes(app: FastAPI) -> list[tuple[str, str]]:
    def _sorted_methods(methods: Iterable[str] | None) -> str:
        if not methods:
            return ""
        ordered = sorted(methods, key=lambda method: (_HTTP_METHOD_ORDER.get(method, 99), method))
        return ", ".join(ordered)

    descriptions: list[tuple[str, str]] = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path:
            continue
        methods = _sorted_methods(getattr(route, "methods", None))
        descriptions.append((path, methods))
    return descriptions


def create_app(engine, *, scheduler=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown()

    app = FastAPI(title="Anna", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestTraceMiddleware)
    app.state.engine = engine
    app.state.scheduler = scheduler
    app.include_router(router)

    @app.exception_handler(AnnaEngineError)
    async def handle_engine_error(_, exc: AnnaEngineError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc),
                    "type": exc.error_type,
                    "code": exc.code,
                }
            },
        )

    return app
