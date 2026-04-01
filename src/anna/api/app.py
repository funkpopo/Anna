from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Iterable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from anna import __version__
from anna.api.routes import router
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
