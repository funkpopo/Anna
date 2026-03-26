from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from anna.api.routes import router
from anna.runtime.engine import AnnaEngineError


def create_app(engine, *, scheduler=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            if scheduler is not None:
                scheduler.shutdown()

    app = FastAPI(title="Anna", version="0.1.0", lifespan=lifespan)
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
