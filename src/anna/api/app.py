from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from anna.api.routes import router
from anna.runtime.engine import AnnaEngineError


def create_app(engine) -> FastAPI:
    app = FastAPI(title="Anna", version="0.1.0")
    app.state.engine = engine
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
