from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from anna.api.routes import router
from anna.runtime.engine import AnnaEngineError


def _error_payload(
    *,
    message: str,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
        }
    }
    if code is not None:
        payload["error"]["code"] = code
    if param is not None:
        payload["error"]["param"] = param
    return payload


def _error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(
            message=message,
            error_type=error_type,
            code=code,
            param=param,
        ),
        headers=headers,
    )


def _extract_api_key(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    header_api_key = request.headers.get("x-api-key")
    if header_api_key is not None and header_api_key.strip():
        return header_api_key.strip()
    return None


def _validation_error_message(exc: RequestValidationError) -> tuple[str, str | None]:
    errors = exc.errors()
    if not errors:
        return "Invalid request body.", None

    first = errors[0]
    location = [str(part) for part in first.get("loc", ()) if part != "body"]
    param = ".".join(location) or None
    detail = first.get("msg", "Invalid request body.")
    if param is not None:
        return f"Invalid value for '{param}': {detail}", param
    return f"Invalid request: {detail}", None


def create_app(engine, *, scheduler=None, api_key: str | None = None) -> FastAPI:
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
    app.state.api_key = api_key
    app.include_router(router)

    @app.middleware("http")
    async def enforce_api_key(request: Request, call_next):
        if api_key is not None and request.url.path.startswith("/v1/"):
            provided = _extract_api_key(request)
            auth_headers = {"WWW-Authenticate": "Bearer"}
            if provided is None:
                return _error_response(
                    status_code=401,
                    message="Missing API key. Provide Authorization: Bearer <key> or X-API-Key.",
                    error_type="authentication_error",
                    code="missing_api_key",
                    headers=auth_headers,
                )
            if provided != api_key:
                return _error_response(
                    status_code=401,
                    message="Invalid API key provided.",
                    error_type="authentication_error",
                    code="invalid_api_key",
                    headers=auth_headers,
                )
        return await call_next(request)

    @app.exception_handler(AnnaEngineError)
    async def handle_engine_error(_, exc: AnnaEngineError):
        return _error_response(
            status_code=exc.status_code,
            message=str(exc),
            error_type=exc.error_type,
            code=exc.code,
            param=exc.param,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_, exc: RequestValidationError):
        message, param = _validation_error_message(exc)
        return _error_response(
            status_code=400,
            message=message,
            error_type="invalid_request_error",
            code="invalid_request",
            param=param,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return _error_response(
            status_code=exc.status_code,
            message=detail,
            error_type="invalid_request_error",
        )

    return app
