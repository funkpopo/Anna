from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

_TRACE_ID: ContextVar[str | None] = ContextVar("anna_trace_id", default=None)


def get_trace_id() -> str | None:
    return _TRACE_ID.get()


def set_trace_id(trace_id: str | None) -> None:
    _TRACE_ID.set(trace_id)


def clear_trace_id() -> None:
    _TRACE_ID.set(None)


class JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per log line for structured ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = get_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        extra_trace = getattr(record, "trace_id", None)
        if extra_trace:
            payload["trace_id"] = extra_trace
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Attach any structured extras passed via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "asctime",
                "trace_id",
            }:
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                payload[key] = repr(value)
            else:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


class TraceIdFilter(logging.Filter):
    """Inject the active request trace id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "trace_id", None):
            trace_id = get_trace_id()
            if trace_id:
                record.trace_id = trace_id  # type: ignore[attr-defined]
        return True


class TraceAwareFormatter(logging.Formatter):
    """Text formatter that appends trace_id when present."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        trace_id = getattr(record, "trace_id", None) or get_trace_id()
        if trace_id:
            return f"{base} trace_id={trace_id}"
        return base


def setup_logging(level: str = "INFO", *, log_format: str = "text") -> None:
    """Configure root logging.

    ``log_format`` is ``text`` (default) or ``json`` for one-line structured logs.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(TraceIdFilter())
    normalized = (log_format or "text").strip().lower()
    if normalized == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            TraceAwareFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root.addHandler(handler)
