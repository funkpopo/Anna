from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol


class HotPathEventRecorder(Protocol):
    def record_cpu_sync(self, *, reason: str, count: int = 1) -> None: ...

    def record_attention_fallback(self, *, reason: str, count: int = 1) -> None: ...

    def record_paged_cache_materialization(self, *, reason: str, count: int = 1) -> None: ...

    def record_sampler_full_vocab_sort(self, *, reason: str, count: int = 1) -> None: ...

    def record_moe_host_offset(self, *, reason: str, count: int = 1) -> None: ...

    def record_moe_stage(self, *, stage: str, seconds: float) -> None: ...


_RECORDER: ContextVar[HotPathEventRecorder | None] = ContextVar("anna_hotpath_event_recorder", default=None)


@contextmanager
def hotpath_event_recorder(recorder: HotPathEventRecorder | None) -> Iterator[None]:
    token = _RECORDER.set(recorder)
    try:
        yield
    finally:
        _RECORDER.reset(token)


def _record(method_name: str, *, reason: str, count: int = 1) -> None:
    recorder = _RECORDER.get()
    if recorder is None:
        return
    method = getattr(recorder, method_name, None)
    if callable(method):
        method(reason=reason, count=count)


def record_cpu_sync(reason: str, *, count: int = 1) -> None:
    _record("record_cpu_sync", reason=reason, count=count)


def record_attention_fallback(reason: str, *, count: int = 1) -> None:
    _record("record_attention_fallback", reason=reason, count=count)


def record_paged_cache_materialization(reason: str, *, count: int = 1) -> None:
    _record("record_paged_cache_materialization", reason=reason, count=count)


def record_sampler_full_vocab_sort(reason: str, *, count: int = 1) -> None:
    _record("record_sampler_full_vocab_sort", reason=reason, count=count)


def record_moe_host_offset(reason: str, *, count: int = 1) -> None:
    _record("record_moe_host_offset", reason=reason, count=count)


def record_moe_stage(stage: str, seconds: float) -> None:
    recorder = _RECORDER.get()
    if recorder is None:
        return
    method = getattr(recorder, "record_moe_stage", None)
    if callable(method):
        method(stage=stage, seconds=seconds)
