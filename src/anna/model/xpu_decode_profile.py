"""Optional XPU decode stepping: accumulate GPU time per component via torch.xpu.Event.

Used when ``EngineOptimizationConfig.profile_runtime`` is enabled. Categories:
``attention``, ``moe``, ``gated_delta``, ``conv``. Percentages are relative to the
sum of recorded categories (excludes layernorm, LM head, etc.).

When wrapped with ``steady_decode_accumulation`` during Qwen3.5 generation, the
engine also logs **steady-state** averages over ``decode[2+]`` only (skips
``decode[1]`` compile/warmup skew) for apples-to-apples comparisons across batch,
KV mode, or kernel changes. ``scheduler_decode`` batches are not split yet.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_session: ContextVar["DecodeProfileSession | None"] = ContextVar("anna_xpu_decode_profile", default=None)
# Steady-state accum uses thread-local storage so generator close() / torch generator_context
# cannot hit ContextVar.reset() from a different logical Context (ValueError).
_steady_tls = threading.local()
_DECODE_STAGE_RE = re.compile(r"^decode\[(\d+)\]$")


def _steady_tls_get_accum() -> SteadyDecodeAccum | None:
    return getattr(_steady_tls, "accum", None)


def _steady_tls_set_accum(accum: SteadyDecodeAccum | None) -> None:
    if accum is None:
        if hasattr(_steady_tls, "accum"):
            delattr(_steady_tls, "accum")
    else:
        _steady_tls.accum = accum


@dataclass
class DecodeProfileSession:
    """Collects (category, start_event, end_event) pairs for one forward pass."""

    _pairs: list[tuple[str, object, object]] = field(default_factory=list)
    _timing_supported: bool = True

    def record_pair(self, category: str, start_ev: object, end_ev: object) -> None:
        self._pairs.append((category, start_ev, end_ev))

    def finalize_ms(self) -> dict[str, float]:
        if not self._pairs:
            return {}
        import torch

        if not hasattr(torch, "xpu") or not self._timing_supported:
            return {}
        torch.xpu.synchronize()
        totals: dict[str, float] = defaultdict(float)
        for category, start_ev, end_ev in self._pairs:
            try:
                elapsed = start_ev.elapsed_time(end_ev)
            except Exception:
                logger.debug("xpu.Event.elapsed_time failed for category=%s", category, exc_info=True)
                continue
            totals[category] += float(elapsed)
        return dict(totals)

    def log_summary(self, *, log: logging.Logger | None = None) -> dict[str, float]:
        """Finalize GPU timers, log per-forward breakdown, return milliseconds per category (or {})."""
        lg = log or logger
        ms = self.finalize_ms()
        if not ms:
            return {}
        tracked = sum(ms.values())
        if tracked <= 0:
            return {}
        pct = {k: 100.0 * v / tracked for k, v in sorted(ms.items(), key=lambda kv: -kv[1])}
        lg.info(
            "xpu_decode_component_ms (GPU elapsed, sum of layers) %s | pct_of_tracked %s | tracked_total_ms=%.3f",
            {k: round(v, 3) for k, v in ms.items()},
            {k: round(v, 1) for k, v in pct.items()},
            tracked,
        )
        return ms


def active_session() -> DecodeProfileSession | None:
    return _session.get()


@contextmanager
def decode_profile_session() -> Generator[DecodeProfileSession, None, None]:
    session = DecodeProfileSession()
    token = _session.set(session)
    try:
        yield session
    finally:
        _session.reset(token)


@contextmanager
def xpu_profile_region(category: str) -> Generator[None, None, None]:
    session = _session.get()
    if session is None:
        yield
        return
    import torch

    if not hasattr(torch, "xpu"):
        yield
        return
    event_cls = getattr(torch.xpu, "Event", None)
    if event_cls is None:
        yield
        return
    try:
        stream = torch.xpu.current_stream()
        start_ev = event_cls(enable_timing=True)
        end_ev = event_cls(enable_timing=True)
    except Exception:
        session._timing_supported = False
        yield
        return
    start_ev.record(stream)
    try:
        yield
    finally:
        end_ev.record(stream)
        session.record_pair(category, start_ev, end_ev)


@dataclass
class SteadyDecodeAccum:
    """Sums decode[2+] component ms within one generation (excludes decode[1] compile/warmup skew)."""

    totals: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    step_count: int = 0

    def add_step(self, ms: dict[str, float]) -> None:
        for key, value in ms.items():
            self.totals[key] += float(value)
        self.step_count += 1

    def log_avg(self, *, lg: logging.Logger | None = None) -> None:
        sink = lg or logger
        if self.step_count <= 0:
            return
        inv = 1.0 / float(self.step_count)
        avg = {k: v * inv for k, v in self.totals.items()}
        tracked = sum(avg.values())
        if tracked <= 0:
            return
        pct = {k: 100.0 * v / tracked for k, v in sorted(avg.items(), key=lambda kv: -kv[1])}
        sink.info(
            "xpu_decode_steady_state_avg_ms_per_step (decode[2+], n=%d) %s | pct_of_tracked %s | tracked_avg_total_ms=%.3f",
            self.step_count,
            {k: round(v, 3) for k, v in avg.items()},
            {k: round(v, 1) for k, v in pct.items()},
            tracked,
        )


def record_steady_decode_step_if_applicable(stage: str, ms: dict[str, float]) -> None:
    if not ms:
        return
    accum = _steady_tls_get_accum()
    if accum is None:
        return
    matched = _DECODE_STAGE_RE.match(stage)
    if matched is None:
        return
    if int(matched.group(1)) < 2:
        return
    accum.add_step(ms)


@contextmanager
def steady_decode_accumulation(*, enabled: bool, log: logging.Logger | None = None) -> Generator[None, None, None]:
    """Wrap one generation: after the block, log average component ms for decode[2+] only."""
    if not enabled:
        yield
        return
    accum = SteadyDecodeAccum()
    prev = _steady_tls_get_accum()
    _steady_tls_set_accum(accum)
    try:
        yield
    finally:
        try:
            accum.log_avg(lg=log)
        finally:
            if _steady_tls_get_accum() is accum:
                _steady_tls_set_accum(prev)
