"""Optional XPU decode stepping: accumulate GPU time per component via torch.xpu.Event.

Used when ``EngineOptimizationConfig.profile_runtime`` is enabled. Categories:
``attention``, ``moe``, ``gated_delta``, ``conv``. Percentages are relative to the
sum of recorded categories (excludes layernorm, LM head, etc.).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_session: ContextVar["DecodeProfileSession | None"] = ContextVar("anna_xpu_decode_profile", default=None)


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

    def log_summary(self, *, log: logging.Logger | None = None) -> None:
        lg = log or logger
        ms = self.finalize_ms()
        if not ms:
            return
        tracked = sum(ms.values())
        if tracked <= 0:
            return
        pct = {k: 100.0 * v / tracked for k, v in sorted(ms.items(), key=lambda kv: -kv[1])}
        lg.info(
            "xpu_decode_component_ms (GPU elapsed, sum of layers) %s | pct_of_tracked %s | tracked_total_ms=%.3f",
            {k: round(v, 3) for k, v in ms.items()},
            {k: round(v, 1) for k, v in pct.items()},
            tracked,
        )


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
