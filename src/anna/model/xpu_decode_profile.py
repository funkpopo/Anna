"""Optional XPU decode stepping: accumulate GPU time per component via torch.xpu.Event.

Used when ``EngineOptimizationConfig.profile_runtime`` is enabled. Categories:
``attention``, ``moe``, ``gated_delta``, ``conv``, ``lm_head``, ``int4_matmul``,
``rmsnorm``, ``rotary``, ``turboquant_dequant``. Percentages are relative to the
sum of recorded categories.

Derived CPU-launch-gap: ``DecodeProfileSession.cpu_launch_gap_ms(wall_ms)``
returns the wall-clock time of the forward that is NOT covered by tracked GPU
categories — kernel-launch gaps, host-side dispatch, sampling, and untracked
ops. The engine logs it per profiled forward (``cpu_launch_gap_ms``), which is
how the P0-#1 ~79ms decode-step breakdown is produced.

When wrapped with ``steady_decode_accumulation`` during Qwen3.5 generation, the
engine also logs **steady-state** averages over ``decode[2+]`` only (skips
``decode[1]`` compile/warmup skew) for apples-to-apples comparisons across batch,
KV mode, or kernel changes.

Continuous-batch ``scheduler_decode`` steps feed a separate process-level
``SchedulerDecodeSteadyAccum`` keyed by batch size (amortized per-request ms =
batch totals / batch_size). The first ``warmup_batches`` scheduler decode
forwards are skipped to drop compile skew.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import defaultdict
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_session: ContextVar["DecodeProfileSession | None"] = ContextVar("anna_xpu_decode_profile", default=None)
# Steady-state accum uses thread-local storage so generator close() / torch generator_context
# cannot hit ContextVar.reset() from a different logical Context (ValueError).
_steady_tls = threading.local()
_scheduler_steady_lock = threading.Lock()
_scheduler_steady: "SchedulerDecodeSteadyAccum | None" = None
_DECODE_STAGE_RE = re.compile(r"^decode\[(\d+)\]$")
_SCHEDULER_DECODE_STAGE = "scheduler_decode"
DEFAULT_SCHEDULER_DECODE_WARMUP_BATCHES = 1


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

    @staticmethod
    def cpu_launch_gap_ms(wall_seconds: float, component_ms: dict[str, float]) -> float:
        """Wall-clock forward time not covered by tracked GPU component timers (ms).

        Large gaps indicate CPU-side launch overhead / host stalls rather than
        device bandwidth — the primary signal for P0-#1 decode-step analysis.
        """
        if wall_seconds <= 0:
            return 0.0
        return max(0.0, wall_seconds * 1000.0 - float(sum(component_ms.values())))


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

    def average_ms(self) -> dict[str, float]:
        if self.step_count <= 0:
            return {}
        inv = 1.0 / float(self.step_count)
        return {k: v * inv for k, v in self.totals.items()}

    def log_avg(self, *, lg: logging.Logger | None = None) -> None:
        sink = lg or logger
        avg = self.average_ms()
        if not avg:
            return
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


def _amortized_per_request_ms(ms: Mapping[str, float], batch_size: int) -> dict[str, float]:
    size = max(1, int(batch_size))
    return {k: float(v) / float(size) for k, v in ms.items()}


@dataclass
class SchedulerDecodeSteadyAccum:
    """Process-level steady-state component ms for continuous-batch decode, keyed by batch size."""

    by_batch: dict[int, SteadyDecodeAccum] = field(default_factory=dict)
    warmup_batches: int = DEFAULT_SCHEDULER_DECODE_WARMUP_BATCHES
    total_seen: int = 0
    skipped_warmup: int = 0

    def record(
        self,
        ms: dict[str, float],
        *,
        batch_size: int,
    ) -> bool:
        """Record one scheduler decode forward. Returns True when counted in steady-state."""
        if not ms:
            return False
        self.total_seen += 1
        if self.total_seen <= max(0, int(self.warmup_batches)):
            self.skipped_warmup += 1
            return False
        size = max(1, int(batch_size))
        bucket = self.by_batch.get(size)
        if bucket is None:
            bucket = SteadyDecodeAccum()
            self.by_batch[size] = bucket
        bucket.add_step(ms)
        return True

    def snapshot(self) -> dict[str, object]:
        batches: dict[str, object] = {}
        for batch_size, accum in sorted(self.by_batch.items()):
            avg = accum.average_ms()
            if not avg:
                continue
            batches[str(batch_size)] = {
                "n": accum.step_count,
                "avg_ms": {k: round(v, 3) for k, v in avg.items()},
                "avg_ms_per_req": {k: round(v, 3) for k, v in _amortized_per_request_ms(avg, batch_size).items()},
                "tracked_avg_total_ms": round(sum(avg.values()), 3),
            }
        return {
            "total_seen": self.total_seen,
            "skipped_warmup": self.skipped_warmup,
            "warmup_batches": self.warmup_batches,
            "by_batch_size": batches,
        }

    def log_summary(self, *, lg: logging.Logger | None = None) -> None:
        sink = lg or logger
        snapshot = self.snapshot()
        by_batch = snapshot.get("by_batch_size") or {}
        if not by_batch:
            return
        for batch_size, stats in by_batch.items():
            sink.info(
                "xpu_scheduler_decode_steady_avg_ms (batch_size=%s, n=%s) %s | "
                "avg_ms_per_req %s | tracked_avg_total_ms=%.3f",
                batch_size,
                stats["n"],
                stats["avg_ms"],
                stats["avg_ms_per_req"],
                float(stats["tracked_avg_total_ms"]),
            )


def get_or_create_scheduler_decode_steady_accum(
    *,
    enabled: bool,
    warmup_batches: int = DEFAULT_SCHEDULER_DECODE_WARMUP_BATCHES,
) -> SchedulerDecodeSteadyAccum | None:
    global _scheduler_steady
    if not enabled:
        return None
    with _scheduler_steady_lock:
        if _scheduler_steady is None:
            _scheduler_steady = SchedulerDecodeSteadyAccum(warmup_batches=max(0, int(warmup_batches)))
        return _scheduler_steady


def reset_scheduler_decode_steady_accum() -> None:
    global _scheduler_steady
    with _scheduler_steady_lock:
        _scheduler_steady = None


def record_steady_decode_step_if_applicable(
    stage: str,
    ms: dict[str, float],
    *,
    batch_size: int | None = None,
    token_cost: int | None = None,
    profile_meta: Mapping[str, object] | None = None,
) -> None:
    """Record single-request decode[2+] and/or scheduler_decode steady-state samples."""
    del token_cost  # reserved for future token-cost banding; batch_size is the primary key today
    if not ms:
        return

    if stage == _SCHEDULER_DECODE_STAGE or stage.startswith(f"{_SCHEDULER_DECODE_STAGE}["):
        size = int(batch_size) if batch_size is not None else 1
        if profile_meta is not None and batch_size is None:
            raw = profile_meta.get("batch_size")
            if raw is not None:
                size = int(raw)
        accum = get_or_create_scheduler_decode_steady_accum(enabled=True)
        if accum is not None:
            accum.record(ms, batch_size=size)
        return

    single = _steady_tls_get_accum()
    if single is None:
        return
    matched = _DECODE_STAGE_RE.match(stage)
    if matched is None:
        return
    if int(matched.group(1)) < 2:
        return
    single.add_step(ms)


def log_scheduler_decode_steady_if_any(*, log: logging.Logger | None = None) -> None:
    with _scheduler_steady_lock:
        accum = _scheduler_steady
    if accum is None:
        return
    accum.log_summary(lg=log)


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
