"""Process-level XPU execution serialization (P1-2.8).

Anna ``anna-serve`` loads a single model family per process. When audio engines
(ASR/TTS) and text engines are co-resident in one process (embedding / tests /
future multi-engine serve), they must not run concurrent XPU work or they will
contend for the same Arc device memory pool.

Boundary:
- **Text engines** keep a per-engine ``execution_lock`` for continuous batching.
- **Audio engines** always take the process device gate around upstream inference.
- Optional ``ANNA_XPU_SERIALIZE_ALL=1`` makes text engines take the same gate so
  co-resident modalities cannot overlap on device.

This is serialization, not multi-tenant VRAM partitioning: co-resident models
still share one device; the gate only prevents concurrent kernels / peak spikes.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

logger = logging.getLogger(__name__)

_PROCESS_DEVICE_LOCK = threading.RLock()
_PROCESS_HOLDER: str | None = None
_PROCESS_ACQUIRE_COUNT = 0
_PROCESS_WAIT_SECONDS_TOTAL = 0.0
_PROCESS_STATS_LOCK = threading.Lock()


def _env_serialize_all() -> bool:
    raw = os.getenv("ANNA_XPU_SERIALIZE_ALL", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class DeviceExecutionSnapshot:
    holder: str | None
    acquire_count: int
    wait_seconds_total: float
    serialize_all: bool


def device_execution_snapshot() -> DeviceExecutionSnapshot:
    with _PROCESS_STATS_LOCK:
        return DeviceExecutionSnapshot(
            holder=_PROCESS_HOLDER,
            acquire_count=_PROCESS_ACQUIRE_COUNT,
            wait_seconds_total=_PROCESS_WAIT_SECONDS_TOTAL,
            serialize_all=_env_serialize_all(),
        )


@contextmanager
def process_device_execution(*, owner: str) -> Iterator[None]:
    """Serialize process-wide device work under ``owner`` label."""
    global _PROCESS_HOLDER, _PROCESS_ACQUIRE_COUNT, _PROCESS_WAIT_SECONDS_TOTAL
    wait_started = time.perf_counter()
    acquired = _PROCESS_DEVICE_LOCK.acquire(blocking=True)
    waited = time.perf_counter() - wait_started
    if not acquired:  # pragma: no cover - RLock.acquire(True) always True
        raise RuntimeError("Failed to acquire process device execution lock.")
    with _PROCESS_STATS_LOCK:
        _PROCESS_HOLDER = owner
        _PROCESS_ACQUIRE_COUNT += 1
        _PROCESS_WAIT_SECONDS_TOTAL += max(0.0, waited)
    if waited > 0.05:
        logger.debug("Process device gate waited %.3fs for owner=%s", waited, owner)
    try:
        yield
    finally:
        with _PROCESS_STATS_LOCK:
            if _PROCESS_HOLDER == owner:
                _PROCESS_HOLDER = None
        _PROCESS_DEVICE_LOCK.release()


class DeviceExecutionGate:
    """Per-engine gate that always takes the process lock (audio path).

    Text engines may call :meth:`optional_for_text` so they only serialize when
    ``ANNA_XPU_SERIALIZE_ALL`` is enabled.
    """

    def __init__(self, owner: str) -> None:
        self.owner = owner
        self._local = threading.RLock()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        with self._local:
            with process_device_execution(owner=self.owner):
                yield

    @contextmanager
    def optional_for_text(self) -> Iterator[None]:
        if _env_serialize_all():
            with self.exclusive():
                yield
        else:
            with self._local:
                yield

    def health_fields(self) -> dict[str, object]:
        snap = device_execution_snapshot()
        return {
            "device_execution_owner": self.owner,
            "process_device_holder": snap.holder,
            "process_device_acquire_count": snap.acquire_count,
            "process_device_wait_seconds_total": round(snap.wait_seconds_total, 6),
            "xpu_serialize_all": snap.serialize_all,
            "isolation_mode": (
                "process_serialized"
                if snap.serialize_all
                else "single_family_preferred"
            ),
        }
