from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def release_cpu_memory_caches() -> None:
    for _ in range(2):
        gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            logger.debug("Failed to trim libc malloc arena.", exc_info=True)
    elif sys.platform == "win32":
        try:
            import ctypes

            try:
                ctypes.CDLL("msvcrt")._heapmin()
            except Exception:
                logger.debug("Failed to minimize CRT heap.", exc_info=True)
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            logger.debug("Failed to trim Windows working set.", exc_info=True)


def release_conversion_artifacts(device: torch.device) -> None:
    release_cpu_memory_caches()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    release_cpu_memory_caches()


class AdaptiveMemoryReleaser:
    """Idle-time device memory sweeper (P2-#6 allocator fragmentation control).

    Problem observed on Arc A770 (9.77 GiB reserved vs 4.35 GiB allocated): after a
    request finishes only ~0.24 GiB returns to free memory because cached blocks and
    idle KV pages stay reserved. The engine trims pages per request, but
    ``empty_cache`` was only invoked on explicit recovery/trim paths.

    This sweeper runs a low-frequency background tick (default 30s). When the
    runtime has been idle and either free device memory is below the configured
    floor (default: the admission ``min_free_bytes`` safety threshold) or the
    ``reserved - allocated`` fragmentation gap exceeds a threshold (default 1 GiB),
    it trims idle KV pages, calls ``torch.xpu.empty_cache()`` and logs the
    reserved-allocated gap so the acceptance metric (idle 5 min → gap < 1 GiB) is
    observable.

    Knobs (env):
    - ``ANNA_MEMORY_RELEASE_SWEEP_INTERVAL_S``: sweep period, ``0`` disables the thread.
    - ``ANNA_MEMORY_RELEASE_MIN_FREE_MIB``: free-memory floor; ``0`` inherits the
      device safety policy's ``min_free_bytes`` (default 1024 MiB).
    - ``ANNA_MEMORY_RELEASE_MAX_GAP_MIB``: idle reserved-allocated gap above which
      a release runs (default 1024 MiB; ``0`` disables gap-triggered release).
    """

    def __init__(
        self,
        *,
        snapshot_provider: Callable[[], dict[str, object] | None],
        release_callback: Callable[[], None],
        interval_seconds: float | None = None,
        min_free_bytes: int | None = None,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._release_callback = release_callback
        if interval_seconds is None:
            interval_seconds = _env_float("ANNA_MEMORY_RELEASE_SWEEP_INTERVAL_S", 30.0)
        self.interval_seconds = max(0.0, float(interval_seconds))
        if min_free_bytes is None:
            env_mib = _env_int("ANNA_MEMORY_RELEASE_MIN_FREE_MIB", 0)
            min_free_bytes = 0 if env_mib <= 0 else env_mib << 20
        self.min_free_bytes = max(0, int(min_free_bytes))
        self.max_gap_mib = max(0, _env_int("ANNA_MEMORY_RELEASE_MAX_GAP_MIB", 1024))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_seconds <= 0 or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="anna-memory-releaser", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
            self._thread = None

    def sweep_once(self) -> bool:
        """Run one adaptive release evaluation. Returns True when a release ran."""
        snapshot = self._snapshot_provider()
        if not snapshot or not snapshot.get("idle", False):
            return False
        free_bytes = snapshot.get("free_bytes")
        if free_bytes is None:
            return False
        threshold = int(snapshot.get("min_free_bytes") or 0) or self.min_free_bytes
        free_bytes = int(free_bytes)
        reserved = snapshot.get("reserved_bytes")
        allocated = snapshot.get("allocated_bytes")
        gap_bytes = 0
        if reserved is not None and allocated is not None:
            gap_bytes = max(0, int(reserved) - int(allocated))
        below_floor = threshold > 0 and free_bytes < threshold
        gap_exceeded = self.max_gap_mib > 0 and gap_bytes > (self.max_gap_mib << 20)
        if not below_floor and not gap_exceeded:
            return False
        before = {
            "free_bytes": free_bytes,
            "reserved_bytes": reserved,
            "allocated_bytes": allocated,
        }
        self._release_callback()
        after = self._snapshot_provider() or {}
        gap_before = _gap_bytes(before)
        gap_after = _gap_bytes(after)
        trigger = "free-below-floor" if below_floor else "fragmentation-gap"
        logger.info(
            "Adaptive memory release (idle, trigger=%s): "
            "reserved-allocated gap %.2f GiB -> %.2f GiB, free %.0f -> %.0f MiB",
            trigger,
            gap_before,
            gap_after,
            int(before["free_bytes"]) >> 20,
            int(after.get("free_bytes") or 0) >> 20,
        )
        return True

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.sweep_once()
            except Exception:  # pragma: no cover - background best effort
                logger.debug("Adaptive memory release sweep failed.", exc_info=True)


def _gap_bytes(snapshot: dict[str, object]) -> float:
    """reserved - allocated in GiB (negative/missing values collapse to 0)."""
    reserved = snapshot.get("reserved_bytes")
    allocated = snapshot.get("allocated_bytes")
    if reserved is None or allocated is None:
        return 0.0
    return max(0.0, (int(reserved) - int(allocated)) / (1 << 30))
