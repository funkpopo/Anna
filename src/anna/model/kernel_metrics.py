"""Lightweight counters for fused-kernel / decode-path strategy hits."""

from __future__ import annotations

import threading
from collections import Counter

_lock = threading.Lock()
_hits: Counter[str] = Counter()


def record_kernel_strategy(family: str, strategy: str) -> None:
    """Count a resolved kernel path, e.g. family=gqa_decode strategy=turboquant."""
    key = f"{family}:{strategy}"
    with _lock:
        _hits[key] += 1


def kernel_strategy_snapshot() -> dict[str, int]:
    with _lock:
        return dict(sorted(_hits.items()))


def reset_kernel_strategy_hits() -> None:
    """Test helper to clear counters."""
    with _lock:
        _hits.clear()
