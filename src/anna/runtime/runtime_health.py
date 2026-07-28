"""Process-level runtime admission gate after device-lost / OOM recovery."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeHealthSnapshot:
    accepting_requests: bool
    status: str
    degradation_reason: str | None
    degradation_category: str | None
    degraded_at: float | None
    recovery_attempts: int


class RuntimeAdmissionGate:
    """Reject new work after severe device failures until recovery clears the gate.

    Device-lost always enters degraded mode. OOM enters degraded mode after recovery
    so the service stops accepting new traffic while the operator inspects healthz;
    a successful memory probe can clear the gate (or the process can be restarted).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepting_requests = True
        self._degradation_reason: str | None = None
        self._degradation_category: str | None = None
        self._degraded_at: float | None = None
        self._recovery_attempts = 0

    def snapshot(self) -> RuntimeHealthSnapshot:
        with self._lock:
            accepting = self._accepting_requests
            return RuntimeHealthSnapshot(
                accepting_requests=accepting,
                status="ok" if accepting else "degraded",
                degradation_reason=self._degradation_reason,
                degradation_category=self._degradation_category,
                degraded_at=self._degraded_at,
                recovery_attempts=self._recovery_attempts,
            )

    @property
    def accepting_requests(self) -> bool:
        with self._lock:
            return self._accepting_requests

    def enter_degraded(self, *, category: str, reason: str) -> None:
        with self._lock:
            self._accepting_requests = False
            self._degradation_category = str(category)
            self._degradation_reason = str(reason)
            if self._degraded_at is None:
                self._degraded_at = time.time()
            self._recovery_attempts += 1

    def clear_degraded(self) -> None:
        with self._lock:
            self._accepting_requests = True
            self._degradation_reason = None
            self._degradation_category = None
            self._degraded_at = None

    def to_health_dict(self) -> dict[str, object]:
        snap = self.snapshot()
        return {
            "accepting_requests": snap.accepting_requests,
            "status": snap.status,
            "degradation_reason": snap.degradation_reason,
            "degradation_category": snap.degradation_category,
            "degraded_at": snap.degraded_at,
            "recovery_attempts": snap.recovery_attempts,
        }


# Shared process gate so API routes, engines, and scheduler see one view.
PROCESS_ADMISSION_GATE = RuntimeAdmissionGate()
