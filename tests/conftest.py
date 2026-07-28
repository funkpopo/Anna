"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_process_admission_gate() -> None:
    """Keep unit tests isolated from process-wide degraded admission state."""
    from anna.runtime.runtime_health import PROCESS_ADMISSION_GATE

    PROCESS_ADMISSION_GATE.clear_degraded()
    yield
    PROCESS_ADMISSION_GATE.clear_degraded()
