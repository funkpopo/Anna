"""Phase 4: scheduler hot-path ledger tests (Rust ext + Python fallback)."""

from __future__ import annotations

import pytest

from anna.runtime.scheduler import (
    _PythonSchedulerLedger,
    _create_scheduler_ledger,
)


class TestPythonLedgerFallback:
    def test_register_and_stats(self) -> None:
        ledger = _PythonSchedulerLedger()
        assert ledger.register(1, 100)
        assert ledger.register(2, 50)
        assert not ledger.register(1, 999)
        assert ledger.stats() == (2, 150, 0, 0)

    def test_finish_moves_cost_out_of_active(self) -> None:
        ledger = _PythonSchedulerLedger()
        ledger.register(1, 100)
        ledger.register(2, 50)
        assert ledger.mark_finished(1)
        assert not ledger.mark_finished(1)
        assert ledger.active_cost() == 50
        assert ledger.active_ids() == [2]
        assert ledger.finished_ids() == [1]
        assert ledger.stats() == (1, 50, 1, 0)

    def test_unregister_and_reconcile(self) -> None:
        ledger = _PythonSchedulerLedger()
        for rid, cost in ((1, 10), (2, 20), (3, 30)):
            ledger.register(rid, cost)
        assert ledger.unregister(2)
        assert not ledger.unregister(2)
        ledger.retain([3, 1])
        assert ledger.active_ids() == [3, 1]
        assert ledger.active_cost() == 40

    def test_set_cost(self) -> None:
        ledger = _PythonSchedulerLedger()
        ledger.register(1, 10)
        assert ledger.set_cost(1, 25)
        assert ledger.active_cost() == 25
        ledger.mark_finished(1)
        assert not ledger.set_cost(1, 99)

    def test_decode_step_accounting(self) -> None:
        ledger = _PythonSchedulerLedger()
        ledger.bump_decode_steps()
        ledger.bump_decode_steps()
        assert ledger.decode_steps() == 2
        ledger.clear()
        assert ledger.stats() == (0, 0, 0, 0)


def test_create_scheduler_ledger_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANNA_SCHEDULER_LEDGER", "off")
    assert _create_scheduler_ledger() is None
    monkeypatch.setenv("ANNA_SCHEDULER_LEDGER", "1")
    ledger = _create_scheduler_ledger()
    assert ledger is not None
    # Either the compiled Rust extension or the Python fallback must be live.
    assert ledger.register(7, 3)
    assert ledger.active_ids() == [7]


def _rust_ledger_class():
    try:
        from anna import _rust
    except Exception:
        return None
    return getattr(_rust, "SchedulerLedger", None)


@pytest.mark.skipif(_rust_ledger_class() is None, reason="anna._rust.SchedulerLedger requires rebuilding crates/anna-rust")
class TestRustLedger:
    def test_parity_with_python_fallback(self) -> None:
        rust_cls = _rust_ledger_class()
        rust = rust_cls()
        py = _PythonSchedulerLedger()
        for rid, cost in ((1, 100), (2, 50), (3, 30)):
            assert rust.register(rid, cost) == py.register(rid, cost)
        # Duplicate register must be rejected on both backends.
        assert rust.register(1, 999) is False
        assert py.register(1, 999) is False
        assert rust.mark_finished(1) == py.mark_finished(1)
        # Duplicate finish must be rejected on both backends.
        assert rust.mark_finished(1) is False
        assert py.mark_finished(1) is False
        rust.retain([3, 1])
        py.retain([3, 1])
        assert list(rust.active_ids()) == py.active_ids()
        assert list(rust.finished_ids()) == py.finished_ids()
        assert rust.active_cost() == py.active_cost()
        assert tuple(rust.stats()) == tuple(py.stats())
