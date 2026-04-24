from __future__ import annotations

import logging

import pytest

from anna.model.xpu_decode_profile import (
    SteadyDecodeAccum,
    record_steady_decode_step_if_applicable,
    steady_decode_accumulation,
)


def test_steady_decode_accum_averages() -> None:
    acc = SteadyDecodeAccum()
    acc.add_step({"attention": 50.0, "gated_delta": 10.0})
    acc.add_step({"attention": 40.0, "gated_delta": 20.0})
    assert acc.step_count == 2
    assert acc.totals["attention"] == 90.0
    assert acc.totals["gated_delta"] == 30.0


def test_steady_excludes_decode1_and_non_decode(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    log = logging.getLogger("anna.steady_ctx_test")
    with steady_decode_accumulation(enabled=True, log=log):
        record_steady_decode_step_if_applicable("decode[1]", {"attention": 999.0})
        record_steady_decode_step_if_applicable("prefill", {"attention": 1.0})
        record_steady_decode_step_if_applicable("decode[2]", {"attention": 40.0, "gated_delta": 10.0})
        record_steady_decode_step_if_applicable("decode[3]", {"attention": 60.0, "gated_delta": 30.0})
    summary = next(r.message for r in caplog.records if "xpu_decode_steady_state_avg_ms_per_step" in r.message)
    assert "n=2" in summary
    assert "'attention': 50.0" in summary
    assert "'gated_delta': 20.0" in summary


def test_steady_session_logs_summary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    with steady_decode_accumulation(enabled=True, log=logging.getLogger("anna.steady_test")):
        record_steady_decode_step_if_applicable("decode[2]", {"attention": 10.0})
    assert any("xpu_decode_steady_state_avg_ms_per_step" in r.message for r in caplog.records)


def test_steady_session_disabled_no_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    with steady_decode_accumulation(enabled=False, log=logging.getLogger("anna.steady_test")):
        record_steady_decode_step_if_applicable("decode[2]", {"attention": 10.0})
    assert not any("xpu_decode_steady_state_avg_ms_per_step" in r.message for r in caplog.records)
