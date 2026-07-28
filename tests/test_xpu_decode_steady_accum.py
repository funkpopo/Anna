from __future__ import annotations

import logging

import pytest

from anna.model.xpu_decode_profile import (
    SchedulerDecodeSteadyAccum,
    SteadyDecodeAccum,
    _amortized_per_request_ms,
    record_steady_decode_step_if_applicable,
    reset_scheduler_decode_steady_accum,
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


def test_amortized_per_request_ms_divides_by_batch_size() -> None:
    assert _amortized_per_request_ms({"attention": 40.0, "moe": 20.0}, 2) == {
        "attention": 20.0,
        "moe": 10.0,
    }


def test_scheduler_decode_steady_skips_warmup_and_buckets_by_batch_size(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reset_scheduler_decode_steady_accum()
    accum = SchedulerDecodeSteadyAccum(warmup_batches=1)
    assert accum.record({"attention": 100.0}, batch_size=2) is False  # warmup
    assert accum.record({"attention": 40.0, "gated_delta": 20.0}, batch_size=2) is True
    assert accum.record({"attention": 80.0, "gated_delta": 40.0}, batch_size=4) is True
    assert accum.by_batch[2].step_count == 1
    assert accum.by_batch[4].step_count == 1
    assert accum.by_batch[2].totals["attention"] == 40.0

    caplog.set_level(logging.INFO)
    accum.log_summary(lg=logging.getLogger("anna.scheduler_steady_test"))
    messages = [r.message for r in caplog.records if "xpu_scheduler_decode_steady_avg_ms" in r.message]
    assert len(messages) == 2
    assert any("batch_size=2" in message for message in messages)
    assert any("batch_size=4" in message for message in messages)
    assert any("avg_ms_per_req" in message for message in messages)


def test_record_steady_routes_scheduler_decode_to_process_accum() -> None:
    reset_scheduler_decode_steady_accum()
    record_steady_decode_step_if_applicable(
        "scheduler_decode",
        {"attention": 30.0},
        batch_size=3,
    )
    # First call is warmup (default warmup_batches=1)
    record_steady_decode_step_if_applicable(
        "scheduler_decode",
        {"attention": 60.0},
        batch_size=3,
    )
    from anna.model.xpu_decode_profile import get_or_create_scheduler_decode_steady_accum

    accum = get_or_create_scheduler_decode_steady_accum(enabled=True)
    assert accum is not None
    assert accum.total_seen == 2
    assert accum.by_batch[3].step_count == 1
    assert accum.by_batch[3].totals["attention"] == 60.0
    reset_scheduler_decode_steady_accum()
