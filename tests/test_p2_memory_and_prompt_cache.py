"""P2 tests: adaptive memory release, scheduler prompt-cache wiring, event-driven prefill."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import torch

from anna.model.qwen3_5_text_config import Qwen3_5TextConfig, Qwen3_5TextModelConfig
from anna.runtime.memory_release import AdaptiveMemoryReleaser
from anna.runtime.qwen3_5_text_engine import (
    AnnaQwen3_5TextEngine,
    EngineOptimizationConfig,
    GenerationConfig,
)
from anna.runtime.scheduler import AnnaScheduler
from anna.mm.prepared_inputs import PreparedInputs

from tests.test_scheduler import _FakeDeviceContext, _FakeModel, _FakeTokenizer, _prepared


# ---------------------------------------------------------------------------
# P2-#6: adaptive memory releaser
# ---------------------------------------------------------------------------


def test_adaptive_memory_releaser_triggers_when_idle_and_below_floor() -> None:
    calls: list[dict[str, object]] = []
    state = {"idle": True, "free_bytes": 512 << 20, "reserved_bytes": 10 << 30, "allocated_bytes": 4 << 30}
    snapshot = dict(state)
    snapshot["min_free_bytes"] = 1 << 30

    def provider() -> dict[str, object]:
        return dict(snapshot)

    def release() -> None:
        calls.append(dict(snapshot))
        snapshot["free_bytes"] = 2 << 30
        snapshot["reserved_bytes"] = 4 << 30

    releaser = AdaptiveMemoryReleaser(snapshot_provider=provider, release_callback=release, interval_seconds=0)
    assert releaser.sweep_once() is True
    assert len(calls) == 1
    # Above the floor now: no further release.
    assert releaser.sweep_once() is False


def test_adaptive_memory_releaser_skips_busy_runtime() -> None:
    calls: list[dict[str, object]] = []
    releaser = AdaptiveMemoryReleaser(
        snapshot_provider=lambda: {"idle": False, "free_bytes": 1 << 20, "min_free_bytes": 1 << 30},
        release_callback=lambda: calls.append({}),
        interval_seconds=0,
    )
    assert releaser.sweep_once() is False
    assert calls == []


def test_adaptive_memory_releaser_env_overrides() -> None:
    import os

    os.environ["ANNA_MEMORY_RELEASE_SWEEP_INTERVAL_S"] = "0"
    os.environ["ANNA_MEMORY_RELEASE_MIN_FREE_MIB"] = "2048"
    try:
        releaser = AdaptiveMemoryReleaser(
            snapshot_provider=lambda: None,
            release_callback=lambda: None,
        )
        assert releaser.interval_seconds == 0.0
        assert releaser.min_free_bytes == 2048 << 20
        # interval 0 disables the background thread entirely.
        releaser.start()
        assert releaser._thread is None
    finally:
        os.environ.pop("ANNA_MEMORY_RELEASE_SWEEP_INTERVAL_S", None)
        os.environ.pop("ANNA_MEMORY_RELEASE_MIN_FREE_MIB", None)


# ---------------------------------------------------------------------------
# P2-#7: scheduler exact prompt-cache wiring
# ---------------------------------------------------------------------------


def _text_config() -> Qwen3_5TextModelConfig:
    return Qwen3_5TextModelConfig(
        text_config=Qwen3_5TextConfig(
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            vocab_size=16,
            eos_token_id=9,
            pad_token_id=0,
            cache_block_size=2,
            layer_types=["full_attention"],
        )
    )


def test_scheduler_prompt_cache_hit_skips_second_prefill() -> None:
    fake_model = _FakeModel(_text_config())
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
        optimization_config=EngineOptimizationConfig(prompt_cache_size=2, prompt_cache_max_tokens=0),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=2, batch_wait_ms=5.0)
    engine.set_scheduler(scheduler)
    config = GenerationConfig(max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0)
    try:
        first = scheduler._submit(_prepared([4, 5]), config=config, stream=False)
        assert first.done.wait(timeout=5.0)
        assert first.error is None
        assert fake_model.text_prefill_batch_sizes == [1]

        # Identical prompt: exact cache hit, no second prefill.
        second = scheduler._submit(_prepared([4, 5]), config=config, stream=False)
        assert second.done.wait(timeout=5.0)
        assert second.error is None
        assert fake_model.text_prefill_batch_sizes == [1]

        # Different prompt: still a real prefill.
        third = scheduler._submit(_prepared([6, 7]), config=config, stream=False)
        assert third.done.wait(timeout=5.0)
        assert third.error is None
        assert fake_model.text_prefill_batch_sizes == [1, 1]

        snapshot = engine.service_metrics_snapshot()
        assert snapshot.prompt_cache_queries_total == 3
        assert snapshot.prompt_cache_hits_total == 1
        assert snapshot.prompt_cache_entries >= 1
    finally:
        scheduler.shutdown()


def test_scheduler_prompt_cache_disabled_keeps_default_behaviour() -> None:
    fake_model = _FakeModel(_text_config())
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    scheduler = AnnaScheduler(engine, max_batch_size=1, batch_wait_ms=0.0)
    engine.set_scheduler(scheduler)
    config = GenerationConfig(max_new_tokens=1, temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0)
    try:
        for _ in range(2):
            request = scheduler._submit(_prepared([4, 5]), config=config, stream=False)
            assert request.done.wait(timeout=5.0)
            assert request.error is None
        assert fake_model.text_prefill_batch_sizes == [1, 1]
        snapshot = engine.service_metrics_snapshot()
        assert snapshot.prompt_cache_queries_total == 0
    finally:
        scheduler.shutdown()


# ---------------------------------------------------------------------------
# P2-#9: event-driven prefill insertion
# ---------------------------------------------------------------------------


def _bare_scheduler(**kwargs) -> AnnaScheduler:
    engine_stub = SimpleNamespace(_trim_runtime_cache_if_idle=lambda: None, metrics=None)
    scheduler = AnnaScheduler(engine_stub, max_batch_size=2, batch_wait_ms=0.0, **kwargs)
    return scheduler


def test_event_driven_prefill_insert_admits_pending_immediately() -> None:
    scheduler = _bare_scheduler(prefill_interval_steps=10, event_driven_prefill_insert=True)
    try:
        assert scheduler._should_run_prefill_step() is False  # nothing pending
        request = scheduler._submit(_prepared([1, 2, 3]), config=GenerationConfig(max_new_tokens=1), stream=False)
        assert scheduler._should_run_prefill_step() is True
        scheduler.cancel(request)
    finally:
        scheduler.shutdown()


def test_interval_based_prefill_still_gates_without_event_flag() -> None:
    scheduler = _bare_scheduler(prefill_interval_steps=10, event_driven_prefill_insert=False)
    try:
        request = scheduler._submit(_prepared([1, 2, 3]), config=GenerationConfig(max_new_tokens=1), stream=False)
        assert scheduler._should_run_prefill_step() is False
        scheduler._decode_steps_since_prefill = 10
        assert scheduler._should_run_prefill_step() is True
        scheduler.cancel(request)
    finally:
        scheduler.shutdown()


# ---------------------------------------------------------------------------
# P2-#7: serve CLI prompt-cache defaults
# ---------------------------------------------------------------------------


def test_serve_prompt_cache_defaults_turboquant(monkeypatch: pytest.MonkeyPatch) -> None:
    from anna.cli import serve as serve_cli

    monkeypatch.setattr(serve_cli, "turboquant_is_available", lambda: True)
    args = argparse.Namespace(prompt_cache_size=None, prompt_cache_max_tokens=None, kv_cache_quantization="auto")
    size, max_tokens, auto = serve_cli._resolve_prompt_cache_defaults(args)
    assert (size, max_tokens, auto) == (4, 2048, True)

    # Explicit user values always win.
    args = argparse.Namespace(prompt_cache_size=9, prompt_cache_max_tokens=128, kv_cache_quantization="turboquant")
    size, max_tokens, auto = serve_cli._resolve_prompt_cache_defaults(args)
    assert (size, max_tokens, auto) == (9, 128, False)


def test_serve_prompt_cache_defaults_paged_path_stays_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from anna.cli import serve as serve_cli

    monkeypatch.setattr(serve_cli, "turboquant_is_available", lambda: True)
    # kv_cache_quantization=none keeps the paged prefix-block path as the default reuse strategy.
    args = argparse.Namespace(prompt_cache_size=None, prompt_cache_max_tokens=None, kv_cache_quantization="none")
    size, max_tokens, auto = serve_cli._resolve_prompt_cache_defaults(args)
    assert (size, max_tokens, auto) == (0, 0, True)

    monkeypatch.setattr(serve_cli, "turboquant_is_available", lambda: False)
    args = argparse.Namespace(prompt_cache_size=None, prompt_cache_max_tokens=None, kv_cache_quantization="auto")
    assert serve_cli._resolve_prompt_cache_defaults(args) == (0, 0, True)


# ---------------------------------------------------------------------------
# P2-#7: KV usage stats report turboquant rows when paging is unused
# ---------------------------------------------------------------------------


def test_kv_cache_page_counts_reports_turboquant_rows() -> None:
    fake_model = _FakeModel(_text_config())
    engine = AnnaQwen3_5TextEngine(
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        processor=object(),
        model_id="fake",
        device_context=_FakeDeviceContext(),
    )
    from anna.model.ops import Qwen3DynamicCache
    cache = Qwen3DynamicCache(
        fake_model.config.text_config,
        allocator=engine.cache_allocator,
        batch_size=1,
        kv_cache_quantization="turboquant",
        kv_cache_quant_bits=4,
        kv_cache_residual_len=8,
    )
    # Simulate a turboquant row (quantized full-attention KV bypasses the pager).
    from anna.model.turboquant import TurboQuantKVRow

    cache.turboquant_rows[0][0] = TurboQuantKVRow(bits=4, residual_len=8)
    cache.layer_lengths[0][0] = 4
    used_pages, total_pages, rows, tokens = engine._kv_cache_page_counts()
    assert used_pages >= 0 and total_pages >= 0
    assert rows == 1
    assert tokens == 4
    cache.release()
    _, _, rows, tokens = engine._kv_cache_page_counts()
    assert rows == 0 and tokens == 0
