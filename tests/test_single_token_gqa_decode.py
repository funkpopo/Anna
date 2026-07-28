from __future__ import annotations

import pytest
import torch

import anna.model.ops as model_ops
from anna.model.ops import single_token_gqa_decode


class _FakeCache:
    def __init__(self) -> None:
        self.turboquant_calls = 0
        self.paged_calls = 0
        self.gather_calls = 0

    def turboquant_single_token_decode_attention(self, *args, **kwargs):
        self.turboquant_calls += 1
        query = args[1]
        return torch.zeros_like(query)

    def paged_attention_state(self, layer_idx: int):
        self.paged_calls += 1
        del layer_idx
        key_pages = torch.zeros(1, 1, 16, 8)
        value_pages = torch.zeros(1, 1, 16, 8)
        page_table = torch.zeros(1, 1, dtype=torch.long)
        visible = torch.ones(1, dtype=torch.long)
        return key_pages, value_pages, page_table, visible

    def _gather_layer_cache(self, layer_idx: int):
        self.gather_calls += 1
        del layer_idx
        key = torch.zeros(1, 1, 4, 8)
        value = torch.zeros(1, 1, 4, 8)
        lengths = torch.zeros(1, dtype=torch.long)
        return key, value, lengths


def test_single_token_gqa_decode_prefers_turboquant(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _FakeCache()
    query = torch.zeros(1, 2, 1, 8)
    gate = torch.zeros(1, 1, 16)
    past_lengths = torch.zeros(1, dtype=torch.long)
    called = {"paged": 0, "dense": 0}

    def _paged(*_args, **_kwargs):
        called["paged"] += 1
        return torch.zeros_like(query)

    def _dense(*_args, **_kwargs):
        called["dense"] += 1
        return torch.zeros_like(query)

    monkeypatch.setattr(model_ops, "paged_kv_single_token_decode_attention", _paged)
    monkeypatch.setattr(model_ops, "materialized_kv_single_token_decode_attention", _dense)

    out = single_token_gqa_decode(
        query_states=query,
        scaling=1.0,
        num_key_value_groups=2,
        gate=gate,
        past_key_values=cache,  # type: ignore[arg-type]
        layer_idx=0,
        past_lengths=past_lengths,
        key_states=None,
        value_states=None,
        prefer_paged_decode=True,
        use_turboquant_cache=True,
    )
    assert out is not None
    assert cache.turboquant_calls == 1
    assert called["paged"] == 0
    assert called["dense"] == 0


def test_single_token_gqa_decode_uses_paged_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _FakeCache()
    query = torch.zeros(1, 2, 1, 8)
    gate = torch.zeros(1, 1, 16)
    past_lengths = torch.zeros(1, dtype=torch.long)
    called = {"paged": 0}

    def _paged(*_args, **_kwargs):
        called["paged"] += 1
        return torch.zeros_like(query)

    monkeypatch.setattr(model_ops, "paged_kv_single_token_decode_attention", _paged)

    out = single_token_gqa_decode(
        query_states=query,
        scaling=1.0,
        num_key_value_groups=2,
        gate=gate,
        past_key_values=cache,  # type: ignore[arg-type]
        layer_idx=0,
        past_lengths=past_lengths,
        key_states=None,
        value_states=None,
        prefer_paged_decode=True,
        use_turboquant_cache=False,
    )
    assert out is not None
    assert called["paged"] == 1
    assert cache.paged_calls == 1


def test_single_token_gqa_decode_returns_none_for_multi_token() -> None:
    cache = _FakeCache()
    query = torch.zeros(1, 2, 3, 8)
    out = single_token_gqa_decode(
        query_states=query,
        scaling=1.0,
        num_key_value_groups=2,
        gate=None,
        past_key_values=cache,  # type: ignore[arg-type]
        layer_idx=0,
        past_lengths=torch.zeros(1, dtype=torch.long),
        key_states=None,
        value_states=None,
        prefer_paged_decode=False,
        use_turboquant_cache=False,
    )
    assert out is None
