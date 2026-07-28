from __future__ import annotations

import torch

from anna.model.gemma4_config import Gemma4Config
from anna.model.gemma4_ops_parity import gemma4_ops_parity_inventory, gemma4_ops_parity_summary
from anna.model.gemma4_text_model import Gemma4DynamicCache


def test_gemma4_ops_parity_inventory_covers_core_shared_areas() -> None:
    rows = gemma4_ops_parity_inventory()
    areas = {row["area"] for row in rows}
    assert "scheduler_continuous_batch" in areas
    assert "dense_xpu_int4" in areas
    assert "paged_kv_allocator" in areas
    assert "gqa_decode_fused" in areas
    summary = gemma4_ops_parity_summary()
    assert summary["family"] == "gemma4"
    assert summary["counts"]["shared"] >= 4
    assert "paged_kv_allocator" in summary["intentionally_qwen_only"]


def test_gemma4_dynamic_cache_compact_batch_rows_preserves_selected() -> None:
    config = Gemma4Config.from_dict(
        {
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "vocab_size": 128,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "global_head_dim": 16,
                "hidden_size_per_layer_input": 4,
                "vocab_size_per_layer_input": 128,
                "sliding_window": 8,
                "layer_types": ["sliding_attention", "full_attention"],
                "rope_parameters": {
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "full_attention": {
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "original_max_position_embeddings": 8,
                        "factor": 2.0,
                    },
                },
            },
        }
    )
    rows = []
    for _ in range(3):
        row = Gemma4DynamicCache(config.text_config, batch_size=1)
        key = torch.randn(1, 2, 3, 8)
        value = torch.randn(1, 2, 3, 8)
        row.update(key, value, layer_idx=0)
        row.advance_sequence(3)
        rows.append(row)
    stacked = Gemma4DynamicCache.stack(rows, config.text_config)
    assert stacked.get_batch_size() == 3

    compacted = stacked.compact_batch_rows([0, 2], release_unselected=True)
    assert compacted.get_batch_size() == 2
    assert compacted.request_lengths == [3, 3]
    assert stacked._released is True


def test_gemma4_kv_auto_resolves_via_shared_turboquant_helper(monkeypatch) -> None:
    from anna.runtime.gemma4_text_engine import AnnaGemma4TextEngine

    monkeypatch.setattr(
        "anna.runtime.gemma4_text_engine.turboquant_is_available",
        lambda: True,
    )
    mode, bits, residual, preset = AnnaGemma4TextEngine._resolve_kv_cache_quantization(
        requested_mode="auto",
        requested_bits=4,
        requested_residual_len=128,
        bits_explicit=False,
        residual_explicit=False,
        weight_bytes=2 * (1 << 30),
    )
    assert mode == "turboquant"
    assert bits in {2, 3, 4}
    assert residual >= 1
    assert preset in {"small", "medium", "large", "xlarge"}
