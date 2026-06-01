from __future__ import annotations

from typing import Any


def sampler_capability_report() -> dict[str, Any]:
    """Describe the current sampler backend and its optimized coverage."""

    return {
        "backend": "torch_tensor_fallback",
        "custom_xpu_kernel": False,
        "batch_params": True,
        "candidate_sampler": True,
        "candidate_penalty_overfetch": True,
        "candidate_penalty_overfetch_requires": {
            "top_k_gt": 0,
            "presence_penalty_gte": 0.0,
            "repetition_penalty_gte": 1.0,
        },
        "direct_prefill_candidates": True,
        "full_vocab_fallback_metric": "sampler_full_vocab_sort_count",
    }
