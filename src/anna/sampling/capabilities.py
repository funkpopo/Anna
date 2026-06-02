from __future__ import annotations

from typing import Any


def sampler_capability_report() -> dict[str, Any]:
    """Describe the current sampler backend and its optimized coverage."""

    return {
        "backend": "torch_tensor_fallback",
        "custom_xpu_kernel": False,
        "xpu_kernel_ready": False,
        "xpu_kernel_reason": "custom_xpu_sampler_kernel_not_implemented",
        "batch_params": True,
        "batch_params_cache": True,
        "batch_params_cache_benchmark": "sampling_params_cache",
        "candidate_sampler": True,
        "candidate_sampler_coverage": {
            "top_k": True,
            "top_k_one_deterministic": True,
            "top_k_top_p_min_p": True,
            "positive_penalty_overfetch": True,
        },
        "candidate_penalty_overfetch": True,
        "candidate_penalty_overfetch_requires": {
            "top_k_gt": 0,
            "presence_penalty_gte": 0.0,
            "repetition_penalty_gte": 1.0,
        },
        "direct_prefill_candidates": True,
        "full_vocab_fallback_metric": "sampler_full_vocab_fallback_count",
        "legacy_full_vocab_sort_metric": "sampler_full_vocab_sort_count",
        "full_vocab_fallback_reasons": (
            "top_p_full_logits_sort",
            "min_p_full_logits_softmax",
            "plain_full_logits_multinomial",
        ),
        "full_vocab_fallback_requires_xpu_kernel": (
            "top_p_full_logits_sort",
            "min_p_full_logits_softmax",
            "plain_full_logits_multinomial",
        ),
    }
