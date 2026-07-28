"""Gemma4 vs Qwen3.5 shared-ops parity inventory (P1-2.7).

Anna's Gemma4 runtime reuses the Qwen text engine shell (scheduler, sampling,
prompt cache, health metrics) but keeps a distinct KV layout (sliding window +
shared full-attention rows). This module is the single source of truth for what
is shared, what is deliberately different, and what remains out of scope.
"""

from __future__ import annotations

from typing import Any, Literal

ParityStatus = Literal["shared", "partial", "gemma_only", "qwen_only", "n_a"]


def gemma4_ops_parity_inventory() -> list[dict[str, Any]]:
    """Return a stable list of capability rows for docs, health, and tests."""
    return [
        {
            "area": "scheduler_continuous_batch",
            "status": "shared",
            "qwen": "AnnaScheduler + DynamicCache stack/split/compact",
            "gemma4": "Same scheduler via AnnaGemma4TextEngine inheritance; Gemma4DynamicCache stack/split/compact",
            "notes": "Serve enables continuous batching for both families when --scheduler-max-batch-size > 1.",
        },
        {
            "area": "dense_xpu_int4",
            "status": "shared",
            "qwen": "convert_module_linears_to_xpu_int4 + layout cache",
            "gemma4": "Same path in from_model_dir / weight_quant auto",
            "notes": "Vision tower skipped when offload_vision is set.",
        },
        {
            "area": "kv_turboquant",
            "status": "partial",
            "qwen": "TurboQuantKVRow + fused split-KV decode on full_attention / paged layers",
            "gemma4": "TurboQuantTensorRow on full_attention only; decode materializes then fused GQA",
            "notes": "Gemma sliding layers stay dense; no paged TurboQuant mirror.",
        },
        {
            "area": "paged_kv_allocator",
            "status": "qwen_only",
            "qwen": "Qwen3PageAllocator + paged_gqa_decode_fused + PrefixBlockPool",
            "gemma4": "NullCacheAllocator; per-request DynamicCache rows",
            "notes": "Paged pages do not map cleanly onto sliding-window + shared-KV Gemma layers.",
        },
        {
            "area": "prefix_block_cache",
            "status": "qwen_only",
            "qwen": "PrefixBlockPool hit/miss metrics",
            "gemma4": "Not wired (no page tables)",
            "notes": "Exact prompt cache still works for text-only Gemma prompts.",
        },
        {
            "area": "gqa_decode_fused",
            "status": "shared",
            "qwen": "single_token_gqa_decode unified entry",
            "gemma4": "run_gqa_decode_fused on full_attention single-token XPU decode",
            "notes": "Sliding attention stays on masked grouped_query_attention.",
        },
        {
            "area": "qk_norm_rope_rmsnorm_fused",
            "status": "shared",
            "qwen": "run_qk_norm_rotary_fused / run_rmsnorm_fused",
            "gemma4": "run_qk_norm_rotary_fused_ex / run_rmsnorm_fused_ex on XPU",
            "notes": None,
        },
        {
            "area": "gated_delta_network",
            "status": "qwen_only",
            "qwen": "GDN prefill/decode + FlashQLA + Arc strategy tables",
            "gemma4": "Architecture has no GDN linear-attention layers",
            "notes": "N/A for Gemma4 model family.",
        },
        {
            "area": "moe_expert_offload_int4",
            "status": "qwen_only",
            "qwen": "Sparse MoE offload, heat resident, grouped int4 MLP",
            "gemma4": "Dense decoder; expert_* knobs rejected",
            "notes": None,
        },
        {
            "area": "lm_head_int4_topk_fused",
            "status": "partial",
            "qwen": "forward_text_only_topk + XPU int4 LM head top-k",
            "gemma4": "Engine sampling path ready; model has no forward_text_only_topk yet",
            "notes": "Falls back to full logits via _compute_logits / tied embed.",
        },
        {
            "area": "prompt_cache",
            "status": "shared",
            "qwen": "Exact prompt KV reuse",
            "gemma4": "Inherited OrderedDict prompt cache",
            "notes": "Text-only prompts only.",
        },
        {
            "area": "chunked_prefill",
            "status": "shared",
            "qwen": "Token-chunked prefill with page reserve",
            "gemma4": "Token-chunked prefill with DynamicCache.reserve_sequence_capacity",
            "notes": "Auto chunk sizing is Gemma-specific (sliding + global head dims).",
        },
    ]


def gemma4_ops_parity_summary() -> dict[str, Any]:
    rows = gemma4_ops_parity_inventory()
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    shared_or_partial = [
        str(row["area"]) for row in rows if row["status"] in {"shared", "partial"}
    ]
    qwen_only = [str(row["area"]) for row in rows if row["status"] == "qwen_only"]
    return {
        "version": 1,
        "family": "gemma4",
        "compared_to": "qwen3_5_text",
        "counts": counts,
        "wired_or_partial": shared_or_partial,
        "intentionally_qwen_only": qwen_only,
        "items": rows,
    }
