from __future__ import annotations

import time

import torch

from anna.core.hotpath_events import record_moe_host_offset, record_moe_stage


def plan_moe_host_fallback(
    *,
    usage: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> tuple[list[int], torch.Tensor]:
    """CPU/debug-only MoE wave planning after XPU callers pass an explicit fallback gate."""

    stage_started_at = time.perf_counter()
    hit_experts = usage.nonzero(as_tuple=False).flatten()
    hit_expert_list = [int(expert_idx) for expert_idx in hit_experts.tolist()]
    if usage.device.type == "xpu":
        record_moe_host_offset("qwen3_moe_hit_experts_cpu")
        record_moe_stage("cpu_sync", time.perf_counter() - stage_started_at)
    if expert_offsets.device.type == "xpu":
        stage_started_at = time.perf_counter()
        expert_offsets_host = expert_offsets.to(device="cpu")
        record_moe_host_offset("qwen3_moe_expert_offsets_cpu")
        record_moe_stage("cpu_sync", time.perf_counter() - stage_started_at)
    else:
        expert_offsets_host = expert_offsets
    return hit_expert_list, expert_offsets_host
