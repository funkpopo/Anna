from __future__ import annotations

from typing import Protocol

import torch


class _CpuSyncRecorder(Protocol):
    def record_cpu_sync(self, reason: str = "", count: int = 1) -> None:
        ...


def stage_token_ids_to_host(
    token_ids: torch.Tensor,
    *,
    metrics: _CpuSyncRecorder | None = None,
    reason: str = "token_id_cpu_staging",
) -> list[int]:
    """Stage a compact token-id tensor to host in one explicit transfer."""
    if metrics is not None:
        metrics.record_cpu_sync(reason=reason)
    host_tokens = token_ids.detach().reshape(-1).to(device="cpu")
    return [int(token_id) for token_id in host_tokens.tolist()]


def stage_single_token_id_to_host(
    token_id: torch.Tensor,
    *,
    metrics: _CpuSyncRecorder | None = None,
    reason: str = "token_id_cpu_staging",
) -> int:
    staged = stage_token_ids_to_host(token_id, metrics=metrics, reason=reason)
    if not staged:
        raise ValueError("Expected at least one token id to stage.")
    return staged[0]
