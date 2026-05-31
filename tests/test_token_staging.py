from __future__ import annotations

import torch

from anna.runtime.service_metrics import AnnaServiceMetrics
from anna.runtime.token_staging import stage_single_token_id_to_host, stage_token_ids_to_host


def test_stage_token_ids_to_host_records_one_cpu_sync_for_batch() -> None:
    metrics = AnnaServiceMetrics()

    staged = stage_token_ids_to_host(torch.tensor([[4], [5], [6]], dtype=torch.long), metrics=metrics)

    assert staged == [4, 5, 6]
    assert metrics.snapshot().cpu_sync_count == 1


def test_stage_single_token_id_to_host_reuses_batch_staging() -> None:
    metrics = AnnaServiceMetrics()

    staged = stage_single_token_id_to_host(torch.tensor([9], dtype=torch.long), metrics=metrics)

    assert staged == 9
    assert metrics.snapshot().cpu_sync_count == 1
