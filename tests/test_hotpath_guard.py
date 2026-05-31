from __future__ import annotations

from pathlib import Path

from anna.core.hotpath_events import (
    hotpath_event_recorder,
    record_attention_fallback,
    record_cpu_sync,
    record_moe_host_offset,
    record_paged_cache_materialization,
    record_sampler_full_vocab_sort,
)
from anna.runtime.hotpath_guard import scan_hotpath_files, unexpected_findings
from anna.runtime.service_metrics import AnnaServiceMetrics


def test_decode_hotpath_cpu_sync_findings_are_allowlisted() -> None:
    findings = scan_hotpath_files(root=Path.cwd())
    unexpected = unexpected_findings(findings)

    assert unexpected == []


def test_hotpath_event_context_records_service_metrics() -> None:
    metrics = AnnaServiceMetrics()

    with hotpath_event_recorder(metrics):
        record_cpu_sync("token_id_cpu_staging", count=2)
        record_attention_fallback("grouped_attention")
        record_paged_cache_materialization("gather_layer_cache")
        record_sampler_full_vocab_sort("top_p_full_logits_sort")
        record_moe_host_offset("expert_offsets_cpu")

    snapshot = metrics.snapshot()
    assert snapshot.cpu_sync_count == 2
    assert snapshot.attention_fallback_count == 1
    assert snapshot.paged_cache_materialize_count == 1
    assert snapshot.sampler_full_vocab_sort_count == 1
    assert snapshot.moe_host_offset_count == 1
