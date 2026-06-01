from __future__ import annotations

from pathlib import Path

from anna.core.hotpath_events import (
    hotpath_event_recorder,
    record_attention_fallback,
    record_cpu_sync,
    record_moe_host_offset,
    record_moe_stage,
    record_paged_cache_materialization,
    record_sampler_full_vocab_sort,
)
from anna.runtime.hotpath_guard import scan_hotpath_file, scan_hotpath_files, unexpected_findings
from anna.runtime.hotpath_guard import DEFAULT_ALLOWLIST, summarize_findings
from anna.runtime.service_metrics import AnnaServiceMetrics


def test_decode_hotpath_cpu_sync_findings_are_allowlisted() -> None:
    findings = scan_hotpath_files(root=Path.cwd())
    unexpected = unexpected_findings(findings)

    assert unexpected == []
    assert summarize_findings(findings) == DEFAULT_ALLOWLIST


def test_hotpath_guard_detects_direct_cpu_calls(tmp_path: Path) -> None:
    source = tmp_path / "hotpath_sample.py"
    source.write_text(
        "\n".join(
            [
                "def decode_step(tensor):",
                "    a = tensor.cpu()",
                "    b = tensor.to('cpu')",
                "    c = tensor.to(device='cpu')",
                "    d = tensor.to(device=torch.device('cpu'))",
                "    return a, b, c, d",
            ]
        ),
        encoding="utf-8",
    )

    findings = scan_hotpath_file(source, root=tmp_path)

    assert [(finding.scope, finding.op) for finding in findings] == [
        ("decode_step", "cpu"),
        ("decode_step", "to_cpu"),
        ("decode_step", "to_cpu"),
        ("decode_step", "to_cpu"),
    ]


def test_hotpath_guard_detects_tensor_bool_reductions(tmp_path: Path) -> None:
    source = tmp_path / "hotpath_bool_reduce.py"
    source.write_text(
        "\n".join(
            [
                "import torch",
                "def decode_step(tensor):",
                "    if bool(torch.any(tensor < 0)):",
                "        return 1",
                "    if bool(torch.all(tensor > 0)):",
                "        return 2",
                "    return 0",
            ]
        ),
        encoding="utf-8",
    )

    findings = scan_hotpath_file(source, root=tmp_path)

    assert [(finding.scope, finding.op) for finding in findings] == [
        ("decode_step", "tensor_bool_reduce"),
        ("decode_step", "tensor_bool_reduce"),
    ]


def test_hotpath_guard_detects_implicit_tensor_bool_conditions(tmp_path: Path) -> None:
    source = tmp_path / "hotpath_implicit_bool_reduce.py"
    source.write_text(
        "\n".join(
            [
                "import torch",
                "def decode_step(tensor):",
                "    if torch.any(tensor < 0):",
                "        return 1",
                "    while tensor.all():",
                "        break",
                "    assert tensor.any()",
                "    return 2 if torch.all(tensor > 0) else 0",
            ]
        ),
        encoding="utf-8",
    )

    findings = scan_hotpath_file(source, root=tmp_path)

    assert [(finding.scope, finding.op) for finding in findings] == [
        ("decode_step", "tensor_bool_reduce"),
        ("decode_step", "tensor_bool_reduce"),
        ("decode_step", "tensor_bool_reduce"),
        ("decode_step", "tensor_bool_reduce"),
    ]


def test_hotpath_guard_detects_torch_is_nonzero(tmp_path: Path) -> None:
    source = tmp_path / "hotpath_is_nonzero.py"
    source.write_text(
        "\n".join(
            [
                "import torch",
                "def decode_step(tensor):",
                "    if torch.is_nonzero(tensor):",
                "        return 1",
                "    return 2 if tensor.is_nonzero() else 0",
            ]
        ),
        encoding="utf-8",
    )

    findings = scan_hotpath_file(source, root=tmp_path)

    assert [(finding.scope, finding.op) for finding in findings] == [
        ("decode_step", "torch_is_nonzero"),
        ("decode_step", "torch_is_nonzero"),
    ]


def test_hotpath_guard_skips_tensor_bool_reductions_inside_cpu_guards(tmp_path: Path) -> None:
    source = tmp_path / "hotpath_cpu_guard.py"
    source.write_text(
        "\n".join(
            [
                "import torch",
                "def validate_cpu_only(tensor, device):",
                "    if device.type == 'cpu' and bool(torch.any(tensor < 0)):",
                "        raise ValueError('negative')",
                "    if device.type == 'cpu':",
                "        if bool(torch.all(tensor == 0)):",
                "            raise ValueError('zero')",
                "        if torch.is_nonzero(tensor[0]):",
                "            raise ValueError('nonzero')",
            ]
        ),
        encoding="utf-8",
    )

    assert scan_hotpath_file(source, root=tmp_path) == []


def test_hotpath_guard_requires_cpu_guard_to_short_circuit_first(tmp_path: Path) -> None:
    source = tmp_path / "hotpath_reversed_cpu_guard.py"
    source.write_text(
        "\n".join(
            [
                "import torch",
                "def validate_cpu_only(tensor, device):",
                "    if bool(torch.any(tensor < 0)) and device.type == 'cpu':",
                "        raise ValueError('negative')",
            ]
        ),
        encoding="utf-8",
    )

    findings = scan_hotpath_file(source, root=tmp_path)

    assert [(finding.scope, finding.op) for finding in findings] == [
        ("validate_cpu_only", "tensor_bool_reduce"),
    ]


def test_hotpath_event_context_records_service_metrics() -> None:
    metrics = AnnaServiceMetrics()

    with hotpath_event_recorder(metrics):
        record_cpu_sync("token_id_cpu_staging", count=2)
        record_attention_fallback("grouped_attention")
        record_paged_cache_materialization("gather_layer_cache")
        record_sampler_full_vocab_sort("top_p_full_logits_sort")
        record_moe_host_offset("expert_offsets_cpu")
        record_moe_stage("router", 0.001)
        record_moe_stage("dispatch", 0.002)
        record_moe_stage("expert_gemm", 0.003)
        record_moe_stage("scatter", 0.004)
        record_moe_stage("staging", 0.005)
        record_moe_stage("cpu_sync", 0.006)

    snapshot = metrics.snapshot()
    assert snapshot.cpu_sync_count == 2
    assert snapshot.attention_fallback_count == 1
    assert snapshot.paged_cache_materialize_count == 1
    assert snapshot.sampler_full_vocab_sort_count == 1
    assert snapshot.moe_host_offset_count == 1
    assert snapshot.moe_router_count == 1
    assert snapshot.moe_dispatch_count == 1
    assert snapshot.moe_expert_gemm_count == 1
    assert snapshot.moe_scatter_count == 1
    assert snapshot.moe_staging_count == 1
    assert snapshot.moe_cpu_sync_count == 1
