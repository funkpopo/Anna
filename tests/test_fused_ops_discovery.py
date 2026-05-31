from __future__ import annotations

from pathlib import Path

import pytest
import torch

from anna.model import fused_ops


def test_default_library_candidates_include_current_workdir_build(
    monkeypatch,
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / ".build" / "anna_gated_delta_fused"
    build_dir.mkdir(parents=True)
    library = build_dir / "anna_gated_delta_fused.pyd"
    library.write_bytes(b"fake")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANNA_PROJECT_ROOT", raising=False)

    candidates = fused_ops._default_library_candidates()

    assert str(library) in candidates


def test_default_library_candidates_include_project_root_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    build_dir = project_root / ".build" / "anna_gated_delta_fused"
    build_dir.mkdir(parents=True)
    library = build_dir / "anna_gated_delta_fused.pyd"
    library.write_bytes(b"fake")
    monkeypatch.setenv("ANNA_PROJECT_ROOT", str(project_root))

    candidates = fused_ops._default_library_candidates()

    assert str(library) in candidates


def test_fused_op_health_report_exposes_key_kernel_availability(monkeypatch) -> None:
    monkeypatch.setattr(fused_ops, "_paged_gqa_decode_op", lambda: object())
    monkeypatch.setattr(fused_ops, "_lm_head_int4_topk_op", lambda: object())
    monkeypatch.setattr(fused_ops, "_moe_grouped_int4_mlp_op", lambda: object())
    monkeypatch.setenv("ANNA_GATED_DELTA_OP_LIB", "D:/fake/anna_gated_delta_fused.pyd")

    report = fused_ops.fused_op_health_report()

    assert report["env"]["ANNA_GATED_DELTA_OP_LIB"] == "D:/fake/anna_gated_delta_fused.pyd"
    assert report["available"]["paged_gqa_decode_fused"] is True
    assert report["available"]["lm_head_int4_topk_fused"] is True
    assert report["available"]["moe_grouped_int4_mlp_fused"] is True
    assert "rmsnorm_fused" in report["available"]
    assert "qk_norm_rotary_fused" in report["available"]


@pytest.mark.parametrize(
    ("op_attr", "call", "op_name"),
    [
        (
            "_paged_gqa_decode_op",
            lambda: fused_ops.run_paged_gqa_decode_fused(
                query=torch.empty(1, 1, 1, 4),
                key_pages=torch.empty(2, 1, 2, 4),
                value_pages=torch.empty(2, 1, 2, 4),
                page_table=torch.zeros(1, 1, dtype=torch.int32),
                visible_lengths=torch.ones(1, dtype=torch.int32),
                scaling=1.0,
            ),
            "paged_gqa_decode_fused",
        ),
        (
            "_lm_head_topk_op",
            lambda: fused_ops.run_lm_head_topk_fused(
                hidden_states=torch.empty(1, 4),
                weight=torch.empty(8, 4),
                top_k=2,
            ),
            "lm_head_topk_fused",
        ),
        (
            "_lm_head_int4_topk_op",
            lambda: fused_ops.run_lm_head_int4_topk_fused(
                hidden_states=torch.empty(1, 4),
                qweight=torch.empty(8, 1, dtype=torch.int32),
                qscale=torch.empty(8, 1),
                qzeros=torch.empty(8, 1),
                group_size=4,
                in_features=4,
                top_k=2,
            ),
            "lm_head_int4_topk_fused",
        ),
        (
            "_moe_grouped_int4_mlp_op",
            lambda: fused_ops.run_moe_grouped_int4_mlp_fused(
                compact_hidden_states=torch.empty(1, 4),
                compact_routing_weights=torch.empty(1),
                compact_outputs=torch.empty(1, 4),
                expert_offsets=torch.zeros(2, dtype=torch.int32),
                active_experts=torch.zeros(1, dtype=torch.int32),
                active_slots=torch.zeros(1, dtype=torch.int32),
                gate_qweight=torch.empty(1, 1, dtype=torch.int32),
                gate_qscale=torch.empty(1, 1),
                gate_qzeros=torch.empty(1, 1),
                up_qweight=torch.empty(1, 1, dtype=torch.int32),
                up_qscale=torch.empty(1, 1),
                up_qzeros=torch.empty(1, 1),
                down_qweight=torch.empty(1, 1, dtype=torch.int32),
                down_qscale=torch.empty(1, 1),
                down_qzeros=torch.empty(1, 1),
                group_size=4,
                max_routes_per_expert=1,
            ),
            "moe_grouped_int4_mlp_fused",
        ),
        (
            "_rmsnorm_op",
            lambda: fused_ops.run_rmsnorm_fused(
                input=torch.empty(1, 4),
                weight=torch.empty(4),
                eps=1e-6,
            ),
            "rmsnorm_fused",
        ),
        (
            "_qk_norm_rotary_op",
            lambda: fused_ops.run_qk_norm_rotary_fused(
                query=torch.empty(1, 1, 1, 4),
                key=torch.empty(1, 1, 1, 4),
                query_norm_weight=torch.empty(4),
                key_norm_weight=torch.empty(4),
                cos=torch.empty(1, 1, 4),
                sin=torch.empty(1, 1, 4),
                query_norm_eps=1e-6,
                key_norm_eps=1e-6,
            ),
            "qk_norm_rotary_fused",
        ),
    ],
)
def test_key_fused_wrappers_raise_clear_error_without_registered_ops(
    monkeypatch: pytest.MonkeyPatch,
    op_attr: str,
    call,
    op_name: str,
) -> None:
    monkeypatch.setattr(fused_ops, op_attr, lambda: None)
    monkeypatch.setattr(fused_ops, "maybe_load_gated_delta_library", lambda *args, **kwargs: False)

    with pytest.raises(RuntimeError) as exc_info:
        call()

    message = str(exc_info.value)
    assert f"Anna {op_name} op is not registered" in message
    assert "Build/load the custom op first" in message
    assert "ANNA_GATED_DELTA_OP_LIB" in message
