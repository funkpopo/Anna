from __future__ import annotations

from pathlib import Path

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
