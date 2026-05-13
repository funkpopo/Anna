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
