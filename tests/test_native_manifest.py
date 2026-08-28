from __future__ import annotations

from pathlib import Path

import pytest

from anna.core import native
from anna.core.native import (
    SafetensorsShardPlan,
    SafetensorsTensorEntry,
    inspect_safetensors_load_plan,
    inspect_safetensors_manifest,
)


class _FakeRust:
    """Stands in for the compiled anna._rust extension module."""

    def __init__(self, manifest, load_plan) -> None:
        self._manifest = manifest
        self._load_plan = load_plan
        self.manifest_calls: list[str] = []
        self.load_plan_calls: list[str] = []

    def inspect_safetensors_manifest(self, model_dir: str):
        self.manifest_calls.append(model_dir)
        return self._manifest

    def inspect_safetensors_load_plan(self, model_dir: str):
        self.load_plan_calls.append(model_dir)
        return self._load_plan


def _install_fake_rust(monkeypatch, fake: _FakeRust) -> None:
    monkeypatch.setattr(native, "_rust_module", lambda: fake)


def test_inspect_safetensors_manifest_requires_rust(monkeypatch, tmp_path: Path) -> None:
    def _raise():
        raise ImportError("anna._rust is not available")

    monkeypatch.setattr(native, "_rust_module", _raise)

    with pytest.raises(ImportError):
        inspect_safetensors_manifest(tmp_path)


def test_inspect_safetensors_manifest_wraps_rust_result(monkeypatch, tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"abc")

    fake = _FakeRust(manifest=([str(shard)], 3), load_plan=None)
    _install_fake_rust(monkeypatch, fake)

    files, total_bytes = inspect_safetensors_manifest(tmp_path)

    assert files == [shard]
    assert total_bytes == 3
    assert fake.manifest_calls == [str(tmp_path)]


def test_inspect_safetensors_load_plan_wraps_rust_result(monkeypatch, tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"abc")

    raw_plan_entry = (
        str(shard),
        3,
        128,
        [
            (
                "model.layers.0.self_attn.q_proj.weight",
                "BF16",
                (8, 8),
                0,
                128,
            )
        ],
    )

    fake = _FakeRust(manifest=([str(shard)], 3), load_plan=([raw_plan_entry], 3))
    _install_fake_rust(monkeypatch, fake)

    plans, total_bytes = inspect_safetensors_load_plan(tmp_path)

    assert total_bytes == 3
    assert len(plans) == 1
    plan = plans[0]
    assert isinstance(plan, SafetensorsShardPlan)
    assert plan.path == shard
    assert plan.size_bytes == 3
    assert plan.header_len == 128
    assert plan.keys == ("model.layers.0.self_attn.q_proj.weight",)
    entry = plan.tensors[0]
    assert isinstance(entry, SafetensorsTensorEntry)
    assert entry.dtype == "BF16"
    assert entry.shape == (8, 8)
    assert (entry.data_start, entry.data_end) == (0, 128)
    assert fake.load_plan_calls == [str(tmp_path)]
