from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from anna.core.native import (
    SafetensorsShardPlan,
    SafetensorsTensorEntry,
    inspect_safetensors_load_plan,
    inspect_safetensors_manifest,
)


def _install_fake_rust(monkeypatch, implementation) -> None:
    module = types.ModuleType("anna._rust")
    module.inspect_safetensors_manifest = implementation
    module.inspect_safetensors_load_plan = implementation
    monkeypatch.setitem(sys.modules, "anna._rust", module)


def test_inspect_safetensors_manifest_requires_rust(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delitem(sys.modules, "anna._rust", raising=False)

    with pytest.raises(ImportError):
        inspect_safetensors_manifest(tmp_path)


def test_inspect_safetensors_manifest_wraps_rust_result(monkeypatch, tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"abc")

    def fake_inspect(model_dir: str):
        assert model_dir == str(tmp_path)
        return [str(shard)], 3

    _install_fake_rust(monkeypatch, fake_inspect)

    files, total_bytes = inspect_safetensors_manifest(tmp_path)

    assert files == [shard]
    assert total_bytes == 3


def test_inspect_safetensors_load_plan_wraps_rust_result(monkeypatch, tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"abc")

    def fake_manifest(_model_dir: str):
        return [str(shard)], 3

    def fake_plan(model_dir: str):
        assert model_dir == str(tmp_path)
        return [
            (
                str(shard),
                3,
                128,
                [
                    ("a.weight", "BF16", [2, 3], 0, 12),
                    ("b.weight", "F32", [1, 3], 12, 24),
                ],
            )
        ], 3

    module = types.ModuleType("anna._rust")
    module.inspect_safetensors_manifest = fake_manifest
    module.inspect_safetensors_load_plan = fake_plan
    monkeypatch.setitem(sys.modules, "anna._rust", module)

    plans, total_bytes = inspect_safetensors_load_plan(tmp_path)

    assert plans == [
        SafetensorsShardPlan(
            path=shard,
            size_bytes=3,
            header_len=128,
            tensors=(
                SafetensorsTensorEntry(name="a.weight", dtype="BF16", shape=(2, 3), data_start=0, data_end=12),
                SafetensorsTensorEntry(name="b.weight", dtype="F32", shape=(1, 3), data_start=12, data_end=24),
            ),
        )
    ]
    assert plans[0].keys == ("a.weight", "b.weight")
    assert total_bytes == 3


def test_qwen_weight_files_use_native_manifest(monkeypatch, tmp_path: Path) -> None:
    from anna.weights import qwen3_5_text_weight_loader as loader

    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"abc")
    plan = [
        SafetensorsShardPlan(
            path=shard,
            size_bytes=123,
            header_len=128,
            tensors=(SafetensorsTensorEntry(name="a.weight", dtype="F32", shape=(1, 1), data_start=0, data_end=4),),
        )
    ]
    monkeypatch.setattr(loader, "inspect_safetensors_manifest", lambda model_dir: ([shard], 123))
    monkeypatch.setattr(loader, "inspect_safetensors_load_plan", lambda model_dir: (plan, 123))

    assert loader._iter_weight_files(tmp_path) == [shard]
    assert loader._load_plan(tmp_path) == (plan, 123)
    assert loader.estimate_qwen3_5_text_model_weight_bytes(tmp_path) == 123


def test_gemma_weight_files_use_native_manifest(monkeypatch, tmp_path: Path) -> None:
    from anna.weights import gemma4_text_weight_loader as loader

    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"abc")
    plan = [
        SafetensorsShardPlan(
            path=shard,
            size_bytes=123,
            header_len=128,
            tensors=(SafetensorsTensorEntry(name="a.weight", dtype="F32", shape=(1, 1), data_start=0, data_end=4),),
        )
    ]
    monkeypatch.setattr(loader, "inspect_safetensors_manifest", lambda model_dir: ([shard], 123))
    monkeypatch.setattr(loader, "inspect_safetensors_load_plan", lambda model_dir: (plan, 123))

    assert loader._iter_weight_files(tmp_path) == [shard]
    assert loader._load_plan(tmp_path) == (plan, 123)
    assert loader.estimate_gemma4_text_model_weight_bytes(tmp_path) == 123


def test_qwen_weight_files_propagates_native_errors(monkeypatch, tmp_path: Path) -> None:
    from anna.weights import qwen3_5_text_weight_loader as loader

    def raise_native(_model_dir: Path):
        raise RuntimeError("native required")

    monkeypatch.setattr(loader, "inspect_safetensors_manifest", raise_native)

    with pytest.raises(RuntimeError, match="native required"):
        loader._iter_weight_files(tmp_path)
