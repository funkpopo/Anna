from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import torch

import anna.cli.xpu_int4_cache as cache_cli
import anna.runtime.qwen3_5_text_engine as qwen_engine
from anna.runtime.device import DeviceMemoryInfo


def test_xpu_int4_cache_cli_reports_safetensors_auto_int4(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    model_dir = tmp_path / "safetensors-model"
    model_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": shard_name}}),
        encoding="utf-8",
    )
    (model_dir / shard_name).write_bytes(b"x" * (20 << 20))
    config = SimpleNamespace(
        quantization_config=None,
        text_config=SimpleNamespace(is_moe_model=False),
    )
    fake_context = SimpleNamespace(
        device=torch.device("xpu"),
        get_memory_info=lambda: DeviceMemoryInfo(
            free_bytes=16 << 30,
            total_bytes=16 << 30,
            allocated_bytes=0,
            reserved_bytes=0,
        ),
    )

    monkeypatch.setattr(cache_cli, "resolve_model_dir", lambda value: model_dir)
    monkeypatch.setattr(cache_cli, "load_qwen3_5_text_model_config", lambda _model_dir: config)
    monkeypatch.setattr(cache_cli, "estimate_qwen3_5_text_model_weight_bytes", lambda _model_dir: 20 << 30)
    monkeypatch.setattr(qwen_engine, "estimate_qwen3_5_text_model_weight_bytes", lambda _model_dir: 20 << 30)
    monkeypatch.setattr(cache_cli, "_device_context_for_memory", lambda _memory_gib: fake_context)

    monkeypatch.setattr(
        "sys.argv",
        [
            "anna-xpu-int4-cache",
            "--model-dir",
            str(model_dir),
            "--weight-quant",
            "auto",
            "--xpu-total-memory-gib",
            "16",
        ],
    )

    cache_cli.main()

    output = capsys.readouterr().out
    assert "uses_gguf=False" in output
    assert "uses_safetensors=True" in output
    assert "resolved_weight_quant=int4" in output
    assert "xpu_int4_cache_enabled=True" in output
    assert f"xpu_int4_cache_dir={model_dir / '.anna' / 'xpu_int4_cache'}" in output
