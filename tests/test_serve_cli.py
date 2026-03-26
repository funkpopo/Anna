from __future__ import annotations

import pytest

from anna.cli.serve import parse_serve_settings


def test_parse_serve_settings_accepts_anna_arguments(tmp_path) -> None:
    model_dir = tmp_path / "models" / "Qwen3.5-2B"
    model_dir.mkdir(parents=True)

    settings = parse_serve_settings(
        [
            str(model_dir),
            "--model-name",
            "qwen3.5",
            "--scheduler-max-batch-size",
            "8",
            "--scheduler-max-batched-tokens",
            "4096",
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.85",
            "--api-key",
            "secret",
        ]
    )

    assert settings.model_dir == model_dir.resolve()
    assert settings.model_id == "qwen3.5"
    assert settings.scheduler_max_batch_size == 8
    assert settings.scheduler_max_batched_tokens == 4096
    assert settings.max_model_len == 32768
    assert settings.gpu_memory_utilization == 0.85
    assert settings.api_key == "secret"


def test_parse_serve_settings_rejects_removed_served_model_name_flag(tmp_path) -> None:
    model_dir = tmp_path / "models" / "Qwen3.5-2B"
    model_dir.mkdir(parents=True)

    with pytest.raises(SystemExit):
        parse_serve_settings(
            [
                str(model_dir),
                "--served-model-name",
                "anna-a",
            ]
        )


def test_parse_serve_settings_rejects_removed_max_num_seqs_flag(tmp_path) -> None:
    model_dir = tmp_path / "models" / "Qwen3.5-2B"
    model_dir.mkdir(parents=True)

    with pytest.raises(SystemExit):
        parse_serve_settings(
            [
                str(model_dir),
                "--max-num-seqs",
                "8",
            ]
        )


def test_parse_serve_settings_rejects_removed_tensor_parallel_flag(tmp_path) -> None:
    model_dir = tmp_path / "models" / "Qwen3.5-2B"
    model_dir.mkdir(parents=True)

    with pytest.raises(SystemExit):
        parse_serve_settings(
            [
                str(model_dir),
                "--tensor-parallel-size",
                "1",
            ]
        )


def test_parse_serve_settings_rejects_mismatched_model_dir_sources(tmp_path) -> None:
    model_a = tmp_path / "models" / "A"
    model_b = tmp_path / "models" / "B"
    model_a.mkdir(parents=True)
    model_b.mkdir(parents=True)

    with pytest.raises(SystemExit):
        parse_serve_settings(
            [
                str(model_a),
                "--model-dir",
                str(model_b),
            ]
        )
