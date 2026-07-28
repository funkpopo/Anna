"""P2-2.10: optional Arc GDN validate gate for CI (skips without XPU / unless opted in)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATE_SCRIPT = REPO_ROOT / "tools" / "validate_arc_gdn_decode.py"


def _xpu_available() -> bool:
    return bool(getattr(torch, "xpu", None) is not None and torch.xpu.is_available())


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@pytest.mark.arc_gdn
def test_arc_gdn_validate_script_exists() -> None:
    assert VALIDATE_SCRIPT.is_file(), f"missing {VALIDATE_SCRIPT}"


@pytest.mark.arc_gdn
def test_arc_gdn_validate_script_help() -> None:
    """Smoke: the gate entrypoint parses and documents presets without hardware."""
    completed = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--presets" in completed.stdout
    assert "quick" in completed.stdout


@pytest.mark.arc_gdn
def test_arc_gdn_ci_gate_skips_without_xpu() -> None:
    """CI unit job: ensure the gate is skippable when no Intel XPU is present."""
    if _xpu_available():
        pytest.skip("XPU is available; full hardware gate is opt-in via ANNA_RUN_ARC_GDN_CI=1")
    assert not _xpu_available()


@pytest.mark.arc_gdn
@pytest.mark.skipif(not _xpu_available(), reason="Intel XPU required for Arc GDN validate gate")
@pytest.mark.skipif(
    not _env_flag("ANNA_RUN_ARC_GDN_CI"),
    reason="Set ANNA_RUN_ARC_GDN_CI=1 to run the live Arc GDN quick validate gate",
)
def test_arc_gdn_ci_gate_runs_quick_when_xpu() -> None:
    """Self-hosted / optional CI job: run the quick Arc GDN validate preset.

    Opt-in only: normal pytest on developer machines with XPU must not spend
    tens of minutes in the full quick bench suite.
    """
    if _env_flag("ANNA_SKIP_ARC_GDN_CI"):
        pytest.skip("ANNA_SKIP_ARC_GDN_CI set")

    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    python = shutil.which("python") or sys.executable
    cmd = [
        python,
        str(VALIDATE_SCRIPT),
        "--presets",
        "quick",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=int(os.getenv("ANNA_ARC_GDN_CI_TIMEOUT_SEC", "3600")),
    )
    if completed.returncode != 0:
        pytest.fail(
            "Arc GDN validate gate failed\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )
