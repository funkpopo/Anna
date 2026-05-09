from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    return output or None


def _path_exists(value: str | None) -> bool:
    return bool(value) and Path(value).exists()


def collect_xpu_env() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "env": {
            key: os.environ.get(key)
            for key in (
                "ANNA_DPCPP",
                "ANNA_VCVARS64",
                "ANNA_ENABLE_INT4_LM_HEAD_TOPK_FUSED",
                "ANNA_XPU_INT4_MATMUL",
                "SYCL_DEVICE_FILTER",
                "ONEAPI_DEVICE_SELECTOR",
            )
        },
    }
    info["env_path_exists"] = {
        "ANNA_DPCPP": _path_exists(os.environ.get("ANNA_DPCPP")),
        "ANNA_VCVARS64": _path_exists(os.environ.get("ANNA_VCVARS64")),
    }
    info["tools"] = {
        "icx": shutil.which("icx"),
        "dpcpp": shutil.which("dpcpp"),
        "sycl-ls": shutil.which("sycl-ls"),
    }
    if info["tools"]["sycl-ls"]:
        info["sycl_ls"] = _run_command([info["tools"]["sycl-ls"]])

    try:
        import torch
    except Exception as exc:
        info["torch_error"] = repr(exc)
        return info

    info["torch"] = {
        "version": torch.__version__,
        "has_xpu": hasattr(torch, "xpu"),
        "xpu_available": bool(hasattr(torch, "xpu") and torch.xpu.is_available()),
    }
    if not hasattr(torch, "xpu"):
        return info

    try:
        device_count = int(torch.xpu.device_count())
    except Exception as exc:
        info["torch"]["xpu_device_count_error"] = repr(exc)
        device_count = 0
    info["torch"]["xpu_device_count"] = device_count
    devices: list[dict[str, Any]] = []
    for idx in range(device_count):
        device: dict[str, Any] = {"index": idx}
        try:
            device["name"] = torch.xpu.get_device_name(idx)
        except Exception as exc:
            device["name_error"] = repr(exc)
        try:
            props = torch.xpu.get_device_properties(idx)
            for key in (
                "name",
                "platform_name",
                "type",
                "driver_version",
                "total_memory",
                "max_compute_units",
                "gpu_eu_count",
                "gpu_subslice_count",
                "max_work_group_size",
                "max_num_sub_groups",
                "sub_group_sizes",
                "has_fp16",
                "has_fp64",
                "has_atomic64",
            ):
                value = getattr(props, key, None)
                if hasattr(value, "tolist"):
                    value = value.tolist()
                device[key] = value
            total_memory = getattr(props, "total_memory", None)
            if total_memory is not None:
                device["total_memory_mib"] = int(total_memory) // (1024 * 1024)
        except Exception as exc:
            device["properties_error"] = repr(exc)
        devices.append(device)
    info["torch"]["xpu_devices"] = devices
    info["anna_recommendations"] = {
        "dense_int4_backend": "torch_int4pack_default",
        "avoid": ["--xpu-int4-matmul sycl", "ANNA_XPU_INT4_GEMV_*"],
        "enable_for_qwen3_5": ["ANNA_ENABLE_INT4_LM_HEAD_TOPK_FUSED=1"],
    }
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect local Torch XPU hardware and Anna runtime environment.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    info = collect_xpu_env()
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return
    print(f"Python: {info.get('python')} ({info.get('executable')})")
    torch_info = info.get("torch", {})
    print(f"Torch: {torch_info.get('version', 'unavailable')}")
    print(f"XPU available: {torch_info.get('xpu_available', False)}")
    for device in torch_info.get("xpu_devices", []):
        print(
            "XPU {index}: {name}, driver={driver}, memory={memory} MiB, EUs={eus}, subgroups={subgroups}".format(
                index=device.get("index"),
                name=device.get("name"),
                driver=device.get("driver_version"),
                memory=device.get("total_memory_mib"),
                eus=device.get("gpu_eu_count"),
                subgroups=device.get("sub_group_sizes"),
            )
        )
    print("Environment:")
    for key, value in info.get("env", {}).items():
        print(f"  {key}={value}")
    print("Tool paths:")
    for key, value in info.get("tools", {}).items():
        print(f"  {key}={value}")
    print("Anna backend: dense int4 defaults to PyTorch int4pack; do not use retired SYCL GEMV knobs.")


if __name__ == "__main__":
    main()
