from __future__ import annotations

from contextlib import contextmanager
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


_DISABLE_ENV_VAR = "ANNA_DISABLE_XPU_FUSION"
_EXTENSION_NAME = "anna_xpu_fusion_ext"
_LOAD_LOCK = threading.Lock()
_LOAD_ATTEMPTED = False
_LOAD_ERROR: str | None = None

_ACTIVATION_IDS: dict[str | None, int] = {
    None: 0,
    "none": 0,
    "silu": 1,
    "swish": 1,
    "relu": 2,
    "gelu": 3,
}


@dataclass(slots=True)
class XPUFusionStatus:
    enabled: bool
    extension_loaded: bool
    reason: str
    torch_xpu_available: bool
    oneapi_compiler_on_path: bool
    msvc_on_path: bool


def _env_flag_is_true(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _torch_xpu_available() -> bool:
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def _native_source_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "native"


def _native_sources_exist() -> bool:
    source_dir = _native_source_dir()
    return (source_dir / "anna_onednn_extension.cpp").exists() and (source_dir / "anna_onednn_extension.sycl").exists()


def _repo_build_dir() -> Path:
    return Path(__file__).resolve().parents[3] / ".build" / _EXTENSION_NAME


def _candidate_paths(*values: str | Path | None) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        if value is None:
            continue
        raw = str(value).strip()
        if not raw:
            continue
        paths.append(Path(raw))
    return paths


def _discover_oneapi_root() -> Path | None:
    candidates = _candidate_paths(
        os.getenv("ANNA_ONEAPI_ROOT"),
        os.getenv("ONEAPI_ROOT"),
        r"D:\Intel\oneAPI",
        r"C:\Program Files (x86)\Intel\oneAPI",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _discover_oneapi_icx() -> Path | None:
    direct = shutil.which("icx")
    if direct:
        return Path(direct)

    root = _discover_oneapi_root()
    if root is None:
        return None

    known = [
        root / "compiler" / "latest" / "bin" / "icx.exe",
        root / "compiler" / "bin" / "icx.exe",
    ]
    for candidate in known:
        if candidate.exists():
            return candidate

    matches = sorted(root.glob("compiler/*/bin/icx.exe"), reverse=True)
    return matches[0] if matches else None


def _discover_oneapi_setvars() -> Path | None:
    root = _discover_oneapi_root()
    if root is None:
        return None
    candidates = [
        root / "compiler" / "latest" / "env" / "vars.bat",
        root / "setvars.bat",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _discover_msvc_root() -> Path | None:
    candidates = _candidate_paths(
        os.getenv("ANNA_MSVC_BUILD_TOOLS_ROOT"),
        os.getenv("VSINSTALLDIR"),
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    vctools = os.getenv("VCToolsInstallDir")
    if vctools:
        vctools_path = Path(vctools)
        if vctools_path.exists():
            return vctools_path
    return None


def _discover_msvc_cl() -> Path | None:
    direct = shutil.which("cl")
    if direct:
        return Path(direct)

    root = _discover_msvc_root()
    if root is None:
        return None
    if root.is_file() and root.name.lower() == "cl.exe":
        return root

    candidates = [root / "cl.exe"]
    if "VC" in root.parts and "MSVC" in root.parts:
        candidates.append(root / "bin" / "Hostx64" / "x64" / "cl.exe")
    else:
        candidates.extend(
            [
                root / "VC" / "Tools" / "MSVC" / "14.44.35207" / "bin" / "Hostx64" / "x64" / "cl.exe",
                root / "VC" / "Tools" / "MSVC" / "latest" / "bin" / "Hostx64" / "x64" / "cl.exe",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(root.glob("VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"), reverse=True)
    if matches:
        return matches[0]
    if root.name == "MSVC":
        versioned = sorted(root.glob("*/bin/Hostx64/x64/cl.exe"), reverse=True)
        if versioned:
            return versioned[0]
    return None


def _discover_vcvars64() -> Path | None:
    root = _discover_msvc_root()
    if root is None:
        return None
    candidates = [
        root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat",
        root.parent.parent.parent / "Auxiliary" / "Build" / "vcvars64.bat" if root.name == "MSVC" else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    matches = sorted(root.glob("VC/Auxiliary/Build/vcvars64.bat"))
    return matches[0] if matches else None


def _toolchain_on_path() -> tuple[bool, bool]:
    return _discover_oneapi_icx() is not None, _discover_msvc_cl() is not None


def _collect_batch_environment(batch_files: list[Path]) -> dict[str, str]:
    if not batch_files:
        return {}
    script_lines = ["@echo off"]
    script_lines.extend(f'call "{path}" >nul' for path in batch_files)
    script_lines.append("set")
    script_content = "\r\n".join(script_lines) + "\r\n"
    batch_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".bat", encoding="utf-8", delete=False) as handle:
            handle.write(script_content)
            batch_path = handle.name
        completed = subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", batch_path],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if batch_path is not None:
            Path(batch_path).unlink(missing_ok=True)
    environment: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        environment[key] = value
    return environment


@contextmanager
def _temporary_toolchain_environment():
    previous: dict[str, str | None] = {}
    try:
        previous.setdefault("VSLANG", os.environ.get("VSLANG"))
        os.environ["VSLANG"] = "1033"
        if "TORCH_XPU_ARCH_LIST" not in os.environ:
            previous.setdefault("TORCH_XPU_ARCH_LIST", None)
            os.environ["TORCH_XPU_ARCH_LIST"] = ""
        icx_path = _discover_oneapi_icx()
        cl_path = _discover_msvc_cl()
        batch_files = [path for path in (_discover_vcvars64(), _discover_oneapi_setvars()) if path is not None]
        if batch_files:
            try:
                captured = _collect_batch_environment(batch_files)
            except Exception as exc:
                warnings.warn(
                    f"Failed to import compiler environment from batch scripts; trying direct PATH injection instead: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                captured = {}
            for key, value in captured.items():
                previous.setdefault(key, os.environ.get(key))
                os.environ[key] = value

        tool_dirs: list[str] = []
        python_dir = str(Path(sys.executable).resolve().parent)
        scripts_dir = str(Path(sys.executable).resolve().parent / "Scripts")
        for extra_dir in (scripts_dir, python_dir):
            if Path(extra_dir).exists() and extra_dir not in tool_dirs:
                tool_dirs.append(extra_dir)
        for tool_path in (icx_path, cl_path):
            if tool_path is None:
                continue
            tool_dir = str(tool_path.parent)
            if tool_dir not in tool_dirs:
                tool_dirs.append(tool_dir)
        if tool_dirs:
            previous.setdefault("PATH", os.environ.get("PATH"))
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = os.pathsep.join(tool_dirs + ([current_path] if current_path else []))
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _temporary_cpp_extension_overrides():
    from torch.utils import cpp_extension

    previous_decode_args = cpp_extension.SUBPROCESS_DECODE_ARGS
    previous_sycl_home = cpp_extension.SYCL_HOME
    cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
    icx_path = _discover_oneapi_icx()
    if icx_path is not None:
        cpp_extension.SYCL_HOME = str(icx_path.parent.parent)
    try:
        yield
    finally:
        cpp_extension.SUBPROCESS_DECODE_ARGS = previous_decode_args
        cpp_extension.SYCL_HOME = previous_sycl_home


def _activation_id(activation: str | None) -> int:
    normalized = None if activation is None else activation.lower()
    activation_id = _ACTIVATION_IDS.get(normalized)
    if activation_id is None:
        raise ValueError(f"Unsupported XPU fusion activation: {activation}")
    return activation_id


def _apply_pointwise_fallback(
    output: torch.Tensor,
    *,
    activation: str | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = None if activation is None else activation.lower()
    if normalized in {None, "none"}:
        pass
    elif normalized in {"silu", "swish"}:
        output = F.silu(output)
    elif normalized == "relu":
        output = F.relu(output)
    elif normalized == "gelu":
        output = F.gelu(output)
    else:
        raise ValueError(f"Unsupported XPU fusion activation: {activation}")

    if residual is not None:
        if residual.device != output.device or residual.dtype != output.dtype:
            residual = residual.to(device=output.device, dtype=output.dtype)
        output = output + residual
    return output


def _op_is_registered() -> bool:
    namespace = getattr(torch.ops, "anna_xpu", None)
    return bool(namespace is not None and hasattr(namespace, "linear_pointwise"))


def describe_xpu_fusion_status() -> XPUFusionStatus:
    oneapi_on_path, msvc_on_path = _toolchain_on_path()
    if _env_flag_is_true(_DISABLE_ENV_VAR):
        reason = "disabled_by_env"
        enabled = False
    elif not _torch_xpu_available():
        reason = "xpu_unavailable"
        enabled = False
    elif not _native_sources_exist():
        reason = "native_sources_missing"
        enabled = False
    elif not oneapi_on_path:
        reason = "oneapi_compiler_missing"
        enabled = False
    elif not msvc_on_path:
        reason = "msvc_build_tools_missing"
        enabled = False
    elif _LOAD_ERROR is not None:
        reason = "extension_load_failed"
        enabled = False
    else:
        reason = "ready"
        enabled = True
    return XPUFusionStatus(
        enabled=enabled,
        extension_loaded=_op_is_registered(),
        reason=reason,
        torch_xpu_available=_torch_xpu_available(),
        oneapi_compiler_on_path=oneapi_on_path,
        msvc_on_path=msvc_on_path,
    )


def _load_extension_if_available() -> bool:
    global _LOAD_ATTEMPTED, _LOAD_ERROR

    if _op_is_registered():
        return True
    status = describe_xpu_fusion_status()
    if not status.enabled:
        return False
    if _LOAD_ATTEMPTED:
        return False

    with _LOAD_LOCK:
        if _op_is_registered():
            return True
        if _LOAD_ATTEMPTED:
            return False
        _LOAD_ATTEMPTED = True

        source_dir = _native_source_dir()
        build_dir = _repo_build_dir()
        build_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.cpp_extension import load

            extra_cflags = ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]
            extra_sycl_cflags = ["-std=c++20"]
            with _temporary_toolchain_environment(), _temporary_cpp_extension_overrides():
                load(
                    name=_EXTENSION_NAME,
                    sources=[
                        str(source_dir / "anna_onednn_extension.cpp"),
                        str(source_dir / "anna_onednn_extension.sycl"),
                    ],
                    build_directory=str(build_dir),
                    extra_cflags=extra_cflags,
                    extra_sycl_cflags=extra_sycl_cflags,
                    with_sycl=True,
                    is_python_module=False,
                    verbose=_env_flag_is_true("ANNA_VERBOSE_XPU_FUSION"),
                )
        except Exception as exc:
            _LOAD_ERROR = str(exc)
            warnings.warn(
                f"Failed to load Anna XPU fusion extension, falling back to eager PyTorch path: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False

        if not _op_is_registered():
            _LOAD_ERROR = "extension_loaded_but_operator_missing"
            warnings.warn(
                "Anna XPU fusion extension loaded without registering anna_xpu::linear_pointwise; using eager fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        return True


def linear_pointwise(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    activation_code = _activation_id(activation)
    if (
        input_tensor.device.type == "xpu"
        and weight.device.type == "xpu"
        and not _env_flag_is_true(_DISABLE_ENV_VAR)
        and _load_extension_if_available()
    ):
        return torch.ops.anna_xpu.linear_pointwise(
            input_tensor,
            weight,
            bias,
            residual,
            activation_code,
        )

    output = F.linear(
        input_tensor,
        weight.to(dtype=input_tensor.dtype),
        None if bias is None else bias.to(device=input_tensor.device, dtype=input_tensor.dtype),
    )
    return _apply_pointwise_fallback(output, activation=activation, residual=residual)


def apply_linear_pointwise(
    module: nn.Module,
    input_tensor: torch.Tensor,
    *,
    activation: str | None = None,
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        return linear_pointwise(
            input_tensor,
            module.weight,
            module.bias,
            activation=activation,
            residual=residual,
        )
    output = module(input_tensor)
    return _apply_pointwise_fallback(output, activation=activation, residual=residual)
