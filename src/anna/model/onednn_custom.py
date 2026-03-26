from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import torch


_EXTENSION_NAME = "anna_onednn_xpu_ext"
_BOOTSTRAP_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()
_BOOTSTRAPPED = False
_BOOTSTRAP_ERROR: str | None = None
_EXTENSION_MODULE: ModuleType | None = None
_EXTENSION_ERROR: str | None = None
_DLL_DIRECTORY_HANDLES: list[object] = []
_DLL_DIRECTORY_PATHS: set[str] = set()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _oneapi_root() -> Path:
    return Path(os.getenv("ANNA_ONEAPI_ROOT", r"D:\Intel\oneAPI"))


def _compiler_root() -> Path | None:
    root = _oneapi_root()
    latest = root / "compiler" / "latest"
    if latest.exists():
        return latest
    compiler_dir = root / "compiler"
    if not compiler_dir.exists():
        return None
    candidates = sorted(
        [path for path in compiler_dir.iterdir() if path.is_dir() and (path / "bin" / "dpcpp.exe").exists()],
        reverse=True,
    )
    return None if not candidates else candidates[0]


def _vs2022_install_dir() -> Path | None:
    configured = os.getenv("ANNA_VS2022INSTALLDIR")
    if configured:
        path = Path(configured)
        return path if path.exists() else None
    default = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools")
    return default if default.exists() else None


def _bootstrap_build_environment() -> bool:
    global _BOOTSTRAPPED, _BOOTSTRAP_ERROR
    if _BOOTSTRAPPED:
        return True
    if _BOOTSTRAP_ERROR is not None:
        return False

    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAPPED:
            return True
        if _BOOTSTRAP_ERROR is not None:
            return False

        setvars = _oneapi_root() / "setvars.bat"
        compiler_root = _compiler_root()
        if not setvars.exists():
            _BOOTSTRAP_ERROR = f"Missing oneAPI environment script: {setvars}"
            return False
        if compiler_root is None:
            _BOOTSTRAP_ERROR = f"Could not locate oneAPI compiler under {_oneapi_root()}"
            return False

        env = os.environ.copy()
        vs2022 = _vs2022_install_dir()
        if vs2022 is not None:
            env["VS2022INSTALLDIR"] = str(vs2022)

        try:
            proc = subprocess.run(
                ["cmd", "/d", "/s", "/c", f"call {setvars} >nul && set"],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            stdout = exc.stdout.strip() if exc.stdout else ""
            details = stderr or stdout or str(exc)
            _BOOTSTRAP_ERROR = f"Failed to bootstrap oneAPI build environment: {details}"
            return False

        for line in proc.stdout.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key] = value

        path_parts = [os.environ.get("PATH", "")]
        for extra in (
            Path(sys.prefix) / "Library" / "bin",
            Path(sys.prefix) / "Scripts",
        ):
            if extra.exists():
                path_parts.insert(0, str(extra))
        os.environ["PATH"] = ";".join(part for part in path_parts if part)
        os.environ.setdefault("TORCH_DONT_CHECK_COMPILER_ABI", "1")
        _BOOTSTRAPPED = True
        return True


def _extension_sources() -> tuple[Path, Path]:
    native_dir = _repo_root() / "src" / "anna" / "native"
    return (
        native_dir / "anna_onednn_extension.cpp",
        native_dir / "anna_onednn_extension.sycl",
    )


def _register_dll_directory(path: Path) -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if not path.exists():
        return
    resolved = path.resolve()
    resolved_str = str(resolved)
    if resolved_str in _DLL_DIRECTORY_PATHS:
        return
    handle = os.add_dll_directory(str(resolved))
    _DLL_DIRECTORY_HANDLES.append(handle)
    _DLL_DIRECTORY_PATHS.add(resolved_str)


def _load_extension() -> ModuleType | None:
    global _EXTENSION_MODULE, _EXTENSION_ERROR
    if _EXTENSION_MODULE is not None:
        return _EXTENSION_MODULE
    if _EXTENSION_ERROR is not None:
        return None
    if os.getenv("ANNA_DISABLE_ONEDNN_CUSTOM_OP", "").strip() == "1":
        _EXTENSION_ERROR = "Disabled by ANNA_DISABLE_ONEDNN_CUSTOM_OP=1."
        return None
    if not torch.xpu._is_compiled():
        _EXTENSION_ERROR = "PyTorch was not built with XPU support."
        return None

    with _LOAD_LOCK:
        if _EXTENSION_MODULE is not None:
            return _EXTENSION_MODULE
        if _EXTENSION_ERROR is not None:
            return None
        if not _bootstrap_build_environment():
            _EXTENSION_ERROR = _BOOTSTRAP_ERROR or "Failed to bootstrap oneAPI build environment."
            return None

        try:
            import torch.utils.cpp_extension as cpp_extension

            compiler_root = _compiler_root()
            if compiler_root is None:
                _EXTENSION_ERROR = f"Could not locate oneAPI compiler under {_oneapi_root()}"
                return None

            cpp_extension.SYCL_HOME = str(compiler_root)
            cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")
            source_cpp, source_sycl = _extension_sources()
            build_dir = _repo_root() / ".build" / _EXTENSION_NAME
            build_dir.mkdir(parents=True, exist_ok=True)
            ext_lib_dir = Path(sys.prefix) / "Library" / "lib"
            for dll_dir in (
                build_dir,
                Path(sys.prefix) / "Library" / "bin",
                Path(torch.__file__).resolve().parent / "lib",
                compiler_root / "bin",
            ):
                _register_dll_directory(dll_dir)

            _EXTENSION_MODULE = cpp_extension.load(
                name=_EXTENSION_NAME,
                sources=[str(source_cpp), str(source_sycl)],
                with_sycl=True,
                build_directory=str(build_dir),
                extra_sycl_cflags=["-std=c++20"],
                extra_ldflags=[f"/LIBPATH:{ext_lib_dir}", "dnnl.lib"],
                verbose=os.getenv("ANNA_VERBOSE_ONEDNN_BUILD", "").strip() == "1",
            )
            return _EXTENSION_MODULE
        except Exception as exc:  # pragma: no cover - exercised on target machine
            _EXTENSION_ERROR = str(exc)
            return None


def custom_linear_pointwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str = "none",
    algorithm: str | None = None,
    other: torch.Tensor | None = None,
    binary: str | None = None,
) -> torch.Tensor | None:
    module = _load_extension()
    if module is None:
        return None
    activation = (activation or "none").lower()
    algorithm = (algorithm or "").lower()
    binary = (binary or "none").lower()
    return module.linear_pointwise(x, weight, bias, activation, algorithm, other, binary)


def custom_linear_int4_weight_only(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor | None:
    module = _load_extension()
    if module is None:
        return None
    return module.linear_int4_weight_only(x, packed_weight, scale, zero, int(group_size))


def custom_linear_status() -> dict[str, object]:
    compiler_root = _compiler_root()
    has_module = _EXTENSION_MODULE is not None
    return {
        "enabled": os.getenv("ANNA_DISABLE_ONEDNN_CUSTOM_OP", "").strip() != "1",
        "oneapi_root": str(_oneapi_root()),
        "compiler_root": None if compiler_root is None else str(compiler_root),
        "bootstrapped": _BOOTSTRAPPED,
        "bootstrap_error": _BOOTSTRAP_ERROR,
        "loaded": has_module,
        "load_error": _EXTENSION_ERROR,
        "features": {
            "linear_pointwise": has_module and hasattr(_EXTENSION_MODULE, "linear_pointwise"),
            "linear_int4_weight_only": has_module and hasattr(_EXTENSION_MODULE, "linear_int4_weight_only"),
        },
    }
