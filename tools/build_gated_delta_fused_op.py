from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch


def _load_msvc_build_environment(vcvars_bat: Path) -> None:
    command = f'call "{vcvars_bat}" >nul && set'
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        shell=True,
    )
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def _prepend_env_path(key: str, *paths: Path) -> None:
    existing = os.environ.get(key, "")
    entries = [str(path) for path in paths if path.exists()]
    if existing:
        entries.append(existing)
    os.environ[key] = os.pathsep.join(entries)


def _write_runtime_paths(build_dir: Path, *paths: Path) -> None:
    runtime_paths = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        runtime_paths.append(resolved)
        seen.add(resolved)
    (build_dir / "runtime_paths.txt").write_text("\n".join(runtime_paths) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python_root = Path(sys.executable).resolve().parent
    torch_root = Path(torch.__file__).resolve().parent
    torch_lib_dir = torch_root / "lib"
    compiler_bin = Path(r"D:\Intel\oneAPI\compiler\2025.3\bin")
    compiler_exe = compiler_bin / "dpcpp.exe"
    compiler_lib_dir = compiler_bin.parent / "lib"
    vcvars_bat = Path(
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    build_dir = repo_root / ".build" / "anna_gated_delta_fused"
    source = repo_root / "src" / "anna" / "model" / "custom_ops" / "gated_delta_fused_op.cpp"
    output = build_dir / "anna_gated_delta_fused.pyd"

    build_dir.mkdir(parents=True, exist_ok=True)
    _load_msvc_build_environment(vcvars_bat)
    _prepend_env_path("PATH", python_root / "Library" / "bin", compiler_bin, torch_lib_dir)

    compile_command = [
        str(compiler_exe),
        "-fsycl",
        "-fsycl-targets=spir64",
        "-std=c++17",
        "-shared",
        "-O2",
        "-Wno-ignored-attributes",
        "-Wno-deprecated-declarations",
        str(source),
        f"-I{torch_root / 'include'}",
        f"-I{torch_root / 'include' / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-I{python_root / 'Include'}",
        f"-L{torch_lib_dir}",
        f"-L{compiler_lib_dir}",
        "-lc10",
        "-lc10_xpu",
        "-ltorch_cpu",
        "-ltorch_xpu",
        "-ltorch",
        "-o",
        str(output),
    ]

    print("Compiling Anna fused XPU/SYCL ops...")
    print(" ".join(compile_command))
    subprocess.run(compile_command, check=True, cwd=str(build_dir), env=os.environ.copy())

    _write_runtime_paths(build_dir, build_dir, python_root / "Library" / "bin", torch_lib_dir, compiler_bin)
    _prepend_env_path("PATH", build_dir, python_root / "Library" / "bin", torch_lib_dir, compiler_bin)

    sys.path.insert(0, str(repo_root / "src"))
    from anna.model.fused_ops import (  # noqa: PLC0415
        causal_conv1d_fused_is_available,
        gated_delta_fused_is_available,
        maybe_load_gated_delta_library,
        qk_norm_rotary_fused_is_available,
        rmsnorm_fused_is_available,
    )

    if not maybe_load_gated_delta_library(output):
        raise RuntimeError(f"Failed to load compiled Anna fused-op library: {output}")

    print(f"library_path={output}")
    print(f"rmsnorm_registered={rmsnorm_fused_is_available()}")
    print(f"qk_norm_rotary_registered={qk_norm_rotary_fused_is_available()}")
    print(f"gated_delta_registered={gated_delta_fused_is_available()}")
    print(f"causal_conv1d_registered={causal_conv1d_fused_is_available()}")


if __name__ == "__main__":
    main()
