from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


_WINDOWS_VISUAL_STUDIO_VERSIONS = ("2022", "2019")
_WINDOWS_VISUAL_STUDIO_EDITIONS = ("BuildTools", "Community", "Professional", "Enterprise")


def _platform_name() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(f"Unsupported platform for Anna fused-op builds: os.name={os.name!r} sys.platform={sys.platform!r}")


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


def _resolve_command_path(candidate: str) -> Path | None:
    expanded = Path(candidate).expanduser()
    if expanded.exists():
        return expanded.resolve()
    resolved = shutil.which(candidate)
    if resolved:
        return Path(resolved).resolve()
    return None


def _split_env_paths(name: str) -> list[Path]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    return [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]


def _unique_existing_paths(*paths: Path) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        unique.append(Path(resolved))
        seen.add(resolved)
    return unique


def _existing_paths_in_order(paths: list[Path]) -> list[Path]:
    return _unique_existing_paths(*paths)


def _env_path(name: str) -> Path | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _windows_program_files_dirs() -> list[Path]:
    candidates: list[Path] = []
    for name in ("ProgramFiles(x86)", "ProgramFiles"):
        raw = os.getenv(name, "").strip()
        if raw:
            candidates.append(Path(raw).expanduser())
    return _existing_paths_in_order(candidates)


def _glob_existing(patterns: list[tuple[Path, str]]) -> list[Path]:
    matches: list[Path] = []
    for root, pattern in patterns:
        if not root.exists():
            continue
        matches.extend(sorted(root.glob(pattern), key=lambda item: str(item), reverse=True))
    return _existing_paths_in_order(matches)


def _standard_windows_dpcpp_candidates(search_roots: list[Path] | None = None) -> list[Path]:
    roots: list[Path] = []
    oneapi_root = _env_path("ONEAPI_ROOT")
    if oneapi_root is not None:
        roots.append(oneapi_root)
    if search_roots is not None:
        roots.extend(search_roots)
    else:
        roots.extend(base / "Intel" / "oneAPI" for base in _windows_program_files_dirs())

    candidates: list[Path] = []
    glob_patterns: list[tuple[Path, str]] = []
    for root in _existing_paths_in_order(roots):
        candidates.append(root / "compiler" / "latest" / "bin" / "dpcpp.exe")
        glob_patterns.append((root, "compiler/*/bin/dpcpp.exe"))
    return _existing_paths_in_order(candidates + _glob_existing(glob_patterns))


def _standard_windows_vcvars_candidates(search_roots: list[Path] | None = None) -> list[Path]:
    candidates: list[Path] = []
    vs_install_dir = _env_path("VSINSTALLDIR")
    if vs_install_dir is not None:
        candidates.append(vs_install_dir / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")

    roots = list(search_roots or _windows_program_files_dirs())
    for root in _existing_paths_in_order(roots):
        for version in _WINDOWS_VISUAL_STUDIO_VERSIONS:
            for edition in _WINDOWS_VISUAL_STUDIO_EDITIONS:
                candidates.append(
                    root / "Microsoft Visual Studio" / version / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                )
    return _existing_paths_in_order(candidates)


def _standard_linux_dpcpp_candidates() -> list[Path]:
    roots: list[Path] = []
    oneapi_root = _env_path("ONEAPI_ROOT")
    if oneapi_root is not None:
        roots.append(oneapi_root)
    roots.append(Path("/opt/intel/oneapi"))

    candidates: list[Path] = []
    glob_patterns: list[tuple[Path, str]] = []
    for root in _existing_paths_in_order(roots):
        candidates.append(root / "compiler" / "latest" / "bin" / "dpcpp")
        glob_patterns.append((root, "compiler/*/bin/dpcpp"))
    return _existing_paths_in_order(candidates + _glob_existing(glob_patterns))


def _resolve_python_include_dir(python_root: Path) -> Path:
    config_paths = sysconfig.get_paths()
    candidates = [
        Path(path)
        for path in (
            config_paths.get("include"),
            config_paths.get("platinclude"),
        )
        if path
    ]
    candidates.extend(
        [
            python_root / "Include",
            python_root / "include",
            python_root.parent / "Include",
            python_root.parent / "include",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not locate the active Python include directory. Searched: {searched}")


def _resolve_compiler_executable(platform_name: str) -> Path:
    env_candidate = os.getenv("ANNA_DPCPP")
    if env_candidate:
        resolved = _resolve_command_path(env_candidate)
        if resolved is None:
            raise FileNotFoundError(f"ANNA_DPCPP points to a missing compiler: {env_candidate}")
        return resolved

    if platform_name == "windows":
        path_candidates = ["dpcpp.exe", "dpcpp"]
        discovered_candidates = _standard_windows_dpcpp_candidates()
    else:
        path_candidates = ["dpcpp"]
        discovered_candidates = _standard_linux_dpcpp_candidates()

    for candidate in path_candidates:
        resolved = _resolve_command_path(candidate)
        if resolved is not None:
            return resolved
    for candidate in discovered_candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join([*path_candidates, *[str(candidate) for candidate in discovered_candidates]])
    raise FileNotFoundError(
        "Could not locate the Intel oneAPI DPC++ compiler. "
        f"Tried: {searched}. Set ANNA_DPCPP to the compiler path if it is installed elsewhere."
    )


def _resolve_vcvars_bat() -> Path | None:
    env_candidate = os.getenv("ANNA_VCVARS64")
    if env_candidate:
        resolved = Path(env_candidate).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"ANNA_VCVARS64 points to a missing file: {resolved}")
        return resolved.resolve()
    for candidate in _standard_windows_vcvars_candidates():
        if candidate.exists():
            return candidate.resolve()
    return None


def _python_runtime_dirs(python_root: Path) -> list[Path]:
    return _unique_existing_paths(
        python_root / "Library" / "bin",
        python_root / "DLLs",
    )


def _compiler_runtime_dirs(compiler_exe: Path) -> list[Path]:
    parents = [compiler_exe.parent]
    current = compiler_exe.parent
    for _ in range(3):
        current = current.parent
        parents.append(current)

    candidates: list[Path] = [compiler_exe.parent]
    for parent in parents:
        candidates.extend(
            [
                parent / "lib",
                parent / "lib" / "x64",
                parent / "lib" / "intel64_win",
                parent / "lib" / "intel64_lin",
            ]
        )
    candidates.extend(_split_env_paths("ANNA_ONEAPI_RUNTIME_PATHS"))
    return _unique_existing_paths(*candidates)


def _shared_library_name(platform_name: str) -> str:
    return "anna_gated_delta_fused.pyd" if platform_name == "windows" else "anna_gated_delta_fused.so"


def _runtime_linker_flags(platform_name: str, *runtime_dirs: Path) -> list[str]:
    if platform_name != "linux":
        return []

    unique_flags: list[str] = []
    seen: set[str] = set()
    for flag in ["-Wl,-rpath,$ORIGIN", *[f"-Wl,-rpath,{path}" for path in runtime_dirs if path.exists()]]:
        if flag in seen:
            continue
        unique_flags.append(flag)
        seen.add(flag)
    return unique_flags


def _build_compile_command(
    *,
    platform_name: str,
    compiler_exe: Path,
    source: Path,
    output: Path,
    torch_root: Path,
    torch_lib_dir: Path,
    python_include_dir: Path,
    compiler_runtime_dirs: list[Path],
) -> list[str]:
    command = [
        str(compiler_exe),
        "-fsycl",
        "-fsycl-targets=spir64",
        "-std=c++17",
        "-shared",
        "-O2",
        "-Wno-ignored-attributes",
        "-Wno-deprecated-declarations",
    ]
    if platform_name == "linux":
        command.append("-fPIC")

    command.extend(
        [
            str(source),
            f"-I{torch_root / 'include'}",
            f"-I{torch_root / 'include' / 'torch' / 'csrc' / 'api' / 'include'}",
            f"-I{python_include_dir}",
            f"-L{torch_lib_dir}",
        ]
    )

    for runtime_dir in compiler_runtime_dirs:
        if runtime_dir != torch_lib_dir:
            command.append(f"-L{runtime_dir}")

    command.extend(_runtime_linker_flags(platform_name, torch_lib_dir, *compiler_runtime_dirs))
    command.extend(
        [
            "-lc10",
            "-lc10_xpu",
            "-ltorch_cpu",
            "-ltorch_xpu",
            "-ltorch",
            "-o",
            str(output),
        ]
    )
    return command


def main() -> None:
    import torch

    platform_name = _platform_name()
    repo_root = Path(__file__).resolve().parents[1]
    python_root = Path(sys.executable).resolve().parent
    python_include_dir = _resolve_python_include_dir(python_root)
    torch_root = Path(torch.__file__).resolve().parent
    torch_lib_dir = torch_root / "lib"
    compiler_exe = _resolve_compiler_executable(platform_name)
    compiler_runtime_dirs = _compiler_runtime_dirs(compiler_exe)
    build_dir = repo_root / ".build" / "anna_gated_delta_fused"
    source = repo_root / "src" / "anna" / "model" / "custom_ops" / "gated_delta_fused_op.cpp"
    output = build_dir / _shared_library_name(platform_name)

    build_dir.mkdir(parents=True, exist_ok=True)
    if platform_name == "windows":
        vcvars_bat = _resolve_vcvars_bat()
        if vcvars_bat is not None:
            _load_msvc_build_environment(vcvars_bat)
        else:
            print("vcvars64.bat was not found; assuming the current shell already has the MSVC build environment.")

    python_runtime_dirs = _python_runtime_dirs(python_root)
    runtime_dirs = _unique_existing_paths(build_dir, *python_runtime_dirs, torch_lib_dir, *compiler_runtime_dirs)
    _prepend_env_path("PATH", compiler_exe.parent, *python_runtime_dirs, torch_lib_dir, *compiler_runtime_dirs)
    if platform_name == "linux":
        _prepend_env_path("LD_LIBRARY_PATH", torch_lib_dir, *compiler_runtime_dirs)

    compile_command = _build_compile_command(
        platform_name=platform_name,
        compiler_exe=compiler_exe,
        source=source,
        output=output,
        torch_root=torch_root,
        torch_lib_dir=torch_lib_dir,
        python_include_dir=python_include_dir,
        compiler_runtime_dirs=compiler_runtime_dirs,
    )

    print("Compiling Anna fused XPU/SYCL ops...")
    print(" ".join(compile_command))
    subprocess.run(compile_command, check=True, cwd=str(build_dir), env=os.environ.copy())

    _write_runtime_paths(build_dir, *runtime_dirs)
    _prepend_env_path("PATH", *runtime_dirs)

    sys.path.insert(0, str(repo_root / "src"))
    from anna.model.fused_ops import (  # noqa: PLC0415
        causal_conv1d_fused_is_available,
        gated_delta_fused_is_available,
        gqa_decode_fused_is_available,
        maybe_load_gated_delta_library,
        moe_dispatch_fused_is_available,
        moe_grouped_int4_mlp_fused_is_available,
        moe_router_fused_is_available,
        moe_scatter_fused_is_available,
        paged_gqa_decode_fused_is_available,
        qk_norm_rotary_fused_ex_is_available,
        qk_norm_rotary_fused_is_available,
        rmsnorm_fused_ex_is_available,
        rmsnorm_gated_fused_is_available,
        rmsnorm_fused_is_available,
    )

    if not maybe_load_gated_delta_library(output):
        raise RuntimeError(f"Failed to load compiled Anna fused-op library: {output}")

    print(f"library_path={output}")
    print(f"gqa_decode_registered={gqa_decode_fused_is_available()}")
    print(f"paged_gqa_decode_registered={paged_gqa_decode_fused_is_available()}")
    print(f"moe_router_registered={moe_router_fused_is_available()}")
    print(f"moe_dispatch_registered={moe_dispatch_fused_is_available()}")
    print(f"moe_scatter_registered={moe_scatter_fused_is_available()}")
    print(f"moe_grouped_int4_mlp_registered={moe_grouped_int4_mlp_fused_is_available()}")
    print(f"rmsnorm_registered={rmsnorm_fused_is_available()}")
    print(f"rmsnorm_ex_registered={rmsnorm_fused_ex_is_available()}")
    print(f"rmsnorm_gated_registered={rmsnorm_gated_fused_is_available()}")
    print(f"qk_norm_rotary_registered={qk_norm_rotary_fused_is_available()}")
    print(f"qk_norm_rotary_ex_registered={qk_norm_rotary_fused_ex_is_available()}")
    print(f"gated_delta_registered={gated_delta_fused_is_available()}")
    print(f"causal_conv1d_registered={causal_conv1d_fused_is_available()}")


if __name__ == "__main__":
    main()
