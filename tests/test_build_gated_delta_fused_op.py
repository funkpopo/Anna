from __future__ import annotations

from pathlib import Path

from tools.build_gated_delta_fused_op import (
    _build_compile_command,
    _runtime_linker_flags,
    _shared_library_name,
    _standard_windows_dpcpp_candidates,
    _standard_windows_vcvars_candidates,
)


def test_shared_library_name_matches_platform() -> None:
    assert _shared_library_name("windows") == "anna_gated_delta_fused.pyd"
    assert _shared_library_name("linux") == "anna_gated_delta_fused.so"


def test_runtime_linker_flags_adds_linux_rpath_entries(tmp_path: Path) -> None:
    torch_lib_dir = tmp_path / "torch-lib"
    oneapi_lib_dir = tmp_path / "oneapi-lib"
    torch_lib_dir.mkdir()
    oneapi_lib_dir.mkdir()

    flags = _runtime_linker_flags("linux", torch_lib_dir, oneapi_lib_dir)

    assert flags == [
        "-Wl,-rpath,$ORIGIN",
        f"-Wl,-rpath,{torch_lib_dir.resolve()}",
        f"-Wl,-rpath,{oneapi_lib_dir.resolve()}",
    ]


def test_build_compile_command_adds_linux_pic_and_rpath(tmp_path: Path) -> None:
    compiler_exe = tmp_path / "bin" / "dpcpp"
    source = tmp_path / "src" / "gated_delta_fused_op.cpp"
    output = tmp_path / "build" / "anna_gated_delta_fused.so"
    torch_root = tmp_path / "torch"
    torch_lib_dir = tmp_path / "torch-lib"
    python_include_dir = tmp_path / "python-include"
    compiler_runtime_dir = tmp_path / "oneapi-lib"

    compiler_exe.parent.mkdir()
    output.parent.mkdir()
    torch_lib_dir.mkdir()
    python_include_dir.mkdir()
    compiler_runtime_dir.mkdir()

    command = _build_compile_command(
        platform_name="linux",
        compiler_exe=compiler_exe,
        source=source,
        output=output,
        torch_root=torch_root,
        torch_lib_dir=torch_lib_dir,
        python_include_dir=python_include_dir,
        compiler_runtime_dirs=[compiler_runtime_dir],
    )

    assert "-fPIC" in command
    assert f"-Wl,-rpath,{torch_lib_dir.resolve()}" in command
    assert f"-Wl,-rpath,{compiler_runtime_dir.resolve()}" in command
    assert str(output) in command


def test_standard_windows_dpcpp_candidates_scan_standard_layout(tmp_path: Path) -> None:
    oneapi_root = tmp_path / "Intel" / "oneAPI"
    latest = oneapi_root / "compiler" / "latest" / "bin" / "dpcpp.exe"
    versioned = oneapi_root / "compiler" / "2025.3" / "bin" / "dpcpp.exe"
    latest.parent.mkdir(parents=True)
    versioned.parent.mkdir(parents=True)
    latest.touch()
    versioned.touch()

    candidates = _standard_windows_dpcpp_candidates([oneapi_root])

    assert latest.resolve() in candidates
    assert versioned.resolve() in candidates


def test_standard_windows_vcvars_candidates_scan_standard_layout(tmp_path: Path) -> None:
    program_files_x86 = tmp_path / "Program Files (x86)"
    vcvars = (
        program_files_x86
        / "Microsoft Visual Studio"
        / "2022"
        / "BuildTools"
        / "VC"
        / "Auxiliary"
        / "Build"
        / "vcvars64.bat"
    )
    vcvars.parent.mkdir(parents=True)
    vcvars.touch()

    candidates = _standard_windows_vcvars_candidates([program_files_x86])

    assert vcvars.resolve() in candidates
