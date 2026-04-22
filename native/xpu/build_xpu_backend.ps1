param(
    [string]$Source = "native\xpu\sycl_backend.cpp",
    [string]$OutDir = "zig-out\bin",
    [string]$Compiler = $env:ANNA_SYCL_CXX
)

$ErrorActionPreference = "Stop"

function Resolve-SyclCompiler {
    param([string]$RequestedCompiler)

    if ($RequestedCompiler) {
        return $RequestedCompiler
    }

    foreach ($name in @("icpx", "dpcpp")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
    }

    $roots = @(
        "C:\Program Files (x86)\Intel\oneAPI\compiler",
        "C:\Program Files\Intel\oneAPI\compiler",
        "D:\Intel\oneAPI\compiler"
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) {
            continue
        }
        foreach ($name in @("icpx.exe", "dpcpp.exe")) {
            $match = Get-ChildItem $root -Recurse -Filter $name -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending |
                Select-Object -First 1
            if ($match) {
                return $match.FullName
            }
        }
    }

    throw "No SYCL compiler found. Install Intel oneAPI DPC++ or set ANNA_SYCL_CXX to dpcpp/icpx."
}

function Resolve-OneApiSetvars {
    param([string]$CompilerPath)

    $cursor = Split-Path -Path $CompilerPath -Parent
    while ($cursor) {
        $candidate = Join-Path $cursor "setvars.bat"
        if (Test-Path $candidate) {
            return $candidate
        }
        $parent = Split-Path -Path $cursor -Parent
        if ($parent -eq $cursor) {
            break
        }
        $cursor = $parent
    }

    $fallbacks = @(
        "D:\Intel\oneAPI\setvars.bat",
        "C:\Program Files (x86)\Intel\oneAPI\setvars.bat",
        "C:\Program Files\Intel\oneAPI\setvars.bat"
    )
    foreach ($fallback in $fallbacks) {
        if (Test-Path $fallback) {
            return $fallback
        }
    }
    return $null
}

function Resolve-VisualStudioInstall {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installPath) {
            return ($installPath | Select-Object -First 1).Trim()
        }
    }

    $fallback = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    if (Test-Path $fallback) {
        return $fallback
    }

    return $null
}

function Resolve-VisualStudioVcVars {
    $install = Resolve-VisualStudioInstall
    if (-not $install) {
        return $null
    }
    $candidate = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $candidate) {
        return $candidate
    }
    return $null
}

function Import-BatchEnvironment {
    param([string]$BatchFile)

    if (-not $BatchFile) {
        return
    }

    $vsInstall = Resolve-VisualStudioInstall
    $vcvars = Resolve-VisualStudioVcVars
    if ($vsInstall) {
        [Environment]::SetEnvironmentVariable("VS2022INSTALLDIR", $vsInstall, "Process")
    }

    $cmd = if ($vcvars) {
        "set VS2022INSTALLDIR=$vsInstall && call `"$vcvars`" >nul 2>&1 && call `"$BatchFile`" intel64 vs2022 >nul 2>&1 && set"
    } else {
        "set VS2022INSTALLDIR=$vsInstall && call `"$BatchFile`" intel64 vs2022 >nul 2>&1 && set"
    }

    $cmdOutput = & cmd.exe /d /s /c $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to import oneAPI environment from $BatchFile"
    }

    foreach ($line in $cmdOutput) {
        $parts = $line -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }
        [Environment]::SetEnvironmentVariable($parts[0], $parts[1], "Process")
    }
}

$compilerPath = Resolve-SyclCompiler $Compiler
$setvarsPath = Resolve-OneApiSetvars $compilerPath
Import-BatchEnvironment $setvarsPath
$sourcePath = Resolve-Path $Source
New-Item -ItemType Directory -Force $OutDir | Out-Null
$outputPath = Join-Path $OutDir "anna-xpu-backend.dll"

& $compilerPath -fsycl -std=c++20 -O3 -shared $sourcePath.Path -o $outputPath
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Built $outputPath"
