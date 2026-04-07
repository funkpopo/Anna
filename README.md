# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna is a from-scratch Qwen3.5 inference service for Intel Arc GPUs with an OpenAI-compatible API.

## Requirements (local development and testing; not validated on Linux)

- Windows + Intel Arc GPU
- PyTorch with `xpu`
- Intel oneAPI DPC++/C++
- Ability to build and load the custom `anna_gated_delta_fused` operator locally

If you want to run a local Qwen3.5 model on an Arc A770 or A750, this README is written for that path.

## Main Features

- OpenAI-compatible API
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `POST /v1/audio/speech`
- Text generation and streaming responses
- Qwen3-TTS speech synthesis
- `anna-speak` CLI for local TTS generation
- `xpu` inference
- Linear-attention SYCL fused ops
- Service address and available routes printed on startup
- Aggregated runtime metrics printed in the terminal
  - Idle periods do not keep spamming logs

## Recommended Environment

The current `main` branch is expected to run in an environment like this:

- Windows 11
- Python 3.11 or 3.12
- Intel Arc A770 / A750
- Intel GPU driver installed correctly
- PyTorch installed with `xpu` support
- Intel oneAPI DPC++/C++ Compiler
- Visual Studio 2022 Build Tools

## Setup

### 1. Clone the repository (`main` already includes SYCL)

```powershell
git clone -b main <your-repo-url> Anna
cd Anna
```

### 2. Create a Conda environment

Using Miniforge or Conda is recommended:

```powershell
conda create -n anna python=3.12 -y
conda activate anna
python -m pip install -U pip
```

### 3. Install PyTorch

After installation, verify it first:

```powershell
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

The minimum expectation is:

- `import torch` works
- `torch.xpu.is_available()` prints `True`

### 4. Install the project itself

Runtime only:

```powershell
pip install -e .
```

Notes:

- `pip install -e .` only installs the Python package, it does not compile the SYCL fused op
- After installation, you must still run `python tools\build_gated_delta_fused_op.py`

### 5. Prepare the compiler environment

The current build script defaults to these two paths:

- oneAPI compiler: `D:\Intel\oneAPI\compiler\2025.3\bin\dpcpp.exe`
- MSVC environment script: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`

If your installation uses different paths, update those values in `tools/build_gated_delta_fused_op.py`.

### 6. Build the SYCL fused op

This is a required step on the current `main` branch when running on Intel Arc with `xpu`:

```powershell
python tools\build_gated_delta_fused_op.py
```

If the build succeeds, the dynamic library will be generated here:

- `.build\anna_gated_delta_fused\anna_gated_delta_fused.pyd`

The terminal usually prints output similar to:

```text
Compiling Anna fused XPU/SYCL ops...
library_path=D:\Projects\Anna\.build\anna_gated_delta_fused\anna_gated_delta_fused.pyd
gated_delta_registered=True
causal_conv1d_registered=True
```

### 7. Optional verification

```powershell
pytest tests\test_fused_op_xpu.py -q
```

For a full regression run:

```powershell
pytest -q
```

### 8. SoX (optional)

The upstream `qwen-tts` runtime may print a warning if the SoX CLI is not on your `PATH` (`'sox' is not recognized...`). Inference still runs; installing SoX silences the warning and enables SoX-based audio helpers when the stack invokes them.

On Windows, pick one:

- **Chocolatey** (Administrator shell): `choco install sox`
- **Scoop**: `scoop install sox`
- **Manual**: Install a build that provides `sox.exe` (see [SoX](http://sox.sourceforge.net/)), then add its directory to your user or system `PATH`.

Open a **new** terminal and confirm:

```powershell
sox --version
```

## Local Model Directory Requirements

### Text / multimodal Qwen models

Your local model directory should contain at least:

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors` or sharded `*.safetensors`

For example:

```text
D:\Projects\Anna\models\Qwen\Qwen3___5-2B
```

### Qwen3-TTS 12Hz models

Anna also supports official `Qwen3-TTS` 12Hz model directories through the `qwen-tts` runtime path. A local `qwen3_tts` model directory should contain at least:

- `config.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `model.safetensors`
- `speech_tokenizer\config.json`
- `speech_tokenizer\model.safetensors`

For example:

```text
D:\Projects\Anna\models\Qwen\Qwen3-TTS-12Hz-1___7B-Base
```

Notes:

- `anna-generate` and `anna-bench` remain for the `qwen3_5_text` family; use `anna-speak` or `POST /v1/audio/speech` for the `qwen3_tts` family
- current TTS support is aimed at official 12Hz model layouts such as `Base`, `CustomVoice`, and `VoiceDesign`
- optional **SoX** install: see [Step 8: SoX (optional)](#8-sox-optional); missing SoX only triggers a warning and does not block 12Hz inference here

## Command Examples

### 1. Start the server

Minimal example:

```powershell
anna-serve --model-dir D:\path\to\model --device xpu --dtype bfloat16
```

Example aligned with the current `main` branch and environment:

```powershell
anna-serve `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B `
  --model-name Qwen3.5-2B `
  --device xpu `
  --dtype bfloat16 `
  --offload-mode none `
  --weight-quant none `
  --host 127.0.0.1 `
  --port 8000
```

If you do not want periodic metrics logs:

```powershell
anna-serve `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B `
  --model-name Qwen3.5-2B `
  --device xpu `
  --dtype bfloat16 `
  --metrics-log-interval-seconds 0
```

### 2. What the terminal prints on startup

After the server starts, the terminal prints:

- The service address
- The currently available routes
- Aggregated runtime metrics

Example:

```text
INFO anna.cli.serve: Starting Anna server on http://127.0.0.1:8000
INFO anna.cli.serve: Available routes are:
INFO anna.cli.serve: Route: /openapi.json, Methods: HEAD, GET
INFO anna.cli.serve: Route: /healthz, Methods: GET
INFO anna.cli.serve: Route: /v1/models, Methods: GET
INFO anna.cli.serve: Route: /v1/chat/completions, Methods: POST
INFO anna.cli.serve: Route: /v1/completions, Methods: POST
INFO anna.cli.serve: Route: /v1/audio/speech, Methods: POST
```

### 3. Generate text directly

```powershell
anna-generate `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B `
  --model-name Qwen3.5-2B `
  --device xpu `
  --dtype bfloat16 `
  --offload-mode none `
  --weight-quant none `
  --prompt "Hello" `
  --max-new-tokens 32
```

### 4. Run a benchmark

```powershell
anna-bench `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B `
  --model-name Qwen3.5-2B `
  --device xpu `
  --dtype bfloat16 `
  --prompt "Hello, please introduce yourself." `
  --runs 5
```

### 5. Generate speech directly with Qwen3-TTS Base

This path is intended for local voice-clone models such as `Qwen3-TTS-12Hz-1.7B-Base`.

```powershell
anna-speak `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3-TTS-12Hz-1___7B-Base `
  --device xpu `
  --dtype bfloat16 `
  --input "Anna CLI smoke test on Intel Arc." `
  --output D:\Projects\Anna\.build\anna_tts_cli_smoke.wav `
  --language English `
  --ref-audio D:\Projects\Anna\.build\tts_ref.wav `
  --ref-text "Hello Anna. This is a reference voice for local synthesis testing." `
  --max-new-tokens 1024
```

For `CustomVoice` models, replace the voice-clone arguments with `--speaker` and optional `--instruct`.

For `VoiceDesign` models, provide `--instruct` without `--speaker`.

### 6. Serve a Qwen3-TTS model

```powershell
anna-serve `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3-TTS-12Hz-1___7B-Base `
  --model-name Qwen3-TTS-1.7B-Base `
  --device xpu `
  --dtype bfloat16 `
  --host 127.0.0.1 `
  --port 8000
```

### 7. Call the speech API

The server returns raw audio bytes. For `Base` voice-clone models, include `ref_audio` and `ref_text`.

In **PowerShell**, pass JSON with a **single-quoted** `-d` argument so the body is valid JSON: use `"` inside the JSON and **do not** write `\"` (those backslashes would be sent literally and break parsing). Prefer forward slashes in `ref_audio` paths on Windows.

```powershell
curl.exe http://127.0.0.1:8000/v1/audio/speech `
  -H "Content-Type: application/json" `
  --output speech.wav `
  -d '{"model":"Qwen3-TTS-1.7B-Base","input":"This request goes through the Anna FastAPI speech route.","language":"English","ref_audio":"D:/Projects/Anna/.build/tts_ref.wav","ref_text":"Hello Anna. This is a reference voice for local synthesis testing.","response_format":"wav","max_new_tokens":1024}'
```

Supported request fields include:

- `input`: text to synthesize
- `response_format`: `wav`, `flac`, or `pcm`
- `language`
- `speaker` or `voice` for `CustomVoice`
- `instruct` for `VoiceDesign` or optional style control on `CustomVoice`
- `ref_audio`, `ref_text`, `x_vector_only_mode` for `Base`
- `max_new_tokens`, `do_sample`, `temperature`, `top_p`, `top_k`, `repetition_penalty`
- `subtalker_do_sample`, `subtalker_temperature`, `subtalker_top_p`, `subtalker_top_k`
- `non_streaming_mode`

## API Examples

### Chat Completions

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"Qwen3.5-2B","messages":[{"role":"user","content":"Hello, please introduce yourself."}],"max_completion_tokens":128}'
```

### Completions

```powershell
curl.exe http://127.0.0.1:8000/v1/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"Qwen3.5-2B","prompt":"Hello","max_tokens":32}'
```

### List models

```powershell
curl.exe http://127.0.0.1:8000/v1/models
```

### Health check

```powershell
curl.exe http://127.0.0.1:8000/healthz
```

## Troubleshooting

### 1. `torch.xpu.is_available()` returns `False`

Check these first:

- Whether your PyTorch build really supports `xpu`
- Whether the Intel GPU driver is installed correctly
- Whether you accidentally installed a CPU-only `torch` in the current environment

### 2. The fused op build fails

Check these first:

- Whether the `dpcpp.exe` path is correct
- Whether the `vcvars64.bat` path is correct
- Whether both oneAPI and MSVC are installed
- Whether the active environment's `torch` includes the required headers and `torch_xpu` related libraries

### 3. The server starts, but generation fails with fused-op related errors

Rebuild the custom operator once:

```powershell
python tools\build_gated_delta_fused_op.py
```

Then restart the server.

### 4. I do not want the metrics logs

Disable them directly:

```powershell
anna-serve --model-dir D:\path\to\model --device xpu --dtype bfloat16 --metrics-log-interval-seconds 0
```

## Development Notes

- After changing the SYCL source, rerun `python tools\build_gated_delta_fused_op.py`
- Then run `pytest tests\test_fused_op_xpu.py -q`
- Restart `anna-serve` after that
