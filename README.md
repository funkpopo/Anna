# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna is a local inference runtime and OpenAI-compatible API server focused on PyTorch/XPU execution for Intel Arc Alchemist . The current project can serve Qwen3.5 text-generation checkpoints, Gemma 4 multimodal checkpoints, and Qwen3-TTS speech synthesis checkpoints from local model directories.

## Features

- OpenAI-compatible routes:
  - `GET /healthz`
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `POST /v1/audio/speech`
- Non-streaming and streaming generation for chat and text completions
- Optional reasoning separation through `reasoning_content`
- Tool/function calling for chat models, including streamed tool-call deltas
- Local multimodal chat input support
  - Qwen3.5: text, image, video
  - Gemma4: text, image, video, audio
- Local speech synthesis for `qwen3_tts` through API and `anna-speak`
- CLI entry points: `anna-serve`, `anna-generate`, `anna-bench`, `anna-speak`
- XPU-oriented runtime controls:
  - `torch.compile`
  - prefill chunking
  - exact prompt KV-cache reuse
  - TurboQuant KV-cache compression
  - runtime int4 weight quantization
  - Qwen MoE expert offload and expert caching
  - continuous batching
  - optional SYCL fused custom ops
- Health, memory, cache, and request metrics exposed through `/healthz` and terminal logs

## Supported Model Families

Anna selects the runtime from the top-level `model_type` in `config.json`:

- `qwen3_tts` -> Qwen3-TTS runtime
- `gemma4` -> Gemma4 runtime
- everything else -> Qwen3.5 text runtime

| Family | `config.json` top-level `model_type` | Other recognized model types/configs | Capabilities | Main commands |
| --- | --- | --- | --- | --- |
| Qwen3.5 text runtime | Any non-`qwen3_tts` and non-`gemma4` value; known values include `qwen3_5`, `qwen3_5_text`, `qwen3_5_moe`, `qwen3_5_vl` | Optional `vision_config`; Qwen quantization configs such as AWQ and AutoRound are parsed from model files | Text completions, chat completions, streaming, reasoning output, tool calling, multimodal chat with image/video | `anna-serve`, `anna-generate`, `anna-bench` |
| Gemma4 runtime | `gemma4` | `text_config.model_type=gemma4_text`; optional `vision_config.model_type=gemma4_vision`; optional `audio_config.model_type=gemma4_audio` | Text completions, chat completions, streaming, reasoning output, tool calling, multimodal chat with image/video/audio | `anna-serve`, `anna-generate`, `anna-bench` |
| Qwen3-TTS runtime | `qwen3_tts` | Runtime `tts_model_type` can be `base`, `custom_voice`, or `voice_design` | Speech synthesis through API or CLI | `anna-serve`, `anna-speak` |

Notes:

- `anna-generate` and `anna-bench` are only for text-generation families (`qwen3_5_text` and `gemma4`).
- `anna-speak` is only for `qwen3_tts`.
- `/v1/audio/speech` only succeeds when the loaded model supports speech synthesis.
- `anna-bench` supports `--image` and `--video`, but not audio benchmarking.
- Multimodal media input is available through chat-style message content arrays and the benchmark CLI; `/v1/completions` remains text-only.

## Requirements

- Python `3.11+`
- A local model directory containing `config.json`, tokenizer files, and weights
- PyTorch installed separately for your platform
  - CPU works for debugging and tests
  - Intel Arc + a PyTorch build with `xpu` support is the intended runtime path
- `imageio` and `imageio-ffmpeg` for video input support
- `qwen-tts` for `qwen3_tts` models
- To build the optional fused XPU operator:
  - Intel oneAPI DPC++/C++ Compiler
  - Visual Studio Build Tools on Windows

The Python package dependencies above are installed by `pip install -e .`; PyTorch and oneAPI are environment-specific and should be installed separately.

## Installation

### 1. Clone and enter the repository

```bash
git clone <your-repo-url> Anna
cd Anna
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3. Install Anna

```bash
python -m pip install -e .
```

Optional development dependencies:

```bash
python -m pip install -e ".[dev]"
```

### 4. Install and verify PyTorch

Install a PyTorch build that matches your target device. If you plan to use Intel Arc, install a build with `xpu` support and verify it:

```bash
python -c "import torch; print(torch.__version__); print(hasattr(torch, 'xpu')); print(torch.xpu.is_available() if hasattr(torch, 'xpu') else False)"
```

### 5. Build the fused XPU operator

This is recommended for XPU performance work, but not required for basic CPU use.

```bash
python tools/build_gated_delta_fused_op.py
```

The build output is written under `.build/anna_gated_delta_fused`.

## Example Local Model Directories

If you already downloaded checkpoints into the repository-local `models/` directory, examples currently present in this workspace include:

- `models/Intel/Qwen3___5-2B-int4-AutoRound`
- `models/Intel/Qwen3___5-35B-A3B-int4-AutoRound`
- `models/google/gemma-4-E4B-it`
- `models/Qwen/Qwen3-TTS-12Hz-1___7B-Base`

Any command below expects `--model-dir` to point at a directory like the ones above.

## Quick Start

### Serve an OpenAI-compatible API

Minimal example:

```bash
anna-serve --model-dir <path-to-model> --host 127.0.0.1 --port 8000
```

Typical XPU-oriented example:

```bash
anna-serve \
  --model-dir <path-to-model> \
  --device xpu \
  --dtype bfloat16 \
  --kv-cache-quantization turboquant \
  --kv-cache-quant-bits 4 \
  --weight-quant auto \
  --prompt-cache-size 4
```

For large Qwen MoE models, you can additionally tune options such as:

- `--offload-mode experts`
- `--expert-quant int4`
- `--cached-experts-per-layer <N>` (64 recommended)
- `--resident-expert-layers <N>`

### Generate text locally

`anna-generate` is a prompt-to-text CLI for text-generation families:

```bash
anna-generate --model-dir <path-to-text-model> --prompt "Write a short summary of KV cache."
```

### Benchmark a model

Text benchmark:

```bash
anna-bench --model-dir <path-to-text-model> --prompt "Explain grouped-query attention." --warmup 1 --runs 3
```

Image benchmark:

```bash
anna-bench --model-dir <path-to-multimodal-model> --prompt "Describe the image." --image ./demo.png
```

Video benchmark:

```bash
anna-bench --model-dir <path-to-multimodal-model> --prompt "Summarize the clip." --video ./demo.mp4
```

### Synthesize speech with Qwen3-TTS

Base voice-clone model:

```bash
anna-speak \
  --model-dir <path-to-qwen3-tts-base> \
  --input "Hello from Anna." \
  --output out.wav \
  --ref-audio ref.wav \
  --ref-text "Reference transcript."
```

CustomVoice model:

```bash
anna-speak \
  --model-dir <path-to-qwen3-tts-custom> \
  --input "Hello from Anna." \
  --output out.wav \
  --speaker Vivian \
  --instruct "Speak with energy."
```

VoiceDesign model:

```bash
anna-speak \
  --model-dir <path-to-qwen3-tts-voice-design> \
  --input "Hello from Anna." \
  --output out.wav \
  --instruct "A calm, warm female voice."
```

The CLI supports `wav` and `flac` output through `--response-format`.

## OpenAI-Compatible API

The server always exposes the same routes, but request success depends on the loaded model family.

### Routes

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/audio/speech`

### Example requests

List models:

```bash
curl http://127.0.0.1:8000/v1/models
```

Chat completion:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Write a haiku about Intel Arc."}],
    "reasoning_format": "deepseek"
  }'
```

Text completion:

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "prompt": "Anna is",
    "max_tokens": 64
  }'
```

Speech synthesis with a Base voice-clone TTS model:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "input": "Hello from Anna.",
    "ref_audio": "ref.wav",
    "ref_text": "Reference transcript.",
    "response_format": "wav"
  }' \
  --output speech.wav
```

For `custom_voice` models, pass `speaker` or `voice`; for `voice_design` models, pass `instruct`.

Chat requests also accept:

- multimodal `content` arrays with `text`, `image_url`, `video_url`, and `audio_url` parts
- `tools`, `tool_choice`, and `parallel_tool_calls`
- `enable_thinking` or `chat_template_kwargs.enable_thinking`

## Useful Runtime Flags

| Area | Options | Notes |
| --- | --- | --- |
| Common | `--device`, `--dtype`, `--compile-mode`, `--compile-fullgraph` | `device` supports `auto`, `cpu`, and `xpu`; `dtype` supports `auto`, `float32`, `float16`, and `bfloat16` |
| Prefill and prompt cache | `--prefill-chunk-size`, `--prompt-cache-size`, `--prompt-cache-max-tokens`, `--profile-runtime` | Available on text-generation CLIs and server |
| KV cache | `--kv-cache-quantization`, `--kv-cache-quant-bits`, `--kv-cache-residual-len` | TurboQuant is supported by the Qwen3.5 and Gemma4 runtimes |
| Qwen large-model controls | `--offload-mode`, `--offload-vision`, `--expert-quant`, `--weight-quant`, `--resident-expert-layers`, `--resident-expert-layer-indices`, `--cached-experts-per-layer` | Mainly useful for large Qwen3.5 MoE or vision-capable checkpoints |
| Server defaults and safety | `--enable-thinking`, `--disable-thinking`, `--reasoning-format`, `--max-completion-tokens`, `--min-free-memory-mib`, `--reserve-memory-mib`, `--max-estimated-usage-ratio`, `--generation-memory-safety-factor` | Controls default chat behavior and request admission |
| Server batching and metrics | `--scheduler-max-batch-size`, `--scheduler-batch-wait-ms`, `--metrics-log-interval-seconds` | Continuous batching is enabled only when batch size is above `1` |
| Speech synthesis | `--language`, `--speaker`, `--instruct`, `--ref-audio`, `--ref-text`, `--x-vector-only-mode`, `--response-format` | `anna-speak` supports `wav` and `flac`; the API also supports `pcm` |

## Development

Run the test suite:

```bash
python -m pytest
```

If you are working on XPU fused kernels, build the custom operator first and then use the project CLIs or tests against the compiled library in `.build/anna_gated_delta_fused`.
