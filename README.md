# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna is a local inference runtime for large language, multimodal, and speech models. It provides an OpenAI-compatible HTTP API and command-line tools for local generation, serving, benchmarking, and Qwen3-TTS speech synthesis.

The runtime is built on PyTorch and is optimized for Intel Arc / XPU. CPU execution is useful for development and smaller tests.

## Features

- OpenAI-compatible endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/audio/speech`, `/v1/models`
- Non-streaming and streaming text generation
- Chat, plain text completion, multimodal chat, function calling, and reasoning output
- Qwen3-TTS speech synthesis
- Intel XPU options for `torch.compile`, KV-cache quantization, int4 weight quantization, MoE expert offload, prompt cache, and continuous batching
- CLI tools: `anna-serve`, `anna-generate`, `anna-bench`, `anna-speak`, `anna-xpu-int4-cache`

## Supported Models

Anna loads a local model directory. A normal Hugging Face-style directory should contain `config.json` and model weights. A Qwen GGUF layout is also supported for compatible Qwen3.5 MoE models.

Model family detection is based on `config.json`:

| `model_type` | Runtime | Main use |
| --- | --- | --- |
| `qwen3_tts` | Qwen3-TTS | Speech synthesis |
| `gemma4` | Gemma 4 | Text, chat, multimodal chat with audio |
| anything else | Qwen3.5 text / VL | Text, chat, image/video multimodal chat |

Use `anna-generate` and `anna-bench` for text-generation models. Use `anna-speak` for Qwen3-TTS. Use `anna-serve` for any supported family; unsupported routes return an API error for the loaded model.

## Requirements

- Python 3.11+
- PyTorch 2.7+ installed for your hardware
- A local model directory
- For Intel Arc / XPU: a PyTorch build with XPU support and the Intel GPU runtime
- Optional fused XPU operator build: Intel oneAPI DPC++ compiler, and Visual Studio Build Tools on Windows

The package declares Python dependencies in `pyproject.toml`. PyTorch and Intel GPU drivers should be installed according to your target machine.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Anna.git
cd Anna
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Install Anna:

```bash
python -m pip install -U pip
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

Check PyTorch and XPU availability:

```bash
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available() if hasattr(torch, 'xpu') else False)"
```

Optional: build the fused XPU operator:

```bash
python tools/build_gated_delta_fused_op.py
```

## Quick Start

Start the OpenAI-compatible server:

```bash
anna-serve --model-dir /path/to/model --host 127.0.0.1 --port 8000
```

Send a chat request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "Explain KV cache in one paragraph."}
    ],
    "max_completion_tokens": 128
  }'
```

Use streaming:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a short haiku about local AI."}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Run one-shot generation from the CLI:

```bash
anna-generate \
  --model-dir /path/to/text-model \
  --prompt "Explain KV cache in one paragraph." \
  --max-new-tokens 128
```

Run a benchmark:

```bash
anna-bench \
  --model-dir /path/to/model \
  --prompt "Hello" \
  --warmup 1 \
  --runs 3
```

Synthesize speech with Qwen3-TTS:

```bash
anna-speak \
  --model-dir /path/to/qwen3-tts \
  --input "Hello from Anna." \
  --output out.wav
```

## Intel XPU Examples

Select a specific XPU device:

```bash
anna-serve \
  --model-dir /path/to/model \
  --device xpu \
  --xpu-device-index 0 \
  --dtype bfloat16
```

Use memory-saving runtime options:

```bash
anna-serve \
  --model-dir /path/to/model \
  --device xpu \
  --dtype bfloat16 \
  --kv-cache-quantization turboquant \
  --kv-cache-quant-bits 4 \
  --weight-quant auto \
  --prompt-cache-size 4
```

Enable continuous batching for API serving:

```bash
anna-serve \
  --model-dir /path/to/model \
  --device xpu \
  --scheduler-max-batch-size 4 \
  --scheduler-batch-wait-ms 2
```

Check whether Anna will create an XPU int4 sidecar cache:

```bash
anna-xpu-int4-cache \
  --model-dir /path/to/model \
  --weight-quant auto \
  --xpu-total-memory-gib 16
```

## Multimodal Requests

For supported vision models, use OpenAI-style content parts:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image."},
          {"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}
        ]
      }
    ],
    "max_completion_tokens": 128
  }'
```

`image_url`, `video_url`, and `audio_url` content parts are accepted by the API schema. Actual support depends on the loaded model family.

## API Routes

| Route | Method | Purpose |
| --- | --- | --- |
| `/healthz` | `GET` | Runtime health and model status |
| `/v1/models` | `GET` | List the loaded model ID |
| `/v1/chat/completions` | `POST` | Chat and multimodal chat |
| `/v1/completions` | `POST` | Plain text completion |
| `/v1/audio/speech` | `POST` | Qwen3-TTS speech synthesis |

## `anna-serve` Options

`anna-serve` has one required option. All other options are optional and have runtime defaults.

| Required option | Meaning |
| --- | --- |
| `--model-dir PATH` | Local model directory. Use a Hugging Face-style model folder with `config.json` and weights, or a compatible GGUF layout. |

Common service options:

| Optional option | Default | Meaning |
| --- | --- | --- |
| `--model-name NAME` | derived from path | Model ID exposed by `/v1/models` and API responses. |
| `--host HOST` | `127.0.0.1` | Server bind address. Use `0.0.0.0` to listen on all interfaces. |
| `--port PORT` | `8000` | Server port. |
| `--log-level LEVEL` | `info` | Uvicorn and Anna logging level. |
| `--device DEVICE` | `auto` | `auto`, `cpu`, or `xpu`. `auto` prefers XPU when available. |
| `--dtype DTYPE` | `auto` | Compute dtype such as `auto`, `float32`, `float16`, or `bfloat16`. |
| `--max-completion-tokens N` | model/default estimate | Default output token cap for API requests that omit `max_tokens` / `max_completion_tokens`. |
| `--temperature FLOAT` | `0.7` | Default sampling temperature when the request omits it. |
| `--top-p FLOAT` | `0.8` | Default nucleus sampling probability. |
| `--top-k N` | `20` | Default top-k sampling limit. Set `0` to disable. |
| `--min-p FLOAT` | `0.0` | Default min-p sampling threshold. |
| `--presence-penalty FLOAT` | `1.5` | Default additive presence penalty. |
| `--repetition-penalty FLOAT` | `1.0` | Default multiplicative repetition penalty. |
| `--enable-thinking` / `--disable-thinking` | enabled | Default thinking behavior for chat requests that omit thinking fields. |
| `--reasoning-format none\|deepseek` | `deepseek` | Reasoning output format. `deepseek` returns `reasoning_content` separately when available. |

Performance and memory options:

| Optional option | Default | Meaning |
| --- | --- | --- |
| `--compile-mode MODE` | `auto` | `none`, `auto`, `default`, `reduce-overhead`, or `max-autotune`. First requests can pay compile cost. |
| `--compile-fullgraph` | off | Request full-graph capture when `torch.compile` is enabled. |
| `--prefill-chunk-size N` | `0` | Split long text-only prefills into chunks. `0` lets Anna auto-size on XPU. |
| `--prompt-cache-size N` | `0` | Keep up to N exact text prompt KV caches resident. `0` disables prompt cache. |
| `--prompt-cache-max-tokens N` | `0` | Only cache prompts up to N tokens. `0` means no token limit. |
| `--kv-cache-quantization none\|turboquant` | `none` | Quantize compatible KV caches. |
| `--kv-cache-quant-bits 2\|3\|4` | `4` | TurboQuant KV-cache bit width. |
| `--kv-cache-residual-len N` | `128` | Keep the newest N KV tokens in full precision. |
| `--offload-mode auto\|none\|experts` | `auto` | MoE expert offload strategy. |
| `--offload-vision` | off | Keep the vision tower on CPU even when the language model runs on XPU. |
| `--expert-quant auto\|none\|int4` | `auto` | Quantization for MoE expert weights on XPU. |
| `--weight-quant auto\|none\|int4` | `auto` | Quantization for dense language-model weights on XPU. |
| `--resident-expert-layers N` | auto | Keep the first N sparse MoE layers fully resident on the execution device. |
| `--resident-expert-layer-indices LIST` | unset | Comma-separated 0-based sparse layer indices to keep resident. Overrides `--resident-expert-layers`. |
| `--cached-experts-per-layer N` | auto | Max offloaded experts cached on XPU per sparse MoE layer. `0` disables. |

XPU and server runtime options:

| Optional option | Default | Meaning |
| --- | --- | --- |
| `--xpu-device-index N` | unset | Select an Intel XPU with `ONEAPI_DEVICE_SELECTOR=level_zero:N`. |
| `--no-xpu-env-defaults` | off | Do not set Anna's recommended Level Zero environment defaults before XPU startup. |
| `--xpu-int4-matmul auto\|torch\|dequant` | runtime default | XPU int4 dense linear execution strategy. |
| `--enable-flashqla-gdn-prefill` | off | Enable the Intel FlashQLA-compatible GDN prefill path on XPU. Unsupported shapes/devices/dtypes raise immediately. |
| `--no-inference-warmup` | off | Skip the small post-load XPU warmup. First client request may then pay lazy kernel load. |
| `--warmup-prefill-tokens N` | `2` | Text token count used by post-load XPU warmup prefill. |
| `--warmup-decode-steps N` | `1` | Decode steps used by post-load XPU warmup. |
| `--warmup-batch-size N` | `1` | Batch size used by post-load XPU warmup. |
| `--profile-runtime` | off | Log synchronized XPU timing and memory stats. |
| `--min-free-memory-mib N` | `1024` | Minimum free XPU memory required before generation starts. |
| `--reserve-memory-mib N` | `512` | Extra XPU memory margin preserved during request admission. |
| `--max-estimated-usage-ratio R` | `0.9` | Reject requests whose estimated usage exceeds this fraction of total XPU memory. |
| `--generation-memory-safety-factor R` | `2.0` | Multiplier applied to estimated generation memory. |
| `--scheduler-max-batch-size N` | `1` | Enable continuous batching when greater than `1`. |
| `--scheduler-batch-wait-ms MS` | `2.0` | Wait time used to coalesce requests when batching is enabled. |
| `--scheduler-prefill-interval-steps N` | `1` | Prefill scheduling interval while continuous batching is active. |
| `--metrics-log-interval-seconds S` | `10.0` | Emit aggregated runtime metrics every S seconds. `0` disables metrics logging. |

For the full option list, run:

```bash
anna-serve --help
anna-generate --help
anna-bench --help
anna-speak --help
```

## Troubleshooting

- If XPU is not detected, confirm that your PyTorch build supports XPU and that Intel GPU drivers are installed.
- If the wrong GPU is selected on a system with multiple Intel GPUs, pass `--xpu-device-index N`.
- If the first request is slow, it may be paying model load, kernel load, or `torch.compile` cost.
- If memory is tight on XPU, try `--dtype bfloat16`, `--kv-cache-quantization turboquant`, `--weight-quant auto`, `--offload-mode experts`, or a lower token limit.
- If a route fails, verify that the loaded model family supports that task.

## License

See [LICENSE](LICENSE).
