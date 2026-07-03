# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna is a PyTorch-based local inference runtime designed to provide high-throughput, low-latency OpenAI-compatible serving on Intel Arc / XPU. The current runtime focuses on Qwen3.5 text and multimodal inference, Gemma4 text inference, Qwen3-TTS speech synthesis, and Qwen3-ASR speech recognition.

Models under `models/` are used for local testing, model-family analysis, and architecture inspection. Anna is not tied to those exact model directories. At runtime, pass any compatible local model directory.

## Overview

- OpenAI-compatible HTTP API for chat, completion, speech synthesis, speech transcription, and model listing.
- CLI tools: `anna-serve`, `anna-generate`, `anna-bench`, `anna-speak`, `anna-transcribe`, `anna-xpu-int4-cache`.
- Intel XPU optimization paths: continuous batching, token-budget scheduling, TurboQuant KV cache, XPU int4 weights, prompt cache, and fused SYCL custom operators.
- Qwen3.5 inference hotspot work includes Gated Delta, attention, RMSNorm, rotary, LM head, and related fused-kernel paths.
- Includes local HTTP concurrency benchmarking and XPU hotspot microbench tooling.

## Supported Models

Anna detects the model family from `config.json`; it does not rely on the directory name.

| `model_type` | Runtime | Entry points |
| --- | --- | --- |
| `qwen3_tts` | Qwen3-TTS | `anna-speak`, `/v1/audio/speech` |
| `qwen3_asr` | Qwen3-ASR | `anna-transcribe`, `/v1/audio/transcriptions` |
| `gemma4` | Gemma4 | `anna-serve`, `anna-generate`, `anna-bench` |
| Other compatible configs | Qwen3.5 text / VL | `anna-serve`, `anna-generate`, `anna-bench` |

A model directory typically contains `config.json`, tokenizer files, and weight files. Compatible Qwen3.5 MoE models can also use the Qwen GGUF layout.

## Environment Setup

Base requirements:

- Python 3.11+
- PyTorch 2.7+; XPU inference requires a PyTorch build with Intel XPU support
- Intel GPU driver and oneAPI Level Zero runtime
- Windows custom XPU-operator builds require Intel oneAPI DPC++ and Visual Studio Build Tools

Install a development environment:

```powershell
conda activate anna
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Check XPU availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else None)"
```

Build the fused XPU operator on Windows + oneAPI:

```powershell
$env:ANNA_DPCPP = "D:\Intel\oneAPI\compiler\latest\bin\dpcpp.exe"
$env:ANNA_VCVARS64 = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
python tools\build_gated_delta_fused_op.py
```

When running directly from source in development mode, you can set:

```powershell
$env:PYTHONPATH = "D:\Projects\anna\src"
$env:ANNA_GATED_DELTA_OP_LIB = "D:\Projects\anna\.build\anna_gated_delta_fused\anna_gated_delta_fused.pyd"
```

## Quick Start

### Start the OpenAI-Compatible Server

```powershell
anna-serve `
  --model-dir D:\Models\Qwen3.5 `
  --model-name qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --host 127.0.0.1 `
  --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

Chat request:

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{`"model`":`"qwen3.5`",`"messages`":[{`"role`":`"user`",`"content`":`"Explain KV cache in three sentences.`"}],`"max_completion_tokens`":128}"
```

Streaming chat:

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{`"model`":`"qwen3.5`",`"messages`":[{`"role`":`"user`",`"content`":`"Write a technical summary of local inference serving.`"}],`"stream`":true,`"stream_options`":{`"include_usage`":true}}"
```

### High-Throughput XPU Serving Example

```powershell
anna-serve `
  --model-dir D:\Models\Qwen3.5 `
  --model-name qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --compile-mode none `
  --weight-quant int4 `
  --kv-cache-quantization turboquant `
  --kv-cache-quant-bits 2 `
  --kv-cache-residual-len 128 `
  --enable-flashqla-gdn-prefill `
  --scheduler-max-batch-size 8 `
  --scheduler-batch-wait-ms 8 `
  --scheduler-prefill-interval-steps 4 `
  --scheduler-max-prefill-tokens 2048 `
  --scheduler-max-decode-tokens 4096 `
  --metrics-log-interval-seconds 2 `
  --profile-runtime `
  --host 127.0.0.1 `
  --port 8000
```

This profile is intended for throughput testing. For interactive low-latency usage, reduce `--scheduler-batch-wait-ms` or lower `--scheduler-max-batch-size`.

### One-Shot Text Generation

```powershell
anna-generate `
  --model-dir D:\Models\Qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --prompt "Explain why separating prefill and decode can reduce tail latency." `
  --max-new-tokens 128
```

### Local Benchmark

```powershell
anna-bench `
  --model-dir D:\Models\Qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --prompt "Hello" `
  --warmup 1 `
  --runs 3 `
  --max-new-tokens 128
```

### HTTP Concurrency Benchmark

Start `anna-serve` first, then run:

```powershell
python tools\bench_api_concurrency.py `
  --base-url http://127.0.0.1:8000 `
  --model qwen3.5 `
  --scenario concurrent-short `
  --requests 16 `
  --concurrency 8 `
  --max-tokens 64 `
  --temperature 0 `
  --top-k 1 `
  --healthz
```

The output includes success count, RPS, output tokens/s, TTFT, ITL, and latency percentiles.

### XPU Hotspot Profiling

Gated Delta decode strategy sweep:

```powershell
python tools\bench_xpu_hotspots.py `
  --gdn-decode-only `
  --batch-size 4 `
  --num-heads 32 `
  --head-dim 128 `
  --gdn-value-head-dim 128 `
  --dtype bf16 `
  --warmup 20 `
  --iters 100 `
  --gdn-decode-value-blocks 1,2,4,8,16,32
```

General hotspot suite:

```powershell
python tools\bench_xpu_hotspots.py `
  --batch-size 1 `
  --seq-len 1 `
  --hidden-size 2560 `
  --num-heads 32 `
  --num-kv-heads 8 `
  --head-dim 128 `
  --kv-len 512 `
  --dtype bf16 `
  --warmup 20 `
  --iters 100
```

### Qwen3-TTS

```powershell
anna-speak `
  --model-dir D:\Models\Qwen3-TTS `
  --input "Hello from Anna." `
  --output out.wav
```

### Qwen3-ASR

```powershell
anna-transcribe `
  --model-dir D:\Models\Qwen3-ASR `
  --audio input.wav `
  --device xpu `
  --language English
```

Upload audio through HTTP:

```powershell
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions `
  -F model=qwen3-asr `
  -F file=@input.wav `
  -F language=English `
  -F response_format=verbose_json
```

## API Routes

- `GET /healthz`: runtime, model, memory, KV cache, and service metrics.
- `GET /v1/models`: currently loaded model ID.
- `POST /v1/chat/completions`: chat, multimodal chat, streaming output, and tool-call-compatible responses.
- `POST /v1/completions`: text completion.
- `POST /v1/audio/speech`: Qwen3-TTS speech synthesis.
- `POST /v1/audio/transcriptions`: Qwen3-ASR speech recognition.

## `anna-serve` Options

### Base Options

- `--model-dir PATH`: required local model directory.
- `--model-name NAME`: model ID exposed by the API; derived from the path when omitted.
- `--host HOST`: bind address, default `127.0.0.1`; use `0.0.0.0` for LAN access.
- `--port PORT`: bind port, default `8000`.
- `--log-level LEVEL`: logging level, default `info`.
- `--device auto|cpu|xpu`: execution device; `auto` prefers XPU.
- `--xpu-device-index N`: select a specific XPU on systems with multiple Intel GPUs.
- `--no-xpu-env-defaults`: do not set Anna's recommended Level Zero environment variables.
- `--dtype DTYPE`: compute dtype, such as `auto`, `bf16`, `bfloat16`, `float16`, or `float32`.

### Generation Defaults

These values only apply when an API request omits the matching field.

- `--max-completion-tokens N`: default output token cap.
- `--temperature FLOAT`: default sampling temperature.
- `--top-p FLOAT`: default nucleus sampling probability.
- `--top-k N`: default top-k; `0` disables top-k.
- `--min-p FLOAT`: default min-p threshold.
- `--presence-penalty FLOAT`: default presence penalty.
- `--repetition-penalty FLOAT`: default repetition penalty.
- `--enable-thinking` / `--disable-thinking`: default thinking behavior for chat requests.
- `--reasoning-format none|deepseek`: reasoning output format; `deepseek` separates reasoning into `reasoning_content`.

### Compilation and Warmup

- `--compile-mode none|auto|default|reduce-overhead|max-autotune`: `torch.compile` mode; serving usually uses `none` or `auto`.
- `--compile-fullgraph`: request fullgraph capture when compile is enabled.
- `--no-inference-warmup`: skip the small post-load XPU warmup.
- `--warmup-prefill-tokens N`: prefill token count used for warmup, default `2`.
- `--warmup-decode-steps N`: decode steps used for warmup, default `1`.
- `--warmup-batch-size N`: warmup batch size, default `1`.

### Memory and Weight Strategy

- `--prefill-chunk-size N`: chunk size for long prompt prefill; `0` enables XPU auto-sizing.
- `--prompt-cache-size N`: number of exact text prompt KV caches to keep; `0` disables prompt cache.
- `--prompt-cache-max-tokens N`: only cache prompts up to N tokens; `0` means no limit.
- `--kv-cache-quantization none|turboquant`: KV-cache quantization mode.
- `--kv-cache-quant-bits 2|3|4`: TurboQuant KV bit width.
- `--kv-cache-residual-len N`: keep the newest N KV tokens in full precision.
- `--weight-quant auto|none|int4`: dense weight quantization strategy.
- `--expert-quant auto|none|int4`: MoE expert weight quantization strategy.
- `--offload-mode auto|none|experts`: MoE expert offload strategy.
- `--offload-vision`: keep the vision tower on CPU to reduce XPU memory use.
- `--resident-expert-layers N`: keep the first N sparse MoE layers resident on the execution device.
- `--resident-expert-layer-indices LIST`: explicit resident sparse MoE layer indices; overrides `--resident-expert-layers`.
- `--cached-experts-per-layer N`: number of offloaded experts cached per layer; `0` disables caching.
- `--min-free-memory-mib N`: minimum free XPU memory required before generation.
- `--reserve-memory-mib N`: memory margin reserved during request admission.
- `--max-estimated-usage-ratio R`: reject requests whose estimate exceeds ratio R of total XPU memory.
- `--generation-memory-safety-factor R`: safety multiplier for generation memory estimates.

### XPU Fused Ops and Int4 Kernels

- `--enable-flashqla-gdn-prefill`: enable the XPU SYCL Gated Delta prefill path; unsupported shapes, dtypes, or devices raise immediately.
- `--xpu-int4-matmul auto|torch|dequant`: XPU int4 dense linear execution strategy.
- `ANNA_GATED_DELTA_OP_LIB`: explicitly point to a fused-op `.pyd` / `.so`.
- `ANNA_XPU_GATED_DELTA_DECODE_STRATEGY=auto|single|single_group|untiled|tiled|tiled_value`: Gated Delta decode kernel strategy.
- `ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK=N`: value block size for tiled decode.
- `ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS=N`: optional override for `auto`; when set, bypass the device/shape lookup and use this single-group element threshold.

### Continuous Batching and Token Budgets

- `--scheduler-max-batch-size N`: enable continuous batching when greater than `1`.
- `--scheduler-batch-wait-ms MS`: wait time for request coalescing; higher values may improve throughput but increase tail latency.
- `--scheduler-prefill-interval-steps N`: insert pending prefill scheduling every N decode steps.
- `--scheduler-max-prefill-tokens N`: prompt-token budget for one prefill admission wave; `0` disables the budget.
- `--scheduler-max-decode-tokens N`: cached-sequence-token budget for one decode batch; `0` disables the budget.
- `--metrics-log-interval-seconds S`: emit aggregate runtime metrics periodically; `0` disables metrics logging.

### ASR Serving Options

- `--asr-max-inference-batch-size N`: number of Qwen3-ASR audio chunks per XPU inference batch.
- `--asr-max-new-tokens N`: maximum generated text tokens per Qwen3-ASR chunk.

## `anna-generate` Options

- `--model-dir PATH`: text model directory.
- `--prompt TEXT`: input prompt.
- `--max-new-tokens N`: output token cap.
- `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`: sampling options.
- `--device`, `--dtype`, `--compile-mode`, `--kv-cache-*`, `--weight-quant`: same meaning as `anna-serve`.

## `anna-bench` Options

- `--model-dir PATH`: text model directory.
- `--prompt TEXT`: benchmark prompt.
- `--warmup N`: warmup runs.
- `--runs N`: measured runs.
- `--max-new-tokens N`: output token cap per run.
- `--profile-runtime`: print XPU component timing.
- `--image PATH` / `--video PATH`: multimodal benchmark input.

## Benchmarking and Profiling Options

`tools/bench_api_concurrency.py`:

- `--base-url URL`: Anna server URL.
- `--route /v1/chat/completions|/v1/completions`: target route.
- `--model NAME`: model ID in the request body.
- `--scenario custom|concurrent-short|single-long|mixed|repeated-system`: built-in prompt scenario.
- `--requests N`: total request count.
- `--concurrency N`: concurrent worker count.
- `--max-tokens N`: output token cap per request.
- `--stream` / `--no-stream`: whether to use streaming.
- `--healthz`: fetch `/healthz` before and after the run.
- `--json`: print the final summary as JSON.

`tools/bench_xpu_hotspots.py`:

- `--batch-size N`, `--seq-len N`, `--hidden-size N`: synthetic input dimensions.
- `--num-heads N`, `--num-kv-heads N`, `--head-dim N`, `--kv-len N`: attention and GDN shapes.
- `--dtype fp16|bf16|fp32`: benchmark dtype.
- `--warmup N`, `--iters N`: warmup and measured iterations.
- `--gdn-decode-only`: only run the Gated Delta decode strategy sweep.
- `--gdn-decode-value-blocks LIST`: test multiple value-block sizes.
- `--gdn-decode-single-min-elements N`: override the auto-strategy threshold.
- `--arc-profile`: add Arc A770/A750-oriented int4 profile rows.
- `--csv-output PATH`: save general hotspot benchmark results.

## Troubleshooting

- XPU unavailable: confirm that your PyTorch build supports XPU and that Intel GPU drivers and Level Zero runtime are installed.
- Slow first request: usually caused by weight loading, lazy kernel loading, fused-op initialization, or `torch.compile`.
- XPU out of memory: try `--dtype bf16`, `--weight-quant int4`, `--kv-cache-quantization turboquant`, a lower output token cap, or expert offload.
- Low throughput with decode batch average near 1: increase `--scheduler-max-batch-size` and `--scheduler-batch-wait-ms`, then inspect TTFT/ITL.
- High cache stack/split/compact cost: inspect KV cache row management and batch membership changes before writing new kernels.
- High decode p95/p99 with stable batch/cache metrics: inspect `--profile-runtime` component timing for attention, Gated Delta, LM head, and sampling.

## License

See [LICENSE](LICENSE).
