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
  --kv-cache-quantization auto `
  --enable-flashqla-gdn-prefill `
  --scheduler-profile throughput `
  --metrics-log-interval-seconds 2 `
  --profile-runtime `
  --host 127.0.0.1 `
  --port 8000
```

`--scheduler-profile throughput` maps to larger batches, longer coalesce waits, and dynamic token budgets.
For interactive low-latency usage, use `--scheduler-profile interactive` (or override individual `--scheduler-*` knobs).
`--kv-cache-quantization auto` enables TurboQuant when installed and picks bits/residual from model-size presets.

### Interactive (Low-Latency) XPU Serving Example

```powershell
anna-serve `
  --model-dir D:\Models\Qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --weight-quant int4 `
  --kv-cache-quantization auto `
  --scheduler-profile interactive `
  --host 127.0.0.1 `
  --port 8000
```

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
  --gdn-decode-auto-compare `
  --batch-size 4 `
  --num-heads 32 `
  --head-dim 128 `
  --gdn-value-head-dim 128 `
  --dtype bf16 `
  --warmup 20 `
  --iters 100 `
  --gdn-decode-value-blocks 1,2,4,8,16,32
```

Multi-shape Gated Delta decode compare matrix:

```powershell
python tools\bench_xpu_hotspots.py `
  --gdn-decode-only `
  --gdn-decode-auto-compare `
  --gdn-decode-batch-head-cases 1x16,1x32,2x32,4x32 `
  --gdn-value-head-dims 128,256 `
  --head-dim 128 `
  --gdn-decode-value-blocks 4,8,16,32 `
  --dtype bf16 `
  --warmup 20 `
  --iters 100
```

Arc default-path preset with explicit compare:

```powershell
python tools\bench_xpu_hotspots.py `
  --gdn-decode-only `
  --gdn-decode-default-compare `
  --gdn-decode-compare-only `
  --gdn-decode-seeds 20260716,20260717 `
  --gdn-decode-shape-presets arc-default `
  --head-dim 128 `
  --dtype bf16 `
  --warmup 20 `
  --iters 100
```

Arc legacy row-band preset with multi-seed aggregation:

```powershell
python tools\bench_xpu_hotspots.py `
  --gdn-decode-only `
  --gdn-decode-auto-compare `
  --gdn-decode-compare-only `
  --gdn-decode-seeds 20260716,20260717 `
  --gdn-decode-shape-presets arc-legacy-v256-block4 `
  --head-dim 128 `
  --dtype bf16 `
  --warmup 20 `
  --iters 100
```

One-command Arc decode validation:

```powershell
python tools\validate_arc_gdn_decode.py `
  --build-first `
  --json-output bench_logs\arc_gdn_decode_baseline_a770.json `
  --warmup 20 `
  --iters 100
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
- `--prompt-cache-size N`: number of **exact** text prompt KV caches to keep; `0` disables prompt cache.
- `--prompt-cache-max-tokens N`: only exact-cache prompts up to N tokens; `0` means no limit.
- **Cache roles**: exact full-prompt reuse → prompt cache; shared system-prefix across different prompts → paged `PrefixBlockPool`. When prompt cache is enabled, exact-cache-eligible prompts skip prefix-block registration to avoid dual residency. Disable prefix sharing entirely with `ANNA_PREFIX_KV_SHARE=0`. Prefix hit/miss rates are exposed on `/healthz` and metrics logs.
- `--kv-cache-quantization none|turboquant|auto`: KV-cache quantization. `auto` enables TurboQuant when the optional dependency is installed and selects bits/residual from size-tier presets unless bits/residual are set explicitly.
- `--kv-cache-quant-bits 2|3|4`: TurboQuant KV bit width. When omitted with turboquant/auto: small≈4, medium≈3, large/xlarge≈2.
- `--kv-cache-residual-len N`: keep the newest N KV tokens in full precision. When omitted with turboquant/auto: 128/128/96/64 by tier.
- `--weight-quant auto|none|int4`: dense weight quantization strategy. `auto` promotes to int4 when estimated weights exceed **85%** of total XPU memory (**70%** for MoE models or experts offload).
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

- `--enable-flashqla-gdn-prefill`: enable the XPU SYCL Gated Delta prefill path in **strict** mode (no fallback).
- `--flashqla-gdn-prefill-mode off|strict|prefer`: FlashQLA policy. `strict` hard-fails on unsupported device/shape/dtype/op; `prefer` degrades to the default fused or torch prefill with a one-shot warning per reason. Env `ANNA_XPU_FLASHQLA_GDN_PREFILL` accepts the same values (`1`/`true`/`on` ≡ `strict`).
- `--xpu-int4-matmul auto|torch|dequant|gemv`: XPU int4 dense linear execution strategy.
  - **`auto` (default)**: PyTorch `int4pack` on Arc — no M-row GEMV/dequant threshold.
  - **`torch`**: force int4pack.
  - **`gemv`**: opt-in SYCL GEMV (decode experiments); not selected by auto.
  - **`dequant`**: full dequant + `F.linear` (debug).
- LM head int4 top-k fused is **on by default** for XPU int4 (`top_k ≤ 16`). Disable with `ANNA_XPU_DISABLE_LM_HEAD_INT4_TOPK=1`. Legacy `ANNA_ENABLE_INT4_LM_HEAD_TOPK_FUSED=0|1` still forces off/on.
- XPU int4 layout cache: on first int4 conversion Anna writes `{model}/.anna/xpu_int4_cache` (override with `ANNA_XPU_INT4_CACHE_DIR`). Version/fingerprint mismatch rebuilds; load/save failures fall back to live quantize. Disable with `ANNA_XPU_DISABLE_INT4_CACHE=1`. Inspect with `anna-xpu-int4-cache --model-dir ...`.
- `ANNA_GATED_DELTA_OP_LIB`: explicitly point to a fused-op `.pyd` / `.so`.
- `ANNA_XPU_GATED_DELTA_DECODE_STRATEGY=auto|single|single_group|untiled|tiled|tiled_value`: Gated Delta decode kernel strategy. Leave unset (or set `auto`) for the built-in Arc shape lookup; force `single` / `tiled` only when debugging.
- `ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK=N`: optional override for the tiled decode value block. When unset, Anna picks a device/shape default from the baked-in Arc table (no manual env tuning required for common Qwen3.5 shapes).
- `ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS=N`: optional override for `auto`; when set, bypass the device/shape strategy lookup and use this single-group element threshold.

Arc A770 decode defaults currently baked into the fused op (rows = `batch * heads`). Primary gates cover **K=128**; K∈{64,256} reuse the same V-based defaults. Uncommon K/V fall back to an interpretable power-of-two block in `{1,2,4}`:

| V (value head dim) | Default value block | Default strategy | Notes |
| --- | --- | --- | --- |
| 64 | 8 | `single` | Also keeps forced block=16 on `single` |
| 128 | 8 | `tiled` | Prefer block=8 over block=16 |
| 256 | 4 | `tiled` | Rows 264..304 use value block=8 |

Validate the K=128 table with `python tools/validate_arc_gdn_decode.py --preset quick` (or `full` / `watch`).

### Continuous Batching and Token Budgets

- `--scheduler-profile none|interactive|throughput`: built-in continuous-batching preset. Explicit `--scheduler-*` flags override the profile.
  - **interactive**: batch=2, wait=0.5ms, prefill interval=1, prefill/decode budgets 1024/2048, max waiting=32, dynamic budgets on, skip idle coalesce wait (TTFT-friendly).
  - **throughput**: batch=8, wait=8ms, prefill interval=4, budgets 2048/4096, max waiting=128, dynamic budgets on, idle coalesce wait kept.
- `--scheduler-max-batch-size N`: enable continuous batching when greater than `1`.
- `--scheduler-batch-wait-ms MS`: wait time for request coalescing; higher values may improve throughput but increase tail latency. Idle GPU skips this wait under interactive defaults (TTFT).
- `--scheduler-prefill-interval-steps N`: insert pending prefill scheduling every N decode steps.
- `--scheduler-max-prefill-tokens N`: prompt-token budget for one prefill admission wave; `0` disables the budget (unless dynamic budgets derive a soft default).
- `--scheduler-max-decode-tokens N`: cached-sequence-token budget for one decode batch; `0` disables the budget.
- `--scheduler-max-waiting-requests N`: reject new work with HTTP 429 when the waiting queue reaches N (backpressure); `0` = unlimited.
- `--scheduler-dynamic-token-budget` / `--no-scheduler-dynamic-token-budget`: scale prefill/decode budgets from free XPU memory and running sequence lengths.
- `--scheduler-max-queue-wait-ms MS`: fairness — force prefill admission when the oldest waiter exceeds this age, so long decodes cannot starve new prompts.
- `--metrics-log-interval-seconds S`: emit aggregate runtime metrics periodically; `0` disables metrics logging. Logs include prompt-cache and prefix-block hit rates plus queue rejections.

### ASR Serving Options

- `--asr-max-inference-batch-size N`: maximum number of concurrent transcription requests Anna will coalesce into one upstream XPU call (server-side continuous batch), and the chunk batch size passed to `qwen-asr`.
- `--asr-max-new-tokens N`: maximum generated text tokens per Qwen3-ASR chunk.

### TTS / ASR wrapper boundary

Qwen3-TTS and Qwen3-ASR run as **upstream wrappers** (`qwen-tts` / `qwen-asr`):

- **Anna owns:** OpenAI-compatible API/CLI, XPU-only load policy (ASR), process device execution gate, metrics, OOM/device-lost mapping, and ASR continuous request batching.
- **Upstream owns:** audio encoder / talker / vocoder kernels and internal generate loops.
- **Not planned:** porting TTS/ASR heavy kernels into Anna SYCL fused ops while Qwen3.5 text remains the primary Arc optimization surface.
- **Process isolation:** `anna-serve` loads one model family per process. Co-resident audio+text engines must serialize on the process device gate (`DeviceExecutionGate`; set `ANNA_XPU_SERIALIZE_ALL=1` to force text engines onto the same gate). This is serialization, not multi-tenant VRAM partitioning.

### Gemma4 serve baseline

Gemma4 reuses the Qwen text engine shell (scheduler, sampling, prompt cache, int4 dense weights, TurboQuant size-tier presets on full-attention layers). Deliberate gaps vs Qwen3.5 (paged KV, prefix block pool, GDN, MoE) are listed under `/healthz` → `ops_parity`.

Local throughput baseline (same prompts as Qwen text scenarios):

```powershell
anna-bench --model-dir D:\Models\Gemma4 --scenario gemma-text-short --device xpu --runs 5
anna-bench --model-dir D:\Models\Gemma4 --scenario gemma-text-long --device xpu --runs 3
```

Serve concurrency baseline (start `anna-serve` with `--scheduler-profile interactive` or `throughput` first):

```powershell
python tools\bench_api_concurrency.py --scenario gemma-concurrent-short --concurrency 4 --requests 16 --healthz
python tools\bench_api_concurrency.py --scenario gemma-mixed --concurrency 4 --requests 16
```

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
- `--scenario custom|concurrent-short|single-long|mixed|repeated-system|gemma-concurrent-short|gemma-single-long|gemma-mixed|multimodal-*`: built-in prompt scenario (gemma-* aliases are the Gemma4 serve baseline).
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
- `--gdn-decode-batch-head-cases LIST`: run multiple `batch x heads` cases in one decode-profile sweep, for example `1x16,1x32,4x32`.
- `--gdn-decode-shape-presets LIST`: run named Arc decode preset matrices such as `arc-default`, `arc-legacy-v128-block8`, and `arc-legacy-v256-block4`.
- `--gdn-decode-value-blocks LIST`: test multiple value-block sizes; when omitted with `--gdn-decode-shape-presets`, Anna uses the preset-recommended blocks.
- `--gdn-value-head-dims LIST`: run multiple value-head dims in one decode-profile sweep; overrides `--gdn-value-head-dim`.
- `--gdn-decode-single-min-elements N`: override the auto-strategy threshold.
- `--gdn-decode-seed N`: fix the decode-profile inputs for repeatable A/B comparisons; negative keeps per-run random inputs.
- `--gdn-decode-seeds LIST`: aggregate each decode-profile case across multiple fixed seeds; overrides `--gdn-decode-seed`.
- `--gdn-decode-timing-repeats N`: take N timing samples per candidate and report the median.
- `--gdn-decode-auto-compare`: add per-value-block `auto` vs best-explicit summary rows after the decode sweep.
- `--gdn-decode-default-compare`: compare the default decode path against explicit `single` and `tiled`.
- `--gdn-decode-default-block-compare`: compare the default decode path against forced value-block overrides.
- `--gdn-decode-compare-only`: skip the full strategy sweep rows and print only the compare summaries.
- `--arc-profile`: add Arc A770/A750-oriented int4 profile rows.
- `--csv-output PATH`: save general hotspot benchmark results.

`tools/validate_arc_gdn_decode.py`:

- Runs the standard Arc decode benchmark presets plus targeted decode regressions in one command.
- `--presets LIST`: choose any subset of `arc-default`, `arc-legacy-v128-block8`, `arc-legacy-v256-block4`.
- `--build-first`: rebuild the fused-op library before validation.
- `--json-output PATH`: write the benchmark gate summaries, resolved blocks, and pytest status to a structured JSON artifact.
- By default, benchmark output is checked against preset-specific speed-ratio gates; use `--skip-bench-gates` for quick smoke runs with reduced warmup/iters.
- `--skip-bench` / `--skip-pytest`: run only the benchmark portion or only the targeted regressions.

## Troubleshooting

- XPU unavailable: confirm that your PyTorch build supports XPU and that Intel GPU drivers and Level Zero runtime are installed.
- Slow first request: usually caused by weight loading, lazy kernel loading, fused-op initialization, or `torch.compile`.
- XPU out of memory: try `--dtype bf16`, `--weight-quant int4`, `--kv-cache-quantization turboquant`, a lower output token cap, or expert offload.
- Low throughput with decode batch average near 1: increase `--scheduler-max-batch-size` and `--scheduler-batch-wait-ms`, then inspect TTFT/ITL.
- High cache stack/split/compact cost: inspect KV cache row management and batch membership changes before writing new kernels.
- High decode p95/p99 with stable batch/cache metrics: inspect `--profile-runtime` component timing for attention, Gated Delta, LM head, and sampling.

## License

See [LICENSE](LICENSE).
