# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna is a PyTorch-based local inference runtime for Intel Arc GPUs (XPU). It serves Qwen3.5 (text and multimodal), Gemma4 (text), and Qwen3-TTS (speech synthesis) behind an OpenAI-compatible HTTP API.

## Features

- OpenAI-compatible API: chat (streaming, multimodal, tool calls), completions, speech synthesis, and model listing.
- CLI tools: `anna-serve`, `anna-generate`, `anna-bench`, `anna-speak`, `anna-xpu-int4-cache`.
- XPU optimizations: continuous batching, token-budget scheduling, TurboQuant KV cache, int4 weight quantization, prompt cache, and fused SYCL kernels (Gated Delta, attention, RMSNorm, rotary, LM head).
- Benchmarking tools for HTTP concurrency and XPU kernel hotspots.

## Supported Models

Anna detects the model family from `config.json` — the directory name does not matter.

| `model_type` | Runtime | Entry points |
| --- | --- | --- |
| `qwen3_tts` | Qwen3-TTS | `anna-speak`, `/v1/audio/speech` |
| `gemma4` | Gemma4 | `anna-serve`, `anna-generate`, `anna-bench` |
| Other compatible configs | Qwen3.5 text / VL | `anna-serve`, `anna-generate`, `anna-bench` |

A model directory needs `config.json`, tokenizer files, and weights. Qwen3.5 MoE models can also use the Qwen GGUF layout. Gemma4 text models can run directly from a GGUF file (config, tokenizer, and weights are read from the GGUF; no separate `config.json` / `tokenizer.json` needed).

## Installation

Requirements:

- Python 3.11+
- PyTorch 2.7+ with Intel XPU support
- Intel GPU driver and oneAPI Level Zero runtime
- To build XPU custom operators on Windows: Intel oneAPI DPC++ and Visual Studio Build Tools

```powershell
conda activate anna
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

TurboQuant KV-cache quantization is optional:

```powershell
python -m pip install -e ".[quant]"
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

When running from source, set:

```powershell
$env:PYTHONPATH = "D:\Projects\anna\src"
$env:ANNA_GATED_DELTA_OP_LIB = "D:\Projects\anna\.build\anna_gated_delta_fused\anna_gated_delta_fused.pyd"
```

## Quick Start

### Start the Server

```powershell
anna-serve `
  --model-dir D:\Models\Qwen3.5 `
  --model-name qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --host 127.0.0.1 `
  --port 8000
```

Health check and chat:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz

curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{`"model`":`"qwen3.5`",`"messages`":[{`"role`":`"user`",`"content`":`"Explain KV cache in three sentences.`"}],`"max_completion_tokens`":128}"
```

For streaming responses, add `"stream":true` (and optionally `"stream_options":{"include_usage":true}`).

### Performance Presets

High-throughput serving (larger batches, dynamic token budgets, int4 weights, TurboQuant KV cache):

```powershell
anna-serve `
  --model-dir D:\Models\Qwen3.5 `
  --model-name qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --weight-quant int4 `
  --kv-cache-quantization auto `
  --enable-flashqla-gdn-prefill `
  --scheduler-profile throughput `
  --host 127.0.0.1 `
  --port 8000
```

Interactive, low-latency serving:

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

Notes:

- `--kv-cache-quantization auto` enables TurboQuant when installed and picks bits/residual from model-size presets.
- Individual `--scheduler-*` flags override the profile. See [`docs/tuning.md`](docs/tuning.md) for the full tuning tables.

### One-Shot Generation and Local Benchmark

```powershell
anna-generate `
  --model-dir D:\Models\Qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --prompt "Explain why separating prefill and decode can reduce tail latency." `
  --max-new-tokens 128

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

Start `anna-serve` first, then:

```powershell
python tools\bench_api_concurrency.py `
  --base-url http://127.0.0.1:8000 `
  --model qwen3.5 `
  --scenario concurrent-short `
  --requests 16 `
  --concurrency 8 `
  --max-tokens 64 `
  --healthz
```

The summary reports RPS, output tokens/s, TTFT, ITL, and latency percentiles.

### Speech Synthesis

```powershell
anna-speak `
  --model-dir D:\Models\Qwen3-TTS `
  --input "Hello from Anna." `
  --output out.wav
```

## API Routes

- `GET /healthz` — runtime, model, memory, KV cache, service metrics, and admission status after OOM/device-lost.
- `GET /v1/models` — currently loaded model ID.
- `POST /v1/chat/completions` — chat, multimodal chat, streaming, and OpenAI-style `tool_calls` deltas.
- `POST /v1/completions` — text completion.
- `POST /v1/audio/speech` — Qwen3-TTS speech synthesis.

## `anna-serve` Options

### Base

- `--model-dir PATH` (required) — local model directory.
- `--model-name NAME` — model ID exposed by the API; derived from the path if omitted.
- `--device auto|cpu|xpu` — execution device; `auto` prefers XPU. `--xpu-device-index N` selects a GPU on multi-GPU systems.
- `--dtype DTYPE` — compute dtype: `auto` (default), `bf16`, `float16`, `float32`.
- `--host HOST` / `--port PORT` — bind address and port (default `127.0.0.1:8000`; use `0.0.0.0` for LAN access).
- `--log-level LEVEL` / `--log-format text|json` — `json` logs one structured object per line, with the request `X-Request-Id` / generated `trace_id` attached to logs and response headers.
- `--no-xpu-env-defaults` — skip Anna's recommended Level Zero environment variables.

### Generation Defaults

Applied only when the API request omits the matching field:

- `--max-completion-tokens N`, `--temperature FLOAT`, `--top-p FLOAT`, `--top-k N` (`0` disables), `--min-p FLOAT`, `--presence-penalty FLOAT`, `--repetition-penalty FLOAT`.
- `--enable-thinking` / `--disable-thinking` — default thinking behavior for chat requests.
- `--reasoning-format none|deepseek` — `deepseek` separates reasoning into `reasoning_content`.

### Compilation and Warmup

- `--compile-mode none|auto|default|reduce-overhead|max-autotune` — `torch.compile` mode; default `auto` (= `reduce-overhead` on XPU). Combined with the persistent inductor cache (see [`docs/tuning.md`](docs/tuning.md)), restarts hit the compile cache.
- `--compile-fullgraph` — request fullgraph capture.
- `--no-inference-warmup` — skip post-load warmup (not recommended for serving).
- `--warmup-prefill-tokens N` / `--warmup-decode-steps N` / `--warmup-batch-size N` — override the auto-derived warmup shape table (the effective table is logged at startup).

### Memory and Weights

- `--prefill-chunk-size N` — long-prompt prefill chunk size; `0` = XPU auto-sizing.
- `--weight-quant auto|none|int4` — `auto` promotes dense weights to int4 when they exceed **85%** of XPU memory (**70%** for MoE or experts-offload).
- `--expert-quant auto|none|int4` — MoE expert weight quantization.
- `--offload-mode auto|none|experts` — MoE expert offload. `--offload-vision` keeps the vision tower on CPU.
- `--resident-expert-layers N` / `--resident-expert-layer-indices LIST` — keep the first (or explicit) N sparse MoE layers on device.
- `--cached-experts-per-layer N` — offloaded experts cached per layer; `0` disables.
- `--kv-cache-quantization none|turboquant|auto` — `auto` enables TurboQuant (optional dependency) and selects bits/residual from model-size presets.
- `--kv-cache-quant-bits 2|3|4` / `--kv-cache-residual-len N` — override TurboQuant bit width and the number of newest KV tokens kept in full precision; when omitted, tier presets apply.
- `--prompt-cache-size N` / `--prompt-cache-max-tokens N` — keep exact full-prompt KV caches for cross-request reuse (`0` disables). Shared system prefixes reuse paged prefix blocks instead; enabling prompt cache avoids dual residency. Disable prefix sharing with `ANNA_PREFIX_KV_SHARE=0`.
- `--min-free-memory-mib N` / `--reserve-memory-mib N` / `--max-estimated-usage-ratio R` / `--generation-memory-safety-factor R` — memory guards for request admission and generation.

### XPU Fused Ops and Int4 Kernels

- `--enable-flashqla-gdn-prefill` — enable the SYCL Gated Delta prefill path in **strict** mode (no fallback).
- `--flashqla-gdn-prefill-mode off|strict|prefer` — `strict` hard-fails on unsupported device/shape/dtype; `prefer` degrades to the default path with a one-shot warning. Env `ANNA_XPU_FLASHQLA_GDN_PREFILL` accepts the same values.
- `--xpu-int4-matmul auto|torch|dequant|gemv` — int4 dense-linear strategy. `auto` (default) routes decode-sized rows (M ≤ threshold) to SYCL GEMV and prefill waves to the int4pack GEMM; `dequant` is for debugging.
- `--xpu-int4-gemv-m-threshold N` — `auto` routing threshold, default `2`.
- `--xpu-int4-cache-load-workers N` (default `8`) / `--weight-load-pipeline-workers N` (default `2`) — startup load parallelism.
- `--torchinductor-cache-dir PATH` — persistent inductor compile cache, default `~/.anna/cache/torchinductor` (a user-set `TORCHINDUCTOR_CACHE_DIR` wins).
- Env toggles: `ANNA_XPU_DISABLE_INT4_CACHE=1` (disable int4 layout cache), `ANNA_XPU_INT4_CACHE_DIR` (cache location), `ANNA_XPU_DISABLE_LM_HEAD_INT4_TOPK=1`, `ANNA_GATED_DELTA_OP_LIB` (fused-op library path). `anna-xpu-int4-cache --model-dir ...` inspects the int4 layout cache.
- Debug-only env: `ANNA_XPU_GATED_DELTA_DECODE_STRATEGY`, `ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK`, `ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS` — leave unset to use the built-in Arc shape lookup.

Arc A770 decode defaults baked into the fused op (rows = `batch * heads`; K=128 gated, K∈{64,256} reuse the same V defaults, other K/V fall back to a power-of-two block in `{1,2,4}`):

| V (value head dim) | Default value block | Default strategy |
| --- | --- | --- |
| 64 | 8 | `single` |
| 128 | 8 | `tiled` |
| 256 | 4 | `tiled` |

Validate with `python tools/validate_arc_gdn_decode.py --preset quick` (or `full` / `watch`).

### Continuous Batching

- `--scheduler-profile none|interactive|throughput` — built-in preset; explicit `--scheduler-*` flags win.
  - **interactive**: batch 2, wait 0.5 ms, prefill interval 1, budgets 1024/2048, dynamic budgets on, idle coalesce wait skipped (TTFT-friendly).
  - **throughput**: batch 8, wait 8 ms, prefill interval 4, budgets 2048/4096, dynamic budgets on.
- `--scheduler-max-batch-size N` — enables continuous batching when > 1.
- `--scheduler-batch-wait-ms MS` — request coalescing wait; higher values help throughput but hurt tail latency.
- `--scheduler-prefill-interval-steps N` — insert pending prefill every N decode steps.
- `--scheduler-max-prefill-tokens N` / `--scheduler-max-decode-tokens N` — per-batch token budgets; `0` disables.
- `--scheduler-max-waiting-requests N` — reject new requests with HTTP 429 when the queue reaches N (`0` = unlimited).
- `--scheduler-dynamic-token-budget` / `--no-...` — scale budgets from free XPU memory and running lengths.
- `--scheduler-max-queue-wait-ms MS` — fairness: force prefill admission for the oldest waiter.
- `--metrics-log-interval-seconds S` — periodic aggregate metrics; `0` disables.

### TTS and Gemma4 Boundaries

- **Qwen3-TTS** runs as an upstream wrapper (`qwen-tts`): Anna owns the OpenAI-compatible API/CLI, device gate, metrics, and OOM mapping; upstream owns the audio kernels. Heavy TTS kernels are not ported to Anna SYCL ops.
- **Process isolation**: `anna-serve` loads one model family per process. Co-resident audio + text engines serialize on the device gate (`ANNA_XPU_SERIALIZE_ALL=1` forces text engines onto the same gate). This is serialization, not VRAM partitioning.
- **Gemma4** reuses the Qwen text engine shell (scheduler, sampling, prompt cache, int4, TurboQuant). Gaps vs Qwen3.5 (paged KV, prefix blocks, GDN, MoE) are listed under `/healthz` → `ops_parity`.

Gemma4 baselines:

```powershell
anna-bench --model-dir D:\Models\Gemma4 --scenario gemma-text-short --device xpu --runs 5
python tools\bench_api_concurrency.py --scenario gemma-concurrent-short --concurrency 4 --requests 16 --healthz
```

## Other CLI Tools

`anna-generate` (one-shot text generation):

- `--model-dir PATH`, `--prompt TEXT`, `--max-new-tokens N`.
- `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty` — sampling.
- `--device`, `--dtype`, `--compile-mode`, `--kv-cache-*`, `--weight-quant` — same as `anna-serve`.

`anna-bench` (local throughput/latency):

- `--model-dir PATH`, `--prompt TEXT`, `--warmup N`, `--runs N`, `--max-new-tokens N`.
- `--profile-runtime` — print XPU component timing.
- `--image PATH` / `--video PATH` — multimodal benchmark input.

## Benchmark Tools

`tools/bench_api_concurrency.py` — HTTP concurrency benchmark:

- `--base-url URL`, `--model NAME`, `--route /v1/chat/completions|/v1/completions`.
- `--scenario custom|concurrent-short|single-long|mixed|repeated-system|gemma-*|multimodal-*` — built-in prompt scenarios.
- `--requests N`, `--concurrency N`, `--max-tokens N`, `--stream/--no-stream`.
- `--healthz` — fetch `/healthz` before and after the run; `--json` — print the summary as JSON.

`tools/bench_xpu_hotspots.py` — XPU kernel microbenchmarks (run with `--help` for the full list):

- Shapes: `--batch-size`, `--seq-len`, `--hidden-size`, `--num-heads`, `--num-kv-heads`, `--head-dim`, `--kv-len`, `--dtype`, `--warmup`, `--iters`.
- Gated Delta decode sweeps: `--gdn-decode-only`, `--gdn-decode-batch-head-cases LIST` (e.g. `1x16,1x32,4x32`), `--gdn-decode-value-blocks LIST`, `--gdn-value-head-dims LIST`, `--gdn-decode-shape-presets LIST` (`arc-default`, `arc-legacy-v128-block8`, `arc-legacy-v256-block4`).
- A/B compares: `--gdn-decode-auto-compare`, `--gdn-decode-default-compare`, `--gdn-decode-compare-only`; reproducibility: `--gdn-decode-seed N`, `--gdn-decode-seeds LIST`, `--gdn-decode-timing-repeats N`.
- `--arc-profile` — Arc-oriented int4 rows; `--csv-output PATH` — save results.

`tools/validate_arc_gdn_decode.py` — one-command Arc decode validation (benchmark presets + targeted regressions):

- `--presets LIST`, `--build-first`, `--json-output PATH`, `--skip-bench` / `--skip-pytest`, `--skip-bench-gates` (quick smoke run).

## Troubleshooting

- **XPU unavailable** — confirm the PyTorch build supports XPU and the Intel GPU driver + Level Zero runtime are installed.
- **Slow first request** — usually weight loading, lazy kernel loading, fused-op init, or `torch.compile`; warmup and the persistent inductor cache absorb most of it.
- **XPU out of memory** — try `--weight-quant int4`, `--kv-cache-quantization turboquant`, a lower token cap, or expert offload.
- **Low throughput, decode batch ≈ 1** — raise `--scheduler-max-batch-size` and `--scheduler-batch-wait-ms`, then check TTFT/ITL.
- **High decode p95/p99 with stable batch/cache metrics** — inspect `--profile-runtime` component timing (attention, Gated Delta, LM head, sampling).

## Documentation

- [`docs/tuning.md`](docs/tuning.md) — full tuning tables, environment variables, and `ServeSettings` fields.
- [`docs/xpu-engine-transition.md`](docs/xpu-engine-transition.md) — XPU engine notes.

## License

See [LICENSE](LICENSE).
