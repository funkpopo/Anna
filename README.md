# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna is a **local inference runtime** for large language and speech models, with an **OpenAI-compatible HTTP API**. It is built around **PyTorch** and is especially tuned for **Intel Arc (XPU)**—while **CPU** works for development and tests.

You point Anna at a folder on disk that contains a model (`config.json` + weights, or a Qwen **GGUF** plus optional vision projector). Anna then serves chat, text completion, multimodal chat, or text-to-speech, depending on the model type.

## What you get

- **HTTP API**: `/healthz`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/audio/speech`
- **Streaming and non-streaming** responses; optional **reasoning** split into `reasoning_content`; **tool / function calling**
- **Multimodal chat** (images / video; Gemma 4 also supports audio) when the model supports it
- **CLIs**: `anna-serve`, `anna-generate`, `anna-bench`, `anna-speak`
- **Runtime options** aimed at Arc: `torch.compile`, prefill chunking, prompt KV reuse, TurboQuant KV cache, runtime int4 weights, MoE expert offload and caching, optional continuous batching, optional SYCL fused ops

## Supported model families

Anna picks the backend from the top-level `model_type` in `config.json`:

| When `model_type` is… | Runtime | Typical use |
| --- | --- | --- |
| Anything except `qwen3_tts` and `gemma4` (e.g. `qwen3_5`, `qwen3_5_moe`, `qwen3_5_vl`) | Qwen3.5 text / VL | Chat, completions, multimodal (image/video), tools |
| `gemma4` | Gemma 4 | Same, plus audio in chat |
| `qwen3_tts` | Qwen3-TTS | Speech synthesis only |

**CLI fit:**

- `anna-generate` and `anna-bench`: text-generation families only (not `qwen3_tts`).
- `anna-speak`: `qwen3_tts` only.
- `anna-serve`: any supported family; which API routes succeed depends on the loaded model.

## Requirements

- **Python 3.11+**
- A **local model directory** (Hugging Face–style layout, or Qwen GGUF + optional `mmproj-*.gguf`)
- **PyTorch** installed for your machine (`torch>=2.7` per `pyproject.toml`). For Arc, use a build with **XPU** support.
- **Video** inputs: `imageio`, `imageio-ffmpeg` (installed with the package)
- **Qwen3-TTS**: `qwen-tts` (declared dependency)
- **Optional** fused XPU operator: Intel **oneAPI** DPC++ and, on Windows, **Visual Studio** Build Tools

Install the Python package; PyTorch and oneAPI are **not** installed by `pip install -e .` alone—you choose those for your hardware.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Anna.git
cd Anna
python -m venv .venv
```

**Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`  
**Linux/macOS:** `source .venv/bin/activate`

```bash
python -m pip install -U pip
python -m pip install -e .
# Optional: python -m pip install -e ".[dev]"
```

**Check PyTorch / XPU:**

```bash
python -c "import torch; print(torch.__version__); print(getattr(torch, 'xpu', None) and torch.xpu.is_available())"
```

**Optional fused op** (better XPU performance when available):

```bash
python tools/build_gated_delta_fused_op.py
```

Output goes under `.build/anna_gated_delta_fused`. You can point to the built library with `ANNA_GATED_DELTA_OP_LIB` (see below).

Arc int4 fused-kernel tuning knobs:

- `ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE`: work-group local size for `lm_head_int4_topk_fused` (rounded up to a power of two, max 64; the blocked top-k path uses at least 64 for correctness).
- `ANNA_XPU_INT4_LM_HEAD_BLOCK_TOPK_THRESHOLD`: vocabulary threshold for the two-stage blocked `lm_head_int4_topk_fused` path (default 65536).
- `ANNA_XPU_INT4_LM_HEAD_BLOCK_SIZE`: vocabulary block size for blocked `lm_head_int4_topk_fused` (default 4096).
- `ANNA_XPU_INT4_GEMV_LOCAL_SIZE`: work-group local size for the experimental standalone `XPUInt4Linear` GEMV path used when `ANNA_XPU_INT4_MATMUL=sycl` and decode rows are `<= 4` (rounded up to a power of two, max 256).
- `ANNA_XPU_AUTO_INT4_GEMV`: allow `ANNA_XPU_INT4_MATMUL=auto` to try the standalone GEMV path for the narrow `M=1,K=N=4096,group=128` shape. Default is off until Arc sweep data is consistently positive.
- `ANNA_XPU_INT4_MOE_GATE_LOCAL_SIZE`: local size for grouped int4 MoE gate/up projection (rounded up to a power of two, max 256).
- `ANNA_XPU_INT4_MOE_DOWN_LOCAL_SIZE`: local size for grouped int4 MoE down projection (rounded up to a power of two, max 256).

These default to the existing conservative choices. On Arc A770/A750, sweep these values with `tools/bench_xpu_hotspots.py --arc-profile`; add `--arc-int4-only` for focused int4 sweeps that skip the general attention/router hotspot suite. The report includes ordinary `XPUInt4Linear`, `lm_head_int4_topk_fused`, and `moe_grouped_int4_mlp_fused` rows so kernel-local wins can be checked against decode-critical paths.

When runtime int4 converts `lm_head`, Anna also prepares a top-k-specific scale/zero layout (`[vocab, group_count]`) for `lm_head_int4_topk_fused`; ordinary linear layers keep the standard matmul layout.

## Quick start

**API server:**

```bash
anna-serve --model-dir /path/to/model --host 127.0.0.1 --port 8000
```

**XPU-oriented example:**

```bash
anna-serve \
  --model-dir /path/to/model \
  --device xpu \
  --xpu-device-index 0 \
  --dtype bfloat16 \
  --kv-cache-quantization turboquant \
  --kv-cache-quant-bits 4 \
  --weight-quant auto \
  --prompt-cache-size 4
```

**One-shot text generation:**

```bash
anna-generate --model-dir /path/to/text-model --prompt "Explain KV cache in one paragraph."
```

**Benchmark (text or multimodal):**

```bash
anna-bench --model-dir /path/to/model --prompt "Hello" --warmup 1 --runs 3
```

**Check runtime XPU int4 cache eligibility:**

```bash
anna-xpu-int4-cache --model-dir /path/to/model --weight-quant auto --xpu-total-memory-gib 16
```

For safetensors Qwen3.5 models without an existing quantization config, this reports whether `--weight-quant auto` resolves to `int4` and shows the sidecar cache directory, usually `<model-dir>/.anna/xpu_int4_cache`.

**TTS (example: base voice clone):**

```bash
anna-speak --model-dir /path/to/tts --input "Hello." --output out.wav --ref-audio ref.wav --ref-text "Reference text."
```

---

## Command-line reference (what each flag does)

### Shared ideas (text runtimes: serve / generate / bench)

| Option | Meaning |
| --- | --- |
| `--model-dir` | **Required.** Directory with the model (or GGUF layout Anna understands). |
| `--model-name` | Name shown in logs / API; default is derived from the path. |
| `--device` | `auto`, `cpu`, or `xpu`. `auto` picks XPU when available. |
| `--xpu-device-index N` | Binds **Level Zero** to device index `N` via `ONEAPI_DEVICE_SELECTOR=level_zero:N`. Use when you have both iGPU and Arc and need the right GPU. |
| `--no-xpu-env-defaults` | Do **not** set Anna’s recommended Level Zero env vars (see **XPU environment** below). |
| `--dtype` | `auto`, `float32`, `float16`, or `bfloat16`. |
| `--compile-mode` | `torch.compile` mode: `none`, `auto`, `default`, `reduce-overhead`, `max-autotune`. **Default:** `auto` on **serve**, `none` on **generate** and **bench** (first compile can add latency). |
| `--compile-fullgraph` | Ask for full-graph capture when compile is enabled. |
| `--prefill-chunk-size` | Split long **text-only** prefills into chunks (tokens). `0` = let Anna auto-size on XPU. |
| `--prompt-cache-size` | Keep up to **N** exact text prompts’ KV caches in memory for reuse. `0` = off. |
| `--prompt-cache-max-tokens` | Only cache prompts up to **N** tokens (saves memory on long prompts). `0` = no limit. |
| `--profile-runtime` | Log synchronized XPU timings / memory for prefill vs decode (profiling). |
| `--kv-cache-quantization` | `none` or `turboquant` (Qwen3.5 + Gemma4). |
| `--kv-cache-quant-bits` | `2`, `3`, or `4` for TurboQuant. |
| `--kv-cache-residual-len` | Keep the newest **N** KV positions in full precision before older entries are compressed. |
| `--offload-mode` | `auto`, `none`, or `experts`. `experts` enables MoE expert offload when applicable. |
| `--offload-vision` | Keep the **vision tower on CPU** even if the main model runs on XPU (saves VRAM for text-only or tight budgets). |
| `--expert-quant` | `auto`, `none`, `int4` for **expert** weights on XPU. `auto` can enable int4 under offload. |
| `--weight-quant` | `auto`, `none`, `int4` for **dense** linear weights on XPU. `auto` may enable int4 when memory is tight. |
| `--resident-expert-layers` | Keep the **first N** sparse MoE layers fully on the accelerator; omit for auto in expert offload. |
| `--resident-expert-layer-indices` | Comma-separated **0-based** layer indices to keep resident; **overrides** `--resident-expert-layers`. |
| `--cached-experts-per-layer` | Max **offloaded experts** cached per MoE layer on XPU; `0` disables; omit for auto. |
| `--log-level` | Logging verbosity (e.g. `info`). |

### `anna-serve` only

| Option | Meaning |
| --- | --- |
| `--no-inference-warmup` | Skip a small post-load prefill+decode on XPU (first real request may pay lazy kernel load). |
| `--enable-thinking` / `--disable-thinking` | Default for chat when the client omits thinking flags. |
| `--max-completion-tokens` | Default token cap for requests that omit `max_tokens` / `max_completion_tokens`. |
| `--reasoning-format` | `none` vs `deepseek`-style split (`reasoning_content` vs main `content`). |
| `--min-free-memory-mib` | Minimum **free** XPU memory (MiB) before starting generation (admission). |
| `--reserve-memory-mib` | Extra **reserved** margin (MiB) during admission. |
| `--max-estimated-usage-ratio` | Reject requests when estimated usage exceeds this fraction of total XPU memory (between 0 and 1, exclusive of 0). |
| `--generation-memory-safety-factor` | Multiplier on estimated generation memory (≥ 1). |
| `--scheduler-max-batch-size` | If **&gt; 1**, enables **continuous batching** (batched decode). `1` = one request per step (lower latency, more kernel launches). |
| `--scheduler-batch-wait-ms` | When batching, how long to wait to fill a batch (trades throughput vs tail latency). |
| `--metrics-log-interval-seconds` | Print aggregated metrics every **N** seconds; `0` disables. |
| `--host`, `--port` | Bind address for Uvicorn. |

### `anna-generate` only

| Option | Meaning |
| --- | --- |
| `--prompt` | **Required.** Input text. |
| `--max-new-tokens` | Hard cap on **new** tokens; default follows model config or an internal safe estimate. |
| `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty` | Sampling controls for generation. |

### `anna-bench` only

| Option | Meaning |
| --- | --- |
| `--image` / `--video` | Optional local paths; builds a **multimodal** chat-style prompt (no audio benchmark path yet). |
| `--warmup` | Runs before timed `runs` (JIT / cache warm-up). |
| `--runs` | Timed iterations; prints latency and throughput stats. |
| `--max-new-tokens`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty` | Same role as generate; bench defaults favor deterministic timing (`temperature=0`, `top_p=1`, `top_k=0`). |

### `anna-speak` (Qwen3-TTS)

Does **not** add `--xpu-device-index` / `--no-xpu-env-defaults`; for Arc you may set `ONEAPI_DEVICE_SELECTOR` yourself if needed.

| Option | Meaning |
| --- | --- |
| `--input` | Text to synthesize. |
| `--output` | Output audio path. |
| `--language` | Optional language hint. |
| `--speaker` | CustomVoice: speaker name. |
| `--instruct` | Style / instruction for VoiceDesign or CustomVoice. |
| `--ref-audio`, `--ref-text` | Base **voice clone**: reference clip and transcript. |
| `--x-vector-only-mode` | Base model: embedding only, skip transcript conditioning. |
| `--response-format` | `wav` or `flac`. |
| `--max-new-tokens` | Cap on generated speech tokens. |
| `--do-sample` / `--no-do-sample` | Stochastic vs greedy for main generation. |
| `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty` | Main sampler. |
| `--subtalker-do-sample`, `--no-subtalker-do-sample`, `--subtalker-temperature`, `--subtalker-top-p`, `--subtalker-top-k` | Sub-module sampling. |
| `--non-streaming-mode` (default) vs `--streaming-style-input` | Text feeding mode for the TTS stack. |

### XPU / Level Zero auto-setup

When `--device` is `auto` or `xpu` and you do **not** pass `--no-xpu-env-defaults`, Anna sets recommended variables before load, including:

- `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`
- `ZES_ENABLE_SYSMAN=1`
- `ONEAPI_DEVICE_SELECTOR=level_zero:<index>` if `--xpu-device-index` is set

Use `--no-xpu-env-defaults` if you manage these globally.

---

## Optional environment variables (advanced)

| Variable | Purpose |
| --- | --- |
| `ANNA_XPU_INT4_MATMUL` | `auto` (default), `torch`, `dequant`, or experimental `sycl`. `sycl` uses Anna's standalone int4 GEMV for decode rows `<= 4`; `auto` only tries it when `ANNA_XPU_AUTO_INT4_GEMV=1` and the strict shape guard matches. |
| `ANNA_GATED_DELTA_OP_LIB` | Path to the compiled **gated delta** fused op library if not auto-discovered. |
| `ANNA_XPU_DISABLE_MOE_GROUPED_INT4`, `ANNA_XPU_DISABLE_LM_HEAD_INT4_TOPK` | Set to disable specific fused ops (e.g. `1` / `true`). |
| `ANNA_ENABLE_INT4_LM_HEAD_TOPK_FUSED` | `1` / `true` / `yes` / `on` to opt into int4 LM-head top-k fused path when available. |
| `ANNA_PREFIX_KV_SHARE` | Set to `0` to disable prefix KV sharing behavior in ops (default enabled). |
| `ANNA_DPCPP`, `ANNA_VCVARS64`, `ANNA_ONEAPI_RUNTIME_PATHS` | Build script hints for `tools/build_gated_delta_fused_op.py` on Windows / oneAPI layout. |

---

## API overview

Same routes are always registered; behavior depends on the loaded model.

- `GET /healthz` — health and runtime metrics snapshot where supported  
- `GET /v1/models`  
- `POST /v1/chat/completions` — multimodal `content` arrays, tools, thinking / reasoning options  
- `POST /v1/completions` — **text-only** prompt  
- `POST /v1/audio/speech` — TTS when the model is `qwen3_tts`

**List models:**

```bash
curl http://127.0.0.1:8000/v1/models
```

**Chat:**

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}]}'
```

**TTS (base clone example):**

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","input":"Hello","ref_audio":"ref.wav","ref_text":"...","response_format":"wav"}' \
  --output speech.wav
```

---

## Development

```bash
python -m pytest
```

For fused-kernel work, build the custom op first, then run tests or CLIs against `.build/anna_gated_delta_fused`.

## License

See [LICENSE](LICENSE).
