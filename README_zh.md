# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna 是一个基于 PyTorch 的本地推理运行时，面向 Intel Arc GPU（XPU），为 Qwen3.5（文本/多模态）、Gemma4（文本）和 Qwen3-TTS（语音合成）提供 OpenAI 兼容的 HTTP API 服务。

## 功能特性

- OpenAI 兼容 API：Chat（流式、多模态、函数调用）、Completion、语音合成、模型列表。
- 命令行工具：`anna-serve`、`anna-generate`、`anna-bench`、`anna-speak`、`anna-xpu-int4-cache`。
- XPU 优化：连续批处理、token budget 调度、TurboQuant KV cache、int4 权重量化、prompt cache，以及 fused SYCL 算子（Gated Delta、attention、RMSNorm、rotary、LM head）。
- 提供 HTTP 并发压测和 XPU 热点算子 profiling 工具。

## 支持的模型

Anna 根据 `config.json` 判断模型族，不依赖目录名。

| `model_type` | 运行时 | 入口 |
| --- | --- | --- |
| `qwen3_tts` | Qwen3-TTS | `anna-speak`、`/v1/audio/speech` |
| `gemma4` | Gemma4 | `anna-serve`、`anna-generate`、`anna-bench` |
| 其它兼容配置 | Qwen3.5 text / VL | `anna-serve`、`anna-generate`、`anna-bench` |

模型目录需要包含 `config.json`、tokenizer 文件和权重文件。Qwen3.5 MoE 模型也支持 Qwen GGUF 布局。Gemma4 文本模型支持直接从 GGUF 文件运行（config、tokenizer 和权重均直接读取自 GGUF，无需单独的 `config.json` / `tokenizer.json`）。

## 环境准备

基础要求：

- Python 3.11+
- PyTorch 2.7+（XPU 推理需要支持 Intel XPU 的构建）
- Intel GPU 驱动和 oneAPI Level Zero 运行时
- 在 Windows 上构建 XPU 自定义算子需要 Intel oneAPI DPC++ 和 Visual Studio Build Tools

安装开发环境：

```powershell
conda activate anna
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

TurboQuant KV cache 量化为可选依赖：

```powershell
python -m pip install -e ".[quant]"
```

检查 XPU 可用性：

```powershell
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else None)"
```

在 Windows + oneAPI 上构建 fused XPU 算子：

```powershell
$env:ANNA_DPCPP = "D:\Intel\oneAPI\compiler\latest\bin\dpcpp.exe"
$env:ANNA_VCVARS64 = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
python tools\build_gated_delta_fused_op.py
```

从源码运行时，可显式指定：

```powershell
$env:PYTHONPATH = "D:\Projects\anna\src"
$env:ANNA_GATED_DELTA_OP_LIB = "D:\Projects\anna\.build\anna_gated_delta_fused\anna_gated_delta_fused.pyd"
```

## 快速开始

### 启动服务

```powershell
anna-serve `
  --model-dir D:\Models\Qwen3.5 `
  --model-name qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --host 127.0.0.1 `
  --port 8000
```

健康检查与 Chat 请求：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz

curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{`"model`":`"qwen3.5`",`"messages`":[{`"role`":`"user`",`"content`":`"用三句话解释 KV cache。`"}],`"max_completion_tokens`":128}"
```

需要流式输出时，在请求体中加 `"stream":true`（可选 `"stream_options":{"include_usage":true}`）。

### 性能预设

高吞吐服务（更大 batch、动态 token budget、int4 权重、TurboQuant KV cache）：

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

交互式低延迟服务：

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

说明：

- `--kv-cache-quantization auto` 在已安装 TurboQuant 时启用，并按模型规模预设 bits/residual。
- 单独的 `--scheduler-*` 参数会覆盖 profile。完整调优表见 [`docs/tuning.md`](docs/tuning.md)。

### 单次生成与本地 benchmark

```powershell
anna-generate `
  --model-dir D:\Models\Qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --prompt "说明 prefill 和 decode 分离为什么能降低尾延迟。" `
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

### HTTP 并发压测

先启动 `anna-serve`，再运行：

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

汇总输出包含 RPS、输出 token/s、TTFT、ITL 和延迟分位数。

### 语音合成

```powershell
anna-speak `
  --model-dir D:\Models\Qwen3-TTS `
  --input "你好，这里是 Anna。" `
  --output out.wav
```

## API 路由

- `GET /healthz` — 运行时、模型、显存、KV cache、服务指标，以及 OOM/device-lost 后的准入状态。
- `GET /v1/models` — 当前加载的模型 ID。
- `POST /v1/chat/completions` — Chat、多模态 Chat、流式输出、OpenAI 风格的 `tool_calls` 增量。
- `POST /v1/completions` — 文本补全。
- `POST /v1/audio/speech` — Qwen3-TTS 语音合成。

## `anna-serve` 参数说明

### 基础参数

- `--model-dir PATH`（必填）— 本地模型目录。
- `--model-name NAME` — API 暴露的模型 ID；不传时从路径推断。
- `--device auto|cpu|xpu` — 执行设备；`auto` 优先 XPU。`--xpu-device-index N` 在多 Intel GPU 机器上选择指定卡。
- `--dtype DTYPE` — 计算精度：`auto`（默认）、`bf16`、`float16`、`float32`。
- `--host HOST` / `--port PORT` — 监听地址与端口（默认 `127.0.0.1:8000`；局域网访问用 `0.0.0.0`）。
- `--log-level LEVEL` / `--log-format text|json` — `json` 输出单行结构化日志，请求的 `X-Request-Id` / 生成的 `trace_id` 会写入日志与响应头。
- `--no-xpu-env-defaults` — 不设置 Anna 推荐的 Level Zero 环境变量。

### 生成默认值

只在 API 请求未传入对应字段时生效：

- `--max-completion-tokens N`、`--temperature FLOAT`、`--top-p FLOAT`、`--top-k N`（`0` 关闭）、`--min-p FLOAT`、`--presence-penalty FLOAT`、`--repetition-penalty FLOAT`。
- `--enable-thinking` / `--disable-thinking` — Chat 请求未指定 thinking 时的默认行为。
- `--reasoning-format none|deepseek` — `deepseek` 会把推理内容拆到 `reasoning_content`。

### 编译和预热

- `--compile-mode none|auto|default|reduce-overhead|max-autotune` — `torch.compile` 模式，默认 `auto`（XPU 上解析为 `reduce-overhead`）。配合持久 inductor 缓存（见 [`docs/tuning.md`](docs/tuning.md)），二次启动直接命中编译缓存。
- `--compile-fullgraph` — compile 时请求 fullgraph 捕获。
- `--no-inference-warmup` — 跳过加载后的预热（服务场景不推荐）。
- `--warmup-prefill-tokens N` / `--warmup-decode-steps N` / `--warmup-batch-size N` — 覆盖自动推导的预热形状表（实际生效的表在启动日志打印）。

### 显存和权重

- `--prefill-chunk-size N` — 长 prompt prefill 分块大小；`0` 表示 XPU 自动估算。
- `--weight-quant auto|none|int4` — `auto` 在 dense 权重超过 XPU 显存 **85%** 时提升为 int4（MoE 或 experts offload 时阈值为 **70%**）。
- `--expert-quant auto|none|int4` — MoE expert 权重量化。
- `--offload-mode auto|none|experts` — MoE expert offload。`--offload-vision` 将 vision tower 留在 CPU。
- `--resident-expert-layers N` / `--resident-expert-layer-indices LIST` — 前 N 个（或指定）sparse MoE 层常驻执行设备。
- `--cached-experts-per-layer N` — 每层缓存的 offloaded expert 数量；`0` 关闭。
- `--kv-cache-quantization none|turboquant|auto` — `auto` 在已安装依赖时启用 TurboQuant，并按模型规模预设 bits/residual。
- `--kv-cache-quant-bits 2|3|4` / `--kv-cache-residual-len N` — 覆盖 TurboQuant bit 数与最近 N 个 KV token 的全精度保留数；省略时按规模预设。
- `--prompt-cache-size N` / `--prompt-cache-max-tokens N` — 缓存完全相同 prompt 的 KV，供跨请求复用（`0` 关闭）。共享 system 前缀则走分页 prefix block 复用；开启 prompt cache 时会避免同一 prompt 双重驻留。`ANNA_PREFIX_KV_SHARE=0` 可关闭 prefix 共享。
- `--min-free-memory-mib N` / `--reserve-memory-mib N` / `--max-estimated-usage-ratio R` / `--generation-memory-safety-factor R` — 请求准入与生成阶段的显存保护。

### XPU fused op 和 int4 kernel

- `--enable-flashqla-gdn-prefill` — 以 **strict** 模式启用 SYCL Gated Delta prefill（不降级）。
- `--flashqla-gdn-prefill-mode off|strict|prefer` — `strict` 在不支持的 device/shape/dtype 上硬失败；`prefer` 降级到默认路径并按 reason 一次性告警。环境变量 `ANNA_XPU_FLASHQLA_GDN_PREFILL` 取值相同。
- `--xpu-int4-matmul auto|torch|dequant|gemv` — int4 dense linear 执行策略。`auto`（默认）按 M 行路由：decode 小 batch 走 SYCL GEMV，prefill 大 batch 走 int4pack GEMM；`dequant` 用于调试。
- `--xpu-int4-gemv-m-threshold N` — `auto` 路由阈值，默认 `2`。
- `--xpu-int4-cache-load-workers N`（默认 `8`）/ `--weight-load-pipeline-workers N`（默认 `2`）— 启动加载并行度。
- `--torchinductor-cache-dir PATH` — 持久 inductor 编译缓存目录，默认 `~/.anna/cache/torchinductor`（用户显式设置的 `TORCHINDUCTOR_CACHE_DIR` 优先）。
- 环境变量开关：`ANNA_XPU_DISABLE_INT4_CACHE=1`（关闭 int4 layout cache）、`ANNA_XPU_INT4_CACHE_DIR`（cache 位置）、`ANNA_XPU_DISABLE_LM_HEAD_INT4_TOPK=1`、`ANNA_GATED_DELTA_OP_LIB`（fused op 库路径）。`anna-xpu-int4-cache --model-dir ...` 可检查 int4 layout cache。
- 仅调试用环境变量：`ANNA_XPU_GATED_DELTA_DECODE_STRATEGY`、`ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK`、`ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS` — 不设置时走内置 Arc shape 查表。

fused op 内固化的 Arc A770 decode 默认（rows = `batch * heads`；K=128 有主门禁，K∈{64,256} 复用同一套 V 默认，其它 K/V 走 `{1,2,4}` 的 power-of-two fallback）：

| V (value head dim) | 默认 value block | 默认 strategy |
| --- | --- | --- |
| 64 | 8 | `single` |
| 128 | 8 | `tiled` |
| 256 | 4 | `tiled` |

用 `python tools/validate_arc_gdn_decode.py --preset quick`（或 `full` / `watch`）验收。

### 连续批处理

- `--scheduler-profile none|interactive|throughput` — 内置 preset；显式 `--scheduler-*` 参数优先。
  - **interactive**：batch 2、wait 0.5ms、prefill interval 1、budget 1024/2048、动态 budget、空闲时跳过合批等待（利 TTFT）。
  - **throughput**：batch 8、wait 8ms、prefill interval 4、budget 2048/4096、动态 budget。
- `--scheduler-max-batch-size N` — 大于 `1` 时启用连续批处理。
- `--scheduler-batch-wait-ms MS` — 请求合批等待时间；越大吞吐越高但尾延迟可能上升。
- `--scheduler-prefill-interval-steps N` — 每 N 个 decode step 插入一次 pending prefill。
- `--scheduler-max-prefill-tokens N` / `--scheduler-max-decode-tokens N` — 单批 token 预算；`0` 关闭。
- `--scheduler-max-waiting-requests N` — 等待队列达到 N 时以 HTTP 429 拒绝新请求（`0` 不限制）。
- `--scheduler-dynamic-token-budget` / `--no-...` — 按空闲显存与 running 序列长度自适应 budget。
- `--scheduler-max-queue-wait-ms MS` — 公平性：最老等待请求超时后强制插入 prefill。
- `--metrics-log-interval-seconds S` — 周期性输出聚合指标；`0` 关闭。

### TTS 和 Gemma4 边界

- **Qwen3-TTS** 以上游库封装方式接入（`qwen-tts`）：Anna 负责 OpenAI 兼容 API/CLI、设备门闩、指标与 OOM 映射；上游负责音频内核。TTS 重计算段不迁入 Anna SYCL 算子。
- **进程隔离**：`anna-serve` 单进程单模型族。同机共驻音频 + 文本引擎时在设备门闩上串行化（`ANNA_XPU_SERIALIZE_ALL=1` 可强制文本引擎进入同一门闩）。这是串行化，不是显存分区。
- **Gemma4** 复用 Qwen 文本引擎外壳（scheduler、采样、prompt cache、int4、TurboQuant）。与 Qwen3.5 的刻意差距（paged KV、prefix block、GDN、MoE）见 `/healthz` → `ops_parity`。

Gemma4 基线：

```powershell
anna-bench --model-dir D:\Models\Gemma4 --scenario gemma-text-short --device xpu --runs 5
python tools\bench_api_concurrency.py --scenario gemma-concurrent-short --concurrency 4 --requests 16 --healthz
```

## 其它命令行工具

`anna-generate`（单次文本生成）：

- `--model-dir PATH`、`--prompt TEXT`、`--max-new-tokens N`。
- `--temperature`、`--top-p`、`--top-k`、`--repetition-penalty` — 采样参数。
- `--device`、`--dtype`、`--compile-mode`、`--kv-cache-*`、`--weight-quant` — 含义与 `anna-serve` 相同。

`anna-bench`（本地吞吐/延迟压测）：

- `--model-dir PATH`、`--prompt TEXT`、`--warmup N`、`--runs N`、`--max-new-tokens N`。
- `--profile-runtime` — 输出 XPU 分段耗时。
- `--image PATH` / `--video PATH` — 多模态 benchmark 输入。

## 压测与 Profiling 工具

`tools/bench_api_concurrency.py` — HTTP 并发压测：

- `--base-url URL`、`--model NAME`、`--route /v1/chat/completions|/v1/completions`。
- `--scenario custom|concurrent-short|single-long|mixed|repeated-system|gemma-*|multimodal-*` — 内置 prompt 场景。
- `--requests N`、`--concurrency N`、`--max-tokens N`、`--stream/--no-stream`。
- `--healthz` — 压测前后拉取 `/healthz`；`--json` — 以 JSON 输出汇总。

`tools/bench_xpu_hotspots.py` — XPU 算子 microbenchmark（`--help` 查看完整参数）：

- 形状：`--batch-size`、`--seq-len`、`--hidden-size`、`--num-heads`、`--num-kv-heads`、`--head-dim`、`--kv-len`、`--dtype`、`--warmup`、`--iters`。
- Gated Delta decode sweep：`--gdn-decode-only`、`--gdn-decode-batch-head-cases LIST`（如 `1x16,1x32,4x32`）、`--gdn-decode-value-blocks LIST`、`--gdn-value-head-dims LIST`、`--gdn-decode-shape-presets LIST`（`arc-default`、`arc-legacy-v128-block8`、`arc-legacy-v256-block4`）。
- A/B 对比：`--gdn-decode-auto-compare`、`--gdn-decode-default-compare`、`--gdn-decode-compare-only`；可复现性：`--gdn-decode-seed N`、`--gdn-decode-seeds LIST`、`--gdn-decode-timing-repeats N`。
- `--arc-profile` — Arc 相关 int4 profile 行；`--csv-output PATH` — 保存结果。

`tools/validate_arc_gdn_decode.py` — 一条命令完成 Arc decode 验证（benchmark preset + 定向回归）：

- `--presets LIST`、`--build-first`、`--json-output PATH`、`--skip-bench` / `--skip-pytest`、`--skip-bench-gates`（快速 smoke run）。

## 常见问题

- **XPU 不可用** — 确认 PyTorch 构建支持 XPU，且已安装 Intel GPU 驱动和 Level Zero 运行时。
- **首次请求慢** — 通常来自权重加载、kernel lazy load、fused op 初始化或 `torch.compile`；预热和持久 inductor 缓存会吸收大部分开销。
- **显存不足** — 尝试 `--weight-quant int4`、`--kv-cache-quantization turboquant`、降低输出 token，或开启 expert offload。
- **吞吐低但 decode batch 平均接近 1** — 调大 `--scheduler-max-batch-size` 和 `--scheduler-batch-wait-ms`，再观察 TTFT/ITL。
- **decode p95/p99 高但 batch/cache 指标稳定** — 查看 `--profile-runtime` 中 attention、Gated Delta、LM head、sampling 的分段耗时。

## 文档

- [`docs/tuning.md`](docs/tuning.md) — 完整调优表、环境变量和 `ServeSettings` 字段。
- [`docs/xpu-engine-transition.md`](docs/xpu-engine-transition.md) — XPU 引擎说明。

## License

见 [LICENSE](LICENSE)。
