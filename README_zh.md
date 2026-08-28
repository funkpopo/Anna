# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna 是一个基于 PyTorch 的本地推理运行时，目标是在 Intel Arc / XPU 上提供高吞吐、低延迟的 OpenAI 兼容服务。当前重点支持 Qwen3.5 文本/多模态推理、Gemma4 文本运行时和 Qwen3-TTS 语音合成。

`models/` 目录中的模型用于本地测试、模型类型分析和架构理解；Anna 的运行逻辑不绑定这些具体目录。实际运行时只要求传入兼容的本地模型目录。

## 功能概览

- OpenAI 兼容 HTTP API：Chat、Completion、语音合成、语音识别、模型列表。
- 命令行工具：`anna-serve`、`anna-generate`、`anna-bench`、`anna-speak`、`anna-transcribe`、`anna-xpu-int4-cache`。
- Intel XPU 优化：连续批处理（整批向量化采样）、token budget 调度、TurboQuant KV cache、XPU int4 权重、prompt cache、fused SYCL 自定义算子。
- Qwen3.5 推理路径包含 Gated Delta、attention、RMSNorm、rotary、LM head 等热点算子优化入口。
- 提供本地 HTTP 并发压测和 XPU 热点 microbench 工具。

## 支持的模型

Anna 根据模型目录中的 `config.json` 判断模型族，不依赖目录名。

| `model_type` | 运行时 | 入口 |
| --- | --- | --- |
| `qwen3_tts` | Qwen3-TTS | `anna-speak`、`/v1/audio/speech` |
| `gemma4` | Gemma4 | `anna-serve`、`anna-generate`、`anna-bench` |
| 其它兼容配置 | Qwen3.5 text / VL | `anna-serve`、`anna-generate`、`anna-bench` |

模型目录通常应包含 `config.json`、tokenizer 文件和权重文件。兼容的 Qwen3.5 MoE 模型也支持 Qwen GGUF 布局。

## 环境准备

基础要求：

- Python 3.11+
- PyTorch 2.7+，XPU 推理需要安装支持 Intel XPU 的 PyTorch
- Intel GPU 驱动和 oneAPI Level Zero 运行时
- Windows 构建自定义 XPU 算子时需要 Intel oneAPI DPC++ 和 Visual Studio Build Tools

安装开发环境：

```powershell
conda activate anna
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

TurboQuant KV 量化为可选依赖，需要时用 `quant` extra 安装：

```powershell
python -m pip install -e ".[quant]"
```

检查 XPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else None)"
```

Windows + oneAPI 构建自定义 fused op：

```powershell
$env:ANNA_DPCPP = "D:\Intel\oneAPI\compiler\latest\bin\dpcpp.exe"
$env:ANNA_VCVARS64 = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
python tools\build_gated_delta_fused_op.py
```

开发模式直接从源码运行时，可显式指定：

```powershell
$env:PYTHONPATH = "D:\Projects\anna\src"
$env:ANNA_GATED_DELTA_OP_LIB = "D:\Projects\anna\.build\anna_gated_delta_fused\anna_gated_delta_fused.pyd"
```

## 快速运行

### 启动 OpenAI 兼容服务

```powershell
anna-serve `
  --model-dir D:\Models\Qwen3.5 `
  --model-name qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --host 127.0.0.1 `
  --port 8000
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

Chat 请求：

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{`"model`":`"qwen3.5`",`"messages`":[{`"role`":`"user`",`"content`":`"用三句话解释 KV cache。`"}],`"max_completion_tokens`":128}"
```

流式 Chat：

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{`"model`":`"qwen3.5`",`"messages`":[{`"role`":`"user`",`"content`":`"写一段本地推理服务的技术总结。`"}],`"stream`":true,`"stream_options`":{`"include_usage`":true}}"
```

### 高吞吐 XPU 服务示例

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

`--scheduler-profile throughput` 对应更大 batch、更长合批等待与动态 token budget。
交互式低延迟用 `--scheduler-profile interactive`（也可用单独的 `--scheduler-*` 覆盖）。
`--kv-cache-quantization auto` 在已安装 turboquant 时启用，并按模型规模预设 bits/residual。

### 交互式（低延迟）XPU 服务示例

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

### 单次文本生成

```powershell
anna-generate `
  --model-dir D:\Models\Qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --prompt "说明 prefill 和 decode 分离为什么能降低尾延迟。" `
  --max-new-tokens 128
```

### 本地 benchmark

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
  --temperature 0 `
  --top-k 1 `
  --healthz
```

输出包含成功数、RPS、输出 token/s、TTFT、ITL 和延迟分位数。

### XPU 热点算子 profiling

Gated Delta decode 策略 sweep：

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

多 shape 的 Gated Delta decode compare matrix：

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

Arc 默认路径 preset 的显式对比：

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

Arc legacy row-band preset 的多 seed 聚合：

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

一条命令跑完整 Arc decode 验证：

```powershell
python tools\validate_arc_gdn_decode.py `
  --build-first `
  --json-output bench_logs\arc_gdn_decode_baseline_a770.json `
  --warmup 20 `
  --iters 100
```

完整热点套件：

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
  --input "你好，这里是 Anna。" `
  --output out.wav
```

## API 路由

- `GET /healthz`：运行时、模型、显存、KV cache、服务指标（prefix 命中率、scheduler 队列深度、TTFT/ITL 直方图、kernel strategy 命中）以及 device-lost/OOM 后的准入状态。
- `GET /v1/models`：当前加载的模型 ID。
- `POST /v1/chat/completions`：Chat、多模态 Chat、流式输出、函数调用兼容响应。
- `POST /v1/completions`：文本补全。
- `POST /v1/audio/speech`：Qwen3-TTS 语音合成。

## `anna-serve` 参数说明

### 基础参数

- `--model-dir PATH`：必填，本地模型目录。
- `--model-name NAME`：API 暴露的模型 ID；不传时从路径推断。
- `--host HOST`：监听地址，默认 `127.0.0.1`；局域网访问可用 `0.0.0.0`。
- `--port PORT`：监听端口，默认 `8000`。
- `--log-level LEVEL`：日志级别，默认 `info`。
- `--log-format text|json`：`json` 输出结构化单行日志；请求 `X-Request-Id` / 生成的 `trace_id` 会写入日志与响应头。
- `--device auto|cpu|xpu`：执行设备；`auto` 优先使用 XPU。
- `--xpu-device-index N`：多 Intel GPU 机器上选择指定 XPU。
- `--no-xpu-env-defaults`：不设置 Anna 推荐的 Level Zero 默认环境变量。
- `--dtype DTYPE`：计算精度，例如 `auto`、`bf16`、`bfloat16`、`float16`、`float32`。

### 生成默认值

这些值只在 API 请求没有显式传入对应字段时生效。

- `--max-completion-tokens N`：默认输出 token 上限。
- `--temperature FLOAT`：默认采样温度。
- `--top-p FLOAT`：默认 nucleus sampling 概率。
- `--top-k N`：默认 top-k；`0` 表示关闭。
- `--min-p FLOAT`：默认 min-p 阈值。
- `--presence-penalty FLOAT`：默认 presence penalty。
- `--repetition-penalty FLOAT`：默认 repetition penalty。
- `--enable-thinking` / `--disable-thinking`：Chat 请求未指定 thinking 时的默认行为。
- `--reasoning-format none|deepseek`：reasoning 输出格式；`deepseek` 会把推理内容拆到 `reasoning_content`。

### 编译和预热

- `--compile-mode none|auto|default|reduce-overhead|max-autotune`：`torch.compile` 模式，默认 `auto`（XPU 上解析为 `reduce-overhead`）；配合持久 inductor 缓存二次启动可命中编译缓存。
- `--compile-fullgraph`：启用 compile 时请求 fullgraph 捕获。
- `--no-inference-warmup`：跳过加载后的 XPU 预热（不推荐服务场景）。
- `--warmup-prefill-tokens N`：额外预热 prefill token 数；默认自动推导覆盖真实聊天形状 **13 / 64 / 256 / 2048**。
- `--warmup-decode-steps N`：每个预热形状的 decode 步数，默认 `8`（覆盖稳态 decode 与 turboquant 解量化路径）。
- `--warmup-batch-size N`：额外预热 batch size；默认自动覆盖 `{1,2,4,8} ∩ max_batch_size`。
- 启动日志会打印实际生效的 `Warmup shape table`。全部调优参数见 [`docs/tuning.md`](docs/tuning.md)。

### 显存和权重策略

- `--prefill-chunk-size N`：长 prompt prefill chunk 大小；`0` 表示 XPU 自动估算。
- `--prompt-cache-size N`：缓存**完全相同**文本 prompt 的 KV 数量；`0` 表示关闭。
- `--prompt-cache-max-tokens N`：只对不超过 N token 的 prompt 做精确缓存；`0` 表示不限制。
- **缓存职责**：完全相同 prompt → prompt cache；跨请求共享 system 前缀 → 分页 `PrefixBlockPool`。开启 prompt cache 时，可精确缓存的 prompt 会跳过 prefix 注册，避免双重驻留。`ANNA_PREFIX_KV_SHARE=0` 可全局关闭 prefix 共享。prefix hit/miss 暴露在 `/healthz` 与 metrics 日志。
- `--kv-cache-quantization none|turboquant|auto`：KV 量化。`auto` 在已安装依赖时启用 TurboQuant，并按模型规模预设 bits/residual（显式 bits/residual 优先）。
- `--kv-cache-quant-bits 2|3|4`：TurboQuant bit 数。省略时按规模：small≈4、medium≈3、large/xlarge≈2。
- `--kv-cache-residual-len N`：最近 N 个 KV token 保留全精度。省略时按规模：128/128/96/64。
- `--weight-quant auto|none|int4`：dense 权重量化策略。`auto` 在估算权重大于 XPU 总显存 **85%** 时提升为 int4（MoE 或 experts offload 时阈值为 **70%**）。
- `--expert-quant auto|none|int4`：MoE expert 权重量化策略。
- `--offload-mode auto|none|experts`：MoE expert offload 策略。
- `--offload-vision`：将 vision tower 留在 CPU，降低 XPU 显存占用。
- `--resident-expert-layers N`：前 N 个 sparse MoE 层常驻执行设备。
- `--resident-expert-layer-indices LIST`：指定常驻 sparse MoE 层号，覆盖 `--resident-expert-layers`。
- `--cached-experts-per-layer N`：每层缓存的 offloaded expert 数量；`0` 表示关闭。
- `--min-free-memory-mib N`：生成前要求的最小 XPU 空闲显存。
- `--reserve-memory-mib N`：请求准入时预留的显存余量。
- `--max-estimated-usage-ratio R`：估算用量超过总显存比例 R 时拒绝请求。
- `--generation-memory-safety-factor R`：生成显存估算安全系数。

### XPU fused op 和 int4 kernel

- `--enable-flashqla-gdn-prefill`：以 **strict** 模式启用 XPU SYCL Gated Delta prefill（不降级）。
- `--flashqla-gdn-prefill-mode off|strict|prefer`：FlashQLA 策略。`strict` 在不支持的 device/shape/dtype/op 上硬失败；`prefer` 降级到默认 fused 或 torch prefill，并按 reason 一次性告警。环境变量 `ANNA_XPU_FLASHQLA_GDN_PREFILL` 取值相同（`1`/`true`/`on` ≡ `strict`）。
- `--xpu-int4-matmul auto|torch|dequant|gemv`：XPU int4 dense linear 执行策略。
  - **`auto`（默认）**：按 M 行路由——decode 小 M（≤ `--xpu-int4-gemv-m-threshold`，默认 2）走 SYCL GEMV，prefill 大 M 走 aten int4pack XMX GEMM（依据 `bench_logs/xpu_int4_m1_auto_vs_gemv_p0_2.csv`）。
  - **`torch`**：强制 int4pack。
  - **`gemv`**：强制 SYCL GEMV。
  - **`dequant`**：完整反量化 + `F.linear`（调试）。
- `--xpu-int4-gemv-m-threshold N`：`auto` 路由阈值，默认 `2`；设 `0` 恢复旧的 auto=int4pack 行为。
- `--xpu-int4-cache-load-workers N`：int4 layout cache 并行反序列化线程数，默认 `8`（原串行加载约 39s）。
- `--weight-load-pipeline-workers N`：权重 shard 流水线并发 staging 线程数，默认 `2`（后台预读下一 shard，主线程同时拷贝当前 shard）。
- `--torchinductor-cache-dir PATH`：持久 inductor 编译缓存目录，默认 `~/.anna/cache/torchinductor`（serve 启动自动设置；用户显式设置的 `TORCHINDUCTOR_CACHE_DIR` 优先，torch import 时自动注入的临时目录默认值会被识别并忽略）。
- LM head int4 top-k fused 在 XPU int4 上**默认开启**（`top_k ≤ 16`）。用 `ANNA_XPU_DISABLE_LM_HEAD_INT4_TOPK=1` 关闭。兼容旧变量 `ANNA_ENABLE_INT4_LM_HEAD_TOPK_FUSED=0|1` 强制关/开。
- XPU int4 layout cache：首次 int4 转换写入 `{model}/.anna/xpu_int4_cache`（可用 `ANNA_XPU_INT4_CACHE_DIR` 覆盖）。版本/指纹不匹配自动重建；读写失败回退到现场量化。`ANNA_XPU_DISABLE_INT4_CACHE=1` 关闭。可用 `anna-xpu-int4-cache --model-dir ...` 检查。
- `ANNA_GATED_DELTA_OP_LIB`：显式指定 fused op `.pyd` / `.so` 路径。
- `ANNA_XPU_GATED_DELTA_DECODE_STRATEGY=auto|single|single_group|untiled|tiled|tiled_value`：Gated Delta decode kernel 策略。不设置（或设为 `auto`）时走内置 Arc shape 查表；仅在调试时强制 `single` / `tiled`。
- `ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK=N`：可选覆盖 tiled decode 的 value block。不设置时，Anna 按内置 Arc 表为 device/shape 选默认值；常见 Qwen3.5 shape 无需再手工调环境变量。
- `ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS=N`：`auto` 的可选覆盖项；设置后会跳过设备/shape 策略查表，改用这个 single-group 元素阈值。

当前 fused op 内固化的 Arc A770 decode 默认（rows = `batch * heads`）。主门禁覆盖 **K=128**；K∈{64,256} 复用同一套 V 默认。非常见 K/V 走可解释的 power-of-two fallback（`{1,2,4}`）：

| V (value head dim) | 默认 value block | 默认 strategy | 说明 |
| --- | --- | --- | --- |
| 64 | 8 | `single` | 强制 block=16 时同样走 `single` |
| 128 | 8 | `tiled` | 优先 block=8，避免退回更慢的 block=16 |
| 256 | 4 | `tiled` | rows 264..304 使用 value block=8 |

用 `python tools/validate_arc_gdn_decode.py --preset quick`（或 `full` / `watch`）验收 K=128 表。

### 连续批处理和 token budget

- `--scheduler-profile none|interactive|throughput`：内置连续批处理 preset；显式 `--scheduler-*` 覆盖 profile。
  - **interactive**：batch=2、wait=0.5ms、prefill interval=1、budget 1024/2048、等待队列 32、动态 budget、空闲跳过合批等待（利 TTFT）。
  - **throughput**：batch=8、wait=8ms、prefill interval=4、budget 2048/4096、等待队列 128、动态 budget、保留空闲合批等待。
- `--scheduler-max-batch-size N`：大于 `1` 时启用连续批处理。
- `--scheduler-batch-wait-ms MS`：等待更多请求合批的时间；越大吞吐越高但尾延迟可能上升。interactive 默认在 GPU 空闲时跳过等待以降低 TTFT。
- `--scheduler-prefill-interval-steps N`：每 N 个 decode step 插入一次 pending prefill 调度。
- `--scheduler-max-prefill-tokens N`：单轮 prefill admission 的 prompt token 预算；`0` 表示关闭（动态 budget 开启时会推导软默认）。
- `--scheduler-max-decode-tokens N`：单轮 decode batch 的序列 token 预算；`0` 表示关闭。
- `--scheduler-max-waiting-requests N`：等待队列达到 N 时以 HTTP 429 拒绝新请求（背压）；`0` 表示不限制。
- `--scheduler-dynamic-token-budget` / `--no-scheduler-dynamic-token-budget`：按空闲显存与 running 序列长度自适应 token budget。
- `--scheduler-max-queue-wait-ms MS`：公平性——最老等待请求超过该时延时强制插入 prefill，避免长 decode 饿死新 prompt。
- `--scheduler-max-decode-tokens N`：单个 decode batch 的 cached sequence token 预算；`0` 表示关闭。
- `--metrics-log-interval-seconds S`：周期性输出聚合指标；`0` 表示关闭。

### TTS 封装边界

Qwen3-TTS 以 **上游库封装** 方式接入（`qwen-tts`）：

- **Anna 负责：** OpenAI 兼容 API/CLI、进程级设备执行门闩、指标与 OOM 映射。
- **上游负责：** 音频编码 / talker / vocoder 内核与内部 generate。
- **暂不迁移：** 在 Qwen3.5 文本仍是 Arc 主优化面时，不把 TTS 重计算段迁入 Anna SYCL 融合算子。（Qwen3-ASR 支持已移除：上游 `qwen-asr` 依赖维护停滞。）
- **进程隔离：** `anna-serve` 单进程单模型族。若同进程共驻音频与文本引擎，必须走进程设备门闩串行化（`DeviceExecutionGate`；`ANNA_XPU_SERIALIZE_ALL=1` 可强制文本引擎也进入同一门闩）。这是串行化，不是多租户显存分区。

### Gemma4 压测基线

Gemma4 复用 Qwen 文本引擎外壳（scheduler、采样、prompt cache、dense int4、full-attention TurboQuant 规模预设）。与 Qwen3.5 的刻意差距（paged KV、prefix block、GDN、MoE）见 `/healthz` → `ops_parity`。

```powershell
anna-bench --model-dir D:\Models\Gemma4 --scenario gemma-text-short --device xpu --runs 5
python tools\bench_api_concurrency.py --scenario gemma-concurrent-short --concurrency 4 --requests 16 --healthz
```

## `anna-generate` 参数说明

- `--model-dir PATH`：文本模型目录。
- `--prompt TEXT`：输入 prompt。
- `--max-new-tokens N`：输出 token 上限。
- `--temperature`、`--top-p`、`--top-k`、`--repetition-penalty`：采样参数。
- `--device`、`--dtype`、`--compile-mode`、`--kv-cache-*`、`--weight-quant`：含义与 `anna-serve` 相同。

## `anna-bench` 参数说明

- `--model-dir PATH`：文本模型目录。
- `--prompt TEXT`：benchmark prompt。
- `--warmup N`：预热轮数。
- `--runs N`：计时轮数。
- `--max-new-tokens N`：每轮输出 token 上限。
- `--profile-runtime`：输出 XPU 分段耗时。
- `--image PATH` / `--video PATH`：多模态 benchmark 输入。

## 压测和 profiling 参数说明

`tools/bench_api_concurrency.py`：

- `--base-url URL`：Anna 服务地址。
- `--route /v1/chat/completions|/v1/completions`：压测路由。
- `--model NAME`：请求体中的模型 ID。
- `--scenario custom|concurrent-short|single-long|mixed|repeated-system`：内置 prompt 场景。
- `--requests N`：总请求数。
- `--concurrency N`：并发数。
- `--max-tokens N`：每个请求输出 token 上限。
- `--stream` / `--no-stream`：是否使用流式接口。
- `--healthz`：压测前后拉取 `/healthz`。
- `--json`：以 JSON 输出汇总。

`tools/bench_xpu_hotspots.py`：

- `--batch-size N`、`--seq-len N`、`--hidden-size N`：合成输入尺寸。
- `--num-heads N`、`--num-kv-heads N`、`--head-dim N`、`--kv-len N`：attention/GDN 形状。
- `--dtype fp16|bf16|fp32`：benchmark dtype。
- `--warmup N`、`--iters N`：预热和计时次数。
- `--gdn-decode-only`：只跑 Gated Delta decode 策略 sweep。
- `--gdn-decode-batch-head-cases LIST`：在一次 decode profile 中跑多个 `batch x heads` case，例如 `1x16,1x32,4x32`。
- `--gdn-decode-shape-presets LIST`：运行命名的 Arc decode preset 矩阵，例如 `arc-default`、`arc-legacy-v128-block8`、`arc-legacy-v256-block4`。
- `--gdn-decode-value-blocks LIST`：测试多个 value block；若同时使用 `--gdn-decode-shape-presets` 且不显式传入，则 Anna 会自动选用该 preset 推荐的 block。
- `--gdn-value-head-dims LIST`：在一次 decode profile 中跑多个 value head dim；设置后覆盖 `--gdn-value-head-dim`。
- `--gdn-decode-single-min-elements N`：覆盖 auto 策略阈值。
- `--gdn-decode-seed N`：固定 decode profile 输入，方便做可复现的 A/B 对比；负值表示每次运行都重新随机输入。
- `--gdn-decode-seeds LIST`：让每个 decode profile case 跨多个固定 seed 聚合；设置后覆盖 `--gdn-decode-seed`。
- `--gdn-decode-timing-repeats N`：每个候选重复计时 N 次并输出中位数。
- `--gdn-decode-auto-compare`：在 decode sweep 之后额外输出每个 value block 上 `auto` 对比最优显式策略的汇总行。
- `--gdn-decode-default-compare`：比较默认 decode 路径与显式 `single` / `tiled`。
- `--gdn-decode-default-block-compare`：比较默认 decode 路径与强制 value-block override。
- `--gdn-decode-compare-only`：跳过完整 strategy sweep 行，只输出 compare 汇总行。
- `--arc-profile`：增加 Arc A770/A750 相关 int4 profile 行。
- `--csv-output PATH`：保存通用热点 benchmark 结果。

`tools/validate_arc_gdn_decode.py`：

- 一条命令串起标准 Arc decode benchmark preset 和定向 decode 回归。
- `--presets LIST`：选择 `arc-default`、`arc-legacy-v128-block8`、`arc-legacy-v256-block4` 的任意子集。
- `--build-first`：验证前先重建 fused-op 动态库。
- `--json-output PATH`：把 benchmark gate 摘要、命中的 value block 和 pytest 状态写成结构化 JSON 结果文件。
- 默认会按 preset 专属的 speed-ratio 门槛校验 benchmark 输出；若只是想快速 smoke run 并降低 warmup/iters，可加 `--skip-bench-gates`。
- `--skip-bench` / `--skip-pytest`：只跑 benchmark 部分或只跑定向回归。

## 常见问题

- XPU 不可用：确认安装的是 XPU 版 PyTorch，并检查 Intel GPU 驱动和 Level Zero 运行时。
- 首次请求慢：通常来自权重加载、kernel lazy load、fused op 初始化或 `torch.compile`。
- 显存不足：优先尝试 `--dtype bf16`、`--weight-quant int4`、`--kv-cache-quantization turboquant`、降低输出 token，或开启 expert offload。
- 吞吐低但 batch 平均接近 1：调大 `--scheduler-max-batch-size` 和 `--scheduler-batch-wait-ms`，再观察 TTFT/ITL。
- cache stack/split/compact 高：优先检查 KV cache row 管理和 batch 成员变化，不要先写算子。
- decode p95/p99 高且 batch/cache 稳定：再看 `--profile-runtime` 中 attention、Gated Delta、LM head、sampling 的分段耗时。

## License

见 [LICENSE](LICENSE)。
