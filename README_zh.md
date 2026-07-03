# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna 是一个基于 PyTorch 的本地推理运行时，目标是在 Intel Arc / XPU 上提供高吞吐、低延迟的 OpenAI 兼容服务。当前重点支持 Qwen3.5 文本/多模态推理、Gemma4 文本运行时、Qwen3-TTS 语音合成和 Qwen3-ASR 语音识别。

`models/` 目录中的模型用于本地测试、模型类型分析和架构理解；Anna 的运行逻辑不绑定这些具体目录。实际运行时只要求传入兼容的本地模型目录。

## 功能概览

- OpenAI 兼容 HTTP API：Chat、Completion、语音合成、语音识别、模型列表。
- 命令行工具：`anna-serve`、`anna-generate`、`anna-bench`、`anna-speak`、`anna-transcribe`、`anna-xpu-int4-cache`。
- Intel XPU 优化：连续批处理、token budget 调度、TurboQuant KV cache、XPU int4 权重、prompt cache、fused SYCL 自定义算子。
- Qwen3.5 推理路径包含 Gated Delta、attention、RMSNorm、rotary、LM head 等热点算子优化入口。
- 提供本地 HTTP 并发压测和 XPU 热点 microbench 工具。

## 支持的模型

Anna 根据模型目录中的 `config.json` 判断模型族，不依赖目录名。

| `model_type` | 运行时 | 入口 |
| --- | --- | --- |
| `qwen3_tts` | Qwen3-TTS | `anna-speak`、`/v1/audio/speech` |
| `qwen3_asr` | Qwen3-ASR | `anna-transcribe`、`/v1/audio/transcriptions` |
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

这个配置适合压测吞吐。交互式低延迟场景可以降低 `--scheduler-batch-wait-ms`，或把 `--scheduler-max-batch-size` 调小。

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

带多 seed 聚合的高 rows compare-only matrix：

```powershell
python tools\bench_xpu_hotspots.py `
  --gdn-decode-only `
  --gdn-decode-auto-compare `
  --gdn-decode-compare-only `
  --gdn-decode-seeds 20260716,20260717 `
  --gdn-decode-batch-head-cases 9x32,10x32,12x32,14x32,15x32,16x32,17x32,22x32,24x32 `
  --gdn-value-head-dims 256 `
  --head-dim 128 `
  --gdn-decode-value-blocks 4 `
  --dtype bf16 `
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

### Qwen3-ASR

```powershell
anna-transcribe `
  --model-dir D:\Models\Qwen3-ASR `
  --audio input.wav `
  --device xpu `
  --language Chinese
```

通过 HTTP 上传音频：

```powershell
curl.exe http://127.0.0.1:8000/v1/audio/transcriptions `
  -F model=qwen3-asr `
  -F file=@input.wav `
  -F language=Chinese `
  -F response_format=verbose_json
```

## API 路由

- `GET /healthz`：运行时、模型、显存、KV cache 和服务指标。
- `GET /v1/models`：当前加载的模型 ID。
- `POST /v1/chat/completions`：Chat、多模态 Chat、流式输出、函数调用兼容响应。
- `POST /v1/completions`：文本补全。
- `POST /v1/audio/speech`：Qwen3-TTS 语音合成。
- `POST /v1/audio/transcriptions`：Qwen3-ASR 语音识别。

## `anna-serve` 参数说明

### 基础参数

- `--model-dir PATH`：必填，本地模型目录。
- `--model-name NAME`：API 暴露的模型 ID；不传时从路径推断。
- `--host HOST`：监听地址，默认 `127.0.0.1`；局域网访问可用 `0.0.0.0`。
- `--port PORT`：监听端口，默认 `8000`。
- `--log-level LEVEL`：日志级别，默认 `info`。
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

- `--compile-mode none|auto|default|reduce-overhead|max-autotune`：`torch.compile` 模式；服务场景常用 `none` 或 `auto`。
- `--compile-fullgraph`：启用 compile 时请求 fullgraph 捕获。
- `--no-inference-warmup`：跳过加载后的 XPU 小预热。
- `--warmup-prefill-tokens N`：预热 prefill token 数，默认 `2`。
- `--warmup-decode-steps N`：预热 decode 步数，默认 `1`。
- `--warmup-batch-size N`：预热 batch size，默认 `1`。

### 显存和权重策略

- `--prefill-chunk-size N`：长 prompt prefill chunk 大小；`0` 表示 XPU 自动估算。
- `--prompt-cache-size N`：缓存完全相同文本 prompt 的 KV cache 数量；`0` 表示关闭。
- `--prompt-cache-max-tokens N`：只缓存不超过 N token 的 prompt；`0` 表示不限制。
- `--kv-cache-quantization none|turboquant`：KV cache 量化模式。
- `--kv-cache-quant-bits 2|3|4`：TurboQuant KV bit 数。
- `--kv-cache-residual-len N`：最近 N 个 KV token 保留全精度。
- `--weight-quant auto|none|int4`：dense 权重量化策略。
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

- `--enable-flashqla-gdn-prefill`：启用 XPU SYCL Gated Delta prefill 路径；不支持的 shape/dtype/device 会直接报错。
- `--xpu-int4-matmul auto|torch|dequant`：XPU int4 dense linear 执行策略。
- `ANNA_GATED_DELTA_OP_LIB`：显式指定 fused op `.pyd` / `.so` 路径。
- `ANNA_XPU_GATED_DELTA_DECODE_STRATEGY=auto|single|single_group|untiled|tiled|tiled_value`：Gated Delta decode kernel 策略。
- `ANNA_XPU_GATED_DELTA_DECODE_VALUE_BLOCK=N`：覆盖 tiled decode 的 value block 大小。不设置时，Anna 会使用 device/shape 默认值；当前 Arc 上 K=128、V={64,128,256} 的 decode shape 默认走 `16`。
- `ANNA_XPU_GATED_DELTA_DECODE_SINGLE_MIN_ELEMENTS=N`：`auto` 的可选覆盖项；设置后会跳过设备/shape 查表，改用这个 single-group 元素阈值。

### 连续批处理和 token budget

- `--scheduler-max-batch-size N`：大于 `1` 时启用连续批处理。
- `--scheduler-batch-wait-ms MS`：等待更多请求合批的时间；越大吞吐越高但尾延迟可能上升。
- `--scheduler-prefill-interval-steps N`：每 N 个 decode step 插入一次 pending prefill 调度。
- `--scheduler-max-prefill-tokens N`：单轮 prefill admission 的 prompt token 预算；`0` 表示关闭。
- `--scheduler-max-decode-tokens N`：单个 decode batch 的 cached sequence token 预算；`0` 表示关闭。
- `--metrics-log-interval-seconds S`：周期性输出聚合指标；`0` 表示关闭。

### ASR 服务参数

- `--asr-max-inference-batch-size N`：Qwen3-ASR 服务端每个 XPU batch 的音频 chunk 数。
- `--asr-max-new-tokens N`：Qwen3-ASR 每个 chunk 生成的最大文本 token 数。

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
- `--gdn-decode-value-blocks LIST`：测试多个 value block。
- `--gdn-value-head-dims LIST`：在一次 decode profile 中跑多个 value head dim；设置后覆盖 `--gdn-value-head-dim`。
- `--gdn-decode-single-min-elements N`：覆盖 auto 策略阈值。
- `--gdn-decode-seed N`：固定 decode profile 输入，方便做可复现的 A/B 对比；负值表示每次运行都重新随机输入。
- `--gdn-decode-seeds LIST`：让每个 decode profile case 跨多个固定 seed 聚合；设置后覆盖 `--gdn-decode-seed`。
- `--gdn-decode-timing-repeats N`：每个候选重复计时 N 次并输出中位数。
- `--gdn-decode-auto-compare`：在 decode sweep 之后额外输出每个 value block 上 `auto` 对比最优显式策略的汇总行。
- `--gdn-decode-compare-only`：跳过完整 strategy sweep 行，只输出 compare 汇总行。
- `--arc-profile`：增加 Arc A770/A750 相关 int4 profile 行。
- `--csv-output PATH`：保存通用热点 benchmark 结果。

## 常见问题

- XPU 不可用：确认安装的是 XPU 版 PyTorch，并检查 Intel GPU 驱动和 Level Zero 运行时。
- 首次请求慢：通常来自权重加载、kernel lazy load、fused op 初始化或 `torch.compile`。
- 显存不足：优先尝试 `--dtype bf16`、`--weight-quant int4`、`--kv-cache-quantization turboquant`、降低输出 token，或开启 expert offload。
- 吞吐低但 batch 平均接近 1：调大 `--scheduler-max-batch-size` 和 `--scheduler-batch-wait-ms`，再观察 TTFT/ITL。
- cache stack/split/compact 高：优先检查 KV cache row 管理和 batch 成员变化，不要先写算子。
- decode p95/p99 高且 batch/cache 稳定：再看 `--profile-runtime` 中 attention、Gated Delta、LM head、sampling 的分段耗时。

## License

见 [LICENSE](LICENSE)。
