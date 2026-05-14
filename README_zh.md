# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna 是一个本地推理运行时，面向大语言模型、多模态模型和语音模型。它提供 OpenAI 兼容 HTTP API，也提供命令行工具，用于本地生成、服务部署、性能测试和 Qwen3-TTS 语音合成。

Anna 基于 PyTorch 构建，重点优化 Intel Arc / XPU。CPU 可用于开发、调试和小模型测试。

## 功能

- OpenAI 兼容接口：`/v1/chat/completions`、`/v1/completions`、`/v1/audio/speech`、`/v1/models`
- 支持非流式和流式文本生成
- 支持 Chat、文本补全、多模态 Chat、函数调用和 reasoning 输出
- 支持 Qwen3-TTS 语音合成
- 提供 Intel XPU 相关选项：`torch.compile`、KV cache 量化、int4 权重量化、MoE expert offload、prompt cache、连续批处理
- 命令行工具：`anna-serve`、`anna-generate`、`anna-bench`、`anna-speak`、`anna-xpu-int4-cache`

## 支持的模型

Anna 加载本地模型目录。普通 Hugging Face 风格目录应包含 `config.json` 和模型权重。兼容的 Qwen3.5 MoE 模型也支持 Qwen GGUF 布局。

模型类型根据 `config.json` 自动识别：

| `model_type` | 运行时 | 主要用途 |
| --- | --- | --- |
| `qwen3_tts` | Qwen3-TTS | 语音合成 |
| `gemma4` | Gemma 4 | 文本、Chat、带音频的多模态 Chat |
| 其它值 | Qwen3.5 text / VL | 文本、Chat、图像/视频多模态 Chat |

文本生成模型使用 `anna-generate` 和 `anna-bench`。Qwen3-TTS 使用 `anna-speak`。`anna-serve` 可加载任意支持的模型族；如果当前模型不支持某个接口，该接口会返回 API 错误。

## 环境要求

- Python 3.11+
- 适配当前硬件的 PyTorch 2.7+
- 一个本地模型目录
- Intel Arc / XPU：需要支持 XPU 的 PyTorch，以及 Intel GPU 运行时
- 可选 XPU fused operator：Intel oneAPI DPC++ 编译器；Windows 上还需要 Visual Studio Build Tools

Python 依赖写在 `pyproject.toml` 中。PyTorch 和 Intel GPU 驱动请根据目标机器单独安装。

## 安装

```bash
git clone https://github.com/YOUR_USERNAME/Anna.git
cd Anna
python -m venv .venv
```

激活虚拟环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

安装 Anna：

```bash
python -m pip install -U pip
python -m pip install -e .
```

开发和测试环境：

```bash
python -m pip install -e ".[dev]"
pytest
```

检查 PyTorch 和 XPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available() if hasattr(torch, 'xpu') else False)"
```

可选：构建 XPU fused operator：

```bash
python tools/build_gated_delta_fused_op.py
```

## 快速开始

启动 OpenAI 兼容服务：

```bash
anna-serve --model-dir /path/to/model --host 127.0.0.1 --port 8000
```

发送 Chat 请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "用一段话解释 KV cache。"}
    ],
    "max_completion_tokens": 128
  }'
```

流式输出：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "写一首关于本地 AI 的短诗。"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

命令行单次文本生成：

```bash
anna-generate \
  --model-dir /path/to/text-model \
  --prompt "用一段话解释 KV cache。" \
  --max-new-tokens 128
```

运行性能测试：

```bash
anna-bench \
  --model-dir /path/to/model \
  --prompt "Hello" \
  --warmup 1 \
  --runs 3
```

使用 Qwen3-TTS 合成语音：

```bash
anna-speak \
  --model-dir /path/to/qwen3-tts \
  --input "你好，这里是 Anna。" \
  --output out.wav
```

## Intel XPU 示例

选择指定 XPU 设备：

```bash
anna-serve \
  --model-dir /path/to/model \
  --device xpu \
  --xpu-device-index 0 \
  --dtype bfloat16
```

使用节省显存的运行选项：

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

为 API 服务开启连续批处理：

```bash
anna-serve \
  --model-dir /path/to/model \
  --device xpu \
  --scheduler-max-batch-size 4 \
  --scheduler-batch-wait-ms 2
```

检查 Anna 是否会创建 XPU int4 sidecar cache：

```bash
anna-xpu-int4-cache \
  --model-dir /path/to/model \
  --weight-quant auto \
  --xpu-total-memory-gib 16
```

## 多模态请求

对支持视觉的模型，可以使用 OpenAI 风格 content parts：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "描述这张图片。"},
          {"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}
        ]
      }
    ],
    "max_completion_tokens": 128
  }'
```

API schema 接受 `image_url`、`video_url` 和 `audio_url` content part。实际是否可用取决于加载的模型族。

## API 路由

| 路由 | 方法 | 用途 |
| --- | --- | --- |
| `/healthz` | `GET` | 运行时健康状态和模型状态 |
| `/v1/models` | `GET` | 查看当前加载的模型 ID |
| `/v1/chat/completions` | `POST` | Chat 和多模态 Chat |
| `/v1/completions` | `POST` | 文本补全 |
| `/v1/audio/speech` | `POST` | Qwen3-TTS 语音合成 |

## `anna-serve` 参数

`anna-serve` 只有一个必填参数，其它参数都有运行时默认值。

| 必填参数 | 说明 |
| --- | --- |
| `--model-dir PATH` | 本地模型目录。可以是包含 `config.json` 和权重的 Hugging Face 风格目录，也可以是兼容的 GGUF 布局。 |

常用服务参数：

| 可选参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-name NAME` | 从路径推断 | 对 `/v1/models` 和 API 响应暴露的模型 ID。 |
| `--host HOST` | `127.0.0.1` | 服务监听地址。需要监听所有网卡时用 `0.0.0.0`。 |
| `--port PORT` | `8000` | 服务端口。 |
| `--log-level LEVEL` | `info` | Uvicorn 和 Anna 的日志级别。 |
| `--device DEVICE` | `auto` | `auto`、`cpu` 或 `xpu`。`auto` 会优先使用可用 XPU。 |
| `--dtype DTYPE` | `auto` | 计算精度，例如 `auto`、`float32`、`float16`、`bfloat16`。 |
| `--max-completion-tokens N` | 模型配置/自动估算 | 当 API 请求未传 `max_tokens` / `max_completion_tokens` 时使用的默认输出长度上限。 |
| `--temperature FLOAT` | `0.7` | 请求未传时使用的默认采样温度。 |
| `--top-p FLOAT` | `0.8` | 默认 nucleus sampling 概率。 |
| `--top-k N` | `20` | 默认 top-k 采样上限。设为 `0` 可关闭。 |
| `--min-p FLOAT` | `0.0` | 默认 min-p 采样阈值。 |
| `--presence-penalty FLOAT` | `1.5` | 默认 presence penalty。 |
| `--repetition-penalty FLOAT` | `1.0` | 默认 repetition penalty。 |
| `--enable-thinking` / `--disable-thinking` | 开启 | 当 Chat 请求未传 thinking 字段时的默认行为。 |
| `--reasoning-format none\|deepseek` | `deepseek` | reasoning 输出格式。`deepseek` 会在可用时把推理内容放到 `reasoning_content`。 |

性能和显存参数：

| 可选参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--compile-mode MODE` | `auto` | `none`、`auto`、`default`、`reduce-overhead` 或 `max-autotune`。首次请求可能有编译开销。 |
| `--compile-fullgraph` | 关闭 | 启用 `torch.compile` 时请求 fullgraph 捕获。 |
| `--prefill-chunk-size N` | `0` | 将较长的纯文本 prefill 拆成 token chunk。`0` 表示在 XPU 上自动估算。 |
| `--prompt-cache-size N` | `0` | 保留最多 N 个完全相同文本 prompt 的 KV cache。`0` 表示关闭。 |
| `--prompt-cache-max-tokens N` | `0` | 只缓存不超过 N 个 token 的 prompt。`0` 表示不限制。 |
| `--kv-cache-quantization none\|turboquant` | `none` | 对兼容的 KV cache 做量化。 |
| `--kv-cache-quant-bits 2\|3\|4` | `4` | TurboQuant KV-cache 位宽。 |
| `--kv-cache-residual-len N` | `128` | 最近 N 个 KV token 保持全精度。 |
| `--offload-mode auto\|none\|experts` | `auto` | MoE expert offload 策略。 |
| `--offload-vision` | 关闭 | 即使语言模型在 XPU 上运行，也把 vision tower 留在 CPU。 |
| `--expert-quant auto\|none\|int4` | `auto` | XPU 上 MoE expert 权重的量化策略。 |
| `--weight-quant auto\|none\|int4` | `auto` | XPU 上语言模型 dense 权重的量化策略。 |
| `--resident-expert-layers N` | 自动 | 将前 N 个 sparse MoE 层完整保留在执行设备上。 |
| `--resident-expert-layer-indices LIST` | 未设置 | 逗号分隔、从 0 开始的 sparse 层号列表；会覆盖 `--resident-expert-layers`。 |
| `--cached-experts-per-layer N` | 自动 | 每个 sparse MoE 层最多在 XPU 上缓存多少个 offloaded expert。`0` 表示关闭。 |

XPU 和服务运行参数：

| 可选参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--xpu-device-index N` | 未设置 | 通过 `ONEAPI_DEVICE_SELECTOR=level_zero:N` 选择 Intel XPU。 |
| `--no-xpu-env-defaults` | 关闭 | 不在 XPU 启动前设置 Anna 推荐的 Level Zero 环境默认值。 |
| `--xpu-int4-matmul auto\|torch\|dequant` | 运行时默认 | XPU int4 dense linear 执行策略。 |
| `--enable-flashqla-gdn-prefill` | 关闭 | 在 XPU 上启用 Intel FlashQLA-compatible GDN prefill 路径。不支持的 shape、设备或 dtype 会直接报错。 |
| `--no-inference-warmup` | 关闭 | 跳过模型加载后的 XPU 小预热；首次真实请求可能承担 lazy kernel 加载开销。 |
| `--warmup-prefill-tokens N` | `2` | XPU 预热 prefill 使用的文本 token 数。 |
| `--warmup-decode-steps N` | `1` | XPU 预热 decode 步数。 |
| `--warmup-batch-size N` | `1` | XPU 预热 batch size。 |
| `--profile-runtime` | 关闭 | 打印同步后的 XPU 耗时和显存统计。 |
| `--min-free-memory-mib N` | `1024` | 开始生成前要求的最小 XPU 空闲显存。 |
| `--reserve-memory-mib N` | `512` | 请求准入时保留的额外 XPU 显存余量。 |
| `--max-estimated-usage-ratio R` | `0.9` | 当估算显存用量超过总显存比例 R 时拒绝请求。 |
| `--generation-memory-safety-factor R` | `2.0` | 估算生成显存时使用的安全系数。 |
| `--scheduler-max-batch-size N` | `1` | 大于 `1` 时启用连续批处理。 |
| `--scheduler-batch-wait-ms MS` | `2.0` | 启用批处理时，为合并请求等待的毫秒数。 |
| `--scheduler-prefill-interval-steps N` | `1` | 连续批处理启用时的 prefill 调度间隔。 |
| `--metrics-log-interval-seconds S` | `10.0` | 每 S 秒输出一次聚合运行指标。`0` 表示关闭。 |

查看完整参数：

```bash
anna-serve --help
anna-generate --help
anna-bench --help
anna-speak --help
```

## 常见问题

- 如果检测不到 XPU，先确认 PyTorch 是否支持 XPU，并确认 Intel GPU 驱动已经安装。
- 如果多 Intel GPU 机器选错设备，使用 `--xpu-device-index N` 指定设备。
- 如果第一次请求较慢，通常是模型加载、kernel 加载或 `torch.compile` 的开销。
- 如果 XPU 显存紧张，可以尝试 `--dtype bfloat16`、`--kv-cache-quantization turboquant`、`--weight-quant auto`、`--offload-mode experts`，或降低 token 数量。
- 如果某个 API 路由失败，确认当前加载的模型族是否支持该任务。

## License

见 [LICENSE](LICENSE)。
