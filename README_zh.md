# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna 是一个面向 Intel Arc Alchemist / PyTorch XPU 的本地推理运行时和 OpenAI 兼容 API 服务。当前项目已经可以直接从本地模型目录加载并运行 Qwen3.5 文本生成模型、Gemma 4 多模态模型，以及 Qwen3-TTS 语音合成模型。Qwen3.5 模型目录现在既支持 Hugging Face 风格的 `config.json` / tokenizer / safetensors，也支持 GGUF 主模型文件，并可选配套 `mmproj-*.gguf` 视觉塔文件。

## 项目能力

- OpenAI 兼容接口：
  - `GET /healthz`
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `POST /v1/audio/speech`
- 聊天补全与文本补全，支持非流式和流式返回
- 支持将推理过程拆分到 `reasoning_content`
- 聊天模型支持工具 / 函数调用，以及流式 tool-call 增量输出
- 本地多模态输入支持
  - Qwen3.5：文本、图片、视频
  - Gemma4：文本、图片、视频、音频
- `qwen3_tts` 可通过 API 和 `anna-speak` 完成本地语音合成
- 提供 CLI：`anna-serve`、`anna-generate`、`anna-bench`、`anna-speak`
- 面向 XPU 的运行时优化：
  - `torch.compile`
  - 预填充分块
  - 精确提示词 KV 缓存复用
  - TurboQuant KV 缓存压缩
  - 运行时 int4 权重量化
  - Qwen MoE expert offload / expert cache
  - 连续批处理
  - 可选 SYCL fused 自定义算子
- `/healthz` 与终端日志可查看内存、缓存和请求指标

## 支持的模型类型

Anna 根据 `config.json` 顶层的 `model_type` 自动选择运行时：

- `qwen3_tts` -> Qwen3-TTS 运行时
- `gemma4` -> Gemma4 运行时
- 其他值 -> Qwen3.5 文本运行时

| 模型族 | `config.json` 顶层 `model_type` | 其他已识别类型 / 配置 | 能力 | 主要命令 |
| --- | --- | --- | --- | --- |
| Qwen3.5 文本运行时 | 除 `qwen3_tts`、`gemma4` 之外的值；已知 / 预期值包括 `qwen3_5`、`qwen3_5_text`、`qwen3_5_moe`、`qwen3_5_vl` | 可带 `vision_config`；支持解析 AWQ、AutoRound 等量化配置 | 文本补全、聊天补全、流式输出、推理拆分、工具调用、图片 / 视频多模态聊天 | `anna-serve`、`anna-generate`、`anna-bench` |
| Gemma4 运行时 | `gemma4` | `text_config.model_type=gemma4_text`；可选 `vision_config.model_type=gemma4_vision`；可选 `audio_config.model_type=gemma4_audio` | 文本补全、聊天补全、流式输出、推理拆分、工具调用、图片 / 视频 / 音频多模态聊天 | `anna-serve`、`anna-generate`、`anna-bench` |
| Qwen3-TTS 运行时 | `qwen3_tts` | 运行时的 `tts_model_type` 可为 `base`、`custom_voice`、`voice_design` | 本地语音合成 | `anna-serve`、`anna-speak` |

补充说明：

- `anna-generate` 和 `anna-bench` 仅适用于文本生成模型族（`qwen3_5_text`、`gemma4`）。
- `anna-speak` 仅适用于 `qwen3_tts`。
- `/v1/audio/speech` 只有在当前加载模型支持语音合成时才会成功。
- `anna-bench` 目前支持 `--image` 和 `--video`，还没有音频基准参数。
- 多模态输入走聊天消息内容数组或 benchmark CLI；`/v1/completions` 仍然是纯文本接口。

## 环境要求

- Python `3.11+`
- 一个本地模型目录，满足以下任一形式：
  - 包含 `config.json`、tokenizer 文件和权重
  - 或包含一个 Qwen GGUF 主模型文件，并可选携带 `mmproj-*.gguf`
- 单独安装与你环境匹配的 PyTorch
  - CPU 可用于调试和测试
  - Intel Arc + 带 `xpu` 的 PyTorch 是当前项目的目标运行路径
- 视频输入依赖 `imageio` 和 `imageio-ffmpeg`
- `qwen3_tts` 依赖 `qwen-tts`
- 若需要构建可选的 fused XPU 算子：
  - Intel oneAPI DPC++/C++ Compiler
  - Windows 下还需要 Visual Studio Build Tools

上面的 Python 依赖会由 `pip install -e .` 安装；PyTorch 和 oneAPI 需要根据你的系统单独准备。

## 安装方式

### 1. 克隆仓库

```bash
git clone <your-repo-url> Anna
cd Anna
```

### 2. 创建并激活虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3. 安装 Anna

```bash
python -m pip install -e .
```

可选开发依赖：

```bash
python -m pip install -e ".[dev]"
```

### 4. 安装并验证 PyTorch

请安装与你目标设备匹配的 PyTorch。如果要跑 Intel Arc，请使用带 `xpu` 支持的版本，并先验证：

```bash
python -c "import torch; print(torch.__version__); print(hasattr(torch, 'xpu')); print(torch.xpu.is_available() if hasattr(torch, 'xpu') else False)"
```

### 5. 构建 fused XPU 自定义算子

这个步骤主要用于提升 XPU 性能；基础 CPU 调试不依赖它。

```bash
python tools/build_gated_delta_fused_op.py
```

编译产物会输出到 `.build/anna_gated_delta_fused`。

## 当前工作区内可直接使用的模型目录示例

如果你已经把模型下载到了仓库内的 `models/` 目录，当前工作区里已有的示例包括：

- `models/Intel/Qwen3___5-2B-int4-AutoRound`
- `models/Intel/Qwen3___5-35B-A3B-int4-AutoRound`
- `models/google/gemma-4-E4B-it`
- `models/Qwen/Qwen3-TTS-12Hz-1___7B-Base`

下面所有命令里的 `--model-dir` 都可以替换成类似上面的目录。

## 快速开始

### 启动 OpenAI 兼容 API 服务

最简用法：

```bash
anna-serve --model-dir <path-to-model> --host 127.0.0.1 --port 8000
```

一个更接近 XPU 实战的例子：

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

如果是大尺寸 Qwen MoE 模型，还可以继续调这些参数：

- `--offload-mode experts`
- `--expert-quant int4`
- `--cached-experts-per-layer <N>` (推荐64)
- `--resident-expert-layers <N>`

### 本地文本生成

`anna-generate` 是一个从 prompt 直接生成文本的 CLI，只适用于文本生成模型族：

```bash
anna-generate --model-dir <path-to-text-model> --prompt "Write a short summary of KV cache."
```

### 做模型基准

纯文本基准：

```bash
anna-bench --model-dir <path-to-text-model> --prompt "Explain grouped-query attention." --warmup 1 --runs 3
```

图片基准：

```bash
anna-bench --model-dir <path-to-multimodal-model> --prompt "Describe the image." --image ./demo.png
```

视频基准：

```bash
anna-bench --model-dir <path-to-multimodal-model> --prompt "Summarize the clip." --video ./demo.mp4
```

### 使用 Qwen3-TTS 合成语音

Base 克隆音色模型：

```bash
anna-speak \
  --model-dir <path-to-qwen3-tts-base> \
  --input "Hello from Anna." \
  --output out.wav \
  --ref-audio ref.wav \
  --ref-text "Reference transcript."
```

CustomVoice 模型：

```bash
anna-speak \
  --model-dir <path-to-qwen3-tts-custom> \
  --input "Hello from Anna." \
  --output out.wav \
  --speaker Vivian \
  --instruct "Speak with energy."
```

VoiceDesign 模型：

```bash
anna-speak \
  --model-dir <path-to-qwen3-tts-voice-design> \
  --input "Hello from Anna." \
  --output out.wav \
  --instruct "A calm, warm female voice."
```

CLI 的 `--response-format` 支持 `wav` 和 `flac`。

## OpenAI 兼容 API

服务启动后，路由始终存在，但是否能成功响应，取决于当前加载的模型族是否支持对应能力。

### 路由列表

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/audio/speech`

### 请求示例

列出模型：

```bash
curl http://127.0.0.1:8000/v1/models
```

聊天补全：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "Write a haiku about Intel Arc."}],
    "reasoning_format": "deepseek"
  }'
```

文本补全：

```bash
curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "prompt": "Anna is",
    "max_tokens": 64
  }'
```

Base 克隆音色 TTS 模型语音合成：

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

`custom_voice` 模型请传 `speaker` 或 `voice`；`voice_design` 模型请传 `instruct`。

聊天请求还支持：

- 含 `text`、`image_url`、`video_url`、`audio_url` 的多模态 `content` 数组
- `tools`、`tool_choice`、`parallel_tool_calls`
- `enable_thinking` 或 `chat_template_kwargs.enable_thinking`

## 常用运行参数

| 分类 | 参数 | 说明 |
| --- | --- | --- |
| 通用 | `--device`、`--dtype`、`--compile-mode`、`--compile-fullgraph` | `device` 支持 `auto`、`cpu`、`xpu`；`dtype` 支持 `auto`、`float32`、`float16`、`bfloat16` |
| 预填充与 prompt cache | `--prefill-chunk-size`、`--prompt-cache-size`、`--prompt-cache-max-tokens`、`--profile-runtime` | 文本生成 CLI 和服务均可用 |
| KV cache | `--kv-cache-quantization`、`--kv-cache-quant-bits`、`--kv-cache-residual-len` | Qwen3.5 和 Gemma4 运行时支持 TurboQuant |
| Qwen 大模型控制 | `--offload-mode`、`--offload-vision`、`--expert-quant`、`--weight-quant`、`--resident-expert-layers`、`--resident-expert-layer-indices`、`--cached-experts-per-layer` | 主要用于大尺寸 Qwen3.5 MoE / VL 模型 |
| 服务默认行为与安全阈值 | `--enable-thinking`、`--disable-thinking`、`--reasoning-format`、`--max-completion-tokens`、`--min-free-memory-mib`、`--reserve-memory-mib`、`--max-estimated-usage-ratio`、`--generation-memory-safety-factor` | 控制默认聊天行为与请求准入 |
| 服务批处理与指标 | `--scheduler-max-batch-size`、`--scheduler-batch-wait-ms`、`--metrics-log-interval-seconds` | 连续批处理只有在 batch size 大于 `1` 时才会启用 |
| 语音合成 | `--language`、`--speaker`、`--instruct`、`--ref-audio`、`--ref-text`、`--x-vector-only-mode`、`--response-format` | `anna-speak` 支持 `wav`、`flac`；API 还支持 `pcm` |

## 开发

运行测试：

```bash
python -m pytest
```

如果你在做 XPU fused kernel 相关开发，先构建自定义算子，再使用 `.build/anna_gated_delta_fused` 下的编译结果跑 CLI 或测试会更稳妥。
