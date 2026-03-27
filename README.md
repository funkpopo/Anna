# Anna

`Anna` 是一个可本地部署的 Qwen3.5 推理服务，目标场景是 Intel Arc A770 / A750（`xpu`），并提供 OpenAI 兼容接口，方便你直接接入现有客户端或 SDK。

## 你可以用它做什么

- 本地加载 Hugging Face 风格模型目录（`safetensors` 单文件或分片）
- 提供 OpenAI 兼容 API：
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
- 支持 SSE 流式输出
- 支持文本与多模态（图像/视频）输入路径
- 支持常见采样参数：`temperature`、`top_k`、`top_p`、`repetition_penalty`
- 支持 BF16/FP16 推理，以及 FP8/AWQ/AWQ-4bit 装载路径

## 适用环境

- Windows（推荐）或 Linux
- Python 3.11+
- Intel GPU 运行环境（如 Arc A770）与 `torch+xpu`
- 本地可用模型目录（项目不负责下载模型文件）

## 安装与项目配置

### 1) 克隆项目

```bash
git clone <your-repo-url>
cd Anna
```

### 2) 创建并激活虚拟环境

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3) 安装项目

仅运行服务：

```bash
pip install -e .
```

需要本地开发/测试（含 `pytest`、`httpx`）：

```bash
pip install -e ".[dev]"
```

### 4) 安装 `xpu` 版 PyTorch

本项目要求使用带 Intel `xpu` 支持的 PyTorch。请按你当前系统与驱动版本，从官方渠道安装对应构建，然后执行：

```bash
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

当输出中 `torch.xpu.is_available()` 为 `True` 时，表示运行环境可用。

### 5) （可选）配置环境变量

你可以在本机设置默认模型目录，减少命令输入：

Windows PowerShell:

```powershell
$env:ANNA_MODEL_DIR="D:\path\to\your\model"
```

Linux/macOS:

```bash
export ANNA_MODEL_DIR="/path/to/your/model"
```

> `Anna` 本身不下载模型文件，需你提前准备本地模型目录。

## 快速开始

### 1) 准备模型目录

模型目录至少应包含：

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors`（或分片 `safetensors`）

### 2) 启动服务

```bash
anna-serve --model-dir /path/to/model --device xpu --dtype bf16
```

可选参数示例：

```bash
anna-serve --model-dir /path/to/model --model-name qwen3.5 --device xpu --dtype bf16 --max-completion-tokens 2048
```

显存紧张时，可启用文本线性层 `int4` 量化，并把视觉塔放到 CPU：

```bash
anna-serve --model-dir /path/to/model --model-name qwen3.5 --device xpu --dtype bf16 --weight-quant int4 --offload-vision
```

默认启用思维链

```bash
anna-serve --model-dir /path/to/model --model-name qwen3.5 --device xpu --dtype bf16 --weight-quant int4 --offload-vision --disable-thinking --min-free-memory-mib 256 --reserve-memory-mib 128 --max-estimated-usage-ratio 0.95 --generation-memory-safety-factor 1.25
```

禁用思维链

```bash
anna-serve --model-dir /path/to/model --model-name qwen3.5 --device xpu --dtype bf16 --weight-quant int4 --offload-vision --disable-thinking --reasoning-format deepseek --min-free-memory-mib 256 --reserve-memory-mib 128 --max-estimated-usage-ratio 0.95 --generation-memory-safety-factor 1.25
```

## API 调用示例

### Chat Completions

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role":"user","content":"你好，介绍一下你自己。"}],
    "max_completion_tokens": 256
  }'
```

### 流式输出

在请求体中设置 `"stream": true` 即可使用 SSE 流式返回。

## CLI 用法

### 文本生成

```bash
anna-generate --model-dir /path/to/model --device xpu --dtype bf16 --prompt "你好，介绍一下你自己。"
```

### 基准测试

```bash
anna-bench --model-dir /path/to/model --device xpu --dtype bf16 --prompt "你好，请介绍一下你自己。" --runs 5
```

## 关键行为说明

- 如果不传 `--model-name`，服务对外暴露的模型名默认是 `--model-dir` 的完整路径。
- 服务端默认 completion 上限优先级：
  1. 请求体 `max_completion_tokens`
  2. 请求体 `max_tokens`
  3. 启动参数 `--max-completion-tokens`
  4. 模型配置中的 `max_completion_tokens`（兼容 `max_new_tokens` / `max_tokens`）
  5. 回退到 `256`
- 聊天接口默认会保留模型原始 `<think>...</think>` 内容；若你不希望前端展示，需要在客户端侧自行过滤或拆分渲染。

## 当前状态

- 已完成基础可用性验证：模型加载、文本生成、多模态路径、API 烟测。
- 仍在持续完善 Arc A770 的系统化性能与稳定性验证。
- FP8/AWQ/AWQ-4bit 路径已接入，仍建议结合目标模型做充分数值校验。

## 常见问题

### 为什么服务起不来或推理不走 xpu？

请优先检查：

- `torch` 是否为带 `xpu` 支持的构建
- 驱动与运行时是否与当前 `torch` 版本兼容
- 启动参数是否显式传入 `--device xpu`

### 显存不足怎么办？

- 降低 `max_completion_tokens`
- 使用更小模型或量化模型
- 尝试 `--weight-quant int4`
- 文本为主的场景可加 `--offload-vision`

## 路线图

- 更完整的 Arc A770 性能与稳定性基准
- 量化模型的更广覆盖与校验
- 多模态调度与运行时开销优化

---
🙏 致谢

感谢 linuxdo 社区的交流、分享与反馈
