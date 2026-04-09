# Anna

[English](README.md) | [简体中文](README_zh.md)

## 环境要求（本地开发和测试）

- Windows 或 Linux + Intel Arc 显卡
- PyTorch `xpu`
- Intel oneAPI DPC++/C++
- 可本地编译并加载 `anna_gated_delta_fused` 自定义算子

如果你准备在 Arc A770 / A750 上跑本地 Qwen3.5，这个 README 就是按这条路径写的。Windows 仍是当前主要开发环境；Linux 走同一套运行时，但 fused-op 需要按 Linux 的方式构建。

## 分支特性

- OpenAI 兼容接口
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `POST /v1/audio/speech`
- 文本生成与流式输出
- Qwen3-TTS 语音合成
- `anna-speak` 本地语音生成命令
- `xpu` 设备推理
- 线性注意力 SYCL fused ops
- 启动时终端打印服务地址与可用路由
- 运行时终端打印聚合指标
  - 空闲时不会持续刷日志

## 推荐环境

当前 `main` 分支默认按下面的环境准备：

- Windows 11 或较新的 Linux 发行版
- Python 3.11 或 3.12
- Intel Arc A770 / A750
- 已正确安装 Intel GPU 驱动
- 已正确安装带 `xpu` 的 PyTorch
- Intel oneAPI DPC++/C++ Compiler
- Windows 下额外需要 Visual Studio 2022 Build Tools

## 环境配置

### 1. 克隆仓库（`main` 已包含 SYCL）

```powershell
git clone -b main <your-repo-url> Anna
cd Anna
```

### 2. 创建 Python 环境

Windows 下可继续使用 Conda/Miniforge：

```powershell
conda create -n anna python=3.12 -y
conda activate anna
python -m pip install -U pip
```

Linux 下可直接使用 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 3. 安装 PyTorch

安装完成后先验证：

```powershell
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

预期至少满足：

- 能正常 `import torch`
- `torch.xpu.is_available()` 输出 `True`

### 4. 安装项目本身

仅运行：

```powershell
pip install -e .
```

注意：

- `pip install -e .` 只会安装 Python 包，不会自动编译 SYCL fused op
- 安装完成后必须继续执行下面的 `python tools/build_gated_delta_fused_op.py`

### 5. 准备编译器环境

构建脚本现在会按平台自动选择默认值，也支持通过环境变量覆盖。

Windows 下：

- 最好先把 `dpcpp.exe` 放到 `PATH` 中
- 如果 `PATH` 里没有，构建脚本还会自动搜索 `Program Files` 下的标准 oneAPI 安装目录
- 对 MSVC 环境脚本，构建脚本会搜索标准的 Visual Studio Build Tools / IDE 安装位置；如果你用了非标准安装目录，再设置 `ANNA_VCVARS64`
- 也可以直接用 `ANNA_DPCPP` 显式指定编译器路径

```powershell
$env:ANNA_DPCPP = "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\dpcpp.exe"
$env:ANNA_VCVARS64 = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

Linux 下：

- 确保 `dpcpp` 已在 `PATH` 中；如果没有，就把绝对路径写到 `ANNA_DPCPP`
- 如果 oneAPI 运行时库不在编译器相邻的 `lib` 目录里，可用 `ANNA_ONEAPI_RUNTIME_PATHS` 指定，多个目录用 `:` 分隔

```bash
export ANNA_DPCPP=/opt/intel/oneapi/compiler/latest/bin/dpcpp
export ANNA_ONEAPI_RUNTIME_PATHS=/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/compiler/latest/lib/x64
```

### 6. 编译 SYCL fused op

这是当前 `main` 分支在 Intel Arc `xpu` 环境下的必需步骤：

```powershell
python tools/build_gated_delta_fused_op.py
```

成功后会在下面目录生成动态库：

- Windows：`.build\anna_gated_delta_fused\anna_gated_delta_fused.pyd`
- Linux：`.build/anna_gated_delta_fused/anna_gated_delta_fused.so`

终端通常会看到类似输出：

```text
Compiling Anna fused XPU/SYCL ops...
library_path=<repo>/.build/anna_gated_delta_fused/anna_gated_delta_fused.<pyd|so>
gated_delta_registered=True
causal_conv1d_registered=True
```

### 7. 可选验证

```powershell
pytest tests/test_fused_op_xpu.py -q
```

如果只是做一次完整回归：

```powershell
pytest -q
```

### 8. SoX (optional)

上游 `qwen-tts` 运行时若找不到 SoX 命令行（`PATH` 中无 `sox`，终端可能提示「不是内部或外部命令」），会打出警告。推理仍可继续；安装 SoX 可消除警告，并在栈内调用 SoX 相关音频工具时使其可用。

在 Windows 上任选其一：

- **Chocolatey**（管理员 PowerShell）：`choco install sox`
- **Scoop**：`scoop install sox`
- **手动安装**：安装包含 `sox.exe` 的版本（参见 [SoX](http://sox.sourceforge.net/)），将其所在目录加入用户或系统 `PATH`。

Linux 下请通过发行版自带的软件包管理器安装 `sox`，然后在新 shell 中确认 `sox --version` 正常。

打开**新的**终端验证：

```powershell
sox --version
```

## 模型目录要求

### 文本 / 多模态 Qwen 模型

本地模型目录至少应包含：

- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `model.safetensors` 或分片 `*.safetensors`

例如：

```text
D:\Projects\Anna\models\Qwen\Qwen3___5-2B
```

### Qwen3-TTS 12Hz 模型

Anna 也支持官方 `Qwen3-TTS` 12Hz 模型目录，底层走 `qwen-tts` 运行时。一个本地 `qwen3_tts` 模型目录至少应包含：

- `config.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `model.safetensors`
- `speech_tokenizer\config.json`
- `speech_tokenizer\model.safetensors`

例如：

```text
D:\Projects\Anna\models\Qwen\Qwen3-TTS-12Hz-1___7B-Base
```

说明：

- `anna-generate` 和 `anna-bench` 当前只面向 `qwen3_5_text` 家族；`qwen3_tts` 家族请使用 `anna-speak` 或 `POST /v1/audio/speech`
- 当前 TTS 支持面向官方 12Hz 目录结构，如 `Base`、`CustomVoice`、`VoiceDesign`
- **SoX** 为可选项：见 [步骤 8：SoX (optional)](#8-sox-optional)；未安装时仅告警，不会阻塞此处验证过的 12Hz 推理

## 运行命令示例

### 1. 启动服务

最小示例：

```powershell
anna-serve --model-dir D:\path\to\model --device xpu --dtype bfloat16
```

Linux/bash 示例：

```bash
anna-serve \
  --model-dir /path/to/model \
  --model-name Qwen3.5-2B \
  --device xpu \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000
```

与你当前 `main` 分支/环境一致的示例：

```powershell
anna-serve `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B `
  --model-name Qwen3.5-2B `
  --device xpu `
  --dtype bfloat16 `
  --offload-mode none `
  --weight-quant none `
  --host 127.0.0.1 `
  --port 8000
```

面向 Arc A770 上本地 Qwen3.5 9B distilled 模型、启用 int4 稠密权重并收紧显存保护参数的示例：

```powershell
anna-serve `
  --model-dir D:\Projects\Anna\models\Jackrong\Qwen3___5-9B-Claude-4___6-Opus-Reasoning-Distilled-v2 `
  --model-name qwen3.5 `
  --device xpu `
  --dtype bf16 `
  --weight-quant int4 `
  --offload-vision `
  --disable-thinking `
  --reasoning-format deepseek `
  --min-free-memory-mib 256 `
  --reserve-memory-mib 128 `
  --max-estimated-usage-ratio 0.95 `
  --generation-memory-safety-factor 1.25
```

面向 Arc A770 上本地 Gemma 4 E4B 多模态运行时、沿用同一组显存保护参数的示例：

```powershell
anna-serve `
  --model-dir D:\Projects\Anna\models\google\gemma-4-E4B-it `
  --model-name gemma4 `
  --device xpu `
  --dtype bf16 `
  --weight-quant int4 `
  --offload-vision `
  --disable-thinking `
  --reasoning-format deepseek `
  --min-free-memory-mib 256 `
  --reserve-memory-mib 128 `
  --max-estimated-usage-ratio 0.95 `
  --generation-memory-safety-factor 1.25
```

如果你不想看周期指标日志：

```powershell
anna-serve `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B `
  --model-name Qwen3.5-2B `
  --device xpu `
  --dtype bfloat16 `
  --metrics-log-interval-seconds 0
```

### 2. 启动后终端行为

服务启动后会在终端打印：

- 服务地址
- 当前可用路由
- 运行时聚合指标

示例：

```text
INFO anna.cli.serve: Starting Anna server on http://127.0.0.1:8000
INFO anna.cli.serve: Available routes are:
INFO anna.cli.serve: Route: /openapi.json, Methods: HEAD, GET
INFO anna.cli.serve: Route: /healthz, Methods: GET
INFO anna.cli.serve: Route: /v1/models, Methods: GET
INFO anna.cli.serve: Route: /v1/chat/completions, Methods: POST
INFO anna.cli.serve: Route: /v1/completions, Methods: POST
INFO anna.cli.serve: Route: /v1/audio/speech, Methods: POST
```

### 3. 直接文本生成

```powershell
anna-generate `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B `
  --model-name Qwen3.5-2B `
  --device xpu `
  --dtype bfloat16 `
  --offload-mode none `
  --weight-quant none `
  --prompt "Hello" `
  --max-new-tokens 32
```

### 4. 基准测试

```powershell
anna-bench `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B `
  --model-name Qwen3.5-2B `
  --device xpu `
  --dtype bfloat16 `
  --prompt "你好，请介绍一下你自己。" `
  --runs 5
```

### 5. 直接运行 Qwen3-TTS Base 语音合成

这条路径适合本地语音克隆模型，例如 `Qwen3-TTS-12Hz-1.7B-Base`。

```powershell
anna-speak `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3-TTS-12Hz-1___7B-Base `
  --device xpu `
  --dtype bfloat16 `
  --input "Anna CLI smoke test on Intel Arc." `
  --output D:\Projects\Anna\.build\anna_tts_cli_smoke.wav `
  --language English `
  --ref-audio D:\Projects\Anna\.build\tts_ref.wav `
  --ref-text "Hello Anna. This is a reference voice for local synthesis testing." `
  --max-new-tokens 1024
```

如果你加载的是 `CustomVoice` 模型，就使用 `--speaker`，可选再加 `--instruct`。

如果你加载的是 `VoiceDesign` 模型，就提供 `--instruct`，不需要 `--speaker`。

### 6. 启动 Qwen3-TTS 服务

```powershell
anna-serve `
  --model-dir D:\Projects\Anna\models\Qwen\Qwen3-TTS-12Hz-1___7B-Base `
  --model-name Qwen3-TTS-1.7B-Base `
  --device xpu `
  --dtype bfloat16 `
  --host 127.0.0.1 `
  --port 8000
```

### 7. 调用语音接口

服务端会直接返回音频字节。对于 `Base` 语音克隆模型，需要带上 `ref_audio` 和 `ref_text`。

在 **PowerShell** 中，请用**单引号**包住 `-d` 的整段 JSON，使请求体为合法 JSON：JSON 内直接使用双引号 `"`，**不要**写成 `\"`（否则反斜杠会原样进入请求体，导致解析失败）。在 Windows 上建议 `ref_audio` 使用正斜杠路径。

```powershell
curl.exe http://127.0.0.1:8000/v1/audio/speech `
  -H "Content-Type: application/json" `
  --output speech.wav `
  -d '{"model":"Qwen3-TTS-1.7B-Base","input":"This request goes through the Anna FastAPI speech route.","language":"English","ref_audio":"D:/Projects/Anna/.build/tts_ref.wav","ref_text":"Hello Anna. This is a reference voice for local synthesis testing.","response_format":"wav","max_new_tokens":1024}'
```

当前支持的主要请求字段包括：

- `input`：待合成文本
- `response_format`：`wav`、`flac`、`pcm`
- `language`
- `speaker` 或 `voice`，用于 `CustomVoice`
- `instruct`，用于 `VoiceDesign`，也可用于 `CustomVoice` 的风格控制
- `ref_audio`、`ref_text`、`x_vector_only_mode`，用于 `Base`
- `max_new_tokens`、`do_sample`、`temperature`、`top_p`、`top_k`、`repetition_penalty`
- `subtalker_do_sample`、`subtalker_temperature`、`subtalker_top_p`、`subtalker_top_k`
- `non_streaming_mode`

## API 调用示例

### Chat Completions

```powershell
curl.exe http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"Qwen3.5-2B","messages":[{"role":"user","content":"你好，介绍一下你自己。"}],"max_completion_tokens":128}'
```

### Completions

```powershell
curl.exe http://127.0.0.1:8000/v1/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"Qwen3.5-2B","prompt":"Hello","max_tokens":32}'
```

### 查看模型列表

```powershell
curl.exe http://127.0.0.1:8000/v1/models
```

### 健康检查

```powershell
curl.exe http://127.0.0.1:8000/healthz
```

## 常见问题

### 1. `torch.xpu.is_available()` 是 `False`

优先检查：

- PyTorch 版本是否真的支持 `xpu`
- Intel GPU 驱动是否正确安装
- 当前 Python 环境是否装错了 `cpu` 版 `torch`

### 2. fused op 编译失败

优先检查：

- `ANNA_DPCPP` 是否指向正确编译器，或者 `dpcpp` 是否已在 `PATH` 中
- Windows 下 `ANNA_VCVARS64` 或 `vcvars64.bat` 路径是否正确
- oneAPI 运行时库是否能被找到；Linux 下如果自动探测不完整，可设置 `ANNA_ONEAPI_RUNTIME_PATHS`
- Windows 下 oneAPI 与 MSVC 是否都已安装
- 当前激活环境里的 `torch` 是否包含头文件和 `torch_xpu` 相关库

### 3. 服务能启动，但生成时报 fused-op 相关错误

先重建一次自定义算子：

```powershell
python tools/build_gated_delta_fused_op.py
```

然后重启服务。

### 4. 指标日志不想显示

可以直接关闭：

```powershell
anna-serve --model-dir D:\path\to\model --device xpu --dtype bfloat16 --metrics-log-interval-seconds 0
```

## 开发建议

- 改了 SYCL 源码后，先重新执行 `python tools/build_gated_delta_fused_op.py`
- 再执行 `pytest tests/test_fused_op_xpu.py -q`
- 最后再重启 `anna-serve`
