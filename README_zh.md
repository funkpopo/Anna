# Anna

[English](README.md) | [简体中文](README_zh.md)

Anna 是一个在**本地**跑大语言模型和语音模型的**推理运行时**，对外提供 **OpenAI 格式 HTTP API**。底层基于 **PyTorch**，针对 **Intel Arc（XPU）** 做了较多优化，**CPU** 也可用于开发与测试。

你把磁盘上的一个**模型目录**指给 Anna（常见是带 `config.json` 的 Hugging Face 式目录，或 Qwen 的 **GGUF** 加可选的 `mmproj` 视觉塔），Anna 会按模型类型提供聊天、文本补全、多模态聊天或语音合成。

## 能做什么

- **HTTP 接口**：`/healthz`、`/v1/models`、`/v1/chat/completions`、`/v1/completions`、`/v1/audio/speech`
- **流式 / 非流式**；可把推理过程拆到 `reasoning_content`；支持**工具 / 函数调用**
- **多模态聊天**（图片、视频；Gemma 4 还支持音频），取决于模型能力
- **命令行**：`anna-serve`、`anna-generate`、`anna-bench`、`anna-speak`
- **运行时能力**（偏 XPU）：`torch.compile`、预填充分块、相同提示词 KV 复用、TurboQuant KV、运行时 int4 权重、MoE 专家卸载与缓存、可选连续批处理、可选 SYCL 融合算子

## 支持的模型类型

Anna 根据 `config.json` **顶层**的 `model_type` 选择后端：

| `model_type` 大致情况 | 运行时 | 典型用途 |
| --- | --- | --- |
| 非 `qwen3_tts`、非 `gemma4`（如 `qwen3_5`、`qwen3_5_moe`、`qwen3_5_vl`） | Qwen3.5 文本 / 多模态 | 聊天、补全、图文/视频、工具调用 |
| `gemma4` | Gemma 4 | 同上，聊天里可带音频 |
| `qwen3_tts` | Qwen3-TTS | 仅语音合成 |

**与各 CLI 的对应关系：**

- `anna-generate`、`anna-bench`：只适用于**文本生成类**模型（不能是 `qwen3_tts`）。
- `anna-speak`：只适用于 `qwen3_tts`。
- `anna-serve`：三类都可以；具体哪个 HTTP 接口能用，取决于当前加载的模型。

## 环境要求

- **Python 3.11+**
- 本机上的**模型目录**（HF 式目录，或 Qwen GGUF + 可选 `mmproj-*.gguf`）
- 按你的机器单独安装 **PyTorch**（项目要求 `torch>=2.7`）。用 Arc 时请装带 **XPU** 的版本。
- **视频**：依赖 `imageio`、`imageio-ffmpeg`（随包安装）
- **Qwen3-TTS**：依赖 `qwen-tts`（已在项目依赖里）
- **可选**：融合 XPU 算子需要 Intel **oneAPI** DPC++；Windows 上通常还需要 **Visual Studio** 生成工具

`pip install -e .` 会装 Python 依赖；**PyTorch 和 oneAPI 要按你的硬件自行准备**。

## 安装

```bash
git clone https://github.com/YOUR_USERNAME/Anna.git
cd Anna
python -m venv .venv
```

**Windows（PowerShell）：** `.\.venv\Scripts\Activate.ps1`  
**Linux / macOS：** `source .venv/bin/activate`

```bash
python -m pip install -U pip
python -m pip install -e .
# 可选：python -m pip install -e ".[dev]"
```

**检查 PyTorch / XPU：**

```bash
python -c "import torch; print(torch.__version__); print(getattr(torch, 'xpu', None) and torch.xpu.is_available())"
```

**可选：编译融合算子**（在支持的 XPU 环境上有利于性能）：

```bash
python tools/build_gated_delta_fused_op.py
```

产物在 `.build/anna_gated_delta_fused`。也可用环境变量 `ANNA_GATED_DELTA_OP_LIB` 指定库路径（见下文）。

Arc int4 fused kernel 调参开关：

- `ANNA_XPU_INT4_LM_HEAD_LOCAL_SIZE`：`lm_head_int4_topk_fused` 的 work-group local size（会向上取 2 的幂，最大 64；blocked top-k 路径为了保证正确性至少使用 64）。
- `ANNA_XPU_INT4_LM_HEAD_BLOCK_TOPK_THRESHOLD`：两阶段 blocked `lm_head_int4_topk_fused` 路径的 vocab 阈值（默认 65536）。
- `ANNA_XPU_INT4_LM_HEAD_BLOCK_SIZE`：blocked `lm_head_int4_topk_fused` 的 vocab block 大小（默认 4096）。
- `ANNA_XPU_INT4_GEMV_LOCAL_SIZE`：实验性 standalone `XPUInt4Linear` GEMV 路径的 local size；仅在 `ANNA_XPU_INT4_MATMUL=sycl` 且 decode rows `<= 4` 时使用（会向上取 2 的幂，最大 256）。
- `ANNA_XPU_INT4_GEMV_KERNEL`：实验性 standalone GEMV kernel 模式。默认 `wg` 使用原 work-group 归约路径；`subgroup` 在 M=1 时用一个 subgroup 计算一个输出 tile，并用 subgroup reduction，避免 local-memory partial sums。
- `ANNA_XPU_INT4_GEMV_OUTPUT_TILE`：实验性 M=1 输出通道 tile；默认 `1`。大于 `1` 的值只用于严格 `4096x4096` decode 实验，可能触发 Level Zero 资源限制。`subgroup` 模式会准备 GEMV 专用 `[output_tiles, packed_k, tile]` qweight 布局和 `[group, output_tiles, tile]` scale/zero 布局，让 subgroup 读取相邻输出时连续访问，而不是复用普通 linear layout。
- `ANNA_XPU_AUTO_INT4_GEMV`：允许 `ANNA_XPU_INT4_MATMUL=auto` 在严格的 `M=1,K=N=4096,group=128` shape 上试探 standalone GEMV。默认关闭，直到 Arc sweep 数据稳定证明收益。
- `ANNA_XPU_INT4_MOE_GATE_LOCAL_SIZE`：grouped int4 MoE gate/up projection 的 local size（会向上取 2 的幂，最大 256）。
- `ANNA_XPU_INT4_MOE_DOWN_LOCAL_SIZE`：grouped int4 MoE down projection 的 local size（会向上取 2 的幂，最大 256）。

默认值保持当前保守策略。在 Arc A770/A750 上，建议先配合 `tools/bench_xpu_hotspots.py --arc-profile` 扫描这些值；需要专门扫 int4 时可加 `--arc-int4-only` 跳过 attention/router 等通用热点。可用 `--arc-int4-gemv-kernels wg,subgroup`、`--arc-int4-gemv-local-sizes 32,64,128` 和 `--arc-int4-gemv-output-tiles 1,2,4` 做可复现 standalone GEMV sweep。报告会同时包含普通 `XPUInt4Linear`、`lm_head_int4_topk_fused` 和 `moe_grouped_int4_mlp_fused` 行，便于确认局部 kernel 优化是否真的覆盖 decode 关键路径。
`anna-serve` 可直接用 CLI 参数替代 SYCL int4 环境变量：`--xpu-int4-matmul sycl --xpu-int4-gemv-kernel subgroup --xpu-int4-gemv-output-tile 4 --xpu-int4-gemv-local-size 128`。当同时设置 `--xpu-int4-matmul sycl --xpu-int4-gemv-kernel subgroup` 时，Anna 会把 tiled subgroup SYCL 路径应用到所有 `XPUInt4Linear` row count，而不是只用于 decode rows。Intel FlashQLA-compatible GDN prefill 也可以通过 CLI 启用：`--enable-flashqla-gdn-prefill`；它等价于设置 `ANNA_XPU_FLASHQLA_GDN_PREFILL=1`，并且在 op、设备、dtype 或 shape 不支持时会直接报错，不会 fallback。

运行时 int4 转换 `lm_head` 时，Anna 还会额外准备面向 top-k 的 scale/zero 布局（`[vocab, group_count]`）供 `lm_head_int4_topk_fused` 使用；普通 linear 仍保持标准 matmul 布局。
XPU int4 sidecar cache 也会保存实验性 decode layout：GEMV subgroup 的 tile-major qweight/scale/zero，以及 `lm_head` fused top-k 使用的 tile-major qweight。旧 v1 cache 会自动 miss 并按新 payload 重建。

## 快速开始

**启动 API 服务：**

```bash
anna-serve --model-dir /path/to/model --host 127.0.0.1 --port 8000
```

**偏 XPU 的一例：**

```bash
anna-serve \
  --model-dir /path/to/model \
  --device xpu \
  --xpu-device-index 0 \
  --dtype bfloat16 \
  --kv-cache-quantization turboquant \
  --kv-cache-quant-bits 4 \
  --weight-quant auto \
  --enable-flashqla-gdn-prefill \
  --prompt-cache-size 4
```

**命令行直接生成文本：**

```bash
anna-generate --model-dir /path/to/text-model --prompt "用一段话解释 KV cache。"
```

**性能基准（纯文本或多模态）：**

```bash
anna-bench --model-dir /path/to/model --prompt "你好" --warmup 1 --runs 3
```

**检查运行时 XPU int4 缓存是否会启用：**

```bash
anna-xpu-int4-cache --model-dir /path/to/model --weight-quant auto --xpu-total-memory-gib 16
```

对于没有现成量化配置的 safetensors Qwen3.5 模型，这个命令会报告 `--weight-quant auto` 是否解析为 `int4`，并显示 sidecar 缓存目录，通常是 `<model-dir>/.anna/xpu_int4_cache`。

**语音合成（示例：基础克隆音色）：**

```bash
anna-speak --model-dir /path/to/tts --input "你好。" --output out.wav --ref-audio ref.wav --ref-text "参考句子的文字。"
```

---

## 命令行参数说明（逐项含义）

下面按「谁用得到」分组，方便你对照 `anna-serve -h` 等帮助信息。

### 多条命令共有（文本类：`serve` / `generate` / `bench`）

| 参数 | 作用说明 |
| --- | --- |
| `--model-dir` | **必填**。模型所在目录（或 Anna 能识别的 GGUF 布局）。 |
| `--model-name` | 在日志或对外展示里用的模型名；不设则从路径推断。 |
| `--device` | 执行设备：`auto`（有 XPU 则优先 XPU）、`cpu`、`xpu`。 |
| `--xpu-device-index N` | 通过 `ONEAPI_DEVICE_SELECTOR=level_zero:N` 绑定第 **N** 块 XPU。机器上同时有核显和 Arc 时，用这个指定 Arc。 |
| `--no-xpu-env-defaults` | **不要**自动设置 Anna 推荐的 Level Zero 环境变量（见下文「XPU 环境」）。 |
| `--dtype` | 计算精度：`auto`、`float32`、`float16`、`bfloat16`。 |
| `--compile-mode` | `torch.compile` 模式：`none`、`auto`、`default`、`reduce-overhead`、`max-autotune`。**默认**：`serve` 为 `auto`，`generate` / `bench` 为 `none`（首次编译可能增加延迟）。 |
| `--compile-fullgraph` | 在启用 compile 时请求整图捕获。 |
| `--prefill-chunk-size` | 把较长的**纯文本**预填充分成多段（按 token 数）。`0` 表示在 XPU 上由 Anna 自动决定块大小。 |
| `--prompt-cache-size` | 最多缓存 **N** 条**完全相同**的文本提示的 KV，便于重复请求。`0` 表示关闭。 |
| `--prompt-cache-max-tokens` | 只缓存不超过 **N** 个 token 的提示，减轻长提示的内存压力。`0` 表示不设上限。 |
| `--profile-runtime` | 打开预填 / 解码阶段的 XPU 同步计时与内存等剖析日志。 |
| `--enable-flashqla-gdn-prefill` | 在 XPU 上启用 Intel FlashQLA-compatible GDN prefill 路径。设备、shape、dtype 或自定义 op 不支持时直接报错；不做 fallback。 |
| `--kv-cache-quantization` | KV 量化：`none` 或 `turboquant`（Qwen3.5 与 Gemma4 支持）。 |
| `--kv-cache-quant-bits` | TurboQuant 位宽：`2`、`3` 或 `4`。 |
| `--kv-cache-residual-len` | 最近的 **N** 个 KV 位置保持高精度，更早的位置再压缩。 |
| `--offload-mode` | `auto`、`none`、`experts`；`experts` 在适用时会做 MoE 专家卸载。 |
| `--offload-vision` | 主模型在 XPU 上时，仍把**视觉塔留在 CPU**，省显存（适合纯文本或显存紧张）。 |
| `--expert-quant` | XPU 上**专家权重**的量化：`auto`、`none`、`int4`；`auto` 在卸载场景下可能启用 int4。 |
| `--weight-quant` | XPU 上**稠密线性层**权重量化：`auto`、`none`、`int4`；`auto` 在显存吃紧时可能启用 int4。 |
| `--resident-expert-layers` | 前 **N** 个稀疏 MoE 层整层留在加速器上；不设则在专家卸载模式下自动估算。 |
| `--resident-expert-layer-indices` | 逗号分隔的 **从 0 开始**的层号，指定哪些层常驻；**覆盖** `--resident-expert-layers`。 |
| `--cached-experts-per-layer` | 每层最多在 XPU 上缓存多少个已卸载专家；`0` 关闭；不设则自动。 |
| `--log-level` | 日志级别，例如 `info`。 |

### 仅 `anna-serve`

| 参数 | 作用说明 |
| --- | --- |
| `--no-inference-warmup` | 加载后**不做**一小轮预热推理；融合核可能在**第一次真实请求**时才加载。 |
| `--enable-thinking` / `--disable-thinking` | 客户端未指定思考相关字段时，聊天是否默认开启思考。 |
| `--max-completion-tokens` | 请求里未写 `max_tokens` / `max_completion_tokens` 时的默认输出长度上限。 |
| `--reasoning-format` | 推理内容格式：`none` 或与 `deepseek` 类似地把推理放进 `reasoning_content`。 |
| `--min-free-memory-mib` | 开始生成前要求至少剩余这么多 XPU 显存（MiB），用于准入控制。 |
| `--reserve-memory-mib` | 准入时额外预留的显存余量（MiB）。 |
| `--max-estimated-usage-ratio` | 若估算用量超过总显存的该比例（0–1]，则拒绝请求。 |
| `--generation-memory-safety-factor` | 对估算的生成显存再乘的系数（≥ 1），偏保守则调大。 |
| `--scheduler-max-batch-size` | 大于 **1** 时开启**连续批处理**（解码步合并）。为 **1** 时逐请求解码（延迟更低、核启动更多）。 |
| `--scheduler-batch-wait-ms` | 批处理时，为凑批最多等待的毫秒数；越大吞吐越好、尾延迟可能变差。 |
| `--metrics-log-interval-seconds` | 每隔 **N** 秒在终端打汇总指标；`0` 关闭。 |
| `--host`、`--port` | HTTP 服务监听地址与端口。 |

### 仅 `anna-generate`

| 参数 | 作用说明 |
| --- | --- |
| `--prompt` | **必填**。输入文本。 |
| `--max-new-tokens` | 最多新生成多少个 token；不设则跟模型配置或内部安全估算。 |
| `--temperature`、`--top-p`、`--top-k`、`--repetition-penalty` | 采样与重复惩罚。 |

### 仅 `anna-bench`

| 参数 | 作用说明 |
| --- | --- |
| `--image` / `--video` | 本地图片 / 视频路径；会走多模态聊天式推理（暂无单独音频 benchmark）。 |
| `--warmup` | 正式计时的 `runs` 之前先跑几轮，用于预热。 |
| `--runs` | 计时轮数；输出平均延迟、吞吐等。 |
| `--max-new-tokens`、采样相关参数 | 与 `generate` 含义相同；bench 默认更偏确定性与测速（如 `temperature=0`）。 |

### 仅 `anna-speak`（Qwen3-TTS）

**注意：** 该命令**没有** `--xpu-device-index` / `--no-xpu-env-defaults`；若在 Arc 上要固定设备，可自行设置 `ONEAPI_DEVICE_SELECTOR` 等。

| 参数 | 作用说明 |
| --- | --- |
| `--input` | 要合成的文字。 |
| `--output` | 输出音频路径。 |
| `--language` | 可选语言。 |
| `--speaker` | CustomVoice 模型的说话人名称。 |
| `--instruct` | VoiceDesign / CustomVoice 的风格或指令描述。 |
| `--ref-audio`、`--ref-text` | 基础克隆模型：参考音频与对应文本。 |
| `--x-vector-only-mode` | 基础克隆：只用说话人嵌入，不用参考文本条件。 |
| `--response-format` | 输出容器：`wav` 或 `flac`。 |
| `--max-new-tokens` | 生成长度上限（语音 token 侧）。 |
| `--do-sample` / `--no-do-sample` | 主路径是否随机采样。 |
| `--temperature`、`--top-p`、`--top-k`、`--repetition-penalty` | 主采样参数。 |
| `--subtalker-do-sample` 等 | 子模块的采样参数。 |
| `--non-streaming-mode`（默认）与 `--streaming-style-input` | 文本是整块送入还是模拟流式送入。 |

### XPU / Level Zero 自动环境

在 `--device` 为 `auto` 或 `xpu` 且**未**加 `--no-xpu-env-defaults` 时，Anna 会在加载模型前设置推荐变量，例如：

- `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`
- `ZES_ENABLE_SYSMAN=1`
- 若指定了 `--xpu-device-index`，则设置 `ONEAPI_DEVICE_SELECTOR=level_zero:<index>`

若你在系统里已统一配置 Level Zero，可加 `--no-xpu-env-defaults` 避免覆盖。

---

## 可选环境变量（进阶）

| 变量 | 作用 |
| --- | --- |
| `ANNA_XPU_INT4_MATMUL` | `auto`（默认）、`torch`、`dequant` 或实验性 `sycl`。`sycl` 对 decode rows `<= 4` 使用 Anna standalone int4 GEMV；`auto` 只有在 `ANNA_XPU_AUTO_INT4_GEMV=1` 且严格 shape guard 命中时才会试探该路径。 |
| `ANNA_GATED_DELTA_OP_LIB` | 指定已编译的 gated delta 融合算子动态库路径。 |
| `ANNA_XPU_DISABLE_MOE_GROUPED_INT4`、`ANNA_XPU_DISABLE_LM_HEAD_INT4_TOPK` | 设为 `1` / `true` 等可关闭对应融合算子。 |
| `ANNA_ENABLE_INT4_LM_HEAD_TOPK_FUSED` | 设为 `1`、`true`、`yes`、`on` 等可在可用时启用 int4 LM-head top-k 融合路径。 |
| `ANNA_PREFIX_KV_SHARE` | 设为 `0` 可关闭 ops 中的前缀 KV 共享（默认开启）。 |
| `ANNA_DPCPP`、`ANNA_VCVARS64`、`ANNA_ONEAPI_RUNTIME_PATHS` | Windows / oneAPI 下编译 `tools/build_gated_delta_fused_op.py` 时的路径提示。 |

---

## API 简要说明

路由始终注册；能否成功取决于当前模型是否支持该能力。

- `GET /healthz`：健康检查与运行时指标（若实现支持）
- `GET /v1/models`
- `POST /v1/chat/completions`：支持多模态 `content`、工具、思考 / 推理相关字段
- `POST /v1/completions`：**仅文本** prompt
- `POST /v1/audio/speech`：仅当加载 `qwen3_tts` 时可用

**列出模型：**

```bash
curl http://127.0.0.1:8000/v1/models
```

**聊天：**

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","messages":[{"role":"user","content":"你好"}]}'
```

**语音（基础克隆示例）：**

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","input":"你好","ref_audio":"ref.wav","ref_text":"……","response_format":"wav"}' \
  --output speech.wav
```

---

## 开发

```bash
python -m pytest
```

若修改融合核，请先执行 `tools/build_gated_delta_fused_op.py`，再针对 `.build/anna_gated_delta_fused` 跑测试或 CLI。

## 许可证

见 [LICENSE](LICENSE)。
