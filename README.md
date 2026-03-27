# Anna

从 0 实现的 Qwen3.5 推理服务，目标是在 Intel Arc A770 上运行，并提供 OpenAI 兼容 API。

## 当前实现范围

- Qwen3.5 原生文本塔
  - `linear_attention + full_attention` 混合层
  - KV cache / recurrent state / conv state
- Qwen3.5 原生多模态路径
  - vision tower
  - image/video placeholder 扩展
  - 多模态 3D RoPE 位置编码
- 本地 Hugging Face 风格模型目录加载
  - `config.json`
  - `preprocessor_config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `model.safetensors` 或分片 safetensors
- OpenAI 兼容 API
  - `/v1/models`
  - `/v1/chat/completions`
  - `/v1/completions`
  - SSE 流式输出
  - 原样透传 `<think>...</think>` 输出
- 采样
  - greedy
  - temperature
  - top-k
  - top-p
  - repetition penalty
- 量化装载路径
  - BF16 / FP16 计算
  - FP8
  - AWQ
  - AWQ-4bit

## 设计边界

- 不引用现成推理项目代码，不以 `ipex-llm` / `vLLM` / `llama.cpp` / `exllama` 为基础。
- 用户手动准备模型文件，本项目不提供模型下载逻辑。
- CPU 仅用于必要的图片/视频预处理；prefill 与 decode 的模型执行优先走 `xpu`。
- 多模态首轮 prefill 会把文本、图像、视频张量迁移到执行设备；后续 decode 只发送新 token，缓存留在执行设备。
- 纯文本请求支持服务端连续批处理：同长度 prefill 合批，decode 支持 mixed-length batching。
- full attention KV cache 采用共享 page pool + per-request block table，支持页级回收与复用。
- 运行前会按请求规模预估显存占用；显存预算不足时会提前拒绝请求，避免把 XPU 顶死。
- 运行期如果遇到 `OOM / device lost / out of resources`，会尝试清理 runtime cache 并返回可控错误。

## 当前限制

- 已在 `conda env anna` 中安装 `torch 2.11.0+xpu`，并完成 `pytest`、真实模型加载、文本生成、图片生成、视频生成与 API 烟测。
- 尚未在 Intel Arc A770 上完成系统化性能与稳定性验证。
- FP8 / AWQ / AWQ-4bit 虽已接入装载路径，但仍需真实官方模型做数值校验。
- 多模态请求当前仍走独占生成路径，未接入连续批处理与 paged KV cache 调度。

## 目录

- `src/anna/model`: Qwen3.5 文本、视觉与量化模块
- `src/anna/mm`: 多模态预处理与 placeholder 展开
- `src/anna/weights`: 配置、权重、tokenizer 加载
- `src/anna/runtime`: 设备迁移策略与推理引擎
- `src/anna/api`: OpenAI 兼容 HTTP API
- `src/anna/cli`: 启动服务与命令行生成
- `tests`: 形状、配置、多模态与量化静态测试

## 运行

1. 安装项目依赖。
2. 安装带 `xpu` 支持的 `torch` 构建。
3. 手动准备一个本地模型目录，至少包含：
   - `config.json`
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `model.safetensors` 或分片 safetensors
4. 启动服务：

```bash
anna-serve --model-dir /path/to/model --device xpu --dtype bf16
```

如果要给所有未显式传 `max_completion_tokens` / `max_tokens` 的 API 请求设置默认 completion 上限，可以在启动命令中指定：

```bash
anna-serve --model-dir /path/to/model --device xpu --dtype bf16 --max-completion-tokens 2048
```

如果要显式指定对外暴露的模型名称：

```bash
anna-serve --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B --model-name qwen3.5 --device xpu --dtype bf16
```

对显存吃紧的 dense 多模态模型，可以启用运行时文本线性层 `int4` 量化；如果主要提供纯文本服务，还可以把视觉塔留在 CPU：

```bash
anna-serve --model-dir D:\Projects\Anna\models\Jackrong\Qwen3___5-9B-Claude-4___6-Opus-Reasoning-Distilled-v2 --model-name qwen3.5 --device xpu --dtype bf16 --weight-quant int4 --offload-vision
```

如果不指定 `--model-name`，默认模型名就是 `--model-dir` 的完整路径。

服务端默认 completion 上限的解析顺序如下：

1. 请求体里的 `max_completion_tokens`
2. 请求体里的 `max_tokens`
3. `anna-serve --max-completion-tokens`
4. 模型 `config.json` 顶层的 `max_completion_tokens`（兼容 `max_new_tokens` / `max_tokens`）
5. 回退到 `256`

5. 命令行文本生成：

```bash
anna-generate --model-dir /path/to/model --device xpu --dtype bf16 --prompt "你好，介绍一下你自己。"
```

按模型名称生成：

```bash
anna-generate --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B --model-name qwen3.5 --device xpu --dtype bf16 --prompt "你好，介绍一下你自己。"
```

6. 基准测试：

```bash
anna-bench --model-dir /path/to/model --device xpu --dtype bf16 --prompt "你好，请介绍一下你自己。" --runs 5
```

7. 聊天接口思维开关：

```json
{
  "model": "qwen3.5",
  "messages": [
    {"role": "user", "content": "请解释一下什么是量化推理"}
  ],
  "max_completion_tokens": 256
}
```

聊天接口会默认保留模型原始的 `<think>...</think>` 文本，不在服务端裁剪、拆分或注入 closed-think prompt。客户端如果需要隐藏或单独渲染 reasoning，应自行解析返回的 `message.content` 或流式 `delta.content`。

## 下一步

- 使用官方 FP8 / AWQ 变体校验量化权重布局。
- 在 Arc A770 上做 profiling，定位热点算子，再决定是否补写自定义 XPU 算子。
- 把当前 paged KV cache 从 Python 级 gather 推进到更低开销的 paged attention / fused gather 形态。
