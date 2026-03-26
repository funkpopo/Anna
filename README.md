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
  - `enable_thinking` 开关
  - `reasoning_content` 输出
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
- 运行前会按请求规模预估显存占用；显存预算不足时会提前拒绝请求，避免把 XPU 顶死。
- 运行期如果遇到 `OOM / device lost / out of resources`，会尝试清理 runtime cache 并返回可控错误。

## 当前限制

- 已在 `conda env anna` 中安装 `torch 2.11.0+xpu`，并完成 `pytest`、真实模型加载、文本生成、图片生成、视频生成与 API 烟测。
- 尚未在 Intel Arc A770 上完成系统化性能与稳定性验证。
- FP8 / AWQ / AWQ-4bit 虽已接入装载路径，但仍需真实官方模型做数值校验。
- 当前默认单请求、单 batch，尚未做连续批处理与调度层。

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

如果要显式指定对外暴露的模型名称：

```bash
anna-serve --model-dir D:\Projects\Anna\models\Qwen\Qwen3___5-2B --model-name qwen3.5 --device xpu --dtype bf16
```

如果不指定 `--model-name`，默认模型名就是 `--model-dir` 的完整路径。

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
  "enable_thinking": true,
  "max_completion_tokens": 256
}
```

当 `enable_thinking=true` 时，返回体中的 `message` 和流式 `delta` 会包含独立的 `reasoning_content` 字段；当 `enable_thinking=false` 时，会通过 Qwen3.5 的 closed-think prompt 关闭思维模式，仅返回最终内容。

## 下一步

- 使用官方 FP8 / AWQ 变体校验量化权重布局。
- 在 Arc A770 上做 profiling，定位热点算子，再决定是否补写自定义 XPU 算子。
