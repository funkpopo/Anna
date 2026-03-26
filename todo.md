# Anna TODO

目标：从 0 实现一个可在 Intel Arc A770 上运行 Qwen3.5 架构模型、并暴露 OpenAI 兼容 API 的推理服务项目。

约束：
- 不引用现成推理项目代码，不以 `ipex-llm` / `vLLM` / `llama.cpp` / `exllama` 等项目作为代码基础。
- 允许使用通用基础库，例如 Python、PyTorch、safetensors、FastAPI、tokenizers；模型执行逻辑、缓存、采样、服务协议和权重映射由本项目自行实现。
- 用户手动准备模型目录，项目内不提供模型下载逻辑。
- 后端优先走 `XPU`，尽量避免 CPU 推理，仅允许 CPU 进行必要的图片/视频预处理。

## Phase 0: 范围与架构
- [x] 目标聚焦为 Qwen3.5 原生架构
  - `Qwen3_5ForConditionalGeneration`
  - `linear_attention + full_attention` 混合文本层
  - 原生多模态 vision tower
- [x] 确认基础技术路线
  - Python 3.11+
  - PyTorch + `xpu`
  - FastAPI + Uvicorn
  - safetensors + tokenizers
- [x] 明确模型输入方式
  - 用户自行手动下载模型文件
  - 本项目只消费本地 Hugging Face 风格模型目录

## Phase 1: 项目骨架
- [x] 创建 `pyproject.toml`
- [x] 创建基础目录
  - `src/anna/api`
  - `src/anna/core`
  - `src/anna/mm`
  - `src/anna/model`
  - `src/anna/runtime`
  - `src/anna/sampling`
  - `src/anna/weights`
  - `src/anna/cli`
  - `tests`
- [x] 建立日志、配置、启动入口

## Phase 2: OpenAI 兼容 API
- [x] 实现 `/v1/models`
- [x] 实现 `/v1/chat/completions`
- [x] 实现 `/v1/completions`
- [x] 统一请求/响应 schema
- [x] SSE 流式输出
- [x] OpenAI 风格错误响应
- [x] 支持聊天内容中的 `text` / `image_url` / `video_url`

## Phase 3: 配置、分词与权重加载
- [x] 解析 `config.json`
- [x] 解析 `preprocessor_config.json`
- [x] 加载 `tokenizer.json` / `tokenizer_config.json`
- [x] 加载 safetensors 单文件与分片权重
- [x] 支持原生 vision/text 权重键
- [x] 支持从官方 `quantization_config` 解析量化元数据

## Phase 4: Qwen3.5 文本与多模态执行链路
- [x] 实现基础张量模块
  - `Embedding`
  - `RMSNorm`
  - `RotaryEmbedding`
  - `MLP`
- [x] 实现文本层
  - full attention
  - linear attention / gated delta rule
  - prefilling + decoding 两阶段执行
- [x] 实现 vision tower
  - patch embedding
  - vision transformer blocks
  - patch merger
- [x] 实现 native multimodal prompt 处理
  - image/video placeholder 扩展
  - `mm_token_type_ids`
  - 3D RoPE position id 计算

## Phase 5: Arc A770 / XPU 后端
- [x] 抽象 `DeviceContext`
- [x] 支持 `cpu` / `xpu` 设备解析
- [x] 支持 BF16 / FP16 计算 dtype
- [x] 增加 FP8 dtype 别名与量化模型装载路径
- [x] 补充张量迁移策略
  - CPU 负责媒体预处理
  - prefill 前一次性迁移文本/媒体张量到执行设备
  - decode 阶段仅发送新 token，媒体张量不重复传输
- [x] 补充缓存迁移策略
  - KV cache / recurrent state / conv state 归属请求级缓存对象
  - cache 常驻执行设备，必要时通过 `DeviceContext.move_cache()` 迁移
- [x] Arc A770 真机验证 `torch.xpu` 路径
- [x] 提供 `anna-bench` 基准入口用于测量延迟与 tokens/s
- [ ] 增加基础 profiling
- [ ] 定位 Arc 上的瓶颈算子
- [ ] 如有必要，补写自定义融合算子或权重预打包逻辑

## Phase 6: 生成与采样
- [x] greedy
- [x] temperature
- [x] top-k
- [x] top-p
- [x] repetition penalty
- [x] 停止词与最大长度控制

## Phase 7: 量化格式
- [x] Dense 基线路径
- [x] FP8 权重模块替换与装载接口
- [x] AWQ / AWQ-4bit 权重模块替换与装载接口
- [ ] 使用真实官方 FP8 权重做端到端前向验证
- [ ] 使用真实官方 AWQ / AWQ-4bit 权重做端到端前向验证
- [ ] 针对 Arc A770 调整量化 kernel 和数据布局

## Phase 8: 端到端联调
- [x] CLI 文本生成功能
- [x] API 调用驱动生成
- [x] 补充多模态 / 配置 / 量化静态测试
- [x] 单 batch 正确性验证
- [x] Arc A770 设备探测与启动验证
- [x] 图片输入端到端验证
- [x] 视频输入端到端验证
- [ ] 输出首个可复现运行说明

## 当前优先级
1. 用官方 FP8/AWQ 变体校验量化权重布局与数值正确性。
2. 在 Arc A770 上做基础 profiling，定位热点算子。
3. 视 profiling 结果决定是否手写自定义 XPU 融合算子。
4. 输出首个可复现运行说明与基准结果。

## 当前限制
- [x] 当前环境已在 `conda env anna` 中安装 `torch 2.11.0+xpu` 与项目依赖，可执行真实前向与 pytest。
- [ ] FP8 与 AWQ 当前为项目内自实现装载/反量化路径，尚未完成真实模型数值校验。
- [ ] 尚未做连续批处理与并发调度。
- [ ] 尚未加入自定义 XPU 融合算子或 kernel。
