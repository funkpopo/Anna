# Anna → Intel XPU 推理引擎转型分析

> 评估日期：2026-08-28
> 评估硬件：i5-12400（6C/12T，无 AVX-512 VNNI 之外的加速能力一般）、32GB DDR4、**Intel Arc A770 16GB（ACM-G10）**，驱动 32.0.101.8974。

---

## 一、项目现状摘要

Anna 目前是一个 **PyTorch-based** 的本地推理运行时，XPU 方向已做了大量工作：

- 自研 SYCL 融合算子（Gated Delta decode/prefill、GQA decode split-kv、paged GQA decode、MoE router/dispatch、RMSNorm、int4 GEMV、LM head int4 top-k 等），通过 DPC++ 手工构建 `.pyd` 加载；
- Paged KV cache + PrefixBlockPool、TurboQuant KV 量化、XPU int4 权重（aten int4pack 为主，自研 GEMV 为辅）+ int4 布局磁盘缓存；
- 连续批处理调度器（token budget、动态预算、公平性抢占）、OpenAI 兼容 API、benchmark/profile 工具链；
- Rust（PyO3）承担 safetensors manifest 解析与 CPU 侧 int4 量化。

**结论：XPU 化的"面"已经铺开，但距离一个真正的"XPU 推理引擎"还有代际差。当前形态是"PyTorch eager + 若干 SYCL 补丁"，核心执行路径的控制权仍在 PyTorch dispatcher 手里。**

---

## 二、当前硬件环境的约束（分析前提）

| 硬件 | 规格 | 对引擎设计的影响 |
| --- | --- | --- |
| Arc A770 16GB | ACM-G10，512 EU，XMX bf16/int8，16GB VRAM | bf16 XMX 算力强但 **fp32 吞吐差**；显存够跑 4B int4 + 中等 batch；驱动仍有 device-lost/OOM 稳定性问题（项目已做恢复逻辑，说明踩过坑） |
| i5-12400 | 6C/12T，DDR4 | **CPU 弱是最大约束**：Python 调度、kernel 提交、host sync 全部落在 6 个核上，host overhead 无法被 CPU 并行掩盖 |
| 32GB DDR4 | ~50GB/s 双通道 | MoE expert offload、CPU 预处理（vision tower offload）都受此带宽限制 |
| PCIe 4.0 | 单 GPU | H2D/D2H 每步往返代价高 |

**核心矛盾：在 6 核 CPU + A770 的组合上，"每 decode step 一次 host 往返 + 几百次小 kernel 提交 + Python 调度"的架构，GPU 利用率的上限由 host 决定，而不是由 XMX 决定。** 这正是项目必须"往 XPU 推理引擎方向转变"的根本原因。

---

## 三、不足与不合理之处（按严重程度）

### P0-1 decode 路径存在大量逐层 host sync（最严重）

`Qwen3DynamicCache._update_visible_layer_cache`（`src/anna/model/ops.py:596`）在**每一层**的 cache update 中执行：

- `past_lengths.max().item()` → 每层一次 D2H 同步；
- `past_lengths.tolist()` → 每层一次 D2H 同步 + Python 循环逐行 `copy_`。

36 层模型 × 每 decode step = **~72 次 host-device 同步往返/步**。i5-12400 上每次同步 20–50µs，仅此一项就可能吃掉 decode step 预算的大半。`_sync_page_table_layer_buffer`（paged 路径）同理每步触发。

此外 engine 中仍有每步 `.item()`（stop-token 检查，`_is_stop_token_device`，注释宣称避免 host sync 但实际每个控制流分支仍同步）、每 token 一次 D2H + JSON 序列化的流式路径。

### P0-2 没有 decode step 的图化/命令缓冲执行

- 所有非融合算子走 eager dispatch，每步几百个 kernel 各自提交 Level Zero；
- `torch.compile`（serve 默认 `auto`→reduce-overhead）在 XPU 上被注释明确指出 **Inductor dynamic_shapes 会拉入 fp64 Triton kernel（XPU 不支持）**——即 Inductor 路径不成熟，不能作为主执行路径；
- 项目没有任何 XPU graph capture / Level Zero command-list 复用机制。decode 是天然可图化的静态形状场景，这一层完全缺失。

### P1-3 带 cache 的 prefill 没有 flash attention 路径

- `F.scaled_dot_product_attention` 仅在 `past_key_values is None` 时启用（`ops.py:2023`）；
- 一旦有 cache（chunked prefill、连续批处理下必然如此），走 materialized `grouped_query_attention`：物化 `[B,H,L,S]` 分数矩阵 + **fp32 softmax**。长 prompt（4k+）在 16GB A770 上既爆显存又爆带宽；
- FlashQLA 只覆盖 GDN prefill，full-attention prefill 无 kernel。**这是当前架构里最大的单点性能空洞。**

### P1-4 int4 主路径依赖 `aten._weight_int4pack_mm`

- 该算子不是为 Arc 优化的通道；qscale 为 fp32 `[group, out]`，对 decode GEMV 不友好；
- 自研 SYCL int4 GEMV 存在但 `auto` 永不选择（README 明说 "not selected by auto"），等于自研 kernel 没有进入主路径；
- 没有利用 A770 的 int8 XMX（dp4a/DPAS）做 int8×int8/int4 反量化融合 GEMM。

### P1-5 KV cache 生命周期管理在 Python 侧

- `stack/split_batch/compact_batch_rows/clone` 在 batch 成员变化时于 Python 中搬运/复制整块 KV（README troubleshooting 自己点名 "High cache stack/split/compact cost"）；
- 调度器 `_run_loop` 每步对全部 active requests 做 O(n) Python 对象扫描与分支判断；
- 在 6 核 CPU 上，这套 Python 管理层本身就是延迟项。

### P2-6 硬件抽象只靠设备名字符串

- `XPUDeviceInfo` 仅识别 `arc/a770/a750/acm-g10` 子串；decode 策略表 bake 死 A770 形状；
- Battlemage（B580）、Lunar Lake iGPU、未来 DG* 全部会走 fallback 分支。作为"XPU 推理引擎"，缺少 **capability 抽象**（EU 数、XMX 支持、subgroup 尺寸、L0/驱动版本、显存带宽等级）。

### P2-7 音频链路是"上游黑盒 wrapper"

- `qwen-tts==0.1.1`（钉死）是外部黑盒，内部 kernel 不受控、无法 XPU 化，README 也明确 "Not planned"（`qwen-asr` 依赖已随 ASR 支持移除而删除）；
- 与"打造 XPU 推理引擎"的目标矛盾：文本栈精心优化，音频栈完全不可控；且 TTS 与 text 共存时靠 `DeviceExecutionGate` 全局串行，A770 16GB（4B int4 ≈ 3GB + TTS）其实装得下并发。（ASR 支持已移除。）

### P2-8 其他不合理

- ~~`turboquant>=0.2,<0.3` 为硬依赖但又是可选功能（装不上时 auto 静默降级），依赖声明自相矛盾~~（已修复：移至 `anna[quant]` optional-dependencies）；
- Rust crate 只做 manifest/量化，host 侧最热的地方（tokenizer、调度、token 流水）仍在 Python；
- DPC++ 构建链要求手工 PowerShell 环境变量 + 手工 `.pyd` 路径，无法 CI 化；
- `todo.md` 为空文件，路线图缺失。

---

## 四、转型方案：从 "PyTorch 运行时" 到 "XPU 推理引擎"

目标形态：**host 侧轻量（Rust/最小 Python），device 侧全图化（SYCL + L0 graph），调度与 kernel 由 capability 层驱动，PyTorch 仅作为权重加载与算子后端之一。**

### Phase 0 — 能力抽象层与可测量基线（1–2 周）

1. `runtime/capability.py`：按驱动/设备枚举产出 `DeviceCapability`（arch 代际、EU、subgroup、XMX、bandwidth tier），替换全部字符串匹配；decode 策略表由 capability 生成，A770 数值作为 G10 档位的特例保留。
2. 在 `anna-bench` 中固化三个金标准场景（decode bs=1/4/8、prefill 2k/8k、mixed），落盘 JSON 基线到 `bench_logs/`，后续每个 Phase 用同一把尺子验收。

### Phase 1 — 消灭 decode 路径 host sync（收益最大、风险最低）

1. `Qwen3DynamicCache` 全部长度信息常驻 device tensor；`max().item()/tolist()` 改为 device 端 `cummax`/页表索引，`_update_visible_layer_cache` 的逐行 Python `copy_` 换成一次 device 端 scatter（配合 P1-4 的 fused cache-write kernel）。
2. stop 检查全 batch 化、on-device： EOS 位图与"序列结束标志"维护在 device，host 只在每个 scheduler tick 拉一次聚合向量（每 N 步一次 D2H，而非每 token）。
3. 流式输出 token 批量搬运（`token_ids_to_host` 改为环形缓冲 + 事件驱动批量拷回）。
4. 验收：用 `torch.xpu` 的 profiling 统计每 decode step 的 D2H 同步次数，目标 **≤1 次/步**。

### Phase 2 — decode step 图化（引擎化核心）

1. 引入 SYCL graph / Level Zero command-list 扩展（或先以 `torch.xpu` 上的 capture 等价物验证），把单步 decode 的 kernel 序列捕获为可重放图：shape 持平（bs、seq=1 固定），页表/长度通过 device buffer 传参而非重建图。
2. 对无法图化的片段（采样分支、MoE dispatch）先做"子图 + 提交链"，逐步扩大图覆盖率。
3. 保留 eager 路径作为 fallback（`--decode-executor eager|graph`），由 `ops_parity` 同样的机制做数值一致性回归（复用 `gemma4_ops_parity` 思路）。
4. 验收：bs=1 decode step 时间对比 eager，目标 ≥30% 降低（i5-12400 上 host-bound 场景收益更大）。

### Phase 3 — 补齐 prefill flash attention 与 int4 XMX 路径

1. **Full-attention chunked prefill kernel**：在现有 FlashQLA/SYCL 框架内新增 paged KV 的 causal flash prefill（online softmax，bf16 累积），替换 materialized grouped path；fp32 softmax 仅保留在数值容差兜底路径。
2. **int4 GEMM 走 XMX**：新增 dp4a/DPAS int4→int8 路径 kernel，`auto` 策略改为：prefill（M 大）用 XMX 批量 GEMM，decode（M 小）用自研 GEMV，并把 `gemv` 正式纳入 auto 决策（阈值由 capability 表给出）。
3. 验收：prefill 4k prompt 的 TTFT 与峰值显存对比；int4 prefill/decode 吞吐对比 aten int4pack。

### Phase 4 — host 热路径下沉 Rust + KV 管理 device 化

1. 调度器 tick、token 流水、batch 成员管理移入 `crates/anna-rust`（已有 PyO3 通道），Python 只保留 API/配置层；KV stack/split/compact 改为 device 端 kernel + Rust 索引管理，消灭 Python 大块拷贝。
2. MoE expert offload：pinned memory + 双缓冲异步预取流水（DDR4 带宽打满），expert 缓存改 LRU 并按 router 命中率预取。

### Phase 5 — 多模型并发与音频栈收编（可选、长线）

1. VRAM 分区 + per-stream 并发替代全局 `DeviceExecutionGate`（text 4B-int4 + TTS 同驻 A770 完全可行）；
2. 评估将 TTS 的 encoder/vocoder 用同一 SYCL kernel 框架收编，或以 OpenVINO GenAI 作为音频后端，摆脱 `qwen-tts==0.1.1` 黑盒钉死。（ASR 支持已于 2026-08-28 移除：上游 `qwen-asr` 依赖维护停滞。）

### 依赖与工程治理（并行进行）

- [x] `turboquant` 移到 optional-dependencies（`anna[quant]`），核心路径不强依赖第三方黑盒；
- DPC++ 构建接 CMake + `tools/build_*.py` 统一，产出符号化命名的库并支持 CI（GitHub runner 无 Intel GPU 时跑 CPU parity 测试，XPU gate 用 self-runner）；
- `todo.md` 按本路线图填充。

---

## 五、风险与兼容性

- 图化执行与 TurboQuant/paged cache 的组合需要专门的 parity 回归（形状动态的部分先留在子图外）；
- A770 驱动 device-lost 场景下，graph 需要可重建（保留 eager fallback + 现有 runtime_health 恢复机制）；
- Phase 1/2 与现有调度 profile（interactive/throughput）语义保持不变，API 层零改动。

## 六、验收指标（同机 i5-12400 + A770 16GB）

| 指标 | 现状基线 | Phase 1 目标 | Phase 2/3 目标 |
| --- | --- | --- | --- |
| 每 decode step host 同步次数 | ~2×层数 + 每步若干 | ≤1 | ≤1 |
| decode step 时间（bs=1, 4B int4） | 待基线 | -10~20% | -30% |
| TTFT（4k prompt prefill） | materialized path | -15%（去 fp32 softmax） | -40%（flash prefill） |
| 单步 Python 调度耗时占比 | 待 profile | -50%（Rust 下沉后） | — |
