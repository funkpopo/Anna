# 性能调优参数手册

Anna 在 Intel Arc（XPU）上的性能优化已内置一套默认参数（P0/P1 优化成果）。本文列举所有可调参数：
**默认值已写入代码，无需显式配置即可生效**；每项均保留显式覆盖通道（CLI 参数 > 环境变量 > 代码内置默认）。

覆盖优先级：

```
CLI 显式参数  >  ANNA_* 环境变量  >  代码内置默认值
```

---

## 1. int4 矩阵执行（P0-2：decode 吞吐）

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--xpu-int4-matmul auto\|torch\|dequant\|gemv` | `auto` | int4 dense linear 执行策略。`auto` 按 M 行路由：decode 小 M 走 SYCL GEMV，prefill 大 M 走 aten int4pack XMX GEMM |
| `--xpu-int4-gemv-m-threshold N` | `2` | `auto` 路由阈值：M ≤ N 走 SYCL GEMV。`0` 恢复旧的 auto=int4pack 行为。依据 `bench_logs/xpu_int4_m1_auto_vs_gemv_p0_2.csv`（M=1 gemv 快 1.2–1.9x，M=2 快 1.4x，M≥8 int4pack 胜） |
| env `ANNA_XPU_INT4_MATMUL` | 未设置 | 同 `--xpu-int4-matmul` |
| env `ANNA_XPU_INT4_GEMV_M_THRESHOLD` | 未设置 | 同 `--xpu-int4-gemv-m-threshold` |

代码位置：`src/anna/model/quantization.py`（`_XPU_INT4_GEMV_M_THRESHOLD_DEFAULT = 2`）。

## 2. 启动/加载流水线（P1-5：缩短启动时间）

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--xpu-int4-cache-load-workers N` | `8` | int4 layout cache（`{model}/.anna/xpu_int4_cache`）并行反序列化线程池宽度。原串行加载约 39s；调小可限制小内存机器上的 CPU/RAM 峰值 |
| `--weight-load-pipeline-workers N` | `2` | 权重 shard 流水线并发 staging 线程数：后台线程 mmap 预读下一个 shard，主线程同时向设备拷贝当前 shard（实测 5 shards ~5s，原 ~11s） |
| `--torchinductor-cache-dir PATH` | `~/.anna/cache/torchinductor` | torch.inductor 编译缓存持久目录（serve 启动自动设置）。二次启动命中缓存可省约 75s 的 warmup 编译 |
| env `ANNA_XPU_INT4_CACHE_LOAD_WORKERS` | 未设置 | 同 `--xpu-int4-cache-load-workers` |
| env `ANNA_WEIGHT_LOAD_PIPELINE_WORKERS` | 未设置 | 同 `--weight-load-pipeline-workers` |
| env `TORCHINDUCTOR_CACHE_DIR` | 未设置 | 显式设置时优先生效，`--torchinductor-cache-dir` 不覆盖它。注：torch import 时会自动注入临时目录默认值，Anna 会识别并忽略该注入值（视为未设置），只有用户真实设置的路径才会被尊重 |

代码位置：`src/anna/model/quantization.py`、`src/anna/weights/qwen3_5_text_weight_loader.py`、`src/anna/cli/serve.py`（`configure_compile_cache_environment`）。

## 2b. 调度器（P2-9：单用户 TTFT 与公平性）

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--scheduler-profile throughput` 的 idle-skip | `on`（P2-9 变更） | throughput profile 现也继承 `skip_batch_wait_when_idle`：GPU 空闲时不做 batch 合并等待（实测 idle coalescing 只带来最多 18.5ms 队列等待，无吞吐收益） |
| `--scheduler-event-driven-prefill-insert / --no-scheduler-event-driven-prefill-insert` | profile 默认：throughput=`on`，interactive=off | 事件驱动 prefill 插入：有等待请求时当前 decode 步后立即插入，不再等满 `prefill_interval_steps`（throughput 原为 4 步 ≈ 320ms）。可自定义覆盖：批处理吞吐受 decode 抖动影响时可 `--no-scheduler-event-driven-prefill-insert` 关闭 |

代码位置：`src/anna/runtime/scheduler.py`、`src/anna/runtime/scheduler_profiles.py`。

## 2c. exact prompt cache（P2-7：跨请求 KV 复用）

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--prompt-cache-size N` | **auto**：turboquant KV 开启时 `4`，paged KV 时 `0`（关闭） | 常驻 exact prompt KV cache 条目数（完整 prompt 逐 token 相等才命中）。**TurboQuant 路径下分页 prefix-block 结构上不可用（量化 KV row 不走分页），exact cache 是唯一跨请求复用通道，故默认开启**；paged 路径默认交给 prefix block 共享，避免双驻留 |
| `--prompt-cache-max-tokens N` | **auto**：cache 自动开启时 `2048`，否则 `0`（不限制） | 只缓存不超过 N token 的 prompt，约束 clone 显存开销。缓存长 prompt（如 32k）时请显式调大或调小 |
| `0` 值 | - | `--prompt-cache-size 0` 显式关闭（turboquant 下也生效） |

注：exact cache 与 prefix block 互斥边界由 `prefer_prompt_cache_over_prefix`（`prompt_cache_size > 0` 时自动为 True）控制：exact-cache 命中范围内的 prompt 不再注册 prefix page，避免同一 prompt 双驻留。若你在 **paged KV（未开 turboquant）** 场景下需要共享系统前缀（不同 user 消息、相同 system），保持 prompt cache 关闭即可走 prefix block 复用。

实测（A770，repeated-system 场景）：命中 15/40 查询，TTFT 相应下降（见 `bench_logs/p2_concurrency_a770_20260828.md`）。

代码位置：`src/anna/cli/serve.py`（`_resolve_prompt_cache_defaults`）、`src/anna/runtime/scheduler.py`（continuous batching 下的 cache 查询/登记）、`src/anna/runtime/qwen3_5_text_engine.py`（`lookup_prompt_cache` / `remember_prompt_cache`）。

## 2d. 显存碎片治理（P2-6：空闲自适应释放）

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| env `ANNA_MEMORY_RELEASE_SWEEP_INTERVAL_S` | `30` | 空闲显存巡检周期（秒）。`0` 关闭后台线程 |
| env `ANNA_MEMORY_RELEASE_MIN_FREE_MIB` | `0`（继承安全策略 `--min-free-memory-mib`，默认 1024 MiB） | 空闲且 free 低于该值时触发释放（trim KV 页 + `empty_cache` + 驱逐 prompt cache） |
| env `ANNA_MEMORY_RELEASE_MAX_GAP_MIB` | `1024` | 空闲时 `reserved - allocated` 碎片缺口超过该值即触发释放（实测 A770：空闲 5 分钟 gap 5.33 GiB → 0.92 GiB，回收约 4.4 GiB） |

另外：请求结束后运行时已空闲且 free 低于安全阈值时，`_trim_runtime_cache_if_idle` 会立即做一次自适应释放（不必等巡检）。

代码位置：`src/anna/runtime/memory_release.py`（`AdaptiveMemoryReleaser`）、`src/anna/runtime/qwen3_5_text_engine.py`（`_trim_runtime_cache_if_idle` / `_adaptive_memory_release_*`）。

## 3. 预热形状表（P1-4：消除首请求 TTFT 尖刺）

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--warmup-prefill-tokens N` | 自动推导 | 额外加入预热形状表的 prefill token 数。默认由 scheduler profile / chunk size 推导出 **13 / 64 / 256 / 2048**（覆盖真实聊天 13-token prompt 与 chunked 长 prompt） |
| `--warmup-decode-steps N` | `8` | 每个预热形状的 decode 步数；8 步可覆盖稳态 decode 与 turboquant 解量化路径 |
| `--warmup-batch-size N` | 自动推导 | 额外 batch size。默认覆盖 `{1, 2, 4, 8} ∩ max_batch_size` |
| `--no-inference-warmup` | 关 | 跳过预热（调试用；不推荐服务场景） |

形状表按 scheduler profile / max_batch_size / chunk size 自动推导（`scheduler_profiles.derive_warmup_shape_table`），
启动日志会打印 `Warmup shape table: prefill_lengths=... batch_sizes=... decode_steps=...`。

## 4. 编译模式

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--compile-mode none\|auto\|default\|reduce-overhead\|max-autotune` | `auto` | `auto` 在引擎内解析为 `reduce-overhead`（XPU 上省去逐 kernel dispatch 开销）。配合持久 inductor 缓存，二次启动成本可控；`none` 为逃生口 |
| `--compile-fullgraph` | 关 | compile 时请求 fullgraph 捕获 |
| `--decode-executor auto\|eager\|graph` | `auto` | `auto` 检测到图捕获后端时启用单步 decode 设备图重放；失败自动回退 eager |

## 5. 观测性（默认开启，低开销）

| 机制 | 默认 | 作用 |
| --- | --- | --- |
| decode step 分类计数 | 常开 | metrics 周期日志输出 `Decode step classes`（`pure` / `+prefill_insert` / `+budget_recompute`），用于归因周期性尖刺（P0-3） |
| decode 尖刺 EWMA 告警 | 常开 | 步长超过滚动均值 3x 时输出 `decode step spike ...` 告警并附分类 |
| `--profile-runtime` | 关 | 逐层逐区域 GPU 分段计时（`int4_matmul` / `rmsnorm` / `rotary` / `lm_head` / `turboquant_dequant` / `gated_delta` / `conv` / `attention`）+ `cpu_launch_gap_ms`。**有可测开销，仅诊断用** |

## 6. 环境变量速查（全部可选）

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `ANNA_XPU_INT4_GEMV_M_THRESHOLD` | `2` | auto→GEMV 的 M 行阈值 |
| `ANNA_XPU_INT4_CACHE_LOAD_WORKERS` | `8` | int4 cache 并行加载线程 |
| `ANNA_WEIGHT_LOAD_PIPELINE_WORKERS` | `2` | 权重 shard 流水线并发 |
| `TORCHINDUCTOR_CACHE_DIR` | `~/.anna/cache/torchinductor`（serve 自动设） | inductor 编译缓存目录 |
| `ANNA_XPU_INT4_MATMUL` | `auto` | int4 执行策略 |
| `ANNA_XPU_INT4_CACHE_DIR` | `{model}/.anna/xpu_int4_cache` | int4 layout cache 位置 |
| `ANNA_XPU_DISABLE_INT4_CACHE` | 未设置 | `1` 关闭 int4 layout cache |
| `ANNA_GATED_DELTA_OP_LIB` | 自动探测 `.build/` | fused op 库路径 |
| `ANNA_MEMORY_RELEASE_SWEEP_INTERVAL_S` | `30` | 空闲显存巡检周期（0 关闭） |
| `ANNA_MEMORY_RELEASE_MIN_FREE_MIB` | 继承 `--min-free-memory-mib` | 空闲释放的 free 下限 |
| `ANNA_MEMORY_RELEASE_MAX_GAP_MIB` | `1024` | 空闲 reserved-allocated 缺口触发阈值（0 关闭） |

## 7. 编程接口（ServeSettings）

`anna.core.config.ServeSettings` 暴露同名字段（默认 `None` = 使用内置默认）：

```python
ServeSettings(
    ...,
    xpu_int4_gemv_m_threshold=2,        # None → 内置默认 2
    xpu_int4_cache_load_workers=8,      # None → 内置默认 8
    weight_load_pipeline_workers=2,     # None → 内置默认 2
    torchinductor_cache_dir=Path(...),  # None → ~/.anna/cache/torchinductor
    scheduler_event_driven_prefill_insert=False,  # throughput profile 解析后默认 True
    prompt_cache_size=4,                # turboquant KV 下 serve 自动默认 4（0 = 关闭）
    prompt_cache_max_tokens=2048,       # cache 自动开启时默认 2048
)
```

显式赋值总是覆盖环境变量与内置默认。

## 8. 验收基线

- gemv vs int4pack 路由依据：`bench_logs/xpu_int4_m1_auto_vs_gemv_p0_2.csv`
- decode 分段占比（A770，profiled eager）：`bench_logs/decode_phase_breakdown_a770_p0_1_generate.log`
  （int4_matmul 47% / gated_delta 18% / attention 14% / rmsnorm 12%）
- P2 并发基线 / prompt cache 证据：`bench_logs/p2_concurrency_a770_20260828.md`
- P2 TurboQuant A/B：`bench_logs/turboquant_ab_p2_8.json`
- 当前基线索引：`bench_logs/CURRENT_BASELINE.md`
