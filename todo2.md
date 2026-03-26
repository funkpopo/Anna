实现方式我建议优先级这样排：

第一选择：PyTorch custom SYCL op + oneDNN matmul primitive
第二选择：PyTorch custom SYCL op + oneDNN graph / post-op fusion
第三选择：自己写更底层的 XeTLA / joint_matrix

第一优先级：继续把 text path 的 dense BF16 做成更像大 GEMM，先把 XMX 利用率拉起来。
第二优先级：做 AWQ/INT4 weight-only custom op。这是 Arc A770 上最现实的量化加速路径。
第三优先级：把 FP8 做成“格式兼容 + capability gate”。在 A770 上不要把它当主加速方案。

把 AWQ INT4 路径继续扩到 bias/binary/activation 的更深层融合，以及把 repack 提前到权重加载阶段而不是首次算子调用阶段