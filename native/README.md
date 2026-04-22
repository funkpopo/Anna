# Anna Native Migration

This directory is the Zig migration trunk for Anna's control plane.

Current scope:

- full native serve-CLI option parsing
- model-path resolution and model-family inspection
- artifact manifest scanning for local model directories
- safetensors tensor-metadata parsing and typed tensor views
- safetensors multi-shard index parsing and shard resolution
- Qwen text-config parsing for hybrid linear/full attention stacks
- native dense/AutoRound-int4 linear kernels
- native Qwen full-attention decode, gated-delta linear-attention decode, MLP/MoE blocks
- native token-engine state machine for decoder-only inference
- native Qwen3.5 weight binding from real safetensors shards
- token-level end-to-end runtime loading via `anna-native eval-token`
- native Qwen ByteLevel-BPE tokenizer runtime with chat-template rendering
- scheduler-backed text generation via `anna-native generate`
- OpenAI-compatible chat/completion JSON payloads via `anna-native chat-json` and `anna-native completion-json`
- explicit `--backend xpu` path backed by Zig-loaded native SYCL kernels
- device-side MLP chaining with buffer reuse and fused SiLU-gate multiply on the SYCL path
- device-side Q/K RMSNorm + RoPE postprocessing on the SYCL full-attention path
- OpenAI response/error payload encoding
- service metrics accounting and interval formatting
- incremental text assembly and stop-string handling
- native batch scheduler skeleton for same-length prompt batching
- sampling primitives for repetition penalty / top-k / top-p

Intentional non-goals of this first rewrite chunk:

- no Python fallback bridge
- no minimal mock-only route layer
- no half-hidden subprocess delegation

SYCL XPU backend:

- `zig build xpu-backend` builds `zig-out/bin/anna-xpu-backend.dll` from `native/xpu/sycl_backend.cpp`
- set `ANNA_SYCL_CXX` to an Intel oneAPI `dpcpp` / `icpx` executable if it is not on `PATH`
- `--backend xpu` requires the SYCL DLL and fails XPU initialization when it is missing

The remaining heavy migration work is the model runtime itself:

- GGUF tensor loading
- HTTP serving loop and request JSON parsing
- Gemma4 / Qwen3-TTS kernels
- deeper XPU memory management and fused-kernel coverage beyond current projection/MLP/attention-postprocess kernels
