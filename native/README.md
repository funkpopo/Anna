# Anna Native Migration

This directory is the Zig migration trunk for Anna's control plane.

Current scope:

- full native serve-CLI option parsing
- model-path resolution and model-family inspection
- artifact manifest scanning for local model directories
- OpenAI response/error payload encoding
- service metrics accounting and interval formatting
- incremental text assembly and stop-string handling
- native batch scheduler skeleton for same-length prompt batching
- sampling primitives for repetition penalty / top-k / top-p

Intentional non-goals of this first rewrite chunk:

- no Python fallback bridge
- no minimal mock-only route layer
- no half-hidden subprocess delegation

The remaining heavy migration work is the model runtime itself:

- safetensors / GGUF tensor loading
- tokenizer runtime
- Qwen3.5 / Gemma4 / Qwen3-TTS kernels
- XPU memory management and fused ops integration
