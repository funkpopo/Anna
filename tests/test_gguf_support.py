from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader, GGUFValueType, GGUFWriter, TokenType
from gguf.quants import dequantize

from anna.model.quantization import XPUInt4Linear
from anna.weights.qwen3_5_text_tokenizer import Qwen3_5TextTokenizer
from anna.weights.qwen3_5_text_weight_loader import (
    build_qwen3_5_text_model,
    estimate_qwen3_5_text_model_weight_bytes,
    load_qwen3_5_text_model_config,
    load_qwen3_5_text_model_weights,
)


def _write_tensor(writer: GGUFWriter, name: str, array: np.ndarray) -> None:
    if array.ndim <= 1:
        raw = np.ascontiguousarray(array)
    else:
        raw = np.ascontiguousarray(np.transpose(array, axes=tuple(reversed(range(array.ndim)))))
    writer.add_tensor(name, raw, raw_shape=array.shape)


def _finalize_writer(writer: GGUFWriter, path: Path) -> None:
    writer.write_header_to_file(path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _read_expected_tensors(path: Path) -> dict[str, np.ndarray]:
    reader = GGUFReader(str(path))
    return {
        tensor.name: np.asarray(dequantize(tensor.data, tensor.tensor_type)).copy()
        for tensor in reader.tensors
    }


def _build_test_main_gguf(model_dir: Path) -> dict[str, np.ndarray]:
    path = model_dir / "toy-qwen.gguf"
    writer = GGUFWriter(path, arch="qwen35moe")

    tokens = [
        "a",
        "b",
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<tool_call>",
        "</tool_call>",
    ]
    token_types = [
        int(TokenType.NORMAL),
        int(TokenType.NORMAL),
        int(TokenType.CONTROL),
        int(TokenType.CONTROL),
        int(TokenType.CONTROL),
        int(TokenType.CONTROL),
        int(TokenType.CONTROL),
        int(TokenType.CONTROL),
        int(TokenType.CONTROL),
        int(TokenType.CONTROL),
        int(TokenType.CONTROL),
        int(TokenType.CONTROL),
    ]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")
    writer.add_token_list(tokens)
    writer.add_token_merges([])
    writer.add_token_types(token_types)
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(4)
    writer.add_chat_template("{{ messages }}")
    writer.add_key_value("tokenizer.ggml.padding_token_id", 7, GGUFValueType.INT32)

    writer.add_key_value("qwen35moe.embedding_length", 8, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.block_count", 2, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.context_length", 32, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.full_attention_interval", 2, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.attention.head_count", 2, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.attention.head_count_kv", 1, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.attention.key_length", 2, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6, GGUFValueType.FLOAT32)
    writer.add_key_value("qwen35moe.ssm.group_count", 2, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.ssm.conv_kernel", 2, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.rope.dimension_count", 1, GGUFValueType.UINT32)
    writer.add_key_value(
        "qwen35moe.rope.dimension_sections",
        [1, 1, 0],
        GGUFValueType.ARRAY,
        GGUFValueType.UINT32,
    )
    writer.add_key_value("qwen35moe.rope.freq_base", 10000.0, GGUFValueType.FLOAT32)
    writer.add_key_value("qwen35moe.expert_feed_forward_length", 4, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.expert_shared_feed_forward_length", 4, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.expert_count", 2, GGUFValueType.UINT32)
    writer.add_key_value("qwen35moe.expert_used_count", 1, GGUFValueType.UINT32)

    offset = 1.0

    def arr(shape: tuple[int, ...], *, scale: float = 0.1) -> np.ndarray:
        nonlocal offset
        size = int(np.prod(shape))
        values = (np.arange(size, dtype=np.float32) + offset).reshape(shape) * scale
        offset += size
        return values

    tensors_to_write: dict[str, np.ndarray] = {}
    tensors_to_write["token_embd.weight"] = arr((len(tokens), 8))
    tensors_to_write["output.weight"] = tensors_to_write["token_embd.weight"].copy()
    tensors_to_write["output_norm.weight"] = arr((8,))

    tensors_to_write["blk.0.attn_norm.weight"] = arr((8,))
    tensors_to_write["blk.0.post_attention_norm.weight"] = arr((8,))
    tensors_to_write["blk.0.attn_qkv.weight"] = arr((16, 8))
    tensors_to_write["blk.0.attn_gate.weight"] = arr((8, 8))
    tensors_to_write["blk.0.ssm_alpha.weight"] = arr((4, 8))
    tensors_to_write["blk.0.ssm_beta.weight"] = arr((4, 8))
    tensors_to_write["blk.0.ssm_out.weight"] = arr((8, 8))
    tensors_to_write["blk.0.ssm_conv1d.weight"] = arr((16, 2))
    tensors_to_write["blk.0.ssm_dt.bias"] = arr((4,))
    tensors_to_write["blk.0.ssm_a"] = -np.linspace(0.5, 2.0, 4, dtype=np.float32)
    tensors_to_write["blk.0.ssm_norm.weight"] = arr((2,))
    tensors_to_write["blk.0.ffn_gate_inp.weight"] = arr((2, 8))
    tensors_to_write["blk.0.ffn_gate_inp_shexp.weight"] = arr((8,))
    tensors_to_write["blk.0.ffn_gate_shexp.weight"] = arr((4, 8))
    tensors_to_write["blk.0.ffn_up_shexp.weight"] = arr((4, 8))
    tensors_to_write["blk.0.ffn_down_shexp.weight"] = arr((8, 4))
    tensors_to_write["blk.0.ffn_gate_exps.weight"] = arr((2, 4, 8))
    tensors_to_write["blk.0.ffn_up_exps.weight"] = arr((2, 4, 8))
    tensors_to_write["blk.0.ffn_down_exps.weight"] = arr((2, 8, 4))

    tensors_to_write["blk.1.attn_norm.weight"] = arr((8,))
    tensors_to_write["blk.1.post_attention_norm.weight"] = arr((8,))
    tensors_to_write["blk.1.attn_q.weight"] = arr((8, 8))
    tensors_to_write["blk.1.attn_k.weight"] = arr((2, 8))
    tensors_to_write["blk.1.attn_v.weight"] = arr((2, 8))
    tensors_to_write["blk.1.attn_output.weight"] = arr((8, 4))
    tensors_to_write["blk.1.attn_q_norm.weight"] = arr((2,))
    tensors_to_write["blk.1.attn_k_norm.weight"] = arr((2,))
    tensors_to_write["blk.1.ffn_gate_inp.weight"] = arr((2, 8))
    tensors_to_write["blk.1.ffn_gate_inp_shexp.weight"] = arr((8,))
    tensors_to_write["blk.1.ffn_gate_shexp.weight"] = arr((4, 8))
    tensors_to_write["blk.1.ffn_up_shexp.weight"] = arr((4, 8))
    tensors_to_write["blk.1.ffn_down_shexp.weight"] = arr((8, 4))
    tensors_to_write["blk.1.ffn_gate_exps.weight"] = arr((2, 4, 8))
    tensors_to_write["blk.1.ffn_up_exps.weight"] = arr((2, 4, 8))
    tensors_to_write["blk.1.ffn_down_exps.weight"] = arr((2, 8, 4))

    for name, array in tensors_to_write.items():
        _write_tensor(writer, name, array)

    _finalize_writer(writer, path)
    return _read_expected_tensors(path)


def _build_test_mmproj_gguf(model_dir: Path) -> dict[str, np.ndarray]:
    path = model_dir / "mmproj-F32.gguf"
    writer = GGUFWriter(path, arch="clip")
    writer.add_key_value("clip.has_vision_encoder", True, GGUFValueType.BOOL)
    writer.add_key_value("clip.projector_type", "qwen3vl_merger", GGUFValueType.STRING)
    writer.add_key_value("clip.use_gelu", True, GGUFValueType.BOOL)
    writer.add_key_value("clip.vision.block_count", 1, GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.embedding_length", 4, GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.feed_forward_length", 8, GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.attention.head_count", 2, GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.patch_size", 1, GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.image_size", 2, GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.spatial_merge_size", 1, GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.projection_dim", 8, GGUFValueType.UINT32)
    writer.add_key_value("clip.vision.image_mean", [0.5, 0.5, 0.5], GGUFValueType.ARRAY, GGUFValueType.FLOAT32)
    writer.add_key_value("clip.vision.image_std", [0.5, 0.5, 0.5], GGUFValueType.ARRAY, GGUFValueType.FLOAT32)

    offset = 1000.0

    def arr(shape: tuple[int, ...], *, scale: float = 0.05) -> np.ndarray:
        nonlocal offset
        size = int(np.prod(shape))
        values = (np.arange(size, dtype=np.float32) + offset).reshape(shape) * scale
        offset += size
        return values

    tensors_to_write: dict[str, np.ndarray] = {}
    tensors_to_write["v.patch_embd.weight"] = arr((4, 3, 1, 1))
    tensors_to_write["v.patch_embd.weight.1"] = arr((4, 3, 1, 1))
    tensors_to_write["v.patch_embd.bias"] = arr((4,))
    tensors_to_write["v.position_embd.weight"] = arr((4, 4))
    tensors_to_write["v.blk.0.ln1.weight"] = arr((4,))
    tensors_to_write["v.blk.0.ln1.bias"] = arr((4,))
    tensors_to_write["v.blk.0.ln2.weight"] = arr((4,))
    tensors_to_write["v.blk.0.ln2.bias"] = arr((4,))
    tensors_to_write["v.blk.0.attn_qkv.weight"] = arr((12, 4))
    tensors_to_write["v.blk.0.attn_qkv.bias"] = arr((12,))
    tensors_to_write["v.blk.0.attn_out.weight"] = arr((4, 4))
    tensors_to_write["v.blk.0.attn_out.bias"] = arr((4,))
    tensors_to_write["v.blk.0.ffn_up.weight"] = arr((8, 4))
    tensors_to_write["v.blk.0.ffn_up.bias"] = arr((8,))
    tensors_to_write["v.blk.0.ffn_down.weight"] = arr((4, 8))
    tensors_to_write["v.blk.0.ffn_down.bias"] = arr((4,))
    tensors_to_write["v.post_ln.weight"] = arr((4,))
    tensors_to_write["v.post_ln.bias"] = arr((4,))
    tensors_to_write["mm.0.weight"] = arr((4, 4))
    tensors_to_write["mm.0.bias"] = arr((4,))
    tensors_to_write["mm.2.weight"] = arr((8, 4))
    tensors_to_write["mm.2.bias"] = arr((8,))

    for name, array in tensors_to_write.items():
        _write_tensor(writer, name, array)

    _finalize_writer(writer, path)
    return _read_expected_tensors(path)


def _build_test_gguf_model_dir(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    model_dir = tmp_path / "toy-gguf-model"
    model_dir.mkdir()
    main_arrays = _build_test_main_gguf(model_dir)
    mmproj_arrays = _build_test_mmproj_gguf(model_dir)
    return model_dir, main_arrays, mmproj_arrays


def test_load_qwen_config_and_tokenizer_from_gguf(tmp_path: Path) -> None:
    model_dir, _, _ = _build_test_gguf_model_dir(tmp_path)

    config = load_qwen3_5_text_model_config(model_dir)
    assert config.text_config.hidden_size == 8
    assert config.text_config.num_hidden_layers == 2
    assert config.text_config.layer_types == ["linear_attention", "full_attention"]
    assert config.text_config.linear_num_key_heads == 2
    assert config.text_config.linear_num_value_heads == 4
    assert config.text_config.linear_key_head_dim == 2
    assert config.text_config.linear_value_head_dim == 2
    assert config.text_config.num_experts == 2
    assert config.vision_config is not None
    assert config.vision_config.hidden_size == 4
    assert config.vision_config.temporal_patch_size == 2

    tokenizer = Qwen3_5TextTokenizer.from_model_dir(model_dir)
    assert tokenizer.encode("<|im_start|>") == [3]
    assert tokenizer.decode([3]) == "<|im_start|>"
    assert tokenizer.image_token_id == 8
    assert tokenizer.video_token_id == 9
    assert tokenizer.vision_start_token_id == 5
    assert tokenizer.vision_end_token_id == 6


def test_load_qwen_weights_from_gguf(tmp_path: Path) -> None:
    model_dir, main_arrays, mmproj_arrays = _build_test_gguf_model_dir(tmp_path)
    config = load_qwen3_5_text_model_config(model_dir)
    model, quantized = build_qwen3_5_text_model(config, device=torch.device("cpu"), dtype=torch.float32)
    assert quantized == 0

    report = load_qwen3_5_text_model_weights(model, model_dir)
    assert report.loaded > 0
    assert report.skipped == 1

    assert torch.allclose(model.model.language_model.embed_tokens.weight, torch.tensor(main_arrays["token_embd.weight"]))
    assert model.lm_head.weight.data_ptr() == model.model.language_model.embed_tokens.weight.data_ptr()
    assert torch.allclose(
        model.model.language_model.layers[0].linear_attn.in_proj_qkv.weight,
        torch.tensor(main_arrays["blk.0.attn_qkv.weight"]),
    )
    assert torch.allclose(
        model.model.language_model.layers[1].self_attn.q_proj.weight,
        torch.tensor(main_arrays["blk.1.attn_q.weight"]),
    )
    assert torch.allclose(
        model.model.language_model.layers[0].linear_attn.A_log,
        torch.log(-torch.tensor(main_arrays["blk.0.ssm_a"])),
    )
    assert torch.allclose(
        model.model.language_model.layers[1].mlp.shared_expert_gate.weight,
        torch.tensor(main_arrays["blk.1.ffn_gate_inp_shexp.weight"])[None, :],
    )
    expected_patch = torch.stack(
        [
            torch.tensor(mmproj_arrays["v.patch_embd.weight"]),
            torch.tensor(mmproj_arrays["v.patch_embd.weight.1"]),
        ],
        dim=2,
    )
    assert torch.allclose(model.model.visual.patch_embed.proj.weight, expected_patch)


def test_load_qwen_weights_from_gguf_into_int4_placeholders(tmp_path: Path) -> None:
    model_dir, main_arrays, _ = _build_test_gguf_model_dir(tmp_path)
    config = load_qwen3_5_text_model_config(model_dir)
    model, quantized = build_qwen3_5_text_model(
        config,
        device=torch.device("cpu"),
        dtype=torch.float32,
        int4_placeholder_predicate=lambda module_name, _module: ".mlp.experts." in module_name,
    )
    assert quantized > 0
    assert isinstance(model.model.language_model.layers[0].mlp.experts[0].gate_proj, XPUInt4Linear)

    load_qwen3_5_text_model_weights(model, model_dir)

    expert_gate = model.model.language_model.layers[0].mlp.experts[0].gate_proj
    assert isinstance(expert_gate, XPUInt4Linear)
    assert expert_gate.qweight.numel() > 0
    recovered = expert_gate._dequantize_weight()
    expected = torch.tensor(main_arrays["blk.0.ffn_gate_exps.weight"][0])
    assert recovered.shape == expected.shape
    assert torch.allclose(recovered, expected, atol=0.25, rtol=0.25)


def test_estimate_qwen_weight_bytes_from_gguf_includes_mmproj(tmp_path: Path) -> None:
    model_dir, main_arrays, mmproj_arrays = _build_test_gguf_model_dir(tmp_path)
    expected_bytes = (
        sum(array.size * (4 if array.dtype == np.float32 else 2) for array in main_arrays.values())
        + sum(array.size * (4 if array.dtype == np.float32 else 2) for array in mmproj_arrays.values())
    )
    assert estimate_qwen3_5_text_model_weight_bytes(model_dir) == expected_bytes
