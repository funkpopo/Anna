from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFReader, TokenType
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

from anna.core.gguf_model import GGUFModelFiles, resolve_gguf_model_files
from anna.model.gemma4_config import Gemma4Config, Gemma4RopeParameters, Gemma4TextConfig
from anna.model.gemma4_text_model import Gemma4ForConditionalGeneration
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig, Qwen3_5TextModelConfig, Qwen3_5TextVisionConfig, RopeParameters, VisionPreprocessorConfig
from anna.model.ops import Qwen3SparseMoeBlock
from anna.model.qwen3_5_text_model import Qwen3_5TextForConditionalGeneration, Qwen3VisionModel
from anna.model.quantization import DenseLinear, XPUInt4Linear
from anna.weights.qwen3_5_text_weight_loader import WeightLoadReport

logger = logging.getLogger(__name__)

_GGUF_ROW_CHUNK_BYTES = 16 << 20
_GGUF_SPECIAL_TOKEN_TYPES = frozenset({int(TokenType.UNKNOWN), int(TokenType.CONTROL), int(TokenType.USER_DEFINED)})
_MISSING = object()


def _reader_field(reader: GGUFReader, key: str, default: Any = _MISSING) -> Any:
    field = reader.fields.get(key)
    if field is None:
        if default is not _MISSING:
            return default
        raise KeyError(f"Missing GGUF field: {key}")
    return field.contents()


def _tensor_map(reader: GGUFReader) -> dict[str, Any]:
    return {tensor.name: tensor for tensor in reader.tensors}


def _logical_shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in reversed(tuple(tensor.shape)))


def _token_id(tokens: list[str], token: str, default: int | None = None) -> int | None:
    try:
        return tokens.index(token)
    except ValueError:
        return default


def _preferred_mmproj_rank(path: Path) -> tuple[int, str]:
    lowered = path.name.lower()
    if "bf16" in lowered:
        return (0, lowered)
    if "f16" in lowered:
        return (1, lowered)
    if "f32" in lowered:
        return (2, lowered)
    return (3, lowered)


def select_preferred_mmproj_file(files: GGUFModelFiles) -> Path | None:
    if files.mmproj_file is not None:
        return files.mmproj_file
    if not files.available_mmproj_files:
        return None
    return min(files.available_mmproj_files, key=_preferred_mmproj_rank)


def _resolve_qwen_gguf_files(model_dir: str | Path) -> tuple[GGUFModelFiles, Path | None]:
    files = resolve_gguf_model_files(model_dir)
    return files, select_preferred_mmproj_file(files)


def _build_rope_parameters(reader: GGUFReader, arch: str, head_dim: int) -> RopeParameters:
    sections = tuple(int(value) for value in _reader_field(reader, f"{arch}.rope.dimension_sections", [11, 11, 10]) if int(value) > 0)
    rotary_dims = int(_reader_field(reader, f"{arch}.rope.dimension_count", head_dim // 4))
    partial_rotary_factor = float(rotary_dims) / float(head_dim) if head_dim > 0 else 0.25
    return RopeParameters(
        rope_theta=float(_reader_field(reader, f"{arch}.rope.freq_base", 10_000_000.0)),
        partial_rotary_factor=partial_rotary_factor,
        mrope_section=sections or (11, 11, 10),
        mrope_interleaved=True,
    )


def load_qwen3_5_text_model_config_from_gguf(model_dir: str | Path) -> Qwen3_5TextModelConfig:
    files, mmproj_file = _resolve_qwen_gguf_files(model_dir)
    reader = GGUFReader(str(files.model_file))
    arch = str(_reader_field(reader, "general.architecture"))
    if arch not in _QWEN35_GGUF_ARCHS:
        raise ValueError(f"Unsupported GGUF architecture for Anna Qwen runtime: {arch}")

    tensors = _tensor_map(reader)
    tokens = [str(value) for value in _reader_field(reader, "tokenizer.ggml.tokens")]
    block_count = int(_reader_field(reader, f"{arch}.block_count"))
    layer_types: list[str] = []
    first_linear_layer = None
    for layer_idx in range(block_count):
        if f"blk.{layer_idx}.attn_q.weight" in tensors:
            layer_types.append("full_attention")
            continue
        if f"blk.{layer_idx}.attn_qkv.weight" in tensors:
            if first_linear_layer is None:
                first_linear_layer = layer_idx
            layer_types.append("linear_attention")
            continue
        raise ValueError(f"Could not determine attention type for GGUF decoder layer {layer_idx}.")

    if first_linear_layer is None:
        raise ValueError("GGUF Qwen model does not expose any linear-attention layer for shape inference.")

    linear_qkv = tensors[f"blk.{first_linear_layer}.attn_qkv.weight"]
    linear_norm = tensors[f"blk.{first_linear_layer}.ssm_norm.weight"]
    linear_alpha = tensors[f"blk.{first_linear_layer}.ssm_alpha.weight"]
    linear_qkv_out = _logical_shape(linear_qkv)[0]
    linear_num_value_heads = _logical_shape(linear_alpha)[0]
    linear_value_head_dim = _logical_shape(linear_norm)[0]
    linear_value_dim = linear_num_value_heads * linear_value_head_dim
    linear_num_key_heads = int(_reader_field(reader, f"{arch}.ssm.group_count"))
    linear_key_dim = linear_qkv_out - linear_value_dim
    if linear_key_dim <= 0 or linear_key_dim % 2 != 0:
        raise ValueError(
            f"Invalid GGUF linear-attention projection width: qkv_out={linear_qkv_out} value_dim={linear_value_dim}"
        )
    linear_key_dim //= 2
    if linear_key_dim % max(1, linear_num_key_heads) != 0:
        raise ValueError(
            f"GGUF linear-attention key_dim={linear_key_dim} is not divisible by group_count={linear_num_key_heads}"
        )
    linear_key_head_dim = linear_key_dim // linear_num_key_heads

    head_dim = int(_reader_field(reader, f"{arch}.attention.key_length"))
    endoftext_id = _token_id(tokens, "<|endoftext|>", 248044)
    pad_token_id = _reader_field(reader, "tokenizer.ggml.padding_token_id", endoftext_id)

    # MoE keys are absent for dense exports (arch == "qwen35"); fall back to the
    # dense feed-forward size so is_moe_model resolves to False.
    num_experts = int(_reader_field(reader, f"{arch}.expert_count", 0))
    num_experts_per_tok = int(_reader_field(reader, f"{arch}.expert_used_count", 0))
    moe_intermediate_size = int(_reader_field(reader, f"{arch}.expert_feed_forward_length", 0))
    shared_expert_intermediate_size = int(
        _reader_field(reader, f"{arch}.expert_shared_feed_forward_length", 0)
    )
    dense_intermediate_size = int(_reader_field(reader, f"{arch}.feed_forward_length", 0))
    if num_experts > 0:
        if moe_intermediate_size <= 0:
            raise ValueError(f"GGUF {arch} model declares {num_experts} experts without expert_feed_forward_length.")
        intermediate_size = shared_expert_intermediate_size or dense_intermediate_size
    else:
        intermediate_size = dense_intermediate_size
        if intermediate_size <= 0:
            raise ValueError(f"GGUF {arch} model is missing feed_forward_length.")
        moe_intermediate_size = 0
        shared_expert_intermediate_size = 0
        num_experts_per_tok = 0

    text_config = Qwen3_5TextConfig(
        model_type="qwen3_5_text",
        hidden_size=int(_reader_field(reader, f"{arch}.embedding_length")),
        intermediate_size=intermediate_size,
        num_hidden_layers=block_count,
        num_attention_heads=int(_reader_field(reader, f"{arch}.attention.head_count")),
        num_key_value_heads=int(_reader_field(reader, f"{arch}.attention.head_count_kv")),
        head_dim=head_dim,
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        hidden_act="silu",
        linear_conv_kernel_dim=int(_reader_field(reader, f"{arch}.ssm.conv_kernel")),
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        max_position_embeddings=int(_reader_field(reader, f"{arch}.context_length")),
        rms_norm_eps=float(_reader_field(reader, f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)),
        vocab_size=len(tokens),
        tie_word_embeddings=True,
        eos_token_id=int(endoftext_id or 248044),
        pad_token_id=int(pad_token_id if pad_token_id is not None else (endoftext_id or 248044)),
        dtype="bfloat16",
        cache_block_size=32,
        layer_types=layer_types,
        full_attention_interval=int(_reader_field(reader, f"{arch}.full_attention_interval", 4)),
        rope_parameters=_build_rope_parameters(reader, arch, head_dim=head_dim),
        decoder_sparse_step=1,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        mlp_only_layers=[],
    )

    vision_config = None
    preprocessor_config = VisionPreprocessorConfig()
    if mmproj_file is not None:
        mmproj_reader = GGUFReader(str(mmproj_file))
        mmproj_tensors = _tensor_map(mmproj_reader)
        patch0 = mmproj_tensors["v.patch_embd.weight"]
        temporal_patch_size = 2 if "v.patch_embd.weight.1" in mmproj_tensors else 1
        patch_shape = _logical_shape(patch0)
        image_size = int(_reader_field(mmproj_reader, "clip.vision.image_size"))
        patch_size = int(_reader_field(mmproj_reader, "clip.vision.patch_size"))
        spatial_merge_size = int(_reader_field(mmproj_reader, "clip.vision.spatial_merge_size", 2))
        image_mean = tuple(float(value) for value in _reader_field(mmproj_reader, "clip.vision.image_mean", [0.5, 0.5, 0.5]))
        image_std = tuple(float(value) for value in _reader_field(mmproj_reader, "clip.vision.image_std", [0.5, 0.5, 0.5]))
        vision_config = Qwen3_5TextVisionConfig(
            depth=int(_reader_field(mmproj_reader, "clip.vision.block_count")),
            hidden_size=int(_reader_field(mmproj_reader, "clip.vision.embedding_length")),
            hidden_act="gelu_pytorch_tanh",
            in_channels=int(patch_shape[1]),
            intermediate_size=int(_reader_field(mmproj_reader, "clip.vision.feed_forward_length")),
            num_heads=int(_reader_field(mmproj_reader, "clip.vision.attention.head_count")),
            num_position_embeddings=int(_logical_shape(mmproj_tensors["v.position_embd.weight"])[0]),
            out_hidden_size=int(_logical_shape(mmproj_tensors["mm.2.weight"])[0]),
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            temporal_patch_size=temporal_patch_size,
        )
        preprocessor_config = VisionPreprocessorConfig(
            shortest_edge=image_size * image_size,
            longest_edge=image_size * image_size,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=spatial_merge_size,
            image_mean=image_mean,
            image_std=image_std,
        )

    return Qwen3_5TextModelConfig(
        model_type="qwen3_5",
        text_config=text_config,
        vision_config=vision_config,
        preprocessor_config=preprocessor_config,
        tie_word_embeddings=True,
        image_token_id=int(_token_id(tokens, "<|image_pad|>", 248056) or 248056),
        video_token_id=int(_token_id(tokens, "<|video_pad|>", 248057) or 248057),
        vision_start_token_id=int(_token_id(tokens, "<|vision_start|>", 248053) or 248053),
        vision_end_token_id=int(_token_id(tokens, "<|vision_end|>", 248054) or 248054),
    )


def build_qwen3_5_text_tokenizer_backend_from_gguf(model_dir: str | Path) -> tuple[Tokenizer, dict[str, Any]]:
    files, _ = _resolve_qwen_gguf_files(model_dir)
    reader = GGUFReader(str(files.model_file))
    tokens = [str(value) for value in _reader_field(reader, "tokenizer.ggml.tokens")]
    merges = [str(value) for value in _reader_field(reader, "tokenizer.ggml.merges", [])]
    token_types = [int(value) for value in _reader_field(reader, "tokenizer.ggml.token_type", [int(TokenType.NORMAL)] * len(tokens))]

    vocab = {token: token_id for token_id, token in enumerate(tokens)}
    merge_pairs = []
    for merge in merges:
        left, right = merge.split(" ", 1)
        merge_pairs.append((left, right))

    backend = Tokenizer(models.BPE(vocab=vocab, merges=merge_pairs, unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()

    special_tokens = [
        AddedToken(tokens[token_id], special=True, normalized=False)
        for token_id, token_type in enumerate(token_types)
        if token_type in _GGUF_SPECIAL_TOKEN_TYPES
    ]
    if special_tokens:
        backend.add_special_tokens(special_tokens)

    metadata = {
        "chat_template": _reader_field(reader, "tokenizer.chat_template", None),
        "extra_special_tokens": {
            "vision_bos_token": "<|vision_start|>",
            "vision_eos_token": "<|vision_end|>",
            "image_token": "<|image_pad|>",
            "video_token": "<|video_pad|>",
        },
    }
    return backend, metadata


def _estimate_reader_weight_bytes(reader: GGUFReader) -> int:
    total = 0
    for tensor in reader.tensors:
        logical_shape = _logical_shape(tensor)
        elements = int(np.prod(logical_shape, dtype=np.int64))
        quant_type = GGMLQuantizationType(int(tensor.tensor_type))
        element_size = 4 if quant_type == GGMLQuantizationType.F32 else 2
        total += elements * element_size
    return total


def estimate_qwen3_5_text_model_weight_bytes_from_gguf(model_dir: str | Path) -> int:
    files, mmproj_file = _resolve_qwen_gguf_files(model_dir)
    total = _estimate_reader_weight_bytes(GGUFReader(str(files.model_file)))
    if mmproj_file is not None:
        total += _estimate_reader_weight_bytes(GGUFReader(str(mmproj_file)))
    return total


def _to_torch_array(array: np.ndarray, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(array, copy=True, order="C"))
    if dtype is not None or device is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor


def _copy_vector_(target: torch.Tensor, values: np.ndarray) -> None:
    expected_shape = tuple(target.shape)
    if tuple(values.shape) != expected_shape:
        raise ValueError(f"Shape mismatch while loading GGUF vector: expected {expected_shape}, got {tuple(values.shape)}")
    with torch.no_grad():
        target.copy_(_to_torch_array(values, dtype=target.dtype, device=target.device))


def _row_chunk_rows(num_columns: int) -> int:
    return max(1, int(_GGUF_ROW_CHUNK_BYTES // max(1, num_columns * 4)))


def _copy_quantized_matrix_data_to_parameter_(
    target: torch.Tensor,
    data: Any,
    tensor_type: GGMLQuantizationType,
) -> None:
    rows, columns = tuple(target.shape)
    rows_per_chunk = min(rows, _row_chunk_rows(columns))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(rows, row_start + rows_per_chunk)
        chunk = dequantize(data[row_start:row_end], tensor_type)
        if tuple(chunk.shape) != (row_end - row_start, columns):
            raise ValueError(
                f"GGUF matrix chunk shape mismatch for rows {row_start}:{row_end}: expected {(row_end - row_start, columns)}, got {tuple(chunk.shape)}"
            )
        with torch.no_grad():
            target[row_start:row_end].copy_(_to_torch_array(chunk, dtype=target.dtype, device=target.device))


def _copy_matrix_to_parameter_(target: torch.Tensor, tensor: Any) -> None:
    rows, columns = tuple(target.shape)
    expected_shape = _logical_shape(tensor)
    if expected_shape != (rows, columns):
        raise ValueError(f"GGUF matrix shape mismatch: expected {(rows, columns)}, got {expected_shape}")
    _copy_quantized_matrix_data_to_parameter_(target, tensor.data, tensor.tensor_type)


def _copy_quantized_matrix_data_to_linear_(
    module: torch.nn.Module,
    data: Any,
    tensor_type: GGMLQuantizationType,
) -> None:
    if isinstance(module, XPUInt4Linear):
        rows = int(module.out_features)
        columns = int(module.in_features)
        rows_per_chunk = min(rows, _row_chunk_rows(columns))
        for row_start in range(0, rows, rows_per_chunk):
            row_end = min(rows, row_start + rows_per_chunk)
            chunk = dequantize(data[row_start:row_end], tensor_type)
            if tuple(chunk.shape) != (row_end - row_start, columns):
                raise ValueError(
                    f"GGUF int4 chunk shape mismatch for rows {row_start}:{row_end}: expected {(row_end - row_start, columns)}, got {tuple(chunk.shape)}"
                )
            qweight, qscale, qzeros = XPUInt4Linear._quantize_weight(
                _to_torch_array(chunk, dtype=torch.float32, device=torch.device("cpu")),
                group_size=module.group_size,
                padded_in_features=module.padded_in_features,
            )
            with torch.no_grad():
                module.qweight[row_start:row_end].copy_(qweight.to(device=module.qweight.device))
                module.qscale[:, row_start:row_end].copy_(qscale.to(device=module.qscale.device))
                module.qzeros[:, row_start:row_end].copy_(qzeros.to(device=module.qzeros.device))
        return
    if not isinstance(module, (torch.nn.Linear, DenseLinear)):
        raise TypeError(f"Unsupported GGUF linear target: {type(module)!r}")
    _copy_quantized_matrix_data_to_parameter_(module.weight, data, tensor_type)


def _copy_matrix_to_linear_(module: torch.nn.Module, tensor: Any) -> None:
    expected_shape = _logical_shape(tensor)
    if isinstance(module, XPUInt4Linear):
        module_shape = (int(module.out_features), int(module.in_features))
    elif isinstance(module, (torch.nn.Linear, DenseLinear)):
        module_shape = tuple(module.weight.shape)
    else:
        raise TypeError(f"Unsupported GGUF linear target: {type(module)!r}")
    if expected_shape != module_shape:
        raise ValueError(f"GGUF linear shape mismatch: expected {module_shape}, got {expected_shape}")
    _copy_quantized_matrix_data_to_linear_(module, tensor.data, tensor.tensor_type)


def _copy_bias_parameter_(parameter: torch.Tensor | None, tensor: Any) -> None:
    if parameter is None:
        return
    values = dequantize(tensor.data, tensor.tensor_type)
    _copy_vector_(parameter, values)


def _copy_parameter_from_tensor_(parameter: torch.Tensor, tensor: Any) -> None:
    values = dequantize(tensor.data, tensor.tensor_type)
    if parameter.ndim == 1:
        _copy_vector_(parameter, values)
        return
    if parameter.ndim == 2:
        _copy_matrix_to_parameter_(parameter, tensor)
        return
    expected_shape = tuple(parameter.shape)
    if tuple(values.shape) != expected_shape:
        raise ValueError(f"GGUF tensor shape mismatch: expected {expected_shape}, got {tuple(values.shape)}")
    with torch.no_grad():
        parameter.copy_(_to_torch_array(values, dtype=parameter.dtype, device=parameter.device))


def _load_qwen_moe_expert_tensor_group_(
    experts: torch.nn.ModuleList,
    tensor: Any,
    *,
    linear_name: str,
) -> None:
    if tensor.data.shape[0] != len(experts):
        raise ValueError(f"GGUF expert tensor count mismatch for {tensor.name}: expected {len(experts)}, got {tensor.data.shape[0]}")
    for expert_idx, expert in enumerate(experts):
        _copy_quantized_matrix_data_to_linear_(getattr(expert, linear_name), tensor.data[expert_idx], tensor.tensor_type)


def _copy_vector_or_matrix_to_linear_weight_(module: torch.nn.Module, values: np.ndarray) -> None:
    if isinstance(module, XPUInt4Linear):
        qweight, qscale, qzeros = XPUInt4Linear._quantize_weight(
            _to_torch_array(values, dtype=torch.float32, device=torch.device("cpu")),
            group_size=module.group_size,
            padded_in_features=module.padded_in_features,
        )
        with torch.no_grad():
            module.qweight.copy_(qweight.to(device=module.qweight.device))
            module.qscale.copy_(qscale.to(device=module.qscale.device))
            module.qzeros.copy_(qzeros.to(device=module.qzeros.device))
        return
    if not isinstance(module, (torch.nn.Linear, DenseLinear)):
        raise TypeError(f"Unsupported GGUF linear target: {type(module)!r}")
    expected_shape = tuple(module.weight.shape)
    if tuple(values.shape) != expected_shape:
        raise ValueError(f"GGUF matrix shape mismatch: expected {expected_shape}, got {tuple(values.shape)}")
    with torch.no_grad():
        module.weight.copy_(_to_torch_array(values, dtype=module.weight.dtype, device=module.weight.device))


def _load_main_qwen35moe_weights_(
    model: Qwen3_5TextForConditionalGeneration,
    reader: GGUFReader,
) -> tuple[int, int]:
    tensors = _tensor_map(reader)
    language_model = model.model.language_model
    loaded = 0
    skipped = 0

    _copy_matrix_to_parameter_(language_model.embed_tokens.weight, tensors["token_embd.weight"])
    loaded += 1
    _copy_vector_(language_model.norm.weight, dequantize(tensors["output_norm.weight"].data, tensors["output_norm.weight"].tensor_type))
    loaded += 1

    if not model.config.tie_word_embeddings and "output.weight" in tensors:
        _copy_matrix_to_parameter_(model.lm_head.weight, tensors["output.weight"])
        loaded += 1
    elif "output.weight" in tensors:
        skipped += 1

    for layer_idx, layer in enumerate(language_model.layers):
        prefix = f"blk.{layer_idx}"
        _copy_vector_(layer.input_layernorm.weight, dequantize(tensors[f"{prefix}.attn_norm.weight"].data, tensors[f"{prefix}.attn_norm.weight"].tensor_type))
        _copy_vector_(
            layer.post_attention_layernorm.weight,
            dequantize(tensors[f"{prefix}.post_attention_norm.weight"].data, tensors[f"{prefix}.post_attention_norm.weight"].tensor_type),
        )
        loaded += 2

        if layer.layer_type == "full_attention":
            self_attn = layer.self_attn
            _copy_matrix_to_linear_(self_attn.q_proj, tensors[f"{prefix}.attn_q.weight"])
            _copy_matrix_to_linear_(self_attn.k_proj, tensors[f"{prefix}.attn_k.weight"])
            _copy_matrix_to_linear_(self_attn.v_proj, tensors[f"{prefix}.attn_v.weight"])
            _copy_matrix_to_linear_(self_attn.o_proj, tensors[f"{prefix}.attn_output.weight"])
            _copy_vector_(self_attn.q_norm.weight, dequantize(tensors[f"{prefix}.attn_q_norm.weight"].data, tensors[f"{prefix}.attn_q_norm.weight"].tensor_type))
            _copy_vector_(self_attn.k_norm.weight, dequantize(tensors[f"{prefix}.attn_k_norm.weight"].data, tensors[f"{prefix}.attn_k_norm.weight"].tensor_type))
            loaded += 6
        else:
            linear_attn = layer.linear_attn
            _copy_matrix_to_linear_(linear_attn.in_proj_qkv, tensors[f"{prefix}.attn_qkv.weight"])
            _copy_matrix_to_linear_(linear_attn.in_proj_z, tensors[f"{prefix}.attn_gate.weight"])
            _copy_matrix_to_linear_(linear_attn.in_proj_a, tensors[f"{prefix}.ssm_alpha.weight"])
            _copy_matrix_to_linear_(linear_attn.in_proj_b, tensors[f"{prefix}.ssm_beta.weight"])
            _copy_matrix_to_linear_(linear_attn.out_proj, tensors[f"{prefix}.ssm_out.weight"])
            conv_weight = dequantize(tensors[f"{prefix}.ssm_conv1d.weight"].data, tensors[f"{prefix}.ssm_conv1d.weight"].tensor_type)
            conv_weight = conv_weight[:, None, :]
            _copy_parameter_from_tensor_(linear_attn.conv1d.weight, _ArrayBackedTensor(conv_weight))
            _copy_vector_(linear_attn.dt_bias, dequantize(tensors[f"{prefix}.ssm_dt.bias"].data, tensors[f"{prefix}.ssm_dt.bias"].tensor_type))
            ssm_a = dequantize(tensors[f"{prefix}.ssm_a"].data, tensors[f"{prefix}.ssm_a"].tensor_type)
            _copy_vector_(linear_attn.A_log, np.log(np.clip(-ssm_a, a_min=1e-20, a_max=None)))
            _copy_vector_(linear_attn.norm.weight, dequantize(tensors[f"{prefix}.ssm_norm.weight"].data, tensors[f"{prefix}.ssm_norm.weight"].tensor_type))
            loaded += 8

        mlp = layer.mlp
        if not isinstance(mlp, Qwen3SparseMoeBlock):
            # Dense feed-forward (arch == "qwen35").
            _copy_matrix_to_linear_(mlp.gate_proj, tensors[f"{prefix}.ffn_gate.weight"])
            _copy_matrix_to_linear_(mlp.up_proj, tensors[f"{prefix}.ffn_up.weight"])
            _copy_matrix_to_linear_(mlp.down_proj, tensors[f"{prefix}.ffn_down.weight"])
            loaded += 3
            continue
        _copy_matrix_to_linear_(mlp.gate, tensors[f"{prefix}.ffn_gate_inp.weight"])
        _load_qwen_moe_expert_tensor_group_(mlp.experts, tensors[f"{prefix}.ffn_gate_exps.weight"], linear_name="gate_proj")
        _load_qwen_moe_expert_tensor_group_(mlp.experts, tensors[f"{prefix}.ffn_up_exps.weight"], linear_name="up_proj")
        _load_qwen_moe_expert_tensor_group_(mlp.experts, tensors[f"{prefix}.ffn_down_exps.weight"], linear_name="down_proj")
        _copy_matrix_to_linear_(mlp.shared_expert.gate_proj, tensors[f"{prefix}.ffn_gate_shexp.weight"])
        _copy_matrix_to_linear_(mlp.shared_expert.up_proj, tensors[f"{prefix}.ffn_up_shexp.weight"])
        _copy_matrix_to_linear_(mlp.shared_expert.down_proj, tensors[f"{prefix}.ffn_down_shexp.weight"])
        _copy_vector_or_matrix_to_linear_weight_(
            mlp.shared_expert_gate,
            dequantize(tensors[f"{prefix}.ffn_gate_inp_shexp.weight"].data, tensors[f"{prefix}.ffn_gate_inp_shexp.weight"].tensor_type)[None, :],
        )
        loaded += 7 + (3 * len(mlp.experts))

    return loaded, skipped


class _ArrayBackedTensor:
    def __init__(self, array: np.ndarray):
        self.data = array
        self.tensor_type = GGMLQuantizationType.F32
        self.shape = tuple(reversed(array.shape))


def _load_clip_mmproj_weights_(visual: Qwen3VisionModel, reader: GGUFReader) -> int:
    tensors = _tensor_map(reader)
    loaded = 0

    patch_weight_0 = dequantize(tensors["v.patch_embd.weight"].data, tensors["v.patch_embd.weight"].tensor_type)
    patch_weight_1 = dequantize(tensors["v.patch_embd.weight.1"].data, tensors["v.patch_embd.weight.1"].tensor_type)
    patch_weight = np.stack([patch_weight_0, patch_weight_1], axis=2)
    with torch.no_grad():
        visual.patch_embed.proj.weight.copy_(
            _to_torch_array(patch_weight, dtype=visual.patch_embed.proj.weight.dtype, device=visual.patch_embed.proj.weight.device)
        )
    _copy_bias_parameter_(visual.patch_embed.proj.bias, tensors["v.patch_embd.bias"])
    _copy_matrix_to_parameter_(visual.pos_embed.weight, tensors["v.position_embd.weight"])
    loaded += 3

    for layer_idx, block in enumerate(visual.blocks):
        prefix = f"v.blk.{layer_idx}"
        _copy_vector_(block.norm1.weight, dequantize(tensors[f"{prefix}.ln1.weight"].data, tensors[f"{prefix}.ln1.weight"].tensor_type))
        _copy_vector_(block.norm1.bias, dequantize(tensors[f"{prefix}.ln1.bias"].data, tensors[f"{prefix}.ln1.bias"].tensor_type))
        _copy_vector_(block.norm2.weight, dequantize(tensors[f"{prefix}.ln2.weight"].data, tensors[f"{prefix}.ln2.weight"].tensor_type))
        _copy_vector_(block.norm2.bias, dequantize(tensors[f"{prefix}.ln2.bias"].data, tensors[f"{prefix}.ln2.bias"].tensor_type))
        _copy_matrix_to_linear_(block.attn.qkv, tensors[f"{prefix}.attn_qkv.weight"])
        _copy_bias_parameter_(block.attn.qkv.bias, tensors[f"{prefix}.attn_qkv.bias"])
        _copy_matrix_to_linear_(block.attn.proj, tensors[f"{prefix}.attn_out.weight"])
        _copy_bias_parameter_(block.attn.proj.bias, tensors[f"{prefix}.attn_out.bias"])
        _copy_matrix_to_linear_(block.mlp.linear_fc1, tensors[f"{prefix}.ffn_up.weight"])
        _copy_bias_parameter_(block.mlp.linear_fc1.bias, tensors[f"{prefix}.ffn_up.bias"])
        _copy_matrix_to_linear_(block.mlp.linear_fc2, tensors[f"{prefix}.ffn_down.weight"])
        _copy_bias_parameter_(block.mlp.linear_fc2.bias, tensors[f"{prefix}.ffn_down.bias"])
        loaded += 12

    _copy_vector_(visual.merger.norm.weight, dequantize(tensors["v.post_ln.weight"].data, tensors["v.post_ln.weight"].tensor_type))
    _copy_vector_(visual.merger.norm.bias, dequantize(tensors["v.post_ln.bias"].data, tensors["v.post_ln.bias"].tensor_type))
    _copy_matrix_to_linear_(visual.merger.linear_fc1, tensors["mm.0.weight"])
    _copy_bias_parameter_(visual.merger.linear_fc1.bias, tensors["mm.0.bias"])
    _copy_matrix_to_linear_(visual.merger.linear_fc2, tensors["mm.2.weight"])
    _copy_bias_parameter_(visual.merger.linear_fc2.bias, tensors["mm.2.bias"])
    loaded += 6

    return loaded


# --- Gemma4 GGUF layout (text-only) ---
#
# general.architecture == "gemma4" maps to the Anna Gemma4 text runtime.
# Metadata keys follow the llama.cpp conventions used by the Qwen35moe layout:
#   gemma4.block_count / context_length / embedding_length / feed_forward_length
#   gemma4.full_attention_interval (default 6)
#   gemma4.attention.{head_count, head_count_kv, key_length, global_key_length,
#                     global_head_count_kv, layer_norm_rms_epsilon, sliding_window,
#                     k_eq_v, kv_shared_layers}
#   gemma4.per_layer_input_length / vocab_per_layer_input_length
#   gemma4.rope.{freq_base_sliding, freq_base_global, partial_rotary_factor,
#                global_original_context_length}
#   gemma4.final_logit_softcapping
# Tensors:
#   token_embd.weight / output_norm.weight / output.weight (optional, untied)
#   token_embd_per_layer.weight / per_layer_model_proj.weight / per_layer_proj_norm.weight
#   blk.N.{attn_norm, attn_post_norm, ffn_norm, ffn_post_norm}.weight
#   blk.N.attn_{q,k,v,output}.weight + blk.N.attn_{q,k}_norm.weight
#   blk.N.ffn_{gate,up,down}.weight
#   blk.N.inp_gate.weight / per_layer_proj.weight / per_layer_post_norm.weight

_GEMMA4_ARCH = "gemma4"
_QWEN35_GGUF_ARCHS = ("qwen35", "qwen35moe")


def _gemma4_layer_types(reader: GGUFReader, arch: str, block_count: int) -> list[str]:
    interval = int(_reader_field(reader, f"{arch}.full_attention_interval", 6))
    if interval <= 0:
        interval = 1
    layer_types = [
        "full_attention" if (layer_idx + 1) % interval == 0 else "sliding_attention"
        for layer_idx in range(block_count)
    ]
    if layer_types and layer_types[-1] != "full_attention":
        layer_types[-1] = "full_attention"
    return layer_types


def _gemma4_layer_attention_head_dim(
    tensors: dict[str, Any],
    layer_types: list[str],
    layer_type: str,
    num_kv_heads: int,
) -> int | None:
    """Infer a per-layer-type head dim from the first matching layer's k projection."""
    if layer_type not in layer_types:
        return None
    tensor = tensors.get(f"blk.{layer_types.index(layer_type)}.attn_k.weight")
    if tensor is None or num_kv_heads <= 0:
        return None
    kv_out = _logical_shape(tensor)[0]
    if kv_out % num_kv_heads != 0:
        return None
    return kv_out // num_kv_heads


def load_gemma4_text_model_config_from_gguf(model_dir: str | Path) -> Gemma4Config:
    files, _ = _resolve_qwen_gguf_files(model_dir)
    reader = GGUFReader(str(files.model_file))
    arch = str(_reader_field(reader, "general.architecture"))
    if arch != _GEMMA4_ARCH:
        raise ValueError(f"Unsupported GGUF architecture for Anna Gemma runtime: {arch}")

    tensors = _tensor_map(reader)
    tokens = [str(value) for value in _reader_field(reader, "tokenizer.ggml.tokens")]
    block_count = int(_reader_field(reader, f"{arch}.block_count"))
    hidden_size = int(_reader_field(reader, f"{arch}.embedding_length"))
    num_heads = int(_reader_field(reader, f"{arch}.attention.head_count"))
    num_kv_heads = int(_reader_field(reader, f"{arch}.attention.head_count_kv"))
    layer_types = _gemma4_layer_types(reader, arch, block_count)

    head_dim = int(_reader_field(reader, f"{arch}.attention.key_length", 0))
    if head_dim <= 0:
        head_dim = (
            _gemma4_layer_attention_head_dim(tensors, layer_types, "sliding_attention", num_kv_heads)
            or _gemma4_layer_attention_head_dim(tensors, layer_types, "full_attention", num_kv_heads)
            or (hidden_size // max(1, num_heads))
        )
    global_head_dim = int(_reader_field(reader, f"{arch}.attention.global_key_length", 0))
    if global_head_dim <= 0:
        global_head_dim = (
            _gemma4_layer_attention_head_dim(tensors, layer_types, "full_attention", num_kv_heads)
            or head_dim
        )
    num_global_key_value_heads = int(_reader_field(reader, f"{arch}.attention.global_head_count_kv", num_kv_heads))

    bos_token_id = int(_reader_field(reader, "tokenizer.ggml.bos_token_id", 2))
    eos_token_id = int(_reader_field(reader, "tokenizer.ggml.eos_token_id", 1))
    pad_token_id = int(_reader_field(reader, "tokenizer.ggml.padding_token_id", 0))
    max_position_embeddings = int(_reader_field(reader, f"{arch}.context_length"))

    hidden_size_per_layer_input = int(_reader_field(reader, f"{arch}.per_layer_input_length", 0))
    vocab_size_per_layer_input = int(
        _reader_field(reader, f"{arch}.vocab_per_layer_input_length", len(tokens))
    )

    original_max_position_embeddings = int(
        _reader_field(reader, f"{arch}.rope.global_original_context_length", min(8_192, max_position_embeddings))
    )
    rope_factor = max(1.0, float(max_position_embeddings) / float(max(1, original_max_position_embeddings)))
    rope_parameters = {
        "sliding_attention": Gemma4RopeParameters(
            rope_type="default",
            rope_theta=float(_reader_field(reader, f"{arch}.rope.freq_base_sliding", 10_000.0)),
        ),
        "full_attention": Gemma4RopeParameters(
            rope_type="proportional",
            rope_theta=float(_reader_field(reader, f"{arch}.rope.freq_base_global", 1_000_000.0)),
            partial_rotary_factor=float(_reader_field(reader, f"{arch}.rope.partial_rotary_factor", 0.25)),
            factor=rope_factor,
            original_max_position_embeddings=original_max_position_embeddings,
        ),
    }

    softcapping = _reader_field(reader, f"{arch}.final_logit_softcapping", None)
    text_config = Gemma4TextConfig(
        model_type="gemma4_text",
        vocab_size=len(tokens),
        hidden_size=hidden_size,
        intermediate_size=int(_reader_field(reader, f"{arch}.feed_forward_length")),
        num_hidden_layers=block_count,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=float(_reader_field(reader, f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)),
        pad_token_id=pad_token_id,
        eos_token_ids=(eos_token_id,),
        bos_token_id=bos_token_id,
        tie_word_embeddings=True,
        dtype="bfloat16",
        sliding_window=int(_reader_field(reader, f"{arch}.attention.sliding_window", 512)),
        layer_types=layer_types,
        final_logit_softcapping=None if softcapping is None else float(softcapping),
        vocab_size_per_layer_input=vocab_size_per_layer_input,
        hidden_size_per_layer_input=hidden_size_per_layer_input,
        num_global_key_value_heads=num_global_key_value_heads,
        attention_k_eq_v=bool(_reader_field(reader, f"{arch}.attention.k_eq_v", False)),
        num_kv_shared_layers=int(_reader_field(reader, f"{arch}.attention.kv_shared_layers", 0)),
        rope_parameters=rope_parameters,
    )

    return Gemma4Config(
        model_type="gemma4",
        text_config=text_config,
        vision_config=None,
        audio_config=None,
        tie_word_embeddings=True,
        image_token_id=int(_token_id(tokens, "<image>", 258_880) or 258_880),
        video_token_id=int(_token_id(tokens, "<video>", 258_884) or 258_884),
        audio_token_id=int(_token_id(tokens, "<|audio|>", 258_881) or 258_881),
        boi_token_id=int(_token_id(tokens, "<|image>", 255_999) or 255_999),
        eoi_token_id=int(_token_id(tokens, "<image|>", 258_882) or 258_882),
        boa_token_id=int(_token_id(tokens, "<|audio>", 256_000) or 256_000),
        eoa_token_id=int(_token_id(tokens, "<audio|>", 258_883) or 258_883),
    )


def build_gemma4_text_tokenizer_backend_from_gguf(model_dir: str | Path) -> tuple[Tokenizer, dict[str, Any]]:
    files, _ = _resolve_qwen_gguf_files(model_dir)
    reader = GGUFReader(str(files.model_file))
    tokens = [str(value) for value in _reader_field(reader, "tokenizer.ggml.tokens")]
    scores = [float(value) for value in _reader_field(reader, "tokenizer.ggml.scores", [0.0] * len(tokens))]
    token_types = [int(value) for value in _reader_field(reader, "tokenizer.ggml.token_type", [int(TokenType.NORMAL)] * len(tokens))]

    token_set = set(tokens)
    byte_fallback = all(f"<0x{value:02X}>" in token_set for value in range(256))
    unk_token_id = _reader_field(reader, "tokenizer.ggml.unk_token_id", None)
    backend = Tokenizer(
        models.Unigram(
            [(token, float(score)) for token, score in zip(tokens, scores)],
            unk_id=None if unk_token_id is None else int(unk_token_id),
            byte_fallback=byte_fallback,
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    backend.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

    special_tokens = [
        AddedToken(tokens[token_id], special=True, normalized=False)
        for token_id, token_type in enumerate(token_types)
        if token_type in _GGUF_SPECIAL_TOKEN_TYPES
    ]
    if special_tokens:
        backend.add_special_tokens(special_tokens)

    def _special_string(key: str, default_id: int, default_token: str) -> str:
        token_id = _reader_field(reader, key, default_id)
        if token_id is None or not 0 <= int(token_id) < len(tokens):
            return tokens[default_id] if 0 <= default_id < len(tokens) else default_token
        return tokens[int(token_id)]

    metadata = {
        "bos_token": _special_string("tokenizer.ggml.bos_token_id", 2, "<bos>"),
        "eos_token": _special_string("tokenizer.ggml.eos_token_id", 1, "<eos>"),
    }
    return backend, metadata


def estimate_gemma4_text_model_weight_bytes_from_gguf(model_dir: str | Path) -> int:
    files, _ = _resolve_qwen_gguf_files(model_dir)
    return _estimate_reader_weight_bytes(GGUFReader(str(files.model_file)))


def load_gemma4_text_model_weights_from_gguf(
    model: Gemma4ForConditionalGeneration,
    model_dir: str | Path,
) -> WeightLoadReport:
    files, _ = _resolve_qwen_gguf_files(model_dir)
    reader = GGUFReader(str(files.model_file))
    tensors = _tensor_map(reader)
    language_model = model.model.language_model
    config = language_model.config
    loaded = 0
    skipped = 0

    _copy_matrix_to_parameter_(language_model.embed_tokens.weight, tensors["token_embd.weight"])
    loaded += 1
    _copy_vector_(language_model.norm.weight, dequantize(tensors["output_norm.weight"].data, tensors["output_norm.weight"].tensor_type))
    loaded += 1

    if not (config.tie_word_embeddings or model.config.tie_word_embeddings) and model.lm_head is not None and "output.weight" in tensors:
        _copy_matrix_to_parameter_(model.lm_head.weight, tensors["output.weight"])
        loaded += 1
    elif "output.weight" in tensors:
        skipped += 1

    if config.hidden_size_per_layer_input > 0:
        _copy_matrix_to_parameter_(language_model.embed_tokens_per_layer.weight, tensors["token_embd_per_layer.weight"])
        _copy_matrix_to_linear_(language_model.per_layer_model_projection, tensors["per_layer_model_proj.weight"])
        _copy_vector_(language_model.per_layer_projection_norm.weight, dequantize(tensors["per_layer_proj_norm.weight"].data, tensors["per_layer_proj_norm.weight"].tensor_type))
        loaded += 3

    for layer_idx, layer in enumerate(language_model.layers):
        prefix = f"blk.{layer_idx}"
        self_attn = layer.self_attn
        _copy_vector_(layer.input_layernorm.weight, dequantize(tensors[f"{prefix}.attn_norm.weight"].data, tensors[f"{prefix}.attn_norm.weight"].tensor_type))
        _copy_vector_(layer.post_attention_layernorm.weight, dequantize(tensors[f"{prefix}.attn_post_norm.weight"].data, tensors[f"{prefix}.attn_post_norm.weight"].tensor_type))
        _copy_vector_(layer.pre_feedforward_layernorm.weight, dequantize(tensors[f"{prefix}.ffn_norm.weight"].data, tensors[f"{prefix}.ffn_norm.weight"].tensor_type))
        _copy_vector_(layer.post_feedforward_layernorm.weight, dequantize(tensors[f"{prefix}.ffn_post_norm.weight"].data, tensors[f"{prefix}.ffn_post_norm.weight"].tensor_type))
        loaded += 4

        _copy_matrix_to_linear_(self_attn.q_proj, tensors[f"{prefix}.attn_q.weight"])
        _copy_matrix_to_linear_(self_attn.k_proj, tensors[f"{prefix}.attn_k.weight"])
        if self_attn.v_proj is not None:
            _copy_matrix_to_linear_(self_attn.v_proj, tensors[f"{prefix}.attn_v.weight"])
        _copy_matrix_to_linear_(self_attn.o_proj, tensors[f"{prefix}.attn_output.weight"])
        _copy_vector_(self_attn.q_norm.weight, dequantize(tensors[f"{prefix}.attn_q_norm.weight"].data, tensors[f"{prefix}.attn_q_norm.weight"].tensor_type))
        _copy_vector_(self_attn.k_norm.weight, dequantize(tensors[f"{prefix}.attn_k_norm.weight"].data, tensors[f"{prefix}.attn_k_norm.weight"].tensor_type))
        loaded += 5 + (1 if self_attn.v_proj is not None else 0)

        mlp = layer.mlp
        _copy_matrix_to_linear_(mlp.gate_proj, tensors[f"{prefix}.ffn_gate.weight"])
        _copy_matrix_to_linear_(mlp.up_proj, tensors[f"{prefix}.ffn_up.weight"])
        _copy_matrix_to_linear_(mlp.down_proj, tensors[f"{prefix}.ffn_down.weight"])
        loaded += 3

        if config.hidden_size_per_layer_input > 0:
            _copy_matrix_to_linear_(layer.per_layer_input_gate, tensors[f"{prefix}.inp_gate.weight"])
            _copy_matrix_to_linear_(layer.per_layer_projection, tensors[f"{prefix}.per_layer_proj.weight"])
            _copy_vector_(layer.post_per_layer_input_norm.weight, dequantize(tensors[f"{prefix}.per_layer_post_norm.weight"].data, tensors[f"{prefix}.per_layer_post_norm.weight"].tensor_type))
            loaded += 3

    model.tie_weights()
    return WeightLoadReport(loaded=loaded, skipped=skipped)


def load_qwen3_5_text_model_weights_from_gguf(
    model: Qwen3_5TextForConditionalGeneration,
    model_dir: str | Path,
) -> WeightLoadReport:
    files, mmproj_file = _resolve_qwen_gguf_files(model_dir)
    reader = GGUFReader(str(files.model_file))
    loaded, skipped = _load_main_qwen35moe_weights_(model, reader)
    if model.config.vision_config is not None:
        if mmproj_file is None:
            raise FileNotFoundError(f"GGUF multimodal model requires an mmproj file in {Path(model_dir).resolve()}")
        loaded += _load_clip_mmproj_weights_(model.model.visual, GGUFReader(str(mmproj_file)))
    model.tie_weights()
    return WeightLoadReport(loaded=loaded, skipped=skipped)
