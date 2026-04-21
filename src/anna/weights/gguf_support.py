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
from anna.model.qwen3_5_text_config import Qwen3_5TextConfig, Qwen3_5TextModelConfig, Qwen3_5TextVisionConfig, RopeParameters, VisionPreprocessorConfig
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
    if arch != "qwen35moe":
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

    text_config = Qwen3_5TextConfig(
        model_type="qwen3_5_text",
        hidden_size=int(_reader_field(reader, f"{arch}.embedding_length")),
        intermediate_size=int(_reader_field(reader, f"{arch}.expert_shared_feed_forward_length")),
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
        moe_intermediate_size=int(_reader_field(reader, f"{arch}.expert_feed_forward_length")),
        shared_expert_intermediate_size=int(_reader_field(reader, f"{arch}.expert_shared_feed_forward_length")),
        num_experts=int(_reader_field(reader, f"{arch}.expert_count")),
        num_experts_per_tok=int(_reader_field(reader, f"{arch}.expert_used_count")),
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
