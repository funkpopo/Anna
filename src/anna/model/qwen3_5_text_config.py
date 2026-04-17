from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _int_from_candidates(*values: Any) -> int:
    value = _first_non_null(*values)
    if value is None:
        raise ValueError("Expected an integer config value, but every candidate was null.")
    return int(value)


@dataclass(slots=True)
class RopeParameters:
    rope_type: str = "default"
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    mrope_interleaved: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RopeParameters":
        if not data:
            return cls()
        return cls(
            rope_type=data.get("rope_type", "default"),
            rope_theta=float(data.get("rope_theta", 10_000_000.0)),
            partial_rotary_factor=float(data.get("partial_rotary_factor", 0.25)),
            mrope_section=tuple(data.get("mrope_section", [11, 11, 10])),
            mrope_interleaved=bool(data.get("mrope_interleaved", True)),
        )


@dataclass(slots=True)
class QuantizationConfig:
    quant_method: str | None = None
    bits: int | None = None
    group_size: int | None = None
    zero_point: bool = False
    data_type: str | None = None
    sym: bool | None = None
    packing_format: str | None = None
    autoround_version: str | None = None
    block_name_to_quantize: tuple[str, ...] = ()
    extra_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    modules_to_not_convert: list[str] = field(default_factory=list)
    version: str | None = None

    @property
    def is_enabled(self) -> bool:
        return self.quant_method is not None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "QuantizationConfig":
        if not data:
            return cls()
        raw_blocks = data.get("block_name_to_quantize")
        if raw_blocks is None:
            block_names: tuple[str, ...] = ()
        elif isinstance(raw_blocks, str):
            block_names = (raw_blocks,)
        else:
            block_names = tuple(str(value) for value in raw_blocks)
        raw_extra = data.get("extra_config") or {}
        return cls(
            quant_method=data.get("quant_method"),
            bits=data.get("bits"),
            group_size=data.get("group_size"),
            zero_point=bool(data.get("zero_point", False)),
            data_type=data.get("data_type"),
            sym=None if data.get("sym") is None else bool(data.get("sym")),
            packing_format=data.get("packing_format"),
            autoround_version=data.get("autoround_version"),
            block_name_to_quantize=block_names,
            extra_config={
                str(module_name): dict(module_config)
                for module_name, module_config in raw_extra.items()
                if isinstance(module_config, dict)
            },
            modules_to_not_convert=list(data.get("modules_to_not_convert", [])),
            version=data.get("version"),
        )


@dataclass(slots=True)
class VisionPreprocessorConfig:
    shortest_edge: int = 56 * 56
    longest_edge: int = 28 * 28 * 1280
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: tuple[float, float, float] = (0.5, 0.5, 0.5)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "VisionPreprocessorConfig":
        if not data:
            return cls()
        size = data.get("size", {})
        return cls(
            shortest_edge=int(size.get("shortest_edge", data.get("min_pixels", 56 * 56))),
            longest_edge=int(size.get("longest_edge", data.get("max_pixels", 28 * 28 * 1280))),
            patch_size=int(data.get("patch_size", 16)),
            temporal_patch_size=int(data.get("temporal_patch_size", 2)),
            merge_size=int(data.get("merge_size", data.get("spatial_merge_size", 2))),
            image_mean=tuple(data.get("image_mean", [0.5, 0.5, 0.5])),
            image_std=tuple(data.get("image_std", [0.5, 0.5, 0.5])),
        )


@dataclass(slots=True)
class Qwen3_5TextConfig:
    model_type: str = "qwen3_5_text"
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    hidden_act: str = "silu"
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248320
    tie_word_embeddings: bool = True
    eos_token_id: int = 248044
    pad_token_id: int = 248044
    dtype: str = "bfloat16"
    cache_block_size: int = 32
    layer_types: list[str] = field(default_factory=list)
    full_attention_interval: int = 4
    rope_parameters: RopeParameters = field(default_factory=RopeParameters)
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    num_experts: int = 0
    num_experts_per_tok: int = 0
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] = field(default_factory=list)

    @property
    def is_moe_model(self) -> bool:
        return self.num_experts > 0 and self.num_experts_per_tok > 0

    def uses_sparse_moe(self, layer_idx: int) -> bool:
        if not self.is_moe_model:
            return False
        if layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % max(1, self.decoder_sparse_step) == 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3_5TextConfig":
        text_config = data.get("text_config", data)
        interval = int(text_config.get("full_attention_interval", 4))
        layer_types = list(text_config.get("layer_types", []))
        num_hidden_layers = int(text_config.get("num_hidden_layers", 24))
        moe_intermediate_size = int(text_config.get("moe_intermediate_size", text_config.get("intermediate_size", 3584)))
        shared_expert_intermediate_size = int(
            text_config.get(
                "shared_expert_intermediate_size",
                text_config.get("intermediate_size", moe_intermediate_size),
            )
        )
        intermediate_size = int(text_config.get("intermediate_size", shared_expert_intermediate_size))
        num_experts = int(text_config.get("num_experts", 0))
        if not layer_types:
            layer_types = [
                "linear_attention" if (layer_idx + 1) % interval else "full_attention"
                for layer_idx in range(num_hidden_layers)
            ]
        eos_token_id = _int_from_candidates(
            text_config.get("eos_token_id"),
            data.get("eos_token_id"),
            248044,
        )
        pad_token_id = _int_from_candidates(
            text_config.get("pad_token_id"),
            data.get("pad_token_id"),
            eos_token_id,
        )

        return cls(
            model_type=text_config.get("model_type", "qwen3_5_text"),
            hidden_size=int(text_config["hidden_size"]),
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=int(text_config["num_attention_heads"]),
            num_key_value_heads=int(text_config.get("num_key_value_heads", text_config["num_attention_heads"])),
            head_dim=int(text_config.get("head_dim", text_config["hidden_size"] // text_config["num_attention_heads"])),
            attention_bias=bool(text_config.get("attention_bias", False)),
            attention_dropout=float(text_config.get("attention_dropout", 0.0)),
            attn_output_gate=bool(text_config.get("attn_output_gate", True)),
            hidden_act=text_config.get("hidden_act", "silu"),
            linear_conv_kernel_dim=int(text_config.get("linear_conv_kernel_dim", 4)),
            linear_key_head_dim=int(text_config.get("linear_key_head_dim", 128)),
            linear_value_head_dim=int(text_config.get("linear_value_head_dim", 128)),
            linear_num_key_heads=int(text_config.get("linear_num_key_heads", 16)),
            linear_num_value_heads=int(text_config.get("linear_num_value_heads", 16)),
            max_position_embeddings=int(text_config.get("max_position_embeddings", 262144)),
            rms_norm_eps=float(text_config.get("rms_norm_eps", 1e-6)),
            vocab_size=int(text_config["vocab_size"]),
            tie_word_embeddings=bool(_first_non_null(text_config.get("tie_word_embeddings"), data.get("tie_word_embeddings"), True)),
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            dtype=str(text_config.get("dtype", text_config.get("torch_dtype", "bfloat16"))),
            cache_block_size=int(text_config.get("cache_block_size", 32)),
            layer_types=layer_types,
            full_attention_interval=interval,
            rope_parameters=RopeParameters.from_dict(text_config.get("rope_parameters")),
            decoder_sparse_step=int(text_config.get("decoder_sparse_step", 1)),
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=int(text_config.get("num_experts_per_tok", 8 if num_experts > 0 else 0)),
            norm_topk_prob=bool(text_config.get("norm_topk_prob", num_experts > 0)),
            router_aux_loss_coef=float(text_config.get("router_aux_loss_coef", 0.001)),
            mlp_only_layers=list(text_config.get("mlp_only_layers", [])),
        )


@dataclass(slots=True)
class Qwen3_5TextVisionConfig:
    depth: int = 12
    hidden_size: int = 768
    hidden_act: str = "gelu_pytorch_tanh"
    in_channels: int = 3
    intermediate_size: int = 3072
    num_heads: int = 12
    num_position_embeddings: int = 2304
    out_hidden_size: int = 1024
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Qwen3_5TextVisionConfig":
        if not data:
            return cls()
        return cls(
            depth=int(data.get("depth", 12)),
            hidden_size=int(data.get("hidden_size", 768)),
            hidden_act=data.get("hidden_act", "gelu_pytorch_tanh"),
            in_channels=int(data.get("in_channels", 3)),
            intermediate_size=int(data.get("intermediate_size", 3072)),
            num_heads=int(data.get("num_heads", 12)),
            num_position_embeddings=int(data.get("num_position_embeddings", 2304)),
            out_hidden_size=int(data.get("out_hidden_size", data.get("hidden_size", 768))),
            patch_size=int(data.get("patch_size", 16)),
            spatial_merge_size=int(data.get("spatial_merge_size", 2)),
            temporal_patch_size=int(data.get("temporal_patch_size", 2)),
        )


@dataclass(slots=True)
class Qwen3_5TextModelConfig:
    model_type: str = "qwen3_5"
    text_config: Qwen3_5TextConfig = field(default_factory=Qwen3_5TextConfig)
    vision_config: Qwen3_5TextVisionConfig | None = None
    preprocessor_config: VisionPreprocessorConfig = field(default_factory=VisionPreprocessorConfig)
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)
    default_max_completion_tokens: int | None = None
    tie_word_embeddings: bool = True
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054

    @classmethod
    def from_dict(
        cls,
        config_data: dict[str, Any],
        *,
        preprocessor_data: dict[str, Any] | None = None,
        generation_config_data: dict[str, Any] | None = None,
    ) -> "Qwen3_5TextModelConfig":
        text_config_data = config_data.get("text_config", {})
        text_config = Qwen3_5TextConfig.from_dict(config_data)
        quantization_config = QuantizationConfig.from_dict(config_data.get("quantization_config"))
        if text_config.is_moe_model and quantization_config.is_enabled:
            # Qwen3.5 MoE router gates are stored as dense float weights in the checkpoint even when
            # expert MLP projections are exported in AutoRound format. Keep them out of placeholder
            # replacement so `mlp.gate.weight` tensors still load into nn.Linear modules.
            quantization_config.extra_config.setdefault(
                r".*\.mlp\.gate$",
                {"bits": 16, "data_type": "fp"},
            )
        default_max_completion_tokens_value = _first_non_null(
            config_data.get("max_completion_tokens"),
            config_data.get("max_new_tokens"),
            config_data.get("max_tokens"),
            generation_config_data.get("max_completion_tokens") if generation_config_data else None,
            generation_config_data.get("max_new_tokens") if generation_config_data else None,
            generation_config_data.get("max_tokens") if generation_config_data else None,
            text_config_data.get("max_completion_tokens"),
            text_config_data.get("max_new_tokens"),
            text_config_data.get("max_tokens"),
        )
        default_max_completion_tokens = (
            None if default_max_completion_tokens_value is None else int(default_max_completion_tokens_value)
        )
        return cls(
            model_type=config_data.get("model_type", "qwen3_5"),
            text_config=text_config,
            vision_config=Qwen3_5TextVisionConfig.from_dict(config_data.get("vision_config")),
            preprocessor_config=VisionPreprocessorConfig.from_dict(preprocessor_data),
            quantization_config=quantization_config,
            default_max_completion_tokens=default_max_completion_tokens,
            tie_word_embeddings=bool(_first_non_null(config_data.get("tie_word_embeddings"), True)),
            image_token_id=_int_from_candidates(config_data.get("image_token_id"), 248056),
            video_token_id=_int_from_candidates(config_data.get("video_token_id"), 248057),
            vision_start_token_id=_int_from_candidates(config_data.get("vision_start_token_id"), 248053),
            vision_end_token_id=_int_from_candidates(config_data.get("vision_end_token_id"), 248054),
        )

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen3_5TextModelConfig":
        model_path = Path(model_dir)
        config_data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        quantization_config_path = model_path / "quantization_config.json"
        if quantization_config_path.exists():
            config_data["quantization_config"] = json.loads(quantization_config_path.read_text(encoding="utf-8"))
        preprocessor_data = None
        preprocessor_path = model_path / "preprocessor_config.json"
        if preprocessor_path.exists():
            preprocessor_data = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        generation_config_data = None
        generation_config_path = model_path / "generation_config.json"
        if generation_config_path.exists():
            generation_config_data = json.loads(generation_config_path.read_text(encoding="utf-8"))
        return cls.from_dict(
            config_data,
            preprocessor_data=preprocessor_data,
            generation_config_data=generation_config_data,
        )
