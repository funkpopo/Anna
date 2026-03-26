from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    activation_scheme: str = "dynamic"
    weight_per_tensor: bool = False
    act_per_tensor: bool = False
    weight_block_size: tuple[int, int] | None = None
    modules_to_not_convert: list[str] = field(default_factory=list)
    version: str | None = None

    @property
    def is_enabled(self) -> bool:
        return self.quant_method is not None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "QuantizationConfig":
        if not data:
            return cls()
        block_size = data.get("weight_block_size")
        return cls(
            quant_method=data.get("quant_method"),
            bits=data.get("bits"),
            group_size=data.get("group_size"),
            zero_point=bool(data.get("zero_point", False)),
            activation_scheme=data.get("activation_scheme", "dynamic"),
            weight_per_tensor=bool(data.get("weight_per_tensor", False)),
            act_per_tensor=bool(data.get("act_per_tensor", False)),
            weight_block_size=None if block_size is None else tuple(block_size),
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
class Qwen3TextConfig:
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
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3TextConfig":
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
            tie_word_embeddings=bool(text_config.get("tie_word_embeddings", data.get("tie_word_embeddings", True))),
            eos_token_id=int(text_config.get("eos_token_id", 248044)),
            pad_token_id=int(text_config.get("pad_token_id", text_config.get("eos_token_id", 248044))),
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
class Qwen3VisionConfig:
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
    def from_dict(cls, data: dict[str, Any] | None) -> "Qwen3VisionConfig":
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
class Qwen3Config:
    model_type: str = "qwen3_5"
    text_config: Qwen3TextConfig = field(default_factory=Qwen3TextConfig)
    vision_config: Qwen3VisionConfig | None = None
    preprocessor_config: VisionPreprocessorConfig = field(default_factory=VisionPreprocessorConfig)
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig)
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
    ) -> "Qwen3Config":
        return cls(
            model_type=config_data.get("model_type", "qwen3_5"),
            text_config=Qwen3TextConfig.from_dict(config_data),
            vision_config=Qwen3VisionConfig.from_dict(config_data.get("vision_config")),
            preprocessor_config=VisionPreprocessorConfig.from_dict(preprocessor_data),
            quantization_config=QuantizationConfig.from_dict(config_data.get("quantization_config")),
            tie_word_embeddings=bool(config_data.get("tie_word_embeddings", True)),
            image_token_id=int(config_data.get("image_token_id", 248056)),
            video_token_id=int(config_data.get("video_token_id", 248057)),
            vision_start_token_id=int(config_data.get("vision_start_token_id", 248053)),
            vision_end_token_id=int(config_data.get("vision_end_token_id", 248054)),
        )

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen3Config":
        model_path = Path(model_dir)
        config_data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        preprocessor_data = None
        preprocessor_path = model_path / "preprocessor_config.json"
        if preprocessor_path.exists():
            preprocessor_data = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        return cls.from_dict(config_data, preprocessor_data=preprocessor_data)
