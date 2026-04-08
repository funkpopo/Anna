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


def _normalize_eos_token_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return (1,)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return (int(value),)


@dataclass(slots=True)
class Gemma4RopeParameters:
    rope_type: str = "default"
    rope_theta: float = 10_000.0
    partial_rotary_factor: float = 1.0
    factor: float = 1.0
    attention_factor: float | None = None
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_max_position_embeddings: int | None = None
    truncate: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Gemma4RopeParameters":
        if not data:
            return cls()
        return cls(
            rope_type=str(data.get("rope_type", "default")),
            rope_theta=float(data.get("rope_theta", 10_000.0)),
            partial_rotary_factor=float(data.get("partial_rotary_factor", 1.0)),
            factor=float(data.get("factor", 1.0)),
            attention_factor=None if data.get("attention_factor") is None else float(data.get("attention_factor")),
            beta_fast=float(data.get("beta_fast", 32.0)),
            beta_slow=float(data.get("beta_slow", 1.0)),
            original_max_position_embeddings=None
            if data.get("original_max_position_embeddings") is None
            else int(data.get("original_max_position_embeddings")),
            truncate=bool(data.get("truncate", True)),
        )


@dataclass(slots=True)
class Gemma4TextConfig:
    model_type: str = "gemma4_text"
    vocab_size: int = 262_144
    hidden_size: int = 2_304
    intermediate_size: int = 9_216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    global_head_dim: int = 512
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_ids: tuple[int, ...] = (1,)
    bos_token_id: int = 2
    tie_word_embeddings: bool = True
    dtype: str = "bfloat16"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 512
    layer_types: list[str] = field(default_factory=list)
    final_logit_softcapping: float | None = None
    use_bidirectional_attention: str | None = None
    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256
    num_global_key_value_heads: int | None = None
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None
    rope_parameters: dict[str, Gemma4RopeParameters] = field(default_factory=dict)

    @property
    def is_moe_model(self) -> bool:
        return bool(self.enable_moe_block and self.num_experts and self.top_k_experts and self.moe_intermediate_size)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Gemma4TextConfig":
        text_config = data.get("text_config", data)
        num_hidden_layers = int(text_config.get("num_hidden_layers", 30))
        layer_types = list(text_config.get("layer_types") or [])
        if not layer_types:
            layer_types = [
                "sliding_attention" if (layer_idx + 1) % 6 else "full_attention"
                for layer_idx in range(num_hidden_layers)
            ]
        if layer_types and layer_types[-1] != "full_attention":
            layer_types[-1] = "full_attention"

        rope_parameters_raw = text_config.get("rope_parameters") or {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        }
        rope_parameters: dict[str, Gemma4RopeParameters] = {}
        for layer_type in set(layer_types):
            params = Gemma4RopeParameters.from_dict(rope_parameters_raw.get(layer_type))
            if params.rope_type == "proportional":
                original_max_position_embeddings = params.original_max_position_embeddings
                if original_max_position_embeddings is None:
                    original_max_position_embeddings = min(8_192, int(text_config.get("max_position_embeddings", 131_072)))
                    params.original_max_position_embeddings = original_max_position_embeddings
                if params.factor <= 1.0 and original_max_position_embeddings > 0:
                    params.factor = max(
                        1.0,
                        float(text_config.get("max_position_embeddings", 131_072)) / float(original_max_position_embeddings),
                    )
            rope_parameters[layer_type] = params

        return cls(
            model_type=str(text_config.get("model_type", "gemma4_text")),
            vocab_size=int(text_config.get("vocab_size", 262_144)),
            hidden_size=int(text_config.get("hidden_size", 2_304)),
            intermediate_size=int(text_config.get("intermediate_size", 9_216)),
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=int(text_config.get("num_attention_heads", 8)),
            num_key_value_heads=int(text_config.get("num_key_value_heads", 4)),
            head_dim=int(text_config.get("head_dim", text_config.get("hidden_size", 2_304) // text_config.get("num_attention_heads", 8))),
            global_head_dim=int(text_config.get("global_head_dim", text_config.get("head_dim", 256))),
            hidden_activation=str(text_config.get("hidden_activation", "gelu_pytorch_tanh")),
            max_position_embeddings=int(text_config.get("max_position_embeddings", 131_072)),
            rms_norm_eps=float(text_config.get("rms_norm_eps", 1e-6)),
            use_cache=bool(text_config.get("use_cache", True)),
            pad_token_id=int(_first_non_null(text_config.get("pad_token_id"), data.get("pad_token_id"), 0)),
            eos_token_ids=_normalize_eos_token_ids(_first_non_null(text_config.get("eos_token_id"), data.get("eos_token_id"), 1)),
            bos_token_id=int(_first_non_null(text_config.get("bos_token_id"), data.get("bos_token_id"), 2)),
            tie_word_embeddings=bool(_first_non_null(text_config.get("tie_word_embeddings"), data.get("tie_word_embeddings"), True)),
            dtype=str(text_config.get("dtype", text_config.get("torch_dtype", data.get("dtype", "bfloat16")))),
            attention_bias=bool(text_config.get("attention_bias", False)),
            attention_dropout=float(text_config.get("attention_dropout", 0.0)),
            sliding_window=int(text_config.get("sliding_window", 512)),
            layer_types=layer_types,
            final_logit_softcapping=None
            if text_config.get("final_logit_softcapping") is None
            else float(text_config.get("final_logit_softcapping")),
            use_bidirectional_attention=text_config.get("use_bidirectional_attention"),
            vocab_size_per_layer_input=int(text_config.get("vocab_size_per_layer_input", text_config.get("vocab_size", 262_144))),
            hidden_size_per_layer_input=int(text_config.get("hidden_size_per_layer_input", 0)),
            num_global_key_value_heads=None
            if text_config.get("num_global_key_value_heads") is None
            else int(text_config.get("num_global_key_value_heads")),
            attention_k_eq_v=bool(text_config.get("attention_k_eq_v", False)),
            num_kv_shared_layers=int(text_config.get("num_kv_shared_layers", 0)),
            enable_moe_block=bool(text_config.get("enable_moe_block", False)),
            use_double_wide_mlp=bool(text_config.get("use_double_wide_mlp", False)),
            num_experts=None if text_config.get("num_experts") is None else int(text_config.get("num_experts")),
            top_k_experts=None if text_config.get("top_k_experts") is None else int(text_config.get("top_k_experts")),
            moe_intermediate_size=None
            if text_config.get("moe_intermediate_size") is None
            else int(text_config.get("moe_intermediate_size")),
            rope_parameters=rope_parameters,
        )


@dataclass(slots=True)
class Gemma4VisionConfig:
    model_type: str = "gemma4_vision"
    dtype: str = "bfloat16"
    hidden_size: int = 768
    intermediate_size: int = 3_072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    global_head_dim: int = 64
    head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    patch_size: int = 16
    pooling_kernel_size: int = 3
    max_position_embeddings: int = 131_072
    position_embedding_size: int = 10_240
    rope_parameters: dict[str, Any] = field(default_factory=dict)
    standardize: bool = False
    use_clipped_linears: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Gemma4VisionConfig | None":
        if not data:
            return None
        return cls(
            model_type=str(data.get("model_type", "gemma4_vision")),
            dtype=str(data.get("dtype", "bfloat16")),
            hidden_size=int(data.get("hidden_size", 768)),
            intermediate_size=int(data.get("intermediate_size", 3_072)),
            num_hidden_layers=int(data.get("num_hidden_layers", 16)),
            num_attention_heads=int(data.get("num_attention_heads", 12)),
            num_key_value_heads=int(data.get("num_key_value_heads", 12)),
            global_head_dim=int(data.get("global_head_dim", data.get("head_dim", 64))),
            head_dim=int(data.get("head_dim", 64)),
            hidden_activation=str(data.get("hidden_activation", "gelu_pytorch_tanh")),
            attention_bias=bool(data.get("attention_bias", False)),
            attention_dropout=float(data.get("attention_dropout", 0.0)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            patch_size=int(data.get("patch_size", 16)),
            pooling_kernel_size=int(data.get("pooling_kernel_size", 3)),
            max_position_embeddings=int(data.get("max_position_embeddings", 131_072)),
            position_embedding_size=int(data.get("position_embedding_size", 10_240)),
            rope_parameters=dict(data.get("rope_parameters") or {"rope_type": "default", "rope_theta": 100.0}),
            standardize=bool(data.get("standardize", False)),
            use_clipped_linears=bool(data.get("use_clipped_linears", True)),
        )


@dataclass(slots=True)
class Gemma4AudioConfig:
    model_type: str = "gemma4_audio"
    dtype: str = "bfloat16"
    hidden_size: int = 1_024
    hidden_act: str = "silu"
    num_attention_heads: int = 8
    num_hidden_layers: int = 12
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_invalid_logits_value: float = -1_000_000_000.0
    attention_logit_cap: float = 50.0
    conv_kernel_size: int = 5
    gradient_clipping: float = 10_000_000_000.0
    output_proj_dims: int = 1_536
    residual_weight: float = 0.5
    rms_norm_eps: float = 1e-6
    subsampling_conv_channels: tuple[int, int] = (128, 32)
    use_clipped_linears: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Gemma4AudioConfig | None":
        if not data:
            return None
        return cls(
            model_type=str(data.get("model_type", "gemma4_audio")),
            dtype=str(data.get("dtype", "bfloat16")),
            hidden_size=int(data.get("hidden_size", 1_024)),
            hidden_act=str(data.get("hidden_act", "silu")),
            num_attention_heads=int(data.get("num_attention_heads", 8)),
            num_hidden_layers=int(data.get("num_hidden_layers", 12)),
            attention_chunk_size=int(data.get("attention_chunk_size", 12)),
            attention_context_left=int(data.get("attention_context_left", 13)),
            attention_context_right=int(data.get("attention_context_right", 0)),
            attention_invalid_logits_value=float(data.get("attention_invalid_logits_value", -1_000_000_000.0)),
            attention_logit_cap=float(data.get("attention_logit_cap", 50.0)),
            conv_kernel_size=int(data.get("conv_kernel_size", 5)),
            gradient_clipping=float(data.get("gradient_clipping", 10_000_000_000.0)),
            output_proj_dims=int(data.get("output_proj_dims", 1_536)),
            residual_weight=float(data.get("residual_weight", 0.5)),
            rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
            subsampling_conv_channels=tuple(int(value) for value in data.get("subsampling_conv_channels", [128, 32])),
            use_clipped_linears=bool(data.get("use_clipped_linears", True)),
        )


@dataclass(slots=True)
class Gemma4Config:
    model_type: str = "gemma4"
    text_config: Gemma4TextConfig = field(default_factory=Gemma4TextConfig)
    vision_config: Gemma4VisionConfig | None = None
    audio_config: Gemma4AudioConfig | None = None
    vision_soft_tokens_per_image: int = 280
    default_max_completion_tokens: int | None = None
    tie_word_embeddings: bool = True
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    audio_token_id: int = 258_881
    boi_token_id: int = 255_999
    eoi_token_id: int = 258_882
    boa_token_id: int = 256_000
    eoa_token_id: int = 258_883

    @classmethod
    def from_dict(
        cls,
        config_data: dict[str, Any],
        *,
        generation_config_data: dict[str, Any] | None = None,
    ) -> "Gemma4Config":
        default_max_completion_tokens_value = _first_non_null(
            config_data.get("max_completion_tokens"),
            config_data.get("max_new_tokens"),
            config_data.get("max_tokens"),
            generation_config_data.get("max_completion_tokens") if generation_config_data else None,
            generation_config_data.get("max_new_tokens") if generation_config_data else None,
            generation_config_data.get("max_tokens") if generation_config_data else None,
        )
        default_max_completion_tokens = (
            None if default_max_completion_tokens_value is None else int(default_max_completion_tokens_value)
        )
        return cls(
            model_type=str(config_data.get("model_type", "gemma4")),
            text_config=Gemma4TextConfig.from_dict(config_data),
            vision_config=Gemma4VisionConfig.from_dict(config_data.get("vision_config")),
            audio_config=Gemma4AudioConfig.from_dict(config_data.get("audio_config")),
            vision_soft_tokens_per_image=int(_first_non_null(config_data.get("vision_soft_tokens_per_image"), 280)),
            default_max_completion_tokens=default_max_completion_tokens,
            tie_word_embeddings=bool(_first_non_null(config_data.get("tie_word_embeddings"), True)),
            image_token_id=int(_first_non_null(config_data.get("image_token_id"), 258_880)),
            video_token_id=int(_first_non_null(config_data.get("video_token_id"), 258_884)),
            audio_token_id=int(_first_non_null(config_data.get("audio_token_id"), 258_881)),
            boi_token_id=int(_first_non_null(config_data.get("boi_token_id"), 255_999)),
            eoi_token_id=int(_first_non_null(config_data.get("eoi_token_id"), 258_882)),
            boa_token_id=int(_first_non_null(config_data.get("boa_token_id"), 256_000)),
            eoa_token_id=int(_first_non_null(config_data.get("eoa_token_id"), config_data.get("eoa_token_index"), 258_883)),
        )

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Gemma4Config":
        model_path = Path(model_dir)
        config_data = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        generation_config_data = None
        generation_config_path = model_path / "generation_config.json"
        if generation_config_path.exists():
            generation_config_data = json.loads(generation_config_path.read_text(encoding="utf-8"))
        return cls.from_dict(config_data, generation_config_data=generation_config_data)
