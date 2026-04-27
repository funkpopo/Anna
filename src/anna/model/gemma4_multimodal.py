from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.gemma4_config import Gemma4AudioConfig, Gemma4TextConfig, Gemma4VisionConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _apply_multimodal_activation(hidden_states: torch.Tensor, activation: str) -> torch.Tensor:
    normalized = activation.strip().lower()
    if normalized in {"gelu_pytorch_tanh", "gelu_new", "gelu_fast"}:
        return F.gelu(hidden_states, approximate="tanh")
    if normalized == "gelu":
        return F.gelu(hidden_states)
    if normalized in {"silu", "swish"}:
        return F.silu(hidden_states)
    if normalized == "relu":
        return F.relu(hidden_states)
    raise ValueError(f"Unsupported Gemma4 multimodal activation: {activation}")


class Gemma4ClippableLinear(nn.Module):
    def __init__(
        self,
        config: Gemma4VisionConfig | Gemma4AudioConfig,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.use_clipped_linears = bool(getattr(config, "use_clipped_linears", True))
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.use_clipped_linears:
            self.register_buffer("input_min", torch.tensor(-float("inf")))
            self.register_buffer("input_max", torch.tensor(float("inf")))
            self.register_buffer("output_min", torch.tensor(-float("inf")))
            self.register_buffer("output_max", torch.tensor(float("inf")))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.input_min, self.input_max)
        hidden_states = self.linear(hidden_states)
        if self.use_clipped_linears:
            hidden_states = torch.clamp(hidden_states, self.output_min, self.output_max)
        return hidden_states


class Gemma4MMRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mean_squared = hidden_states.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        normed = hidden_states.float() * torch.pow(mean_squared, -0.5)
        if self.weight is not None:
            normed = normed * self.weight.float()
        return normed.to(dtype=hidden_states.dtype)


@dataclass(slots=True)
class Gemma4AudioModelOutput:
    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor


class Gemma4AudioRelPositionalEncoding(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.max_past_horizon = int(config.attention_context_left) - 1
        self.max_future_horizon = int(config.attention_context_right)
        min_timescale = 1.0
        max_timescale = 10_000.0
        num_timescales = self.hidden_size // 2
        log_timescale_increment = math.log(max_timescale / min_timescale) / max(max(1, num_timescales - 1), 1)
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)
        self.register_buffer("inv_timescales", inv_timescales.unsqueeze(0).unsqueeze(0), persistent=False)

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(
            self.max_past_horizon,
            -self.max_future_horizon - 1,
            -1,
            device=hidden_states.device,
        )[..., None]
        scaled_time = position_ids * self.inv_timescales.to(device=hidden_states.device)
        pos_embed = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        return pos_embed.to(dtype=hidden_states.dtype)


class Gemma4AudioAttention(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        del layer_idx
        self.config = config
        self.attention_logits_soft_cap = float(config.attention_logit_cap)
        self.head_dim = int(config.hidden_size) // int(config.num_attention_heads)
        self.num_heads = int(config.num_attention_heads)
        self.q_scale = (self.head_dim**-0.5) / math.log(2.0)
        self.k_scale = math.log1p(math.e) / math.log(2.0)
        self.chunk_size = int(config.attention_chunk_size)
        self.max_past_horizon = int(config.attention_context_left) - 1
        self.max_future_horizon = int(config.attention_context_right)
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon
        hidden_size = int(config.hidden_size)
        self.q_proj = Gemma4ClippableLinear(config, hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Gemma4ClippableLinear(config, hidden_size, self.num_heads * self.head_dim)
        self.v_proj = Gemma4ClippableLinear(config, hidden_size, self.num_heads * self.head_dim)
        self.post = Gemma4ClippableLinear(config, hidden_size, hidden_size)
        self.relative_k_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))
        self.register_buffer("softcap", torch.tensor(self.attention_logits_soft_cap), persistent=False)

    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim).contiguous()

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(
            hidden_states,
            (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1),
        )
        hidden_states = hidden_states.unfold(1, self.context_size, self.chunk_size)
        hidden_states = torch.movedim(hidden_states, -1, 2)
        return hidden_states.contiguous()

    def _rel_shift(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_blocks, block_size, position_length = hidden_states.shape
        hidden_states = F.pad(hidden_states, (0, self.context_size + 1 - position_length))
        hidden_states = hidden_states.view(batch_size, num_heads, num_blocks, block_size * (self.context_size + 1))
        hidden_states = hidden_states[..., : block_size * self.context_size]
        return hidden_states.view(batch_size, num_heads, num_blocks, block_size, self.context_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)

        query_states = query_states * self.q_scale * F.softplus(self.per_dim_scale)
        key_states = key_states * self.k_scale

        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)

        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap

        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(
                ~attention_mask.to(device=attn_weights.device, dtype=torch.bool),
                float(self.config.attention_invalid_logits_value),
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=value_states.dtype)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, num_blocks * self.chunk_size, -1)
        attn_output = attn_output[:, :seq_length].contiguous()
        return self.post(attn_output.to(dtype=self.post.linear.weight.dtype))


class Gemma4AudioSubSampleConvProjectionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_eps: float):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=1,
            bias=False,
        )
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps, elementwise_affine=True, bias=False)
        self.act = nn.ReLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if mask is not None:
            mask = mask.to(device=hidden_states.device)
            hidden_states = hidden_states * mask[:, None, :, None]

        hidden_states = self.conv(hidden_states.to(dtype=self.conv.weight.dtype))
        hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        hidden_states = self.act(hidden_states)

        if mask is not None:
            mask = mask[:, ::2]
        return hidden_states, mask


class Gemma4AudioSubSampleConvProjection(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        conv_channels = tuple(int(value) for value in config.subsampling_conv_channels)
        self.layer0 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=1,
            out_channels=conv_channels[0],
            norm_eps=float(config.rms_norm_eps),
        )
        self.layer1 = Gemma4AudioSubSampleConvProjectionLayer(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            norm_eps=float(config.rms_norm_eps),
        )
        proj_input_dim = (conv_channels[0] // 4) * conv_channels[1]
        self.input_proj_linear = nn.Linear(proj_input_dim, int(config.hidden_size), bias=False)

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = input_features.unsqueeze(1)
        hidden_states, mask = self.layer0(hidden_states, input_features_mask)
        hidden_states, mask = self.layer1(hidden_states, mask)

        batch_size, _, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous().reshape(batch_size, seq_len, -1)
        projected = self.input_proj_linear(hidden_states.to(dtype=self.input_proj_linear.weight.dtype))
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=projected.device)
        else:
            mask = mask.to(device=projected.device, dtype=torch.bool)
        return projected, mask


class Gemma4AudioFeedForward(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.ffw_layer_1 = Gemma4ClippableLinear(config, hidden_size, hidden_size * 4)
        self.ffw_layer_2 = Gemma4ClippableLinear(config, hidden_size * 4, hidden_size)
        self.pre_layer_norm = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.post_layer_norm = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.hidden_activation = str(config.hidden_act)
        self.gradient_clipping = float(config.gradient_clipping)
        self.post_layer_scale = float(config.residual_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gradient_clipping = min(self.gradient_clipping, torch.finfo(self.ffw_layer_1.linear.weight.dtype).max)
        residual = hidden_states
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.ffw_layer_1(hidden_states)
        hidden_states = _apply_multimodal_activation(hidden_states, self.hidden_activation)
        hidden_states = self.ffw_layer_2(hidden_states)
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.post_layer_norm(hidden_states)
        hidden_states = hidden_states * self.post_layer_scale
        return hidden_states + residual


class Gemma4AudioCausalConv1d(nn.Conv1d):
    @property
    def left_pad(self) -> int:
        effective_kernel_size = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        return effective_kernel_size - self.stride[0]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, (self.left_pad, 0))
        return super().forward(hidden_states)


class Gemma4AudioLightConv1d(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.linear_start = Gemma4ClippableLinear(config, hidden_size, hidden_size * 2)
        self.linear_end = Gemma4ClippableLinear(config, hidden_size, hidden_size)
        self.depthwise_conv1d = Gemma4AudioCausalConv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=int(config.conv_kernel_size),
            groups=hidden_size,
            bias=False,
        )
        self.pre_layer_norm = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps), with_scale=True)
        self.conv_norm = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps), with_scale=True)
        self.hidden_activation = str(config.hidden_act)
        self.gradient_clipping = float(config.gradient_clipping)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.pre_layer_norm(hidden_states)
        hidden_states = self.linear_start(hidden_states)
        hidden_states = F.glu(hidden_states, dim=-1)
        hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)
        gradient_clipping = min(self.gradient_clipping, torch.finfo(self.linear_start.linear.weight.dtype).max)
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.conv_norm(hidden_states)
        hidden_states = _apply_multimodal_activation(hidden_states, self.hidden_activation)
        hidden_states = self.linear_end(hidden_states)
        return hidden_states + residual


class Gemma4AudioLayer(nn.Module):
    def __init__(self, config: Gemma4AudioConfig, layer_idx: int):
        super().__init__()
        self.feed_forward1 = Gemma4AudioFeedForward(config)
        self.feed_forward2 = Gemma4AudioFeedForward(config)
        self.self_attn = Gemma4AudioAttention(config, layer_idx)
        self.lconv1d = Gemma4AudioLightConv1d(config)
        hidden_size = int(config.hidden_size)
        self.norm_pre_attn = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.norm_post_attn = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.norm_out = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.gradient_clipping = float(config.gradient_clipping)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        gradient_clipping = min(self.gradient_clipping, torch.finfo(self.norm_pre_attn.weight.dtype).max)

        hidden_states = self.feed_forward1(hidden_states)
        residual = hidden_states

        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_pre_attn(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        hidden_states = self.norm_post_attn(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states = self.lconv1d(hidden_states)
        hidden_states = self.feed_forward2(hidden_states)
        hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
        return self.norm_out(hidden_states)


class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.patch_size = int(config.patch_size)
        self.position_embedding_size = int(config.position_embedding_size)
        self.input_proj = nn.Linear(3 * self.patch_size**2, int(config.hidden_size), bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, int(config.hidden_size))
        )

    def _position_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        clamped_positions = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(clamped_positions, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(self.position_embedding_table)
        position_embeddings = one_hot @ self.position_embedding_table
        position_embeddings = position_embeddings.sum(dim=1)
        return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden_states = self.input_proj(pixel_values.to(self.input_proj.weight.dtype))
        return hidden_states + self._position_embeddings(pixel_position_ids, padding_positions)


class Gemma4VisionPooler(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.root_hidden_size = float(config.hidden_size) ** 0.5

    def _avg_pool_by_positions(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = int(hidden_states.shape[1])
        kernel = int((input_seq_len // output_length) ** 0.5)
        if kernel * kernel * output_length != input_seq_len:
            raise ValueError(f"Cannot pool {input_seq_len} vision patches into {output_length} soft tokens.")

        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_indices = torch.div(clamped_positions, kernel, rounding_mode="floor")
        kernel_indices = kernel_indices[..., 0] + ((max_x // kernel) * kernel_indices[..., 1])
        weights = F.one_hot(kernel_indices.long(), output_length).float() / float(kernel * kernel)
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(dtype=hidden_states.dtype), mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        pooler_mask = ~padding_positions
        if int(hidden_states.shape[1]) != output_length:
            hidden_states, pooler_mask = self._avg_pool_by_positions(
                hidden_states,
                pixel_position_ids,
                output_length,
            )
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, pooler_mask


class Gemma4VisionMLP(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_proj = Gemma4ClippableLinear(config, hidden_size, intermediate_size)
        self.up_proj = Gemma4ClippableLinear(config, hidden_size, intermediate_size)
        self.down_proj = Gemma4ClippableLinear(config, intermediate_size, hidden_size)
        self.hidden_activation = str(config.hidden_activation).strip().lower()

    def _act(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _apply_multimodal_activation(hidden_states, self.hidden_activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self._act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Gemma4VisionRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        base = float((config.rope_parameters or {}).get("rope_theta", 100.0))
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / float(spatial_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        all_cos: list[torch.Tensor] = []
        all_sin: list[torch.Tensor] = []
        for dim_index in range(2):
            dim_position_ids = position_ids[:, :, dim_index][:, None, :].float()
            freqs = (inv_freq.float() @ dim_position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos())
            all_sin.append(emb.sin())
        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin


def _apply_multidimensional_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    ndim = int(position_ids.shape[-1])
    channels = int(x.shape[-1])
    rotated_per_dim = 2 * (channels // (2 * ndim))
    if rotated_per_dim <= 0:
        raise ValueError("Invalid Gemma4 vision rotary dimensions.")
    split_sizes = [rotated_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    rotated = []
    for x_part, cos_part, sin_part in zip(x_parts, cos_parts, sin_parts):
        rotated.append((x_part * cos_part.unsqueeze(1)) + (_rotate_half(x_part) * sin_part.unsqueeze(1)))
    return torch.cat(rotated, dim=-1)


class Gemma4VisionAttention(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        del layer_idx
        self.head_dim = int(config.head_dim)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = float(config.attention_dropout)
        hidden_size = int(config.hidden_size)
        self.q_proj = Gemma4ClippableLinear(config, hidden_size, self.num_attention_heads * self.head_dim)
        self.k_proj = Gemma4ClippableLinear(config, hidden_size, self.num_key_value_heads * self.head_dim)
        self.v_proj = Gemma4ClippableLinear(config, hidden_size, self.num_key_value_heads * self.head_dim)
        self.o_proj = Gemma4ClippableLinear(config, self.num_attention_heads * self.head_dim, hidden_size)
        self.q_norm = Gemma4MMRMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.k_norm = Gemma4MMRMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.v_norm = Gemma4MMRMSNorm(self.head_dim, eps=float(config.rms_norm_eps), with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_ids is None:
            raise ValueError("Gemma4 vision attention requires pixel position ids.")
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        value_states = self.v_norm(value_states)

        cos, sin = position_embeddings
        query_states = _apply_multidimensional_rope(query_states, cos, sin, position_ids)
        key_states = _apply_multidimensional_rope(key_states, cos, sin, position_ids)

        query_states = query_states.transpose(1, 2)
        key_states = _repeat_kv(key_states.transpose(1, 2), self.num_key_value_groups)
        value_states = _repeat_kv(value_states.transpose(1, 2), self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            valid = attention_mask.to(dtype=torch.bool)
            key_mask = valid[:, None, None, :]
            attn_weights = attn_weights.masked_fill(~key_mask, torch.finfo(attn_weights.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=value_states.dtype)
        if self.attention_dropout > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=True)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        if attention_mask is not None:
            attn_output = attn_output.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        return self.o_proj(attn_output)


class Gemma4VisionEncoderLayer(nn.Module):
    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.self_attn = Gemma4VisionAttention(config, layer_idx)
        self.mlp = Gemma4VisionMLP(config)
        self.input_layernorm = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.post_attention_layernorm = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.pre_feedforward_layernorm = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.post_feedforward_layernorm = Gemma4MMRMSNorm(hidden_size, eps=float(config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class Gemma4VisionEncoder(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.rotary_emb = Gemma4VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [Gemma4VisionEncoderLayer(config, layer_idx) for layer_idx in range(int(config.num_hidden_layers))]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, pixel_position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=pixel_position_ids,
            )
        return hidden_states


class Gemma4VisionModel(nn.Module):
    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.patch_embedder = Gemma4VisionPatchEmbedder(config)
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)
        self.standardize = bool(config.standardize)
        if self.standardize:
            self.register_buffer("std_bias", torch.empty(int(config.hidden_size)))
            self.register_buffer("std_scale", torch.empty(int(config.hidden_size)))

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: int,
    ) -> torch.Tensor:
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
        hidden_states = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )
        hidden_states, pooler_mask = self.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        hidden_states = hidden_states[pooler_mask]
        if self.standardize:
            hidden_states = (hidden_states - self.std_bias) * self.std_scale
        return hidden_states


class Gemma4AudioModel(nn.Module):
    def __init__(self, config: Gemma4AudioConfig):
        super().__init__()
        self.config = config
        self.subsample_conv_projection = Gemma4AudioSubSampleConvProjection(config)
        self.rel_pos_enc = Gemma4AudioRelPositionalEncoding(config)
        self.layers = nn.ModuleList(
            [Gemma4AudioLayer(config, layer_idx) for layer_idx in range(int(config.num_hidden_layers))]
        )
        self.output_proj = nn.Linear(int(config.hidden_size), int(config.output_proj_dims), bias=True)

    def _convert_4d_mask_to_blocked_5d(self, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = attention_mask.shape
        chunk_size = int(self.config.attention_chunk_size)
        max_past_horizon = int(self.config.attention_context_left) - 1
        max_future_horizon = int(self.config.attention_context_right)
        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        padded_seq_len = num_blocks * chunk_size
        pad_amount = padded_seq_len - seq_len

        attention_mask = F.pad(attention_mask, (0, pad_amount, 0, pad_amount), value=False)
        blocked = attention_mask.reshape(batch_size, 1, num_blocks, chunk_size, padded_seq_len)
        blocked = F.pad(blocked, (max_past_horizon, max_future_horizon), value=False)

        device = attention_mask.device
        block_starts = torch.arange(num_blocks, device=device) * chunk_size
        offsets = torch.arange(chunk_size + max_past_horizon + max_future_horizon, device=device)
        kv_indices = block_starts[:, None] + offsets[None, :]
        kv_indices = kv_indices[None, None, :, None, :].expand(batch_size, 1, -1, chunk_size, -1)
        return blocked.gather(-1, kv_indices)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> Gemma4AudioModelOutput:
        hidden_states, output_mask = self.subsample_conv_projection(input_features, attention_mask)
        if output_mask is None:
            output_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            output_mask = output_mask.to(device=hidden_states.device, dtype=torch.bool)

        position_embeddings = self.rel_pos_enc(hidden_states)
        bidirectional_mask = output_mask[:, None, :, None] & output_mask[:, None, None, :]
        blocked_attention_mask = self._convert_4d_mask_to_blocked_5d(bidirectional_mask)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=blocked_attention_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.output_proj(hidden_states.to(dtype=self.output_proj.weight.dtype))
        return Gemma4AudioModelOutput(last_hidden_state=hidden_states, attention_mask=output_mask)


class Gemma4MultimodalEmbedder(nn.Module):
    def __init__(
        self,
        multimodal_config: Gemma4VisionConfig | Gemma4AudioConfig,
        text_config: Gemma4TextConfig,
    ):
        super().__init__()
        multimodal_hidden_size = int(getattr(multimodal_config, "output_proj_dims", multimodal_config.hidden_size))
        text_hidden_size = int(text_config.hidden_size)
        self.embedding_pre_projection_norm = Gemma4MMRMSNorm(
            multimodal_hidden_size,
            eps=float(multimodal_config.rms_norm_eps),
            with_scale=False,
        )
        self.embedding_projection = nn.Linear(multimodal_hidden_size, text_hidden_size, bias=False)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(inputs_embeds))
