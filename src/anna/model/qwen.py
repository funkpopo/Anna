from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.config import Qwen3Config, Qwen3TextConfig, Qwen3VisionConfig
from anna.model.ops import (
    Qwen3DecoderLayer,
    Qwen3DynamicCache,
    Qwen3PageAllocator,
    Qwen3RMSNorm,
    Qwen3SparseMoeBlock,
    Qwen3TextRotaryEmbedding,
    rotate_half,
)


@dataclass(slots=True)
class TextModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Qwen3DynamicCache | None = None


@dataclass(slots=True)
class CausalLMOutput:
    logits: torch.Tensor
    past_key_values: Qwen3DynamicCache | None = None
    rope_deltas: torch.Tensor | None = None


@dataclass(slots=True)
class VisionModelOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor


@dataclass(slots=True)
class MultimodalModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: Qwen3DynamicCache | None = None
    rope_deltas: torch.Tensor | None = None


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _align_tensor_device(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if tensor is None or tensor.device == device:
        return tensor
    return tensor.to(device=device)


class Qwen3TextModel(nn.Module):
    def __init__(self, config: Qwen3TextConfig):
        super().__init__()
        self.config = config
        self.cache_allocator: Qwen3PageAllocator | None = None
        self.execution_device: torch.device | None = None
        padding_idx = config.pad_token_id if 0 <= config.pad_token_id < config.vocab_size else None
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TextRotaryEmbedding(config)

    def configure_runtime(
        self,
        execution_device: torch.device,
        *,
        offload_experts: bool = False,
        offload_token_io: bool = False,
        resident_expert_layers: int = 0,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        expert_quant: str = "none",
        cached_experts_per_layer: int | None = None,
    ) -> None:
        self.execution_device = execution_device
        self.embed_tokens.to(torch.device("cpu") if offload_token_io else execution_device)
        self.rotary_emb.to(execution_device)
        self.norm.to(execution_device)
        resident_sparse_layers_remaining = max(0, int(resident_expert_layers))
        resident_layer_indices = None if resident_expert_layer_indices is None else set(resident_expert_layer_indices)

        for layer_idx, layer in enumerate(self.layers):
            layer.input_layernorm.to(execution_device)
            layer.post_attention_layernorm.to(execution_device)
            if layer.layer_type == "linear_attention":
                layer.linear_attn.to(execution_device)
            else:
                layer.self_attn.to(execution_device)

            if isinstance(layer.mlp, Qwen3SparseMoeBlock):
                layer.mlp.gate.to(execution_device)
                layer.mlp.shared_expert.to(execution_device)
                layer.mlp.shared_expert_gate.to(execution_device)
                if resident_layer_indices is None:
                    resident_experts = offload_experts and resident_sparse_layers_remaining > 0
                else:
                    resident_experts = layer_idx in resident_layer_indices
                if resident_experts:
                    resident_sparse_layers_remaining -= 1
                layer.mlp.configure_runtime(
                    execution_device,
                    offload_experts=offload_experts,
                    resident_experts=resident_experts,
                    expert_quant=expert_quant,
                    cached_experts_per_layer=cached_experts_per_layer,
                )
            else:
                layer.mlp.to(execution_device)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> TextModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            embed_device = self.embed_tokens.weight.device
            if input_ids.device != embed_device:
                input_ids = input_ids.to(device=embed_device)
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3DynamicCache(self.config, allocator=self.cache_allocator)

        execution_device = self.execution_device or _module_device(self.norm)
        if inputs_embeds.device != execution_device:
            inputs_embeds = inputs_embeds.to(device=execution_device)
        if attention_mask is not None and attention_mask.device != execution_device:
            attention_mask = attention_mask.to(device=execution_device)
        if position_ids is not None and position_ids.device != execution_device:
            position_ids = position_ids.to(device=execution_device)

        if position_ids is None:
            seq_len = inputs_embeds.shape[1]
            batch_size = inputs_embeds.shape[0]
            if past_key_values is None or past_key_values.get_batch_size() == 0:
                past_seen_tokens = torch.zeros(batch_size, device=inputs_embeds.device, dtype=torch.long)
            else:
                past_seen_tokens = past_key_values.get_seq_lengths(device=inputs_embeds.device)
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).view(1, -1)
            position_ids = position_ids + past_seen_tokens.view(-1, 1)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        return TextModelOutput(last_hidden_state=hidden_states, past_key_values=past_key_values)


class Qwen3VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


class Qwen3VisionMLP(nn.Module):
    def __init__(self, config: Qwen3VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_state), approximate="tanh"))


class Qwen3VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3VisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)


class Qwen3VisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VisionConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


def apply_rotary_pos_emb_vision(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype = q.dtype
    k_dtype = k.dtype
    q = q.float()
    k = k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(dtype=q_dtype), k_embed.to(dtype=k_dtype)


class Qwen3VisionAttention(nn.Module):
    def __init__(self, config: Qwen3VisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim ** -0.5

    def _attend_chunk(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query.dtype)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        cos, sin = position_embeddings
        outputs: list[torch.Tensor] = []
        for idx in range(cu_seqlens.shape[0] - 1):
            start = int(cu_seqlens[idx].item())
            end = int(cu_seqlens[idx + 1].item())
            outputs.append(self._attend_chunk(hidden_states[start:end], cos[start:end], sin[start:end]))
        return torch.cat(outputs, dim=0)


class Qwen3VisionBlock(nn.Module):
    def __init__(self, config: Qwen3VisionConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VisionAttention(config=config)
        self.mlp = Qwen3VisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VisionModel(nn.Module):
    def __init__(self, config: Qwen3VisionConfig):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.patch_embed = Qwen3VisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen3VisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VisionPatchMerger(config=config)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        total_tokens = sum(t * h * w for t, h, w in grid_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_list:
            merged_h = height // merge_size
            merged_w = width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            token_count = coords.shape[0]
            pos_ids[offset : offset + token_count] = coords
            offset += token_count
        embeddings = freq_table[pos_ids].flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_list]
        grid_hs = [row[1] for row in grid_list]
        grid_ws = [row[2] for row in grid_list]
        device = self.pos_embed.weight.device
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for _, h, w in grid_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute: list[torch.Tensor] = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> VisionModelOutput:
        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        position_embeddings = (rotary_pos_emb.cos(), rotary_pos_emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)

        merged_hidden_states = self.merger(hidden_states)
        return VisionModelOutput(last_hidden_state=hidden_states, pooler_output=merged_hidden_states)


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.visual = Qwen3VisionModel(config.vision_config) if config.vision_config is not None else None
        self.language_model = Qwen3TextModel(config.text_config)

    def configure_runtime(
        self,
        execution_device: torch.device,
        *,
        offload_experts: bool = False,
        offload_vision: bool = False,
        offload_token_io: bool = False,
        resident_expert_layers: int = 0,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        expert_quant: str = "none",
        cached_experts_per_layer: int | None = None,
    ) -> None:
        self.language_model.configure_runtime(
            execution_device,
            offload_experts=offload_experts,
            offload_token_io=offload_token_io,
            resident_expert_layers=resident_expert_layers,
            resident_expert_layer_indices=resident_expert_layer_indices,
            expert_quant=expert_quant,
            cached_experts_per_layer=cached_experts_per_layer,
        )
        if self.visual is not None:
            self.visual.to(torch.device("cpu") if offload_vision else execution_device)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.language_model.embed_tokens = value

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size
        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(llm_grid_h * llm_grid_t)
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        position_temporal = position_temporal * time_interval
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata_device = input_ids.device
        mm_token_type_ids = _align_tensor_device(mm_token_type_ids, metadata_device)
        attention_mask = _align_tensor_device(attention_mask, metadata_device)
        image_grid_thw = _align_tensor_device(image_grid_thw, metadata_device)
        video_grid_thw = _align_tensor_device(video_grid_thw, metadata_device)
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size if self.config.vision_config else 1

        mrope_position_deltas = []
        position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            token_types = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                mask = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[mask]
                token_types = token_types[mask]

            input_type_groups = []
            for key, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
                group = list(group)
                input_type_groups.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            llm_pos_ids_list: list[torch.Tensor] = []
            for modality_type, start_idx, end_idx in input_type_groups:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        1,
                        spatial_merge_size,
                        device=input_ids.device,
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))

        rope_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, rope_deltas

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.LongTensor) -> tuple[torch.Tensor, list[int]]:
        if self.visual is None:
            raise ValueError("The loaded model does not include a vision tower.")
        visual_device = next(self.visual.parameters()).device
        pixel_values = pixel_values.to(dtype=next(self.visual.parameters()).dtype, device=visual_device)
        image_grid_thw = _align_tensor_device(image_grid_thw, visual_device)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return vision_output.pooler_output, split_sizes

    def get_video_features(self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.LongTensor) -> tuple[torch.Tensor, list[int]]:
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor | None = None,
        video_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds)

        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError("Image features and image placeholder token counts do not match.")
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError("Video features and video placeholder token counts do not match.")
        return special_image_mask, special_video_mask

    def compute_3d_position_ids(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> torch.Tensor | None:
        past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
        rope_deltas = None if past_key_values is None else past_key_values.rope_deltas
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

        if can_compute_mrope and (rope_deltas is None or past_key_values_length == 0):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            if past_key_values is not None:
                past_key_values.rope_deltas = rope_deltas
        elif rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
            else:
                position_ids = torch.arange(
                    past_key_values_length,
                    past_key_values_length + seq_length,
                    device=inputs_embeds.device,
                )
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            delta = rope_deltas.repeat_interleave(batch_size // rope_deltas.shape[0], dim=0)
            position_ids = position_ids + delta.to(device=inputs_embeds.device)
        else:
            position_ids = None
        return position_ids

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = None,
    ) -> MultimodalModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")

        if use_cache and past_key_values is None:
            past_key_values = Qwen3DynamicCache(
                self.config.text_config,
                allocator=self.language_model.cache_allocator,
            )

        if inputs_embeds is None:
            embedding_device = self.get_input_embeddings().weight.device
            if input_ids.device != embedding_device:
                input_ids = input_ids.to(device=embedding_device)
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds, _ = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, _ = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            metadata_device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = self.compute_3d_position_ids(
                input_ids=_align_tensor_device(input_ids, metadata_device),
                image_grid_thw=_align_tensor_device(image_grid_thw, metadata_device),
                video_grid_thw=_align_tensor_device(video_grid_thw, metadata_device),
                inputs_embeds=inputs_embeds,
                attention_mask=_align_tensor_device(attention_mask, metadata_device),
                past_key_values=past_key_values,
                mm_token_type_ids=_align_tensor_device(mm_token_type_ids, metadata_device),
            )

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        return MultimodalModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=None if outputs.past_key_values is None else outputs.past_key_values.rope_deltas,
        )


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3TextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def configure_runtime(
        self,
        execution_device: torch.device,
        *,
        offload_experts: bool = False,
        offload_token_io: bool = False,
        resident_expert_layers: int = 0,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        expert_quant: str = "none",
        cached_experts_per_layer: int | None = None,
    ) -> None:
        self.model.configure_runtime(
            execution_device,
            offload_experts=offload_experts,
            offload_token_io=offload_token_io,
            resident_expert_layers=resident_expert_layers,
            resident_expert_layer_indices=resident_expert_layer_indices,
            expert_quant=expert_quant,
            cached_experts_per_layer=cached_experts_per_layer,
        )
        self.lm_head.to(torch.device("cpu") if offload_token_io else execution_device)

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
    ) -> CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        if logits_to_keep is not None and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        lm_head_device = _module_device(self.lm_head)
        if hidden_states.device != lm_head_device:
            hidden_states = hidden_states.to(device=lm_head_device)
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=logits, past_key_values=outputs.past_key_values)


class Qwen3ForConditionalGeneration(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

    def configure_runtime(
        self,
        execution_device: torch.device,
        *,
        offload_experts: bool = False,
        offload_vision: bool = False,
        offload_token_io: bool = False,
        resident_expert_layers: int = 0,
        resident_expert_layer_indices: tuple[int, ...] | None = None,
        expert_quant: str = "none",
        cached_experts_per_layer: int | None = None,
    ) -> None:
        self.model.configure_runtime(
            execution_device,
            offload_experts=offload_experts,
            offload_vision=offload_vision,
            offload_token_io=offload_token_io,
            resident_expert_layers=resident_expert_layers,
            resident_expert_layer_indices=resident_expert_layer_indices,
            expert_quant=expert_quant,
            cached_experts_per_layer=cached_experts_per_layer,
        )
        self.lm_head.to(torch.device("cpu") if offload_token_io else execution_device)

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def forward_text_only(
        self,
        *,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
    ) -> CausalLMOutput:
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        if logits_to_keep is not None and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        lm_head_device = _module_device(self.lm_head)
        if hidden_states.device != lm_head_device:
            hidden_states = hidden_states.to(device=lm_head_device)
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=logits, past_key_values=outputs.past_key_values)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
    ) -> CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state
        if logits_to_keep is not None and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        lm_head_device = _module_device(self.lm_head)
        if hidden_states.device != lm_head_device:
            hidden_states = hidden_states.to(device=lm_head_device)
        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=logits, past_key_values=outputs.past_key_values, rope_deltas=outputs.rope_deltas)
