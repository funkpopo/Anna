from __future__ import annotations

import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.config import Qwen3TextConfig


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


class Qwen3PagedLayerAllocator:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.key_pages: torch.Tensor | None = None
        self.value_pages: torch.Tensor | None = None
        self.free_pages: list[int] = []

    def allocate(
        self,
        num_pages: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> list[int]:
        page_ids: list[int] = []
        while self.free_pages and len(page_ids) < num_pages:
            page_ids.append(self.free_pages.pop())

        missing = num_pages - len(page_ids)
        if missing > 0:
            page_ids.extend(self._grow(missing, key_template=key_template, value_template=value_template))
        return page_ids

    def free(self, page_ids: list[int]) -> None:
        for page_id in reversed(page_ids):
            self.free_pages.append(page_id)

    def _grow(
        self,
        num_pages: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> list[int]:
        old_capacity = 0 if self.key_pages is None else int(self.key_pages.shape[0])
        grow_by = max(num_pages, old_capacity or 16)
        new_capacity = old_capacity + grow_by
        key_shape = (new_capacity, key_template.shape[1], self.block_size, key_template.shape[3])
        value_shape = (new_capacity, value_template.shape[1], self.block_size, value_template.shape[3])
        new_key_pages = key_template.new_empty(key_shape)
        new_value_pages = value_template.new_empty(value_shape)
        if self.key_pages is not None:
            new_key_pages[:old_capacity].copy_(self.key_pages)
        if self.value_pages is not None:
            new_value_pages[:old_capacity].copy_(self.value_pages)
        self.key_pages = new_key_pages
        self.value_pages = new_value_pages

        new_page_ids = list(range(old_capacity, new_capacity))
        allocated = new_page_ids[:num_pages]
        retained_free = new_page_ids[num_pages:]
        for page_id in reversed(retained_free):
            self.free_pages.append(page_id)
        return allocated

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        kwargs: dict[str, object] = {}
        if device is not None:
            kwargs["device"] = device
        if self.key_pages is not None:
            if dtype is not None and self.key_pages.is_floating_point():
                kwargs["dtype"] = dtype
            self.key_pages = self.key_pages.to(**kwargs)
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if self.value_pages is not None:
            if dtype is not None and self.value_pages.is_floating_point():
                kwargs["dtype"] = dtype
            self.value_pages = self.value_pages.to(**kwargs)


class Qwen3PageAllocator:
    def __init__(self, config: Qwen3TextConfig):
        self.config = config
        self.block_size = max(1, int(config.cache_block_size))
        self.layers = [Qwen3PagedLayerAllocator(self.block_size) for _ in range(config.num_hidden_layers)]

    def allocate(
        self,
        layer_idx: int,
        num_pages: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> list[int]:
        return self.layers[layer_idx].allocate(
            num_pages,
            key_template=key_template,
            value_template=value_template,
        )

    def free(self, layer_idx: int, page_ids: list[int]) -> None:
        if page_ids:
            self.layers[layer_idx].free(page_ids)

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Qwen3PageAllocator":
        for layer in self.layers:
            layer.to(device=device, dtype=dtype)
        return self


class Qwen3DynamicCache:
    def __init__(
        self,
        config: Qwen3TextConfig,
        *,
        allocator: Qwen3PageAllocator | None = None,
        batch_size: int = 0,
    ):
        self.config = config
        self.allocator = allocator or Qwen3PageAllocator(config)
        self.conv_states: list[torch.Tensor | None] = [None for _ in range(config.num_hidden_layers)]
        self.recurrent_states: list[torch.Tensor | None] = [None for _ in range(config.num_hidden_layers)]
        self.block_size = max(1, int(config.cache_block_size))
        self.layer_lengths: list[list[int]] = [
            [0 for _ in range(batch_size)]
            for _ in range(config.num_hidden_layers)
        ]
        self.page_tables: list[list[list[int]]] = [
            [[] for _ in range(batch_size)]
            for _ in range(config.num_hidden_layers)
        ]
        self.seen_tokens = 0
        self.rope_deltas: torch.Tensor | None = None
        self._released = False

    @property
    def has_previous_state(self) -> bool:
        return any(state is not None for state in self.conv_states) or any(state is not None for state in self.recurrent_states)

    def _ensure_batch_size(self, batch_size: int) -> None:
        if not self.layer_lengths or not self.layer_lengths[0]:
            self.layer_lengths = [[0 for _ in range(batch_size)] for _ in range(self.config.num_hidden_layers)]
            self.page_tables = [[[] for _ in range(batch_size)] for _ in range(self.config.num_hidden_layers)]
            return
        if len(self.layer_lengths[0]) != batch_size:
            raise ValueError(f"Cache batch size mismatch: expected {len(self.layer_lengths[0])}, got {batch_size}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, append_length, _ = key_states.shape
        self._ensure_batch_size(batch_size)
        past_lengths = torch.tensor(self.layer_lengths[layer_idx], device=key_states.device, dtype=torch.long)

        for batch_idx in range(batch_size):
            current_length = self.layer_lengths[layer_idx][batch_idx]
            required_length = current_length + append_length
            required_blocks = (required_length + self.block_size - 1) // self.block_size
            page_table = self.page_tables[layer_idx][batch_idx]
            if len(page_table) < required_blocks:
                page_table.extend(
                    self.allocator.allocate(
                        layer_idx,
                        required_blocks - len(page_table),
                        key_template=key_states,
                        value_template=value_states,
                    )
                )

            self._write_pages(
                layer_idx=layer_idx,
                page_ids=page_table,
                start_position=current_length,
                key_states=key_states[batch_idx],
                value_states=value_states[batch_idx],
            )
            self.layer_lengths[layer_idx][batch_idx] = required_length

        self.seen_tokens = max(self.get_seq_lengths().tolist(), default=0)
        gathered_key, gathered_value, _ = self._gather_layer_cache(layer_idx)
        return gathered_key, gathered_value, past_lengths

    def _write_pages(
        self,
        *,
        layer_idx: int,
        page_ids: list[int],
        start_position: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        layer = self.allocator.layers[layer_idx]
        if layer.key_pages is None or layer.value_pages is None:
            raise RuntimeError(f"Layer {layer_idx} pages are not initialized.")

        remaining = int(key_states.shape[1])
        src_offset = 0
        current_position = start_position
        while remaining > 0:
            block_idx = current_position // self.block_size
            block_offset = current_position % self.block_size
            take = min(remaining, self.block_size - block_offset)
            page_id = page_ids[block_idx]
            layer.key_pages[page_id, :, block_offset : block_offset + take, :].copy_(
                key_states[:, src_offset : src_offset + take, :]
            )
            layer.value_pages[page_id, :, block_offset : block_offset + take, :].copy_(
                value_states[:, src_offset : src_offset + take, :]
            )
            current_position += take
            src_offset += take
            remaining -= take

    def get_batch_size(self) -> int:
        if self.layer_lengths and self.layer_lengths[0]:
            return len(self.layer_lengths[0])
        for state_group in (self.conv_states, self.recurrent_states):
            for state in state_group:
                if state is not None:
                    return int(state.shape[0])
        if self.rope_deltas is not None:
            return int(self.rope_deltas.shape[0])
        return 0

    def get_seq_length(self, batch_idx: int | None = None) -> int:
        if batch_idx is not None:
            return self._request_seq_length(batch_idx)
        batch_size = self.get_batch_size()
        if batch_size == 0:
            return 0
        if batch_size == 1:
            return self._request_seq_length(0)
        return max(self._request_seq_length(idx) for idx in range(batch_size))

    def get_seq_lengths(self, *, device: torch.device | None = None) -> torch.Tensor:
        lengths = [self._request_seq_length(idx) for idx in range(self.get_batch_size())]
        return torch.tensor(lengths, dtype=torch.long, device=device)

    def _request_seq_length(self, batch_idx: int) -> int:
        for layer_idx in range(self.config.num_hidden_layers):
            length = self.layer_lengths[layer_idx][batch_idx]
            if length > 0:
                return length
        return 0

    def visible_key_cache(self, layer_idx: int) -> torch.Tensor | None:
        key_tensor, _, _ = self._gather_layer_cache(layer_idx)
        return key_tensor

    def visible_value_cache(self, layer_idx: int) -> torch.Tensor | None:
        _, value_tensor, _ = self._gather_layer_cache(layer_idx)
        return value_tensor

    def _gather_layer_cache(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        batch_size = self.get_batch_size()
        if batch_size == 0:
            empty_lengths = torch.zeros(0, dtype=torch.long)
            return None, None, empty_lengths

        visible_lengths = torch.tensor(self.layer_lengths[layer_idx], dtype=torch.long)
        max_length = int(visible_lengths.max().item()) if visible_lengths.numel() > 0 else 0
        if max_length <= 0:
            return None, None, visible_lengths

        layer = self.allocator.layers[layer_idx]
        if layer.key_pages is None or layer.value_pages is None:
            return None, None, visible_lengths

        key_batch = layer.key_pages.new_zeros(
            (batch_size, layer.key_pages.shape[1], max_length, layer.key_pages.shape[3])
        )
        value_batch = layer.value_pages.new_zeros(
            (batch_size, layer.value_pages.shape[1], max_length, layer.value_pages.shape[3])
        )

        for batch_idx, length in enumerate(self.layer_lengths[layer_idx]):
            copied = 0
            for page_id in self.page_tables[layer_idx][batch_idx]:
                if copied >= length:
                    break
                take = min(self.block_size, length - copied)
                key_batch[batch_idx, :, copied : copied + take, :].copy_(layer.key_pages[page_id, :, :take, :])
                value_batch[batch_idx, :, copied : copied + take, :].copy_(layer.value_pages[page_id, :, :take, :])
                copied += take

        return key_batch, value_batch, visible_lengths

    @classmethod
    def stack(cls, caches: list["Qwen3DynamicCache"], config: Qwen3TextConfig) -> "Qwen3DynamicCache":
        if not caches:
            return cls(config)

        if any(cache.allocator is not caches[0].allocator for cache in caches):
            raise ValueError("Cannot stack caches from different allocators.")

        rows: list[Qwen3DynamicCache] = []
        for cache in caches:
            if cache.get_batch_size() <= 1:
                rows.append(cache)
            else:
                rows.extend(cache.split_batch())

        stacked = cls(config, allocator=rows[0].allocator, batch_size=len(rows))
        for layer_idx in range(config.num_hidden_layers):
            stacked.layer_lengths[layer_idx] = [row.layer_lengths[layer_idx][0] for row in rows]
        stacked.seen_tokens = max(stacked.get_seq_lengths().tolist(), default=0)
        stacked.rope_deltas = None if rows[0].rope_deltas is None else torch.cat([row.rope_deltas for row in rows], dim=0)

        for layer_idx in range(config.num_hidden_layers):
            stacked.page_tables[layer_idx] = [list(row.page_tables[layer_idx][0]) for row in rows]

            conv_states = [row.conv_states[layer_idx] for row in rows]
            if all(state is None for state in conv_states):
                stacked.conv_states[layer_idx] = None
            elif any(state is None for state in conv_states):
                raise ValueError(f"Cannot stack partially initialized conv states at layer {layer_idx}")
            else:
                stacked.conv_states[layer_idx] = torch.cat(conv_states, dim=0)

            recurrent_states = [row.recurrent_states[layer_idx] for row in rows]
            if all(state is None for state in recurrent_states):
                stacked.recurrent_states[layer_idx] = None
            elif any(state is None for state in recurrent_states):
                raise ValueError(f"Cannot stack partially initialized recurrent states at layer {layer_idx}")
            else:
                stacked.recurrent_states[layer_idx] = torch.cat(recurrent_states, dim=0)

        return stacked

    def split_batch(self) -> list["Qwen3DynamicCache"]:
        batch_size = self.get_batch_size()
        if batch_size <= 1:
            return [self]

        outputs = [Qwen3DynamicCache(self.config, allocator=self.allocator, batch_size=1) for _ in range(batch_size)]
        for batch_idx, cache in enumerate(outputs):
            for layer_idx in range(self.config.num_hidden_layers):
                cache.layer_lengths[layer_idx][0] = self.layer_lengths[layer_idx][batch_idx]
            cache.seen_tokens = cache.get_seq_length()

        if self.rope_deltas is not None:
            rope_chunks = self.rope_deltas.split(1, dim=0)
            for idx, chunk in enumerate(rope_chunks):
                outputs[idx].rope_deltas = chunk

        for layer_idx in range(self.config.num_hidden_layers):
            for idx in range(batch_size):
                outputs[idx].page_tables[layer_idx][0] = list(self.page_tables[layer_idx][idx])
            if self.conv_states[layer_idx] is not None:
                for idx, chunk in enumerate(self.conv_states[layer_idx].split(1, dim=0)):
                    outputs[idx].conv_states[layer_idx] = chunk.clone()
            if self.recurrent_states[layer_idx] is not None:
                for idx, chunk in enumerate(self.recurrent_states[layer_idx].split(1, dim=0)):
                    outputs[idx].recurrent_states[layer_idx] = chunk.clone()
        return outputs

    def release(self) -> None:
        if self._released:
            return

        for layer_idx in range(self.config.num_hidden_layers):
            for page_ids in self.page_tables[layer_idx]:
                self.allocator.free(layer_idx, page_ids)
                page_ids.clear()
            for batch_idx in range(len(self.layer_lengths[layer_idx])):
                self.layer_lengths[layer_idx][batch_idx] = 0
            self.conv_states[layer_idx] = None
            self.recurrent_states[layer_idx] = None

        self.seen_tokens = 0
        self.rope_deltas = None
        self._released = True

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Qwen3DynamicCache":
        def _move_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            kwargs: dict[str, object] = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None and tensor.is_floating_point():
                kwargs["dtype"] = dtype
            return tensor.to(**kwargs)

        self.conv_states = [_move_tensor(tensor) for tensor in self.conv_states]
        self.recurrent_states = [_move_tensor(tensor) for tensor in self.recurrent_states]
        self.rope_deltas = _move_tensor(self.rope_deltas)
        self.allocator.to(device=device, dtype=dtype)
        return self


class Qwen3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.to(dtype=x.dtype)


class Qwen3RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        hidden_states = self.weight * hidden_states.to(dtype=input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(dtype=input_dtype)


class Qwen3TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3TextConfig):
        super().__init__()
        self.config = config
        inv_freq, attention_scaling = self.compute_default_rope_parameters(config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.attention_scaling = attention_scaling
        self.mrope_section = tuple(config.rope_parameters.mrope_section)

    @staticmethod
    def compute_default_rope_parameters(config: Qwen3TextConfig) -> tuple[torch.Tensor, float]:
        base = config.rope_parameters.rope_theta
        dim = int(config.head_dim * config.rope_parameters.partial_rotary_factor)
        dim = max(2, dim - (dim % 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return inv_freq, 1.0

    def apply_interleaved_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        last_dim = freqs_t.shape[-1]
        for dim, offset in enumerate((1, 2), start=1):
            length = min(last_dim, self.mrope_section[dim] * 3)
            if length <= offset:
                continue
            idx = torch.arange(offset, length, 3, device=freqs.device)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        hidden_states = hidden_states * attention_mask[:, :, None].to(dtype=hidden_states.dtype)
    return hidden_states


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(dtype=weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias=bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(dtype=hidden_states.dtype)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    query = query * (query.shape[-1] ** -0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(dim=-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, key_head_dim, value_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(dtype=value.dtype)
    )
    core_attn_out = torch.zeros_like(value)
    inter_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(inter_mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, value_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(dtype=initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    query = l2norm(query, dim=-1)
    key = l2norm(key, dim=-1)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]

    batch_size, num_heads, sequence_length, key_head_dim = key.shape
    value_head_dim = value.shape[-1]
    query = query * (query.shape[-1] ** -0.5)
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, value_head_dim, device=value.device, dtype=value.dtype)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, key_head_dim, value_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(dtype=value.dtype)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out.transpose(1, 2).contiguous().to(dtype=initial_dtype), last_recurrent_state


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _causal_mask(
        self,
        query_len: int,
        key_len: int,
        past_lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        if query_len == 1:
            return None
        q_positions = past_lengths[:, None] + torch.arange(query_len, device=device)[None, :]
        k_positions = torch.arange(key_len, device=device)[None, None, :]
        return k_positions > q_positions[:, :, None]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_lengths = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.long)
        if past_key_values is not None:
            key_states, value_states, past_lengths = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling
        visible_lengths = past_lengths + seq_len

        causal_mask = self._causal_mask(seq_len, key_states.shape[-2], past_lengths, hidden_states.device)
        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(1), float("-inf"))

        key_positions = torch.arange(key_states.shape[-2], device=hidden_states.device)[None, :]
        visible_mask = key_positions < visible_lengths[:, None]
        attn_scores = attn_scores.masked_fill(~visible_mask[:, None, None, :], float("-inf"))
        if attention_mask is not None and past_key_values is None:
            key_padding_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(~key_padding_mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=query_states.dtype)
        attn_output = torch.matmul(attn_probs, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen3GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3TextConfig, layer_idx: int):
        super().__init__()
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.norm = Qwen3RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(config.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Qwen3DynamicCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state and seq_len == 1
        conv_state = None if cache_params is None else cache_params.conv_states[self.layer_idx]
        recurrent_state = None if cache_params is None else cache_params.recurrent_states[self.layer_idx]

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and conv_state is not None:
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            repeat_factor = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
            )
        else:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
            )

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3TextConfig, intermediate_size: int | None = None):
        super().__init__()
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3SparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3TextConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.offload_experts = False
        self.resident_experts = False
        self.execution_device: torch.device | None = None
        self.cached_experts_per_layer: int = max(self.top_k, 8)
        self._expert_cache: OrderedDict[int, Qwen3MLP] = OrderedDict()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Qwen3MLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.shared_expert = Qwen3MLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def configure_runtime(
        self,
        execution_device: torch.device,
        *,
        offload_experts: bool,
        resident_experts: bool = False,
    ) -> None:
        self.execution_device = execution_device
        self.resident_experts = resident_experts
        self.offload_experts = offload_experts and not resident_experts
        self._expert_cache.clear()

        expert_device = execution_device if resident_experts or not offload_experts else torch.device("cpu")
        for expert in self.experts:
            expert.to(expert_device)

    def _get_cached_expert(self, expert_idx: int) -> Qwen3MLP | None:
        if not self.offload_experts or self.execution_device is None or self.cached_experts_per_layer <= 0:
            return None
        cached = self._expert_cache.get(expert_idx)
        if cached is not None:
            self._expert_cache.move_to_end(expert_idx)
            return cached

        source = self.experts[expert_idx]
        cached = copy.deepcopy(source).to(self.execution_device)
        if len(self._expert_cache) >= self.cached_experts_per_layer:
            self._expert_cache.popitem(last=False)
        self._expert_cache[expert_idx] = cached
        return cached

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        execution_device = hidden_states.device
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(dtype=hidden_states.dtype)

        final_hidden_states = hidden_states.new_zeros((batch_size * sequence_length, hidden_dim))
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        hit_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).flatten()
        gpu_candidates: set[int] = set()
        if self.offload_experts and self.execution_device is not None and hit_experts.numel() > 0:
            usage = expert_mask.sum(dim=(-1, -2))
            gpu_count = min(int(hit_experts.numel()), self.cached_experts_per_layer)
            if gpu_count > 0:
                _, candidate_idx = torch.topk(usage, k=gpu_count)
                gpu_candidates = {int(idx) for idx in candidate_idx.tolist() if int(usage[idx].item()) > 0}

        for expert_idx in hit_experts.tolist():
            expert_layer = self.experts[expert_idx]
            if expert_idx in gpu_candidates:
                cached = self._get_cached_expert(expert_idx)
                if cached is not None:
                    expert_layer = cached
            route_idx, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            expert_device = _module_device(expert_layer)
            current_state = hidden_states.index_select(0, token_idx)
            current_routing_weights = routing_weights[token_idx, route_idx, None]
            if current_state.device != expert_device:
                current_state = current_state.to(device=expert_device)
            if current_routing_weights.device != expert_device:
                current_routing_weights = current_routing_weights.to(device=expert_device)
            current_hidden_states = expert_layer(current_state) * current_routing_weights
            if current_hidden_states.device != execution_device:
                current_hidden_states = current_hidden_states.to(device=execution_device, dtype=hidden_states.dtype)
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(dtype=hidden_states.dtype))

        shared_expert_device = _module_device(self.shared_expert)
        shared_gate_device = _module_device(self.shared_expert_gate)
        shared_input = hidden_states if hidden_states.device == shared_expert_device else hidden_states.to(device=shared_expert_device)
        gate_input = hidden_states if hidden_states.device == shared_gate_device else hidden_states.to(device=shared_gate_device)
        shared_expert_output = self.shared_expert(shared_input)
        shared_expert_output = torch.sigmoid(self.shared_expert_gate(gate_input).to(device=shared_expert_output.device)) * shared_expert_output
        if shared_expert_output.device != execution_device:
            shared_expert_output = shared_expert_output.to(device=execution_device, dtype=hidden_states.dtype)
        final_hidden_states = final_hidden_states + shared_expert_output
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")
        self.mlp = Qwen3SparseMoeBlock(config) if config.uses_sparse_moe(layer_idx) else Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Qwen3DynamicCache | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, cache_params=past_key_values, attention_mask=attention_mask)
        else:
            hidden_states = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        return hidden_states
