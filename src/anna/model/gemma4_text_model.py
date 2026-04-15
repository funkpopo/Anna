from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.gemma4_config import Gemma4Config, Gemma4RopeParameters, Gemma4TextConfig
from anna.model.gemma4_multimodal import Gemma4AudioModel, Gemma4MultimodalEmbedder, Gemma4VisionModel
from anna.model.fused_ops import run_gqa_decode_fused, run_qk_norm_rotary_fused_ex, run_rmsnorm_fused_ex
from anna.model.ops import (
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    grouped_query_attention,
    rotate_half,
)
from anna.model.turboquant import TurboQuantTensorRow


@dataclass(slots=True)
class Gemma4TextModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: "Gemma4DynamicCache | None" = None


@dataclass(slots=True)
class Gemma4CausalLMOutput:
    logits: torch.Tensor
    past_key_values: "Gemma4DynamicCache | None" = None


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _normalize_add_lengths(batch_size: int, append_lengths: torch.Tensor | int) -> list[int]:
    if isinstance(append_lengths, int):
        return [int(append_lengths) for _ in range(batch_size)]
    if append_lengths.ndim == 0:
        value = int(append_lengths.item())
        return [value for _ in range(batch_size)]
    return [int(item) for item in append_lengths.tolist()]


def _apply_hidden_activation(hidden_states: torch.Tensor, activation: str) -> torch.Tensor:
    normalized = activation.strip().lower()
    if normalized in {"gelu_pytorch_tanh", "gelu_new", "gelu_fast"}:
        return F.gelu(hidden_states, approximate="tanh")
    if normalized == "gelu":
        return F.gelu(hidden_states)
    if normalized in {"silu", "swish"}:
        return F.silu(hidden_states)
    raise ValueError(f"Unsupported Gemma4 activation: {activation}")


def _apply_rotary_pos_emb_single(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    hidden_rot, hidden_pass = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    hidden_rot = ((hidden_rot * cos) + (rotate_half(hidden_rot) * sin)).to(dtype=hidden_states.dtype)
    return torch.cat((hidden_rot, hidden_pass), dim=-1)


Gemma4CacheRow = torch.Tensor | TurboQuantTensorRow


def _cache_row_length(row: Gemma4CacheRow | None) -> int:
    if row is None:
        return 0
    if isinstance(row, TurboQuantTensorRow):
        return row.length
    return int(row.shape[1])


def _materialize_cache_row(
    row: Gemma4CacheRow | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if row is None:
        return None
    if isinstance(row, TurboQuantTensorRow):
        return row.materialize(device=device, dtype=dtype)
    if row.device != device or row.dtype != dtype:
        return row.to(device=device, dtype=dtype)
    return row


def _clone_cache_row(row: Gemma4CacheRow | None) -> Gemma4CacheRow | None:
    if row is None:
        return None
    if isinstance(row, TurboQuantTensorRow):
        return row.clone()
    return row.clone()


class _Gemma4SharedLayerState:
    def __init__(
        self,
        *,
        key_rows: list[Gemma4CacheRow],
        value_rows: list[Gemma4CacheRow],
        visible_lengths: list[int],
    ) -> None:
        self.key_rows = key_rows
        self.value_rows = value_rows
        self.visible_lengths = [int(length) for length in visible_lengths]

    def clone(self) -> "_Gemma4SharedLayerState":
        return _Gemma4SharedLayerState(
            key_rows=[_clone_cache_row(row) for row in self.key_rows],
            value_rows=[_clone_cache_row(row) for row in self.value_rows],
            visible_lengths=list(self.visible_lengths),
        )

    def select_batch(self, batch_idx: int) -> "_Gemma4SharedLayerState":
        return _Gemma4SharedLayerState(
            key_rows=[_clone_cache_row(self.key_rows[batch_idx])],
            value_rows=[_clone_cache_row(self.value_rows[batch_idx])],
            visible_lengths=[self.visible_lengths[batch_idx]],
        )

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.key_rows or not self.value_rows:
            raise ValueError("Gemma4 shared-layer state is empty.")
        materialized_keys = [
            _materialize_cache_row(
                row,
                device=row.device if isinstance(row, torch.Tensor) else row.device or torch.device("cpu"),
                dtype=row.dtype if isinstance(row, torch.Tensor) else row.dtype or torch.float32,
            )
            for row in self.key_rows
        ]
        materialized_values = [
            _materialize_cache_row(
                row,
                device=row.device if isinstance(row, torch.Tensor) else row.device or torch.device("cpu"),
                dtype=row.dtype if isinstance(row, torch.Tensor) else row.dtype or torch.float32,
            )
            for row in self.value_rows
        ]
        assert all(row is not None for row in materialized_keys)
        assert all(row is not None for row in materialized_values)
        key_rows = [row for row in materialized_keys if row is not None]
        value_rows = [row for row in materialized_values if row is not None]
        device = key_rows[0].device
        key_dtype = key_rows[0].dtype
        value_dtype = value_rows[0].dtype
        padded_key = Gemma4DynamicCache._pad_rows(key_rows, device=device, dtype=key_dtype)
        padded_value = Gemma4DynamicCache._pad_rows(value_rows, device=device, dtype=value_dtype)
        visible_lengths = torch.tensor(self.visible_lengths, dtype=torch.long, device=device)
        return padded_key, padded_value, visible_lengths

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "_Gemma4SharedLayerState":
        moved_key_rows: list[Gemma4CacheRow] = []
        moved_value_rows: list[Gemma4CacheRow] = []
        for row in self.key_rows:
            if isinstance(row, TurboQuantTensorRow):
                moved_key_rows.append(row.clone().to(device=device, dtype=dtype))
            else:
                kwargs: dict[str, object] = {}
                if device is not None:
                    kwargs["device"] = device
                if dtype is not None:
                    kwargs["dtype"] = dtype
                moved_key_rows.append(row.to(**kwargs) if kwargs else row)
        for row in self.value_rows:
            if isinstance(row, TurboQuantTensorRow):
                moved_value_rows.append(row.clone().to(device=device, dtype=dtype))
            else:
                kwargs = {}
                if device is not None:
                    kwargs["device"] = device
                if dtype is not None:
                    kwargs["dtype"] = dtype
                moved_value_rows.append(row.to(**kwargs) if kwargs else row)
        return _Gemma4SharedLayerState(
            key_rows=moved_key_rows,
            value_rows=moved_value_rows,
            visible_lengths=list(self.visible_lengths),
        )


class Gemma4DynamicCache:
    def __init__(
        self,
        config: Gemma4TextConfig,
        *,
        batch_size: int = 0,
        kv_cache_quantization: str = "none",
        kv_cache_quant_bits: int = 4,
        kv_cache_residual_len: int = 128,
    ):
        self.config = config
        self.kv_cache_quantization = kv_cache_quantization.strip().lower()
        if self.kv_cache_quantization not in {"none", "turboquant"}:
            raise ValueError(f"Unsupported Gemma4 KV-cache quantization mode: {kv_cache_quantization}")
        self.kv_cache_quant_bits = int(kv_cache_quant_bits)
        if self.kv_cache_quant_bits not in {3, 4}:
            raise ValueError(f"Unsupported Gemma4 TurboQuant bit-width: {kv_cache_quant_bits}")
        self.kv_cache_residual_len = max(1, int(kv_cache_residual_len))
        self.request_lengths: list[int] = [0 for _ in range(batch_size)]
        self.key_rows: list[list[Gemma4CacheRow | None]] = [
            [None for _ in range(batch_size)] for _ in range(config.num_hidden_layers)
        ]
        self.value_rows: list[list[Gemma4CacheRow | None]] = [
            [None for _ in range(batch_size)] for _ in range(config.num_hidden_layers)
        ]
        self.shared_layers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | _Gemma4SharedLayerState] = {}
        self.reserved_seq_capacity = 0
        self._released = False

    def reserve_sequence_capacity(self, seq_length: int) -> None:
        self.reserved_seq_capacity = max(0, int(seq_length))

    def _ensure_batch_size(self, batch_size: int) -> None:
        if not self.request_lengths:
            self.request_lengths = [0 for _ in range(batch_size)]
            self.key_rows = [[None for _ in range(batch_size)] for _ in range(self.config.num_hidden_layers)]
            self.value_rows = [[None for _ in range(batch_size)] for _ in range(self.config.num_hidden_layers)]
            return
        if len(self.request_lengths) != batch_size:
            raise ValueError(f"Gemma4 cache batch size mismatch: expected {len(self.request_lengths)}, got {batch_size}")

    def _use_turboquant_for_layer(self, layer_idx: int) -> bool:
        return self.kv_cache_quantization == "turboquant" and self.config.layer_types[layer_idx] == "full_attention"

    def _new_turboquant_row(self) -> TurboQuantTensorRow:
        return TurboQuantTensorRow(bits=self.kv_cache_quant_bits, residual_len=self.kv_cache_residual_len)

    def _ensure_turboquant_row(self, row: Gemma4CacheRow | None) -> TurboQuantTensorRow:
        if isinstance(row, TurboQuantTensorRow):
            return row
        quantized = self._new_turboquant_row()
        if isinstance(row, torch.Tensor) and row.numel() > 0:
            quantized.append(row)
        return quantized

    def get_batch_size(self) -> int:
        return len(self.request_lengths)

    def get_seq_length(self, batch_idx: int | None = None) -> int:
        if batch_idx is not None:
            return self.request_lengths[batch_idx]
        return max(self.request_lengths, default=0)

    def get_seq_lengths(self, *, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(self.request_lengths, dtype=torch.long, device=device)

    def advance_sequence(self, append_lengths: torch.Tensor | int) -> None:
        increments = _normalize_add_lengths(self.get_batch_size(), append_lengths)
        for batch_idx, amount in enumerate(increments):
            self.request_lengths[batch_idx] += max(0, amount)

    @staticmethod
    def _pad_rows(rows: list[torch.Tensor], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not rows:
            raise ValueError("Gemma4DynamicCache expected at least one cache row.")
        max_length = max(int(row.shape[1]) for row in rows)
        head_count = int(rows[0].shape[0])
        head_dim = int(rows[0].shape[2])
        padded = torch.zeros((len(rows), head_count, max_length, head_dim), device=device, dtype=dtype)
        for batch_idx, row in enumerate(rows):
            row_length = int(row.shape[1])
            if row_length > 0:
                padded[batch_idx, :, :row_length, :].copy_(row.to(device=device, dtype=dtype))
        return padded

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(key_states.shape[0])
        self._ensure_batch_size(batch_size)
        device = key_states.device
        dtype = key_states.dtype
        is_sliding = self.config.layer_types[layer_idx] == "sliding_attention"
        sliding_window = max(1, int(self.config.sliding_window))
        past_visible_lengths: list[int] = []
        current_visible_lengths: list[int] = []
        output_keys: list[torch.Tensor] = []
        output_values: list[torch.Tensor] = []
        use_turboquant = self._use_turboquant_for_layer(layer_idx)

        for batch_idx in range(batch_size):
            existing_key_row = self.key_rows[layer_idx][batch_idx]
            existing_value_row = self.value_rows[layer_idx][batch_idx]
            if use_turboquant:
                key_row = self._ensure_turboquant_row(existing_key_row)
                value_row = self._ensure_turboquant_row(existing_value_row)
                past_visible = _cache_row_length(existing_key_row)
                key_row.append(key_states[batch_idx])
                value_row.append(value_states[batch_idx])
                combined_key = key_row.materialize(device=device, dtype=dtype)
                combined_value = value_row.materialize(device=device, dtype=value_states.dtype)
                self.key_rows[layer_idx][batch_idx] = key_row
                self.value_rows[layer_idx][batch_idx] = value_row
            else:
                existing_key = _materialize_cache_row(existing_key_row, device=device, dtype=dtype)
                existing_value = _materialize_cache_row(existing_value_row, device=device, dtype=value_states.dtype)
                if existing_key is None or existing_value is None:
                    combined_key = key_states[batch_idx]
                    combined_value = value_states[batch_idx]
                    past_visible = 0
                else:
                    combined_key = torch.cat((existing_key, key_states[batch_idx]), dim=1)
                    combined_value = torch.cat((existing_value, value_states[batch_idx]), dim=1)
                    past_visible = int(existing_key.shape[1])

            output_keys.append(combined_key)
            output_values.append(combined_value)
            past_visible_lengths.append(past_visible)
            current_visible_lengths.append(int(combined_key.shape[1]))

            if use_turboquant:
                continue

            if is_sliding and int(combined_key.shape[1]) > sliding_window:
                stored_key = combined_key[:, -sliding_window:, :].contiguous()
                stored_value = combined_value[:, -sliding_window:, :].contiguous()
            else:
                stored_key = combined_key.contiguous()
                stored_value = combined_value.contiguous()
            self.key_rows[layer_idx][batch_idx] = stored_key
            self.value_rows[layer_idx][batch_idx] = stored_value

        padded_key = self._pad_rows(output_keys, device=device, dtype=dtype)
        padded_value = self._pad_rows(output_values, device=device, dtype=value_states.dtype)
        return (
            padded_key,
            padded_value,
            torch.tensor(past_visible_lengths, dtype=torch.long, device=device),
            torch.tensor(current_visible_lengths, dtype=torch.long, device=device),
        )

    def set_shared_layer(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        visible_lengths: torch.Tensor,
    ) -> None:
        del key_states, value_states

        batch_size = self.get_batch_size()
        key_rows: list[Gemma4CacheRow] = []
        value_rows: list[Gemma4CacheRow] = []
        lengths: list[int] = []
        if batch_size <= 0:
            self.shared_layers[layer_idx] = (
                torch.empty((0, 0, 0, 0)),
                torch.empty((0, 0, 0, 0)),
                visible_lengths.detach(),
            )
            return

        for batch_idx in range(batch_size):
            key_row = self.key_rows[layer_idx][batch_idx]
            value_row = self.value_rows[layer_idx][batch_idx]
            if key_row is None or value_row is None:
                raise RuntimeError(f"Gemma4 shared KV state for layer {layer_idx} is missing cache rows.")
            key_rows.append(key_row)
            value_rows.append(value_row)
            lengths.append(int(visible_lengths[batch_idx].item()))
        self.shared_layers[layer_idx] = _Gemma4SharedLayerState(
            key_rows=key_rows,
            value_rows=value_rows,
            visible_lengths=lengths,
        )

    def get_shared_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        shared = self.shared_layers.get(layer_idx)
        if shared is None:
            return None
        if isinstance(shared, _Gemma4SharedLayerState):
            return shared.materialize()
        return shared

    def clone(self) -> "Gemma4DynamicCache":
        cloned = Gemma4DynamicCache(
            self.config,
            batch_size=self.get_batch_size(),
            kv_cache_quantization=self.kv_cache_quantization,
            kv_cache_quant_bits=self.kv_cache_quant_bits,
            kv_cache_residual_len=self.kv_cache_residual_len,
        )
        cloned.request_lengths = list(self.request_lengths)
        cloned.reserved_seq_capacity = self.reserved_seq_capacity
        for layer_idx in range(self.config.num_hidden_layers):
            for batch_idx in range(self.get_batch_size()):
                key_row = self.key_rows[layer_idx][batch_idx]
                value_row = self.value_rows[layer_idx][batch_idx]
                if key_row is not None:
                    cloned.key_rows[layer_idx][batch_idx] = _clone_cache_row(key_row)
                if value_row is not None:
                    cloned.value_rows[layer_idx][batch_idx] = _clone_cache_row(value_row)
        for layer_idx, shared in self.shared_layers.items():
            if isinstance(shared, _Gemma4SharedLayerState):
                cloned.shared_layers[layer_idx] = shared.clone()
            else:
                key_states, value_states, visible_lengths = shared
                cloned.shared_layers[layer_idx] = (key_states.clone(), value_states.clone(), visible_lengths.clone())
        return cloned

    def split_batch(self) -> list["Gemma4DynamicCache"]:
        batch_size = self.get_batch_size()
        if batch_size == 0:
            return []
        rows: list[Gemma4DynamicCache] = []
        for batch_idx in range(batch_size):
            row = Gemma4DynamicCache(
                self.config,
                batch_size=1,
                kv_cache_quantization=self.kv_cache_quantization,
                kv_cache_quant_bits=self.kv_cache_quant_bits,
                kv_cache_residual_len=self.kv_cache_residual_len,
            )
            row.request_lengths[0] = self.request_lengths[batch_idx]
            row.reserved_seq_capacity = self.reserved_seq_capacity
            for layer_idx in range(self.config.num_hidden_layers):
                key_row = self.key_rows[layer_idx][batch_idx]
                value_row = self.value_rows[layer_idx][batch_idx]
                if key_row is not None:
                    row.key_rows[layer_idx][0] = _clone_cache_row(key_row)
                if value_row is not None:
                    row.value_rows[layer_idx][0] = _clone_cache_row(value_row)
            for layer_idx, shared in self.shared_layers.items():
                if isinstance(shared, _Gemma4SharedLayerState):
                    row.shared_layers[layer_idx] = shared.select_batch(batch_idx)
                else:
                    key_states, value_states, visible_lengths = shared
                    row.shared_layers[layer_idx] = (
                        key_states[batch_idx : batch_idx + 1].clone(),
                        value_states[batch_idx : batch_idx + 1].clone(),
                        visible_lengths[batch_idx : batch_idx + 1].clone(),
                    )
            rows.append(row)
        return rows

    @classmethod
    def stack(cls, caches: list["Gemma4DynamicCache"], config: Gemma4TextConfig) -> "Gemma4DynamicCache":
        if not caches:
            return cls(config)

        rows: list[Gemma4DynamicCache] = []
        for cache in caches:
            if cache.get_batch_size() <= 1:
                rows.append(cache)
            else:
                rows.extend(cache.split_batch())
        if not rows:
            prototype = caches[0]
            return cls(
                config,
                kv_cache_quantization=prototype.kv_cache_quantization,
                kv_cache_quant_bits=prototype.kv_cache_quant_bits,
                kv_cache_residual_len=prototype.kv_cache_residual_len,
            )

        prototype = rows[0]
        stacked = cls(
            config,
            batch_size=len(rows),
            kv_cache_quantization=prototype.kv_cache_quantization,
            kv_cache_quant_bits=prototype.kv_cache_quant_bits,
            kv_cache_residual_len=prototype.kv_cache_residual_len,
        )
        stacked.request_lengths = [row.request_lengths[0] for row in rows]
        stacked.reserved_seq_capacity = max((row.reserved_seq_capacity for row in rows), default=0)

        for layer_idx in range(config.num_hidden_layers):
            for batch_idx, row in enumerate(rows):
                key_row = row.key_rows[layer_idx][0]
                value_row = row.value_rows[layer_idx][0]
                if key_row is not None:
                    stacked.key_rows[layer_idx][batch_idx] = _clone_cache_row(key_row)
                if value_row is not None:
                    stacked.value_rows[layer_idx][batch_idx] = _clone_cache_row(value_row)

        shared_layer_indices = set()
        for row in rows:
            shared_layer_indices.update(row.shared_layers.keys())
        for layer_idx in shared_layer_indices:
            key_parts: list[torch.Tensor] = []
            value_parts: list[torch.Tensor] = []
            visible_parts: list[torch.Tensor] = []
            for row in rows:
                shared = row.get_shared_layer(layer_idx)
                if shared is None:
                    raise ValueError(f"Gemma4 cache stack mismatch: missing shared layer {layer_idx}.")
                key_states, value_states, visible_lengths = shared
                key_parts.append(key_states)
                value_parts.append(value_states)
                visible_parts.append(visible_lengths)
            stacked.set_shared_layer(
                layer_idx,
                torch.cat(key_parts, dim=0),
                torch.cat(value_parts, dim=0),
                torch.cat(visible_parts, dim=0),
            )
        return stacked

    def release(self) -> None:
        if self._released:
            return
        self.request_lengths.clear()
        self.shared_layers.clear()
        for layer_idx in range(self.config.num_hidden_layers):
            self.key_rows[layer_idx].clear()
            self.value_rows[layer_idx].clear()
        self._released = True

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Gemma4DynamicCache":
        for layer_idx in range(self.config.num_hidden_layers):
            for batch_idx in range(self.get_batch_size()):
                key_row = self.key_rows[layer_idx][batch_idx]
                value_row = self.value_rows[layer_idx][batch_idx]
                if key_row is not None:
                    if isinstance(key_row, TurboQuantTensorRow):
                        self.key_rows[layer_idx][batch_idx] = key_row.to(device=device, dtype=dtype)
                    else:
                        kwargs: dict[str, object] = {}
                        if device is not None:
                            kwargs["device"] = device
                        if dtype is not None:
                            kwargs["dtype"] = dtype
                        if kwargs:
                            self.key_rows[layer_idx][batch_idx] = key_row.to(**kwargs)
                if value_row is not None:
                    if isinstance(value_row, TurboQuantTensorRow):
                        self.value_rows[layer_idx][batch_idx] = value_row.to(device=device, dtype=dtype)
                    else:
                        kwargs = {}
                        if device is not None:
                            kwargs["device"] = device
                        if dtype is not None:
                            kwargs["dtype"] = dtype
                        if kwargs:
                            self.value_rows[layer_idx][batch_idx] = value_row.to(**kwargs)
        moved_shared_layers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor] | _Gemma4SharedLayerState] = {}
        for layer_idx, shared in self.shared_layers.items():
            if isinstance(shared, _Gemma4SharedLayerState):
                moved_shared_layers[layer_idx] = shared.to(device=device, dtype=dtype)
                continue
            key_states, value_states, visible_lengths = shared
            kwargs: dict[str, object] = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            moved_shared_layers[layer_idx] = (
                key_states.to(**kwargs),
                value_states.to(**kwargs),
                visible_lengths.to(device=device) if device is not None else visible_lengths,
            )
        self.shared_layers = moved_shared_layers
        return self


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.with_scale and self.weight is not None and hidden_states.device.type == "xpu":
            return run_rmsnorm_fused_ex(
                input=hidden_states,
                weight=self.weight,
                eps=self.eps,
                add_unit_offset=False,
            )
        output = hidden_states.float()
        output = output * torch.rsqrt(output.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.with_scale and self.weight is not None:
            output = output * self.weight.float()
        return output.to(dtype=hidden_states.dtype)


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, *, embed_scale: float) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = float(embed_scale)
        self.register_buffer("embed_scale", torch.tensor(self.scalar_embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(dtype=self.weight.dtype)

    def reset_runtime_buffers(self) -> None:
        self.embed_scale = torch.tensor(self.scalar_embed_scale, device=self.weight.device)


class Gemma4TextMLP(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        self.hidden_activation = config.hidden_activation
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activated = _apply_hidden_activation(self.gate_proj(hidden_states), self.hidden_activation)
        return self.down_proj(activated * self.up_proj(hidden_states))


class Gemma4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.layer_types = set(config.layer_types)
        self.rope_type: dict[str, str] = {}
        self.attention_scaling: dict[str, float] = {}

        for layer_type in self.layer_types:
            params = self.config.rope_parameters[layer_type]
            self.rope_type[layer_type] = params.rope_type
            inv_freq, scaling = self._compute_inv_freq(params, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.attention_scaling[layer_type] = scaling

    def _head_dim_for_layer(self, layer_type: str) -> int:
        if layer_type == "full_attention" and self.config.global_head_dim:
            return int(self.config.global_head_dim)
        return int(self.config.head_dim)

    def _rotary_dim_for_layer(self, params: Gemma4RopeParameters, *, layer_type: str) -> int:
        dim = int(self._head_dim_for_layer(layer_type) * params.partial_rotary_factor)
        dim = max(2, dim if dim > 0 else self._head_dim_for_layer(layer_type))
        return dim - (dim % 2)

    def _compute_default_rope_parameters(
        self,
        params: Gemma4RopeParameters,
        *,
        layer_type: str,
    ) -> tuple[torch.Tensor, float]:
        dim = self._rotary_dim_for_layer(params, layer_type=layer_type)
        base = float(params.rope_theta)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return inv_freq, 1.0

    def _compute_proportional_rope_parameters(
        self,
        params: Gemma4RopeParameters,
        *,
        layer_type: str,
    ) -> tuple[torch.Tensor, float]:
        dim = self._rotary_dim_for_layer(params, layer_type=layer_type)
        base = float(params.rope_theta)
        original_max_position_embeddings = params.original_max_position_embeddings
        if original_max_position_embeddings is None or original_max_position_embeddings <= 0:
            original_max_position_embeddings = min(8_192, int(self.config.max_position_embeddings))
        factor = float(params.factor)
        if factor <= 1.0 and original_max_position_embeddings > 0:
            factor = max(1.0, float(self.config.max_position_embeddings) / float(original_max_position_embeddings))

        def get_mscale(scale: float, mscale: float = 1.0) -> float:
            if scale <= 1.0:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        attention_factor = params.attention_factor
        if attention_factor is None:
            attention_factor = get_mscale(factor)

        def find_correction_dim(num_rotations: float, rope_dim: int, rope_base: float, max_positions: int) -> float:
            return (rope_dim * math.log(max_positions / (num_rotations * 2 * math.pi))) / (2 * math.log(rope_base))

        def find_correction_range(
            low_rot: float,
            high_rot: float,
            rope_dim: int,
            rope_base: float,
            max_positions: int,
            truncate: bool,
        ) -> tuple[float, float]:
            low = find_correction_dim(low_rot, rope_dim, rope_base, max_positions)
            high = find_correction_dim(high_rot, rope_dim, rope_base, max_positions)
            if truncate:
                low = math.floor(low)
                high = math.ceil(high)
            return max(low, 0.0), min(high, float(rope_dim - 1))

        def linear_ramp_factor(min_value: float, max_value: float, ramp_dim: int) -> torch.Tensor:
            if min_value == max_value:
                max_value += 0.001
            linear = (torch.arange(ramp_dim, dtype=torch.float32) - min_value) / (max_value - min_value)
            return torch.clamp(linear, 0.0, 1.0)

        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)
        low, high = find_correction_range(
            float(params.beta_fast),
            float(params.beta_slow),
            dim,
            base,
            int(original_max_position_embeddings),
            bool(params.truncate),
        )
        inv_freq_extrapolation_factor = 1.0 - linear_ramp_factor(low, high, dim // 2)
        inv_freq = (
            inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )
        return inv_freq, float(attention_factor)

    def _compute_inv_freq(
        self,
        params: Gemma4RopeParameters,
        *,
        layer_type: str,
    ) -> tuple[torch.Tensor, float]:
        if params.rope_type == "proportional":
            return self._compute_proportional_rope_parameters(params, layer_type=layer_type)
        return self._compute_default_rope_parameters(params, layer_type=layer_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = getattr(self, f"{layer_type}_inv_freq").to(device=hidden_states.device)
        scaling = self.attention_scaling[layer_type]
        freqs = position_ids[:, :, None].float() * inv_freq[None, None, :].float()
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * scaling
        sin = emb.sin() * scaling
        return cos, sin

    def reset_runtime_buffers(self) -> None:
        for layer_type in self.layer_types:
            params = self.config.rope_parameters[layer_type]
            inv_freq, scaling = self._compute_inv_freq(params, layer_type=layer_type)
            current = getattr(self, f"{layer_type}_inv_freq")
            setattr(self, f"{layer_type}_inv_freq", inv_freq.to(device=current.device))
            self.attention_scaling[layer_type] = scaling


class Gemma4TextAttention(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = int(config.sliding_window) if self.is_sliding else None
        self.head_dim = int(config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim)
        self.use_alternative_attention = bool(config.attention_k_eq_v and not self.is_sliding)
        self.num_key_value_heads = int(
            config.num_global_key_value_heads
            if self.use_alternative_attention and config.num_global_key_value_heads is not None
            else config.num_key_value_heads
        )
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = 1.0

        first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        if self.is_kv_shared_layer:
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            self.store_full_length_kv = layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(self.layer_type)

        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = (
            None
            if self.use_alternative_attention
            else nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        )
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

    @staticmethod
    def _visible_mask(visible_lengths: torch.Tensor, key_len: int, device: torch.device) -> torch.Tensor:
        key_positions = torch.arange(key_len, device=device)[None, :]
        return key_positions < visible_lengths[:, None]

    def _full_attention_masks(
        self,
        *,
        query_len: int,
        key_len: int,
        past_request_lengths: torch.Tensor,
        past_visible_lengths: torch.Tensor,
        visible_lengths: torch.Tensor,
        attention_mask: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        visible_mask = self._visible_mask(visible_lengths, key_len, device)
        if query_len == 1:
            causal_mask = None
        else:
            query_positions = past_request_lengths[:, None] + torch.arange(query_len, device=device)[None, :]
            key_positions = (
                past_request_lengths[:, None]
                - past_visible_lengths[:, None]
                + torch.arange(key_len, device=device)[None, :]
            )
            causal_mask = key_positions[:, None, :] > query_positions[:, :, None]

        key_padding_mask = None
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and attention_mask.shape[1] == key_len
            and int(past_request_lengths.max().item()) == 0
        ):
            key_padding_mask = attention_mask.to(dtype=torch.bool)
        return causal_mask, visible_mask, key_padding_mask

    def _sliding_attention_masks(
        self,
        *,
        query_len: int,
        key_len: int,
        past_request_lengths: torch.Tensor,
        past_visible_lengths: torch.Tensor,
        visible_lengths: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_positions = past_request_lengths[:, None] + torch.arange(query_len, device=device)[None, :]
        key_positions = (
            past_request_lengths[:, None]
            - past_visible_lengths[:, None]
            + torch.arange(key_len, device=device)[None, :]
        )
        left_bound = query_positions - (int(self.sliding_window) - 1)
        causal_mask = (key_positions[:, None, :] > query_positions[:, :, None]) | (
            key_positions[:, None, :] < left_bound[:, :, None]
        )
        visible_mask = self._visible_mask(visible_lengths, key_len, device)
        return causal_mask, visible_mask

    def _compute_query_states(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        query_states = query_states.transpose(1, 2)
        query_states = self.q_norm(query_states)
        return _apply_rotary_pos_emb_single(query_states, cos, sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Gemma4DynamicCache | None = None,
        past_request_lengths: torch.Tensor | None = None,
        shared_layer_states: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        if past_request_lengths is None:
            if past_key_values is None:
                past_request_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            else:
                past_request_lengths = past_key_values.get_seq_lengths(device=device)

        cos, sin = position_embeddings

        if self.is_kv_shared_layer:
            if shared_layer_states is None or self.kv_shared_layer_index is None:
                raise RuntimeError("Gemma4 shared KV attention requires shared-layer state tracking.")
            shared = shared_layer_states.get(self.kv_shared_layer_index)
            if shared is None and past_key_values is not None:
                shared = past_key_values.get_shared_layer(self.kv_shared_layer_index)
            if shared is None:
                raise RuntimeError(f"Missing shared KV states for layer {self.layer_idx}.")
            key_states, value_states, visible_lengths = shared
            if key_states.device != device:
                key_states = key_states.to(device=device)
            if value_states.device != device:
                value_states = value_states.to(device=device)
            if visible_lengths.device != device:
                visible_lengths = visible_lengths.to(device=device)
            query_states = self._compute_query_states(hidden_states, cos, sin)
            past_visible_lengths = torch.clamp(visible_lengths - seq_len, min=0)
        else:
            hidden_shape = (batch_size, seq_len, -1, self.head_dim)
            key_source = self.k_proj(hidden_states).view(hidden_shape)
            value_source = key_source if self.use_alternative_attention else self.v_proj(hidden_states).view(hidden_shape)
            query_source = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            query_states = query_source.transpose(1, 2)
            key_states = key_source.transpose(1, 2)
            value_states = value_source.transpose(1, 2)

            if device.type == "xpu":
                query_states, key_states = run_qk_norm_rotary_fused_ex(
                    query=query_states,
                    key=key_states,
                    query_norm_weight=self.q_norm.weight,
                    key_norm_weight=self.k_norm.weight,
                    cos=cos,
                    sin=sin,
                    query_norm_eps=self.q_norm.eps,
                    key_norm_eps=self.k_norm.eps,
                    add_unit_offset=False,
                )
            else:
                query_states = self.q_norm(query_states)
                key_states = self.k_norm(key_states)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            value_states = self.v_norm(value_states)

            if past_key_values is not None:
                key_states, value_states, past_visible_lengths, visible_lengths = past_key_values.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                )
            else:
                past_visible_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
                visible_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

            if self.store_full_length_kv:
                if shared_layer_states is not None:
                    shared_layer_states[self.layer_idx] = (key_states, value_states, visible_lengths)
                if past_key_values is not None:
                    past_key_values.set_shared_layer(self.layer_idx, key_states, value_states, visible_lengths)

        if (
            not self.is_sliding
            and attention_mask is None
            and past_key_values is None
            and seq_len > 1
            and device.type == "xpu"
        ):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=self.num_key_value_groups > 1,
            )
        elif (
            not self.is_sliding
            and device.type == "xpu"
            and seq_len == 1
            and past_key_values is not None
        ):
            attn_output = run_gqa_decode_fused(
                query=query_states,
                key=key_states,
                value=value_states,
                visible_lengths=visible_lengths,
                scaling=self.scaling,
            )
        else:
            if self.is_sliding:
                causal_mask, visible_mask = self._sliding_attention_masks(
                    query_len=seq_len,
                    key_len=int(key_states.shape[-2]),
                    past_request_lengths=past_request_lengths,
                    past_visible_lengths=past_visible_lengths,
                    visible_lengths=visible_lengths,
                    device=device,
                )
                key_padding_mask = None
            else:
                causal_mask, visible_mask, key_padding_mask = self._full_attention_masks(
                    query_len=seq_len,
                    key_len=int(key_states.shape[-2]),
                    past_request_lengths=past_request_lengths,
                    past_visible_lengths=past_visible_lengths,
                    visible_lengths=visible_lengths,
                    attention_mask=attention_mask,
                    device=device,
                )
            attn_output = grouped_query_attention(
                query_states,
                key_states,
                value_states,
                scaling=self.scaling,
                causal_mask=causal_mask,
                visible_mask=visible_mask,
                key_padding_mask=key_padding_mask,
            )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
        return self.o_proj(attn_output)


class Gemma4TextDecoderLayer(nn.Module):
    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_activation = config.hidden_activation
        self.self_attn = Gemma4TextAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1), persistent=False)

        self.hidden_size_per_layer_input = int(config.hidden_size_per_layer_input)
        if self.hidden_size_per_layer_input > 0:
            self.per_layer_input_gate = nn.Linear(
                self.hidden_size,
                self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input,
                self.hidden_size,
                bias=False,
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        per_layer_input: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Gemma4DynamicCache | None = None,
        past_request_lengths: torch.Tensor | None = None,
        shared_layer_states: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_request_lengths=past_request_lengths,
            shared_layer_states=shared_layer_states,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input > 0:
            if per_layer_input is None:
                raise RuntimeError("Gemma4 per-layer embeddings are enabled but per_layer_input was not provided.")
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = _apply_hidden_activation(hidden_states, self.hidden_activation)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states * self.layer_scalar.to(dtype=hidden_states.dtype)

    def reset_runtime_buffers(self) -> None:
        self.layer_scalar = torch.ones_like(self.layer_scalar)


class Gemma4TextModel(nn.Module):
    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.execution_device: torch.device | None = None
        self.kv_cache_quantization = "none"
        self.kv_cache_quant_bits = 4
        self.kv_cache_residual_len = 128
        padding_idx = config.pad_token_id if 0 <= config.pad_token_id < config.vocab_size else 0
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList([Gemma4TextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.unique_layer_types = set(config.layer_types)
        self.hidden_size_per_layer_input = int(config.hidden_size_per_layer_input)

        if self.hidden_size_per_layer_input > 0:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                padding_idx,
                embed_scale=self.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * self.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(
                self.hidden_size_per_layer_input,
                eps=config.rms_norm_eps,
            )

    def configure_runtime(
        self,
        execution_device: torch.device,
        *,
        offload_token_io: bool = False,
        offload_per_layer_input_embeddings: bool = False,
        kv_cache_quantization: str = "none",
        kv_cache_quant_bits: int = 4,
        kv_cache_residual_len: int = 128,
    ) -> None:
        self.execution_device = execution_device
        self.kv_cache_quantization = kv_cache_quantization
        self.kv_cache_quant_bits = int(kv_cache_quant_bits)
        self.kv_cache_residual_len = max(1, int(kv_cache_residual_len))
        token_device = torch.device("cpu") if offload_token_io else execution_device
        self.embed_tokens.to(token_device)
        if self.hidden_size_per_layer_input > 0:
            per_layer_embedding_device = (
                torch.device("cpu") if offload_token_io or offload_per_layer_input_embeddings else execution_device
            )
            self.embed_tokens_per_layer.to(per_layer_embedding_device)
            self.per_layer_model_projection.to(execution_device)
            self.per_layer_projection_norm.to(execution_device)
        self.rotary_emb.to(execution_device)
        self.norm.to(execution_device)
        for layer in self.layers:
            layer.to(execution_device)

    def get_per_layer_inputs(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        del inputs_embeds
        if self.hidden_size_per_layer_input <= 0:
            raise RuntimeError("Gemma4 per-layer embeddings are not enabled in this config.")
        if input_ids is None:
            raise RuntimeError("Gemma4 per-layer embeddings require input_ids for this runtime path.")
        embedding_device = self.embed_tokens_per_layer.weight.device
        if input_ids.device != embedding_device:
            input_ids = input_ids.to(device=embedding_device)
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.hidden_size_per_layer_input <= 0:
            raise RuntimeError("Gemma4 per-layer embeddings are not enabled in this config.")
        projection = (self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale).reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        projection = self.per_layer_projection_norm(projection)
        if per_layer_inputs is None:
            return projection
        return (projection + per_layer_inputs) * self.per_layer_input_scale

    def reset_runtime_buffers(self) -> None:
        self.embed_tokens.reset_runtime_buffers()
        if self.hidden_size_per_layer_input > 0:
            self.embed_tokens_per_layer.reset_runtime_buffers()
        self.rotary_emb.reset_runtime_buffers()
        for layer in self.layers:
            layer.reset_runtime_buffers()

    def kv_cache_runtime_info(self) -> dict[str, object]:
        full_attention_layer_indices: list[int] = []
        sliding_attention_layer_indices: list[int] = []
        shared_kv_source_layer_indices: list[int] = []
        shared_kv_consumer_layer_indices: list[int] = []
        for layer_idx, layer in enumerate(self.layers):
            attention = layer.self_attn
            if attention.is_sliding:
                sliding_attention_layer_indices.append(layer_idx)
            else:
                full_attention_layer_indices.append(layer_idx)
            if attention.store_full_length_kv:
                shared_kv_source_layer_indices.append(layer_idx)
            if attention.is_kv_shared_layer:
                shared_kv_consumer_layer_indices.append(layer_idx)

        turboquant_enabled = self.kv_cache_quantization == "turboquant"
        turboquant_quantized_layer_indices = (
            list(full_attention_layer_indices)
            if turboquant_enabled
            else []
        )
        return {
            "mode": self.kv_cache_quantization,
            "turboquant_enabled": turboquant_enabled,
            "turboquant_bits": self.kv_cache_quant_bits if turboquant_enabled else None,
            "turboquant_residual_len": self.kv_cache_residual_len if turboquant_enabled else None,
            "turboquant_applies_to": "full_attention_only" if turboquant_enabled else "disabled",
            "full_attention_layers": len(full_attention_layer_indices),
            "full_attention_layer_indices": full_attention_layer_indices,
            "sliding_attention_layers": len(sliding_attention_layer_indices),
            "sliding_attention_layer_indices": sliding_attention_layer_indices,
            "turboquant_quantized_layers": len(turboquant_quantized_layer_indices),
            "turboquant_quantized_layer_indices": turboquant_quantized_layer_indices,
            "shared_kv_state_mode": "row_reference",
            "shared_kv_row_reuse_enabled": True,
            "shared_kv_source_layers": len(shared_kv_source_layer_indices),
            "shared_kv_source_layer_indices": shared_kv_source_layer_indices,
            "shared_kv_consumer_layers": len(shared_kv_consumer_layer_indices),
            "shared_kv_consumer_layer_indices": shared_kv_consumer_layer_indices,
        }

    def _append_lengths_from_attention_mask(
        self,
        *,
        seq_len: int,
        attention_mask: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | int:
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and attention_mask.shape[1] == seq_len
            and not bool(torch.all(attention_mask == 1).item())
        ):
            return attention_mask.sum(dim=-1).to(dtype=torch.long)
        if batch_size == 1:
            return seq_len
        return torch.full(
            (batch_size,),
            seq_len,
            dtype=torch.long,
            device=attention_mask.device if attention_mask is not None else None,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Gemma4DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> Gemma4TextModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            embed_device = self.embed_tokens.weight.device
            if input_ids.device != embed_device:
                input_ids = input_ids.to(device=embed_device)
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input > 0:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            projection_device = _module_device(self.per_layer_model_projection)
            if inputs_embeds.device != projection_device:
                inputs_embeds = inputs_embeds.to(device=projection_device)
            if per_layer_inputs.device != projection_device:
                per_layer_inputs = per_layer_inputs.to(device=projection_device)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and past_key_values is None:
            past_key_values = Gemma4DynamicCache(
                self.config,
                batch_size=int(inputs_embeds.shape[0]),
                kv_cache_quantization=self.kv_cache_quantization,
                kv_cache_quant_bits=self.kv_cache_quant_bits,
                kv_cache_residual_len=self.kv_cache_residual_len,
            )

        execution_device = self.execution_device or _module_device(self.norm)
        if inputs_embeds.device != execution_device:
            inputs_embeds = inputs_embeds.to(device=execution_device)
        if attention_mask is not None and attention_mask.device != execution_device:
            attention_mask = attention_mask.to(device=execution_device)
        if position_ids is not None and position_ids.device != execution_device:
            position_ids = position_ids.to(device=execution_device)
        if per_layer_inputs is not None and per_layer_inputs.device != execution_device:
            per_layer_inputs = per_layer_inputs.to(device=execution_device)

        batch_size, seq_len, _ = inputs_embeds.shape
        if past_key_values is None or past_key_values.get_batch_size() == 0:
            past_request_lengths = torch.zeros(batch_size, device=inputs_embeds.device, dtype=torch.long)
        else:
            past_request_lengths = past_key_values.get_seq_lengths(device=inputs_embeds.device)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).view(1, -1)
            position_ids = position_ids + past_request_lengths.view(-1, 1)

        position_embeddings = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }
        shared_layer_states: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        hidden_states = inputs_embeds

        for layer_idx, decoder_layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, :, layer_idx, :] if per_layer_inputs is not None else None
            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input=per_layer_input,
                position_embeddings=position_embeddings[self.config.layer_types[layer_idx]],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                past_request_lengths=past_request_lengths,
                shared_layer_states=shared_layer_states,
            )

        hidden_states = self.norm(hidden_states)

        if past_key_values is not None:
            past_key_values.advance_sequence(
                self._append_lengths_from_attention_mask(
                    seq_len=seq_len,
                    attention_mask=attention_mask,
                    batch_size=batch_size,
                )
            )

        return Gemma4TextModelOutput(last_hidden_state=hidden_states, past_key_values=past_key_values)


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.language_model = Gemma4TextModel(config.text_config)
        self.vision_tower = Gemma4VisionModel(config.vision_config) if config.vision_config is not None else None
        self.embed_vision = (
            Gemma4MultimodalEmbedder(config.vision_config, config.text_config)
            if config.vision_config is not None
            else None
        )
        self.audio_tower = Gemma4AudioModel(config.audio_config) if config.audio_config is not None else None
        self.embed_audio = (
            Gemma4MultimodalEmbedder(config.audio_config, config.text_config)
            if config.audio_config is not None
            else None
        )

    def configure_runtime(
        self,
        execution_device: torch.device,
        *,
        offload_vision: bool = False,
        offload_token_io: bool = False,
        offload_per_layer_input_embeddings: bool = False,
        kv_cache_quantization: str = "none",
        kv_cache_quant_bits: int = 4,
        kv_cache_residual_len: int = 128,
    ) -> None:
        self.language_model.configure_runtime(
            execution_device,
            offload_token_io=offload_token_io,
            offload_per_layer_input_embeddings=offload_per_layer_input_embeddings,
            kv_cache_quantization=kv_cache_quantization,
            kv_cache_quant_bits=kv_cache_quant_bits,
            kv_cache_residual_len=kv_cache_residual_len,
        )
        vision_device = torch.device("cpu") if offload_vision else execution_device
        if self.vision_tower is not None:
            self.vision_tower.to(vision_device)
        if self.embed_vision is not None:
            self.embed_vision.to(vision_device)
        if self.audio_tower is not None:
            self.audio_tower.to(execution_device)
        if self.embed_audio is not None:
            self.embed_audio.to(execution_device)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.language_model.embed_tokens = value

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.vision_tower is None or self.embed_vision is None:
            raise ValueError("The loaded Gemma4 model does not include a vision tower.")
        vision_device = next(self.vision_tower.parameters()).device
        pixel_values = pixel_values.to(device=vision_device, dtype=next(self.vision_tower.parameters()).dtype)
        image_position_ids = image_position_ids.to(device=vision_device, dtype=torch.long)
        output_length = int(pixel_values.shape[-2] // (int(self.config.vision_config.pooling_kernel_size) ** 2))
        features = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
            output_length=output_length,
        )
        return self.embed_vision(features)

    def get_video_features(
        self,
        pixel_values_videos: torch.Tensor,
        video_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        flattened_pixels = pixel_values_videos.flatten(0, 1)
        flattened_positions = video_position_ids.flatten(0, 1)
        return self.get_image_features(flattened_pixels, flattened_positions)

    def get_audio_features(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.audio_tower is None or self.embed_audio is None:
            raise ValueError("The loaded Gemma4 model does not include an audio tower.")
        audio_device = _module_device(self.audio_tower)
        audio_dtype = next(self.audio_tower.parameters()).dtype
        input_features = input_features.to(device=audio_device, dtype=audio_dtype)
        input_features_mask = input_features_mask.to(device=audio_device, dtype=torch.bool)
        audio_outputs = self.audio_tower(input_features, input_features_mask)
        return self.embed_audio(audio_outputs.last_hidden_state), audio_outputs.attention_mask

    def reset_runtime_buffers(self) -> None:
        self.language_model.reset_runtime_buffers()

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
    ) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.BoolTensor]:
        image_mask = input_ids == self.config.image_token_id
        video_mask = input_ids == self.config.video_token_id
        audio_mask = input_ids == self.config.audio_token_id
        return image_mask, video_mask, audio_mask


class Gemma4ForConditionalGeneration(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)
        self.lm_head = (
            None
            if (config.tie_word_embeddings or config.text_config.tie_word_embeddings)
            else nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        )

    def configure_runtime(
        self,
        execution_device: torch.device,
        *,
        offload_vision: bool = False,
        offload_token_io: bool = False,
        offload_per_layer_input_embeddings: bool = False,
        kv_cache_quantization: str = "none",
        kv_cache_quant_bits: int = 4,
        kv_cache_residual_len: int = 128,
    ) -> None:
        self.model.configure_runtime(
            execution_device,
            offload_vision=offload_vision,
            offload_token_io=offload_token_io,
            offload_per_layer_input_embeddings=offload_per_layer_input_embeddings,
            kv_cache_quantization=kv_cache_quantization,
            kv_cache_quant_bits=kv_cache_quant_bits,
            kv_cache_residual_len=kv_cache_residual_len,
        )
        if self.lm_head is not None:
            self.lm_head.to(torch.device("cpu") if offload_token_io else execution_device)

    def tie_weights(self) -> None:
        if (
            self.lm_head is not None
            and (self.config.tie_word_embeddings or self.config.text_config.tie_word_embeddings)
        ):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def reset_runtime_buffers(self) -> None:
        self.model.reset_runtime_buffers()

    def kv_cache_runtime_info(self) -> dict[str, object]:
        return self.model.language_model.kv_cache_runtime_info()

    def _compute_logits(self, hidden_states: torch.Tensor, logits_to_keep: int | None) -> torch.Tensor:
        if logits_to_keep is not None and logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        if self.lm_head is None:
            output_weight = self.model.language_model.embed_tokens.weight
            if hidden_states.device != output_weight.device:
                hidden_states = hidden_states.to(device=output_weight.device)
            logits = F.linear(hidden_states, output_weight)
        else:
            lm_head_device = _module_device(self.lm_head)
            if hidden_states.device != lm_head_device:
                hidden_states = hidden_states.to(device=lm_head_device)
            logits = self.lm_head(hidden_states)
        softcap = self.config.text_config.final_logit_softcapping
        if softcap is not None:
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap
        return logits

    def forward_text_only(
        self,
        *,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Gemma4DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
    ) -> Gemma4CausalLMOutput:
        outputs = self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        logits = self._compute_logits(outputs.last_hidden_state, logits_to_keep)
        return Gemma4CausalLMOutput(logits=logits, past_key_values=outputs.past_key_values)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Gemma4DynamicCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | None = None,
    ) -> Gemma4CausalLMOutput:
        del image_grid_thw, video_grid_thw, mm_token_type_ids
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is not None and (
            pixel_values is not None or pixel_values_videos is not None or input_features is not None
        ):
            raise ValueError("Gemma4 multimodal inputs require input_ids so placeholder tokens can be expanded.")
        if input_features is not None and input_features_mask is None:
            raise ValueError("Gemma4 audio inputs require input_features_mask.")
        if input_features_mask is not None and input_features is None:
            raise ValueError("Gemma4 input_features_mask requires input_features.")

        image_mask = video_mask = audio_mask = None
        llm_input_ids = input_ids
        if input_ids is not None:
            if pixel_values is not None or pixel_values_videos is not None or input_features is not None:
                image_mask, video_mask, audio_mask = self.model.get_placeholder_mask(input_ids)
                llm_input_ids = input_ids.clone()
                llm_input_ids[image_mask | video_mask | audio_mask] = self.config.text_config.pad_token_id
            embedding_device = self.get_input_embeddings().weight.device
            if llm_input_ids.device != embedding_device:
                llm_input_ids = llm_input_ids.to(device=embedding_device)
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        per_layer_inputs = None
        if self.config.text_config.hidden_size_per_layer_input > 0 and llm_input_ids is not None:
            per_layer_inputs = self.model.language_model.get_per_layer_inputs(llm_input_ids, inputs_embeds)

        if pixel_values is not None:
            if image_position_ids is None:
                raise ValueError("Gemma4 image inputs require image_position_ids.")
            image_features = self.model.get_image_features(pixel_values, image_position_ids)
            image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[expanded_image_mask].numel() != image_features.numel():
                raise ValueError("Gemma4 image features and image placeholder tokens do not match.")
            inputs_embeds = inputs_embeds.masked_scatter(expanded_image_mask, image_features)

        if pixel_values_videos is not None:
            if video_position_ids is None:
                raise ValueError("Gemma4 video inputs require video_position_ids.")
            video_features = self.model.get_video_features(pixel_values_videos, video_position_ids)
            video_features = video_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            expanded_video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[expanded_video_mask].numel() != video_features.numel():
                raise ValueError("Gemma4 video features and video placeholder tokens do not match.")
            inputs_embeds = inputs_embeds.masked_scatter(expanded_video_mask, video_features)

        if input_features is not None:
            audio_features, encoded_audio_mask = self.model.get_audio_features(input_features, input_features_mask)
            audio_features = audio_features[encoded_audio_mask]
            audio_features = audio_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            expanded_audio_mask = audio_mask.unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[expanded_audio_mask].numel() != audio_features.numel():
                raise ValueError("Gemma4 audio features and audio placeholder tokens do not match.")
            inputs_embeds = inputs_embeds.masked_scatter(expanded_audio_mask, audio_features)

        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            use_cache=use_cache,
        )
        logits = self._compute_logits(outputs.last_hidden_state, logits_to_keep)
        return Gemma4CausalLMOutput(logits=logits, past_key_values=outputs.past_key_values)
