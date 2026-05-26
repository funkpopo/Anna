from __future__ import annotations

import math
import os
from typing import Any, NamedTuple

import numpy as np
import torch

from anna.model.fused_ops import run_gqa_decode_splitkv_fused_out
from anna.model.xpu_decode_profile import xpu_profile_region

_mse_quantizers: dict[tuple[int, int, str], Any] = {}
_ip_quantizers: dict[tuple[int, int, str], Any] = {}
_KEEP_DECODE_CACHE = True
_SYNC_AFTER_QUANTIZE = os.getenv("ANNA_TURBOQUANT_SYNC_AFTER_QUANTIZE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_DECODE_SPLITKV_BLOCK_SIZE = max(16, int(os.getenv("ANNA_XPU_TURBOQUANT_DECODE_BLOCK_SIZE", "64")))

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


class TurboQuantKeyState(NamedTuple):
    mse_indices: torch.Tensor
    norms: torch.Tensor
    qjl_signs: torch.Tensor
    residual_norms: torch.Tensor


class TurboQuantValueState(NamedTuple):
    data: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int
    group_size: int


def turboquant_is_available() -> bool:
    try:
        from turboquant.core import TurboQuantIP, TurboQuantMSE  # noqa: F401
    except ImportError:
        return False
    return True


# KV-cache TurboQuant bit-widths allowed by Anna runtimes (value packing supports 2/3/4 in this module).
VALID_KV_CACHE_QUANT_BITS: frozenset[int] = frozenset((2, 3, 4))


def _require_turboquant_core() -> tuple[Any, Any]:
    try:
        from turboquant.core import TurboQuantIP, TurboQuantMSE
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError(
            "TurboQuant KV-cache compression requires the optional 'turboquant' dependency."
        ) from exc
    return TurboQuantMSE, TurboQuantIP


def _require_turboquant_mse():
    TurboQuantMSE, _ = _require_turboquant_core()
    return TurboQuantMSE


def _require_turboquant_ip():
    _, TurboQuantIP = _require_turboquant_core()
    return TurboQuantIP


def get_turboquant_quantizer(*, head_dim: int, bits: int, device: torch.device) -> Any:
    TurboQuantMSE = _require_turboquant_mse()
    key = (int(head_dim), int(bits), str(device))
    quantizer = _mse_quantizers.get(key)
    if quantizer is None:
        quantizer = TurboQuantMSE(dim=int(head_dim), bits=int(bits), device=str(device), seed=42)
        _mse_quantizers[key] = quantizer
    return quantizer


def get_turboquant_ip_quantizer(*, head_dim: int, bits: int, device: torch.device) -> Any:
    TurboQuantIP = _require_turboquant_ip()
    key = (int(head_dim), int(bits), str(device))
    quantizer = _ip_quantizers.get(key)
    if quantizer is None:
        quantizer = TurboQuantIP(dim=int(head_dim), bits=int(bits), device=str(device), seed=42)
        _ip_quantizers[key] = quantizer
    return quantizer


def _clone_key_state(state: TurboQuantKeyState | None) -> TurboQuantKeyState | None:
    if state is None:
        return None
    return TurboQuantKeyState(
        mse_indices=state.mse_indices.clone(),
        norms=state.norms.clone(),
        qjl_signs=state.qjl_signs.clone(),
        residual_norms=state.residual_norms.clone(),
    )


def _clone_value_state(state: TurboQuantValueState | None) -> TurboQuantValueState | None:
    if state is None:
        return None
    return TurboQuantValueState(
        data=state.data.clone(),
        scales=state.scales.clone(),
        zeros=state.zeros.clone(),
        bits=state.bits,
        group_size=state.group_size,
    )


def _move_key_state(
    state: TurboQuantKeyState | None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TurboQuantKeyState | None:
    if state is None:
        return None
    norm_dtype = torch.float32 if dtype is not None else None
    return TurboQuantKeyState(
        mse_indices=state.mse_indices.to(device=device) if device is not None else state.mse_indices,
        norms=state.norms.to(device=device, dtype=norm_dtype) if (device is not None or norm_dtype is not None) else state.norms,
        qjl_signs=state.qjl_signs.to(device=device) if device is not None else state.qjl_signs,
        residual_norms=state.residual_norms.to(device=device, dtype=norm_dtype)
        if (device is not None or norm_dtype is not None)
        else state.residual_norms,
    )


def _move_value_state(
    state: TurboQuantValueState | None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> TurboQuantValueState | None:
    if state is None:
        return None
    kwargs: dict[str, object] = {}
    if device is not None:
        kwargs["device"] = device
    return TurboQuantValueState(
        data=state.data.to(device=device) if device is not None else state.data,
        scales=state.scales.to(**({**kwargs, "dtype": dtype} if dtype is not None else kwargs)) if kwargs or dtype is not None else state.scales,
        zeros=state.zeros.to(**({**kwargs, "dtype": dtype} if dtype is not None else kwargs)) if kwargs or dtype is not None else state.zeros,
        bits=state.bits,
        group_size=state.group_size,
    )


def _resolve_value_group_size(head_dim: int) -> int:
    for candidate in (32, 16, 8, 4, 2, 1):
        if candidate <= head_dim and head_dim % candidate == 0:
            return candidate
    return 1


def _pack_value_data(values: torch.Tensor, *, bits: int) -> torch.Tensor:
    if bits == 2:
        if values.shape[-1] % 4 != 0:
            raise ValueError(f"2-bit value packing expects a multiple of 4 elements, got {values.shape[-1]}.")
        reshaped = values.reshape(*values.shape[:-1], values.shape[-1] // 4, 4)
        return reshaped[..., 0] | (reshaped[..., 1] << 2) | (reshaped[..., 2] << 4) | (reshaped[..., 3] << 6)
    if bits == 4:
        if values.shape[-1] % 2 != 0:
            raise ValueError(f"4-bit value packing expects a multiple of 2 elements, got {values.shape[-1]}.")
        reshaped = values.reshape(*values.shape[:-1], values.shape[-1] // 2, 2)
        return reshaped[..., 0] | (reshaped[..., 1] << 4)
    raise ValueError(f"Unsupported packed TurboQuant value bit-width: {bits}")


def _unpack_value_data(values: torch.Tensor, *, bits: int, target_dim: int) -> torch.Tensor:
    if bits == 2:
        unpacked = torch.stack(
            (
                values & 0x03,
                (values >> 2) & 0x03,
                (values >> 4) & 0x03,
                (values >> 6) & 0x03,
            ),
            dim=-1,
        ).reshape(*values.shape[:-1], values.shape[-1] * 4)
        return unpacked[..., :target_dim]
    if bits == 4:
        unpacked = torch.stack((values & 0x0F, (values >> 4) & 0x0F), dim=-1).reshape(
            *values.shape[:-1],
            values.shape[-1] * 2,
        )
        return unpacked[..., :target_dim]
    raise ValueError(f"Unsupported packed TurboQuant value bit-width: {bits}")


def quantize_turboquant_values(
    values: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> TurboQuantValueState:
    if values.ndim != 3:
        raise ValueError(f"TurboQuant value quantization expects [heads, tokens, dim], got {tuple(values.shape)}")
    if bits not in {2, 3, 4}:
        raise ValueError(f"Unsupported TurboQuant value bit-width: {bits}. Expected 2, 3, or 4.")
    if group_size <= 0 or values.shape[-1] % group_size != 0:
        raise ValueError(
            f"TurboQuant value group_size={group_size} must evenly divide head_dim={values.shape[-1]}."
        )

    storage_bits = 4 if bits in {3, 4} else bits
    levels = float((1 << bits) - 1)
    num_groups = values.shape[-1] // group_size
    grouped = values.float().reshape(values.shape[0], values.shape[1], num_groups, group_size)
    value_min = grouped.min(dim=-1, keepdim=True).values
    value_max = grouped.max(dim=-1, keepdim=True).values
    scales = ((value_max - value_min) / max(levels, 1.0)).clamp(min=1e-8).to(dtype=values.dtype)
    zeros = value_min.to(dtype=values.dtype)
    quantized = ((grouped - zeros.float()) / scales.float()).round().clamp(0, levels).to(torch.uint8)
    packed = _pack_value_data(quantized.reshape(values.shape[0], values.shape[1], values.shape[-1]), bits=storage_bits)
    return TurboQuantValueState(
        data=packed.contiguous(),
        scales=scales.squeeze(-1).contiguous(),
        zeros=zeros.squeeze(-1).contiguous(),
        bits=bits,
        group_size=group_size,
    )


def dequantize_turboquant_values(
    state: TurboQuantValueState,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_dim = int(state.scales.shape[-1]) * int(state.group_size)
    storage_bits = 4 if state.bits in {3, 4} else state.bits
    unpacked = _unpack_value_data(state.data, bits=storage_bits, target_dim=target_dim).float()
    unpacked = unpacked.reshape(*state.data.shape[:-1], int(state.scales.shape[-1]), int(state.group_size))
    values = unpacked * state.scales.float().unsqueeze(-1) + state.zeros.float().unsqueeze(-1)
    values = values.reshape(*state.data.shape[:-1], target_dim)
    kwargs: dict[str, object] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    if kwargs:
        values = values.to(**kwargs)
    return values


def _quantize_turboquant_keys(
    tensor: torch.Tensor,
    *,
    bits: int,
    device: torch.device,
) -> TurboQuantKeyState:
    if tensor.ndim != 3:
        raise ValueError(f"TurboQuant key quantization expects [heads, tokens, dim], got {tuple(tensor.shape)}")
    quantizer = get_turboquant_ip_quantizer(head_dim=int(tensor.shape[-1]), bits=bits, device=device)
    flat = tensor.reshape(-1, int(tensor.shape[-1]))
    mse_indices, norms, qjl_signs, residual_norms = quantizer.quantize(flat)
    shape = tensor.shape[:2]
    return TurboQuantKeyState(
        mse_indices=mse_indices.reshape(*shape, int(tensor.shape[-1])).detach(),
        norms=norms.reshape(*shape, 1).detach(),
        qjl_signs=qjl_signs.reshape(*shape, int(tensor.shape[-1])).detach(),
        residual_norms=residual_norms.reshape(*shape, 1).detach(),
    )


def dequantize_turboquant_keys(
    state: TurboQuantKeyState,
    *,
    bits: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    head_dim = int(state.mse_indices.shape[-1])
    quantizer = get_turboquant_ip_quantizer(head_dim=head_dim, bits=bits, device=device)
    keys = quantizer.dequantize(state.mse_indices, state.norms, state.qjl_signs, state.residual_norms)
    if dtype is not None:
        keys = keys.to(dtype=dtype)
    return keys


def turboquant_key_attention_scores(
    *,
    query: torch.Tensor,
    state: TurboQuantKeyState,
    bits: int,
    device: torch.device,
    num_key_value_groups: int,
    scaling: float,
) -> torch.Tensor:
    if query.ndim != 3:
        raise ValueError(f"TurboQuant key attention expects [query_heads, tokens, dim], got {tuple(query.shape)}")
    num_key_value_heads = int(state.mse_indices.shape[0])
    query_heads, query_len, head_dim = query.shape
    if query_heads % num_key_value_heads != 0:
        raise ValueError(
            f"Query head count {query_heads} is not compatible with TurboQuant KV heads {num_key_value_heads}."
        )
    if query_heads != num_key_value_heads * num_key_value_groups:
        raise ValueError(
            f"Expected {num_key_value_heads * num_key_value_groups} query heads, got {query_heads}."
        )
    if head_dim != int(state.mse_indices.shape[-1]):
        raise ValueError(
            f"TurboQuant key attention head_dim mismatch: expected {state.mse_indices.shape[-1]}, got {head_dim}."
        )

    quantizer = get_turboquant_ip_quantizer(head_dim=head_dim, bits=bits, device=device)
    grouped_query = query.float().reshape(num_key_value_heads, num_key_value_groups, query_len, head_dim)
    zero_signs = torch.zeros_like(state.qjl_signs)
    zero_residual = torch.zeros_like(state.residual_norms)
    mse_only = quantizer.dequantize(state.mse_indices, state.norms, zero_signs, zero_residual).float()
    mse_scores = torch.matmul(grouped_query, mse_only.unsqueeze(1).transpose(-1, -2))

    qjl_signs = 2.0 * state.qjl_signs.float() - 1.0
    projected_query = torch.matmul(grouped_query, quantizer.S.float().transpose(0, 1))
    residual_scale = (
        (math.sqrt(math.pi / 2.0) / float(head_dim))
        * state.norms.float().squeeze(-1)
        * state.residual_norms.float().squeeze(-1)
    )
    qjl_scores = torch.matmul(projected_query, qjl_signs.unsqueeze(1).transpose(-1, -2))
    qjl_scores = qjl_scores * residual_scale[:, None, None, :]
    return (mse_scores + qjl_scores) * scaling


class TurboQuantTensorRow:
    def __init__(self, *, bits: int, residual_len: int) -> None:
        self.bits = int(bits)
        self.residual_len = max(1, int(residual_len))
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self.head_dim: int | None = None
        self._indices: torch.Tensor | None = None
        self._norms: torch.Tensor | None = None
        self._residual: torch.Tensor | None = None
        self._length = 0

    @property
    def length(self) -> int:
        return self._length

    @property
    def initialized(self) -> bool:
        return self.device is not None and self.dtype is not None and self.head_dim is not None

    def append(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 3:
            raise ValueError(f"TurboQuantTensorRow expects [heads, tokens, dim], got {tuple(tensor.shape)}")
        if tensor.shape[1] <= 0:
            return

        if not self.initialized:
            self.device = tensor.device
            self.dtype = tensor.dtype
            self.head_dim = int(tensor.shape[-1])
            self._indices = None
            self._norms = None
            self._residual = tensor.detach().contiguous()
            self._length = int(tensor.shape[1])
        else:
            if tensor.device != self.device:
                tensor = tensor.to(device=self.device)
            if tensor.dtype != self.dtype:
                tensor = tensor.to(dtype=self.dtype)
            if int(tensor.shape[-1]) != self.head_dim:
                raise ValueError(
                    f"TurboQuantTensorRow head_dim mismatch: expected {self.head_dim}, got {int(tensor.shape[-1])}"
                )
            residual = self._residual
            if residual is None:
                self._residual = tensor.detach().contiguous()
            else:
                self._residual = torch.cat((residual, tensor.detach()), dim=1).contiguous()
            self._length += int(tensor.shape[1])

        residual = self._residual
        if residual is None or residual.shape[1] <= self.residual_len:
            return

        overflow = int(residual.shape[1] - self.residual_len)
        to_quantize = residual[:, :overflow, :]
        quantizer = get_turboquant_quantizer(head_dim=int(self.head_dim), bits=self.bits, device=self.device)
        flat = to_quantize.reshape(-1, int(self.head_dim))
        indices, norms = quantizer.quantize(flat)
        indices = indices.reshape(to_quantize.shape).detach()
        norms = norms.reshape(to_quantize.shape[:-1] + (1,)).detach()
        self._indices = indices if self._indices is None else torch.cat((self._indices, indices), dim=1)
        self._norms = norms if self._norms is None else torch.cat((self._norms, norms), dim=1)
        self._residual = residual[:, overflow:, :].contiguous()

    def materialize(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not self.initialized:
            raise RuntimeError("TurboQuantTensorRow is not initialized.")
        assert self.device is not None
        assert self.dtype is not None
        assert self.head_dim is not None

        parts: list[torch.Tensor] = []
        if self._indices is not None and self._norms is not None and self._indices.numel() > 0:
            quantizer = get_turboquant_quantizer(head_dim=self.head_dim, bits=self.bits, device=self.device)
            dequantized = quantizer.dequantize(
                self._indices.reshape(-1, self.head_dim),
                self._norms.reshape(-1, 1),
            ).reshape(self._indices.shape)
            parts.append(dequantized.to(dtype=self.dtype))
        if self._residual is not None and self._residual.numel() > 0:
            parts.append(self._residual)
        if parts:
            materialized = torch.cat(parts, dim=1)
        else:
            materialized = torch.empty((0, 0, self.head_dim), device=self.device, dtype=self.dtype)

        kwargs: dict[str, object] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        if kwargs:
            materialized = materialized.to(**kwargs)
        return materialized

    def clone(self) -> "TurboQuantTensorRow":
        cloned = TurboQuantTensorRow(bits=self.bits, residual_len=self.residual_len)
        cloned.device = self.device
        cloned.dtype = self.dtype
        cloned.head_dim = self.head_dim
        cloned._length = self._length
        cloned._indices = None if self._indices is None else self._indices.clone()
        cloned._norms = None if self._norms is None else self._norms.clone()
        cloned._residual = None if self._residual is None else self._residual.clone()
        return cloned

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TurboQuantTensorRow":
        if not self.initialized:
            return self
        if self._indices is not None and device is not None:
            self._indices = self._indices.to(device=device)
        if self._norms is not None:
            kwargs: dict[str, object] = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = torch.float32
            if kwargs:
                self._norms = self._norms.to(**kwargs)
        if self._residual is not None:
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            if kwargs:
                self._residual = self._residual.to(**kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self


class TurboQuantKVRow:
    def __init__(self, *, bits: int, residual_len: int, value_group_size: int | None = None) -> None:
        self.bits = int(bits)
        self.residual_len = max(1, int(residual_len))
        self.value_group_size = None if value_group_size is None else max(1, int(value_group_size))
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self.head_dim: int | None = None
        self.num_heads: int | None = None
        self._quantized_keys: TurboQuantKeyState | None = None
        self._quantized_values: TurboQuantValueState | None = None
        self._quantized_length = 0
        self._quantized_capacity = 0
        self._residual_keys: torch.Tensor | None = None
        self._residual_values: torch.Tensor | None = None
        self._decode_quantized_keys: torch.Tensor | None = None
        self._decode_quantized_values: torch.Tensor | None = None
        self._decode_quantized_length = 0
        self._decode_quantized_capacity = 0
        self._decode_full_keys: torch.Tensor | None = None
        self._decode_full_values: torch.Tensor | None = None
        self._decode_full_length = 0
        self._decode_full_capacity = 0
        self._decode_splitkv_output: torch.Tensor | None = None
        self._decode_splitkv_partial_stats: torch.Tensor | None = None
        self._decode_splitkv_partial_values: torch.Tensor | None = None
        self._decode_splitkv_blocks_capacity = 0
        self._length = 0

    @property
    def initialized(self) -> bool:
        return (
            self.device is not None
            and self.dtype is not None
            and self.head_dim is not None
            and self.num_heads is not None
        )

    @property
    def length(self) -> int:
        return self._length

    @property
    def quantized_length(self) -> int:
        if self._quantized_keys is None or self._quantized_values is None:
            return 0
        return self._quantized_length

    @property
    def residual_length(self) -> int:
        if self._residual_keys is None:
            return 0
        return int(self._residual_keys.shape[1])

    def _invalidate_decode_cache(self) -> None:
        self._decode_quantized_keys = None
        self._decode_quantized_values = None
        self._decode_quantized_length = 0
        self._decode_quantized_capacity = 0
        self._invalidate_decode_full_cache()

    def _invalidate_decode_full_cache(self) -> None:
        self._decode_full_keys = None
        self._decode_full_values = None
        self._decode_full_length = 0
        self._decode_full_capacity = 0

    def _invalidate_decode_splitkv_workspace(self) -> None:
        self._decode_splitkv_output = None
        self._decode_splitkv_partial_stats = None
        self._decode_splitkv_partial_values = None
        self._decode_splitkv_blocks_capacity = 0

    def _active_quantized_keys(self) -> TurboQuantKeyState | None:
        if self._quantized_keys is None or self._quantized_length <= 0:
            return None
        length = self._quantized_length
        return TurboQuantKeyState(
            mse_indices=self._quantized_keys.mse_indices[:, :length, :],
            norms=self._quantized_keys.norms[:, :length, :],
            qjl_signs=self._quantized_keys.qjl_signs[:, :length, :],
            residual_norms=self._quantized_keys.residual_norms[:, :length, :],
        )

    def _active_quantized_values(self) -> TurboQuantValueState | None:
        if self._quantized_values is None or self._quantized_length <= 0:
            return None
        length = self._quantized_length
        return TurboQuantValueState(
            data=self._quantized_values.data[:, :length, :],
            scales=self._quantized_values.scales[:, :length, :],
            zeros=self._quantized_values.zeros[:, :length, :],
            bits=self._quantized_values.bits,
            group_size=self._quantized_values.group_size,
        )

    @staticmethod
    def _next_decode_capacity(current_capacity: int, required_length: int) -> int:
        if required_length <= current_capacity:
            return current_capacity
        growth_target = max(16, current_capacity * 2)
        return max(required_length, growth_target)

    def _ensure_quantized_capacity(
        self,
        required_length: int,
        *,
        key_template: TurboQuantKeyState,
        value_template: TurboQuantValueState,
    ) -> None:
        if required_length <= 0:
            return
        if (
            self._quantized_keys is not None
            and self._quantized_values is not None
            and self._quantized_capacity >= required_length
        ):
            return

        new_capacity = self._next_decode_capacity(self._quantized_capacity, required_length)
        new_keys = TurboQuantKeyState(
            mse_indices=key_template.mse_indices.new_empty(
                (key_template.mse_indices.shape[0], new_capacity, key_template.mse_indices.shape[2])
            ),
            norms=key_template.norms.new_empty(
                (key_template.norms.shape[0], new_capacity, key_template.norms.shape[2])
            ),
            qjl_signs=key_template.qjl_signs.new_empty(
                (key_template.qjl_signs.shape[0], new_capacity, key_template.qjl_signs.shape[2])
            ),
            residual_norms=key_template.residual_norms.new_empty(
                (
                    key_template.residual_norms.shape[0],
                    new_capacity,
                    key_template.residual_norms.shape[2],
                )
            ),
        )
        new_values = TurboQuantValueState(
            data=value_template.data.new_empty(
                (value_template.data.shape[0], new_capacity, value_template.data.shape[2])
            ),
            scales=value_template.scales.new_empty(
                (value_template.scales.shape[0], new_capacity, value_template.scales.shape[2])
            ),
            zeros=value_template.zeros.new_empty(
                (value_template.zeros.shape[0], new_capacity, value_template.zeros.shape[2])
            ),
            bits=value_template.bits,
            group_size=value_template.group_size,
        )
        if self._quantized_keys is not None and self._quantized_values is not None and self._quantized_length > 0:
            copy_len = min(self._quantized_length, self._quantized_capacity, new_capacity)
            new_keys.mse_indices[:, :copy_len, :].copy_(self._quantized_keys.mse_indices[:, :copy_len, :])
            new_keys.norms[:, :copy_len, :].copy_(self._quantized_keys.norms[:, :copy_len, :])
            new_keys.qjl_signs[:, :copy_len, :].copy_(self._quantized_keys.qjl_signs[:, :copy_len, :])
            new_keys.residual_norms[:, :copy_len, :].copy_(self._quantized_keys.residual_norms[:, :copy_len, :])
            new_values.data[:, :copy_len, :].copy_(self._quantized_values.data[:, :copy_len, :])
            new_values.scales[:, :copy_len, :].copy_(self._quantized_values.scales[:, :copy_len, :])
            new_values.zeros[:, :copy_len, :].copy_(self._quantized_values.zeros[:, :copy_len, :])
        self._quantized_keys = new_keys
        self._quantized_values = new_values
        self._quantized_capacity = new_capacity

    def _append_quantized_state(
        self,
        key_state: TurboQuantKeyState,
        value_state: TurboQuantValueState,
    ) -> None:
        append_length = int(key_state.mse_indices.shape[1])
        if append_length <= 0:
            return
        if int(value_state.data.shape[1]) != append_length:
            raise ValueError("TurboQuant compressed key/value append length mismatch.")
        if self._quantized_keys is None and self._quantized_values is None and self._quantized_length == 0:
            required_capacity = append_length + max(1, self.residual_len)
            self._ensure_quantized_capacity(required_capacity, key_template=key_state, value_template=value_state)
            if self._quantized_keys is None or self._quantized_values is None:
                self._quantized_keys = key_state
                self._quantized_values = value_state
                self._quantized_capacity = append_length
            else:
                self._quantized_keys.mse_indices[:, :append_length, :].copy_(key_state.mse_indices)
                self._quantized_keys.norms[:, :append_length, :].copy_(key_state.norms)
                self._quantized_keys.qjl_signs[:, :append_length, :].copy_(key_state.qjl_signs)
                self._quantized_keys.residual_norms[:, :append_length, :].copy_(key_state.residual_norms)
                self._quantized_values.data[:, :append_length, :].copy_(value_state.data)
                self._quantized_values.scales[:, :append_length, :].copy_(value_state.scales)
                self._quantized_values.zeros[:, :append_length, :].copy_(value_state.zeros)
            self._quantized_length = append_length
            return

        start = self._quantized_length
        required_length = start + append_length
        self._ensure_quantized_capacity(required_length, key_template=key_state, value_template=value_state)
        if self._quantized_keys is None or self._quantized_values is None:
            self._quantized_keys = key_state
            self._quantized_values = value_state
            self._quantized_length = append_length
            self._quantized_capacity = append_length
            return
        end = required_length
        self._quantized_keys.mse_indices[:, start:end, :].copy_(key_state.mse_indices)
        self._quantized_keys.norms[:, start:end, :].copy_(key_state.norms)
        self._quantized_keys.qjl_signs[:, start:end, :].copy_(key_state.qjl_signs)
        self._quantized_keys.residual_norms[:, start:end, :].copy_(key_state.residual_norms)
        self._quantized_values.data[:, start:end, :].copy_(value_state.data)
        self._quantized_values.scales[:, start:end, :].copy_(value_state.scales)
        self._quantized_values.zeros[:, start:end, :].copy_(value_state.zeros)
        self._quantized_length = required_length

    def _ensure_decode_quantized_capacity(
        self,
        required_length: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> None:
        if required_length <= 0:
            return
        if (
            self._decode_quantized_keys is not None
            and self._decode_quantized_values is not None
            and self._decode_quantized_capacity >= required_length
        ):
            return

        new_capacity = self._next_decode_capacity(self._decode_quantized_capacity, required_length)
        new_keys = key_template.new_empty((key_template.shape[0], new_capacity, key_template.shape[2]))
        new_values = value_template.new_empty((value_template.shape[0], new_capacity, value_template.shape[2]))
        if (
            self._decode_quantized_keys is not None
            and self._decode_quantized_values is not None
            and self._decode_quantized_length > 0
        ):
            copy_len = min(self._decode_quantized_length, self._decode_quantized_keys.shape[1], new_capacity)
            new_keys[:, :copy_len, :].copy_(self._decode_quantized_keys[:, :copy_len, :])
            new_values[:, :copy_len, :].copy_(self._decode_quantized_values[:, :copy_len, :])
        self._decode_quantized_keys = new_keys
        self._decode_quantized_values = new_values
        self._decode_quantized_capacity = new_capacity

    def _append_decode_quantized_cache(
        self,
        *,
        prior_quantized_length: int,
        new_keys: torch.Tensor | None,
        new_values: torch.Tensor | None,
    ) -> None:
        if not _KEEP_DECODE_CACHE:
            self._invalidate_decode_cache()
            return
        if new_keys is None or new_values is None or new_keys.shape[1] <= 0:
            return
        if (
            self._decode_quantized_keys is None
            or self._decode_quantized_values is None
        ):
            if prior_quantized_length != 0:
                self._invalidate_decode_cache()
                return
            self._decode_quantized_length = 0
        elif self._decode_quantized_length != prior_quantized_length:
            self._invalidate_decode_cache()
            return

        required_length = prior_quantized_length + int(new_keys.shape[1])
        self._ensure_decode_quantized_capacity(
            required_length,
            key_template=new_keys,
            value_template=new_values,
        )
        if self._decode_quantized_keys is None or self._decode_quantized_values is None:
            self._invalidate_decode_cache()
            return
        self._decode_quantized_keys[:, prior_quantized_length:required_length, :].copy_(new_keys)
        self._decode_quantized_values[:, prior_quantized_length:required_length, :].copy_(new_values)
        self._decode_quantized_length = required_length

    def _ensure_decode_full_capacity(
        self,
        required_length: int,
        *,
        key_template: torch.Tensor,
        value_template: torch.Tensor,
    ) -> None:
        if required_length <= 0:
            return
        if (
            self._decode_full_keys is not None
            and self._decode_full_values is not None
            and self._decode_full_capacity >= required_length
        ):
            return

        new_capacity = self._next_decode_capacity(self._decode_full_capacity, required_length)
        new_keys = key_template.new_empty((key_template.shape[0], new_capacity, key_template.shape[2]))
        new_values = value_template.new_empty((value_template.shape[0], new_capacity, value_template.shape[2]))
        if self._decode_full_keys is not None and self._decode_full_values is not None and self._decode_full_length > 0:
            copy_len = min(self._decode_full_length, self._decode_full_keys.shape[1], new_capacity)
            new_keys[:, :copy_len, :].copy_(self._decode_full_keys[:, :copy_len, :])
            new_values[:, :copy_len, :].copy_(self._decode_full_values[:, :copy_len, :])
        self._decode_full_keys = new_keys
        self._decode_full_values = new_values
        self._decode_full_capacity = new_capacity

    @staticmethod
    def _splitkv_num_blocks(key_len: int, block_size: int) -> int:
        effective_block_size = min(max(int(block_size), 16), int(key_len))
        return (int(key_len) + effective_block_size - 1) // effective_block_size

    def _ensure_decode_splitkv_workspace(
        self,
        *,
        query: torch.Tensor,
        key_len: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if key_len <= 0:
            raise ValueError("TurboQuant split-KV decode workspace requires a non-empty KV cache.")
        if query.ndim != 4:
            raise ValueError(f"TurboQuant split-KV decode workspace expects 4D query, got {tuple(query.shape)}")
        query_heads = int(query.shape[1])
        head_dim = int(query.shape[3])
        num_blocks = self._splitkv_num_blocks(key_len, block_size)
        need_output = (
            self._decode_splitkv_output is None
            or tuple(self._decode_splitkv_output.shape) != tuple(query.shape)
            or self._decode_splitkv_output.device != query.device
            or self._decode_splitkv_output.dtype != query.dtype
        )
        need_partials = (
            self._decode_splitkv_partial_stats is None
            or self._decode_splitkv_partial_values is None
            or self._decode_splitkv_blocks_capacity < num_blocks
            or self._decode_splitkv_partial_stats.device != query.device
            or self._decode_splitkv_partial_values.device != query.device
            or self._decode_splitkv_partial_stats.shape[:2] != (1, query_heads)
            or self._decode_splitkv_partial_values.shape[:2] != (1, query_heads)
            or int(self._decode_splitkv_partial_stats.shape[3]) != 2
            or int(self._decode_splitkv_partial_values.shape[3]) != head_dim
        )
        if need_output:
            self._decode_splitkv_output = query.new_empty(query.shape)
        if need_partials:
            blocks_capacity = self._next_decode_capacity(self._decode_splitkv_blocks_capacity, num_blocks)
            self._decode_splitkv_partial_stats = torch.empty(
                (1, query_heads, blocks_capacity, 2),
                device=query.device,
                dtype=torch.float32,
            )
            self._decode_splitkv_partial_values = torch.empty(
                (1, query_heads, blocks_capacity, head_dim),
                device=query.device,
                dtype=torch.float32,
            )
            self._decode_splitkv_blocks_capacity = blocks_capacity
        assert self._decode_splitkv_output is not None
        assert self._decode_splitkv_partial_stats is not None
        assert self._decode_splitkv_partial_values is not None
        return (
            self._decode_splitkv_output,
            self._decode_splitkv_partial_stats,
            self._decode_splitkv_partial_values,
        )

    def _copy_decode_full_range(
        self,
        *,
        start: int,
        keys: torch.Tensor | None,
        values: torch.Tensor | None,
    ) -> None:
        if keys is None or values is None or keys.shape[1] <= 0:
            return
        required_length = start + int(keys.shape[1])
        self._ensure_decode_full_capacity(required_length, key_template=keys, value_template=values)
        if self._decode_full_keys is None or self._decode_full_values is None:
            self._invalidate_decode_full_cache()
            return
        self._decode_full_keys[:, start:required_length, :].copy_(keys)
        self._decode_full_values[:, start:required_length, :].copy_(values)
        self._decode_full_length = max(self._decode_full_length, required_length)

    def _materialize_decode_full_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            raise RuntimeError("TurboQuantKVRow is not initialized.")
        if self._length <= 0:
            assert self.device is not None
            assert self.dtype is not None
            assert self.head_dim is not None
            return (
                torch.empty((0, 0, self.head_dim), device=self.device, dtype=self.dtype),
                torch.empty((0, 0, self.head_dim), device=self.device, dtype=self.dtype),
            )
        if (
            _KEEP_DECODE_CACHE
            and self._decode_full_keys is not None
            and self._decode_full_values is not None
            and self._decode_full_length == self._length
        ):
            return (
                self._decode_full_keys[:, : self._length, :],
                self._decode_full_values[:, : self._length, :],
            )

        keys, values = self.materialize()
        if not _KEEP_DECODE_CACHE:
            return keys, values
        self._ensure_decode_full_capacity(self._length, key_template=keys, value_template=values)
        if self._decode_full_keys is None or self._decode_full_values is None:
            return keys, values
        self._decode_full_keys[:, : self._length, :].copy_(keys)
        self._decode_full_values[:, : self._length, :].copy_(values)
        self._decode_full_length = self._length
        return (
            self._decode_full_keys[:, : self._length, :],
            self._decode_full_values[:, : self._length, :],
        )

    def _materialize_quantized_prefix(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        quantized_len = self.quantized_length
        quantized_key_state = self._active_quantized_keys()
        quantized_value_state = self._active_quantized_values()
        if (
            quantized_key_state is None
            or quantized_value_state is None
            or quantized_len <= 0
        ):
            self._invalidate_decode_cache()
            return None, None

        if (
            _KEEP_DECODE_CACHE
            and
            self._decode_quantized_keys is not None
            and self._decode_quantized_values is not None
            and self._decode_quantized_length == quantized_len
        ):
            return (
                self._decode_quantized_keys[:, :quantized_len, :],
                self._decode_quantized_values[:, :quantized_len, :],
            )

        assert self.device is not None
        assert self.dtype is not None
        with xpu_profile_region("turboquant_dequant"):
            quantized_keys = dequantize_turboquant_keys(
                quantized_key_state,
                bits=self.bits,
                device=self.device,
                dtype=self.dtype,
            ).contiguous()
            quantized_values = dequantize_turboquant_values(
                quantized_value_state,
                device=self.device,
                dtype=self.dtype,
            ).contiguous()
        if _KEEP_DECODE_CACHE:
            self._ensure_decode_quantized_capacity(
                quantized_len,
                key_template=quantized_keys,
                value_template=quantized_values,
            )
            if self._decode_quantized_keys is not None and self._decode_quantized_values is not None:
                self._decode_quantized_keys[:, :quantized_len, :].copy_(quantized_keys)
                self._decode_quantized_values[:, :quantized_len, :].copy_(quantized_values)
            self._decode_quantized_length = quantized_len
        else:
            self._invalidate_decode_cache()
        if (
            _KEEP_DECODE_CACHE
            and self._decode_quantized_keys is not None
            and self._decode_quantized_values is not None
        ):
            return (
                self._decode_quantized_keys[:, :quantized_len, :],
                self._decode_quantized_values[:, :quantized_len, :],
            )
        return quantized_keys, quantized_values

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if key_states.ndim != 3 or value_states.ndim != 3:
            raise ValueError(
                "TurboQuantKVRow expects key/value tensors shaped [heads, tokens, dim]."
            )
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"TurboQuantKVRow key/value shape mismatch: {tuple(key_states.shape)} vs {tuple(value_states.shape)}"
            )
        if key_states.shape[1] <= 0:
            return
        append_length = int(key_states.shape[1])
        previous_length = self._length

        if not self.initialized:
            self.device = key_states.device
            self.dtype = key_states.dtype
            self.head_dim = int(key_states.shape[-1])
            self.num_heads = int(key_states.shape[0])
            self.value_group_size = self.value_group_size or _resolve_value_group_size(self.head_dim)
            self._residual_keys = key_states.detach().contiguous()
            self._residual_values = value_states.detach().contiguous()
            self._length = append_length
        else:
            assert self.device is not None
            assert self.dtype is not None
            assert self.head_dim is not None
            assert self.num_heads is not None
            if int(key_states.shape[0]) != self.num_heads or int(key_states.shape[-1]) != self.head_dim:
                raise ValueError(
                    f"TurboQuantKVRow shape mismatch: expected heads={self.num_heads} head_dim={self.head_dim}, "
                    f"got {tuple(key_states.shape)}"
                )
            if key_states.device != self.device:
                key_states = key_states.to(device=self.device)
                value_states = value_states.to(device=self.device)
            if key_states.dtype != self.dtype:
                key_states = key_states.to(dtype=self.dtype)
                value_states = value_states.to(dtype=self.dtype)
            self._residual_keys = (
                key_states.detach().contiguous()
                if self._residual_keys is None
                else torch.cat((self._residual_keys, key_states.detach()), dim=1).contiguous()
            )
            self._residual_values = (
                value_states.detach().contiguous()
                if self._residual_values is None
                else torch.cat((self._residual_values, value_states.detach()), dim=1).contiguous()
            )
            self._length += append_length

        if _KEEP_DECODE_CACHE and self._decode_full_keys is not None and self._decode_full_values is not None:
            self._copy_decode_full_range(
                start=previous_length,
                keys=key_states.detach().contiguous(),
                values=value_states.detach().contiguous(),
            )

        residual_keys = self._residual_keys
        residual_values = self._residual_values
        if residual_keys is None or residual_values is None:
            return
        if residual_keys.shape[1] <= self.residual_len:
            return

        overflow = int(residual_keys.shape[1] - self.residual_len)
        keys_to_quantize = residual_keys[:, :overflow, :]
        values_to_quantize = residual_values[:, :overflow, :]
        assert self.device is not None
        assert self.value_group_size is not None
        prior_quant_len = self.quantized_length
        quantized_keys = _quantize_turboquant_keys(keys_to_quantize, bits=self.bits, device=self.device)
        quantized_values = quantize_turboquant_values(
            values_to_quantize,
            bits=self.bits,
            group_size=self.value_group_size,
        )
        if _KEEP_DECODE_CACHE:
            with xpu_profile_region("turboquant_dequant"):
                new_decode_keys = dequantize_turboquant_keys(
                    quantized_keys,
                    bits=self.bits,
                    device=self.device,
                    dtype=self.dtype,
                ).contiguous()
                new_decode_values = dequantize_turboquant_values(
                    quantized_values,
                    device=self.device,
                    dtype=self.dtype,
                ).contiguous()
        else:
            new_decode_keys = None
            new_decode_values = None
        self._append_quantized_state(quantized_keys, quantized_values)
        self._residual_keys = residual_keys[:, overflow:, :].contiguous()
        self._residual_values = residual_values[:, overflow:, :].contiguous()
        if _SYNC_AFTER_QUANTIZE and self.device is not None and self.device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()

        # Incrementally extend decode tensors instead of invalidating or concatenating the full prefix.
        self._append_decode_quantized_cache(
            prior_quantized_length=prior_quant_len,
            new_keys=new_decode_keys,
            new_values=new_decode_values,
        )
        if _KEEP_DECODE_CACHE and self._decode_full_keys is not None and self._decode_full_values is not None:
            self._copy_decode_full_range(start=prior_quant_len, keys=new_decode_keys, values=new_decode_values)
            self._decode_full_length = self._length

    def materialize(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            raise RuntimeError("TurboQuantKVRow is not initialized.")
        assert self.device is not None
        assert self.dtype is not None
        assert self.head_dim is not None

        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        if self._quantized_keys is not None and self._quantized_values is not None and self.quantized_length > 0:
            quantized_keys, quantized_values = self._materialize_quantized_prefix()
            if quantized_keys is not None and quantized_values is not None:
                key_parts.append(quantized_keys)
                value_parts.append(quantized_values)
        if self._residual_keys is not None and self._residual_values is not None and self._residual_keys.numel() > 0:
            key_parts.append(self._residual_keys)
            value_parts.append(self._residual_values)
        if key_parts:
            keys = torch.cat(key_parts, dim=1)
            values = torch.cat(value_parts, dim=1)
        else:
            keys = torch.empty((0, 0, self.head_dim), device=self.device, dtype=self.dtype)
            values = torch.empty((0, 0, self.head_dim), device=self.device, dtype=self.dtype)

        kwargs: dict[str, object] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        if kwargs:
            keys = keys.to(**kwargs)
            values = values.to(**kwargs)
        return keys, values

    def decode_attention(
        self,
        query: torch.Tensor,
        *,
        scaling: float,
        num_key_value_groups: int,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.initialized:
            raise RuntimeError("TurboQuantKVRow is not initialized.")
        if query.ndim != 3 or query.shape[1] != 1:
            raise ValueError(
                f"TurboQuantKVRow single-token decode expects [query_heads, 1, dim], got {tuple(query.shape)}"
            )
        assert self.device is not None
        assert self.dtype is not None
        assert self.head_dim is not None
        assert self.num_heads is not None
        if int(query.shape[-1]) != self.head_dim:
            raise ValueError(
                f"TurboQuantKVRow decode head_dim mismatch: expected {self.head_dim}, got {int(query.shape[-1])}"
            )
        if gate is not None and (
            gate.ndim != 3 or int(gate.shape[0]) != int(query.shape[0]) or int(gate.shape[1]) != 1 or int(gate.shape[2]) != self.head_dim
        ):
            raise ValueError(
                f"TurboQuantKVRow decode gate expects [query_heads, 1, head_dim], got {tuple(gate.shape)}"
            )

        if self.length > 0:
            decode_keys, decode_values = self._materialize_decode_full_cache()
            if query.device.type == "xpu":
                query_4d = query.unsqueeze(0)
                key_4d = decode_keys.unsqueeze(0)
                value_4d = decode_values.unsqueeze(0)
                output, partial_stats, partial_values = self._ensure_decode_splitkv_workspace(
                    query=query_4d,
                    key_len=int(key_4d.shape[2]),
                    block_size=_DECODE_SPLITKV_BLOCK_SIZE,
                )
                run_gqa_decode_splitkv_fused_out(
                    query=query_4d,
                    key=key_4d,
                    value=value_4d,
                    output=output,
                    partial_stats=partial_stats,
                    partial_values=partial_values,
                    scaling=scaling,
                    block_size=_DECODE_SPLITKV_BLOCK_SIZE,
                    gate=None if gate is None else gate.unsqueeze(0),
                )
                return output.squeeze(0)
            # FP16/BF16 matmul on XPU/GPU is typically much faster than promoting everything to FP32; keep softmax in
            # FP32 for stability on long contexts.
            compute_dtype = (
                query.dtype
                if query.dtype in (torch.float16, torch.bfloat16)
                else torch.float32
            )
            grouped_shape = (self.num_heads, num_key_value_groups, int(query.shape[1]), self.head_dim)
            grouped_query = query.to(dtype=compute_dtype).reshape(grouped_shape)
            attn_scores = torch.matmul(
                grouped_query,
                decode_keys.to(dtype=compute_dtype).unsqueeze(1).transpose(-1, -2),
            ) * scaling
            attn_probs = torch.softmax(attn_scores.to(torch.float32), dim=-1).to(dtype=compute_dtype)
            attn_output = torch.matmul(attn_probs, decode_values.to(dtype=compute_dtype).unsqueeze(1))
            if gate is not None:
                attn_output = attn_output * torch.sigmoid(gate.to(dtype=compute_dtype).reshape(grouped_shape))
            return attn_output.reshape(int(query.shape[0]), 1, self.head_dim).to(dtype=query.dtype)

        compute_dtype = (
            query.dtype
            if query.dtype in (torch.float16, torch.bfloat16)
            else torch.float32
        )
        grouped_shape = (self.num_heads, num_key_value_groups, int(query.shape[1]), self.head_dim)
        grouped_query = query.to(dtype=compute_dtype).reshape(grouped_shape)
        score_parts: list[torch.Tensor] = []
        quantized_len = self.quantized_length
        residual_len = self.residual_length
        quantized_keys = None
        quantized_values = None

        if self._quantized_keys is not None and self._quantized_values is not None and quantized_len > 0:
            quantized_keys, quantized_values = self._materialize_quantized_prefix()
            if quantized_keys is not None:
                quantized_scores = torch.matmul(
                    grouped_query,
                    quantized_keys.to(dtype=compute_dtype).unsqueeze(1).transpose(-1, -2),
                ) * scaling
                score_parts.append(quantized_scores)
        if self._residual_keys is not None and self._residual_values is not None and residual_len > 0:
            residual_scores = torch.matmul(
                grouped_query,
                self._residual_keys.to(dtype=compute_dtype).unsqueeze(1).transpose(-1, -2),
            ) * scaling
            score_parts.append(residual_scores)

        if not score_parts:
            return query.new_zeros((int(query.shape[0]), 1, self.head_dim))

        attn_scores = torch.cat(score_parts, dim=-1)
        attn_probs = torch.softmax(attn_scores.to(torch.float32), dim=-1).to(dtype=compute_dtype)
        attn_output: torch.Tensor | None = None
        offset = 0
        if quantized_values is not None and quantized_len > 0:
            attn_output = torch.matmul(
                attn_probs[..., offset : offset + quantized_len],
                quantized_values.to(dtype=compute_dtype).unsqueeze(1),
            )
            offset += quantized_len
        if self._residual_values is not None and residual_len > 0:
            residual_output = torch.matmul(
                attn_probs[..., offset : offset + residual_len],
                self._residual_values.to(dtype=compute_dtype).unsqueeze(1),
            )
            attn_output = residual_output if attn_output is None else attn_output + residual_output
        if attn_output is None:
            return query.new_zeros((int(query.shape[0]), 1, self.head_dim))
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate.to(dtype=compute_dtype).reshape(grouped_shape))
        return attn_output.reshape(int(query.shape[0]), 1, self.head_dim).to(dtype=query.dtype)

    def clone(self) -> "TurboQuantKVRow":
        cloned = TurboQuantKVRow(
            bits=self.bits,
            residual_len=self.residual_len,
            value_group_size=self.value_group_size,
        )
        cloned.device = self.device
        cloned.dtype = self.dtype
        cloned.head_dim = self.head_dim
        cloned.num_heads = self.num_heads
        cloned._quantized_keys = _clone_key_state(self._active_quantized_keys())
        cloned._quantized_values = _clone_value_state(self._active_quantized_values())
        cloned._quantized_length = self._quantized_length
        cloned._quantized_capacity = self._quantized_length
        cloned._residual_keys = None if self._residual_keys is None else self._residual_keys.clone()
        cloned._residual_values = None if self._residual_values is None else self._residual_values.clone()
        cloned._decode_quantized_keys = (
            None if self._decode_quantized_keys is None else self._decode_quantized_keys.clone()
        )
        cloned._decode_quantized_values = (
            None if self._decode_quantized_values is None else self._decode_quantized_values.clone()
        )
        cloned._decode_quantized_length = self._decode_quantized_length
        cloned._decode_quantized_capacity = self._decode_quantized_capacity
        cloned._decode_full_keys = None if self._decode_full_keys is None else self._decode_full_keys.clone()
        cloned._decode_full_values = None if self._decode_full_values is None else self._decode_full_values.clone()
        cloned._decode_full_length = self._decode_full_length
        cloned._decode_full_capacity = self._decode_full_capacity
        cloned._length = self._length
        return cloned

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "TurboQuantKVRow":
        if not self.initialized:
            return self
        self._quantized_keys = _move_key_state(self._quantized_keys, device=device, dtype=dtype)
        self._quantized_values = _move_value_state(self._quantized_values, device=device, dtype=dtype)
        if self._decode_quantized_keys is not None:
            kwargs: dict[str, object] = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            if kwargs:
                self._decode_quantized_keys = self._decode_quantized_keys.to(**kwargs)
        if self._decode_quantized_values is not None:
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            if kwargs:
                self._decode_quantized_values = self._decode_quantized_values.to(**kwargs)
        if self._decode_full_keys is not None:
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            if kwargs:
                self._decode_full_keys = self._decode_full_keys.to(**kwargs)
        if self._decode_full_values is not None:
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            if kwargs:
                self._decode_full_values = self._decode_full_values.to(**kwargs)
        if device is not None or dtype is not None:
            self._invalidate_decode_splitkv_workspace()
        if self._residual_keys is not None:
            kwargs: dict[str, object] = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            if kwargs:
                self._residual_keys = self._residual_keys.to(**kwargs)
        if self._residual_values is not None:
            kwargs = {}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None:
                kwargs["dtype"] = dtype
            if kwargs:
                self._residual_values = self._residual_values.to(**kwargs)
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self
