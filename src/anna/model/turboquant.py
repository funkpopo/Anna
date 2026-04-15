from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch

_mse_quantizers: dict[tuple[int, int, str], Any] = {}
_ip_quantizers: dict[tuple[int, int, str], Any] = {}

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


@dataclass(slots=True)
class TurboQuantKVUsage:
    total_bytes: int = 0
    dense_equivalent_bytes: int = 0
    quantized_bytes: int = 0
    residual_bytes: int = 0
    quantized_tokens: int = 0
    residual_tokens: int = 0
    rows: int = 0

    def add(self, other: "TurboQuantKVUsage") -> "TurboQuantKVUsage":
        return TurboQuantKVUsage(
            total_bytes=self.total_bytes + other.total_bytes,
            dense_equivalent_bytes=self.dense_equivalent_bytes + other.dense_equivalent_bytes,
            quantized_bytes=self.quantized_bytes + other.quantized_bytes,
            residual_bytes=self.residual_bytes + other.residual_bytes,
            quantized_tokens=self.quantized_tokens + other.quantized_tokens,
            residual_tokens=self.residual_tokens + other.residual_tokens,
            rows=self.rows + other.rows,
        )


def turboquant_is_available() -> bool:
    try:
        from turboquant.core import TurboQuantIP, TurboQuantMSE  # noqa: F401
    except ImportError:
        return False
    return True


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


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


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


def _concat_key_states(left: TurboQuantKeyState | None, right: TurboQuantKeyState) -> TurboQuantKeyState:
    if left is None:
        return right
    return TurboQuantKeyState(
        mse_indices=torch.cat((left.mse_indices, right.mse_indices), dim=1),
        norms=torch.cat((left.norms, right.norms), dim=1),
        qjl_signs=torch.cat((left.qjl_signs, right.qjl_signs), dim=1),
        residual_norms=torch.cat((left.residual_norms, right.residual_norms), dim=1),
    )


def _concat_value_states(left: TurboQuantValueState | None, right: TurboQuantValueState) -> TurboQuantValueState:
    if left is None:
        return right
    return TurboQuantValueState(
        data=torch.cat((left.data, right.data), dim=1),
        scales=torch.cat((left.scales, right.scales), dim=1),
        zeros=torch.cat((left.zeros, right.zeros), dim=1),
        bits=right.bits,
        group_size=right.group_size,
    )


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
        self._residual_keys: torch.Tensor | None = None
        self._residual_values: torch.Tensor | None = None
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
        if self._quantized_keys is None:
            return 0
        return int(self._quantized_keys.mse_indices.shape[1])

    @property
    def residual_length(self) -> int:
        if self._residual_keys is None:
            return 0
        return int(self._residual_keys.shape[1])

    def usage(self) -> TurboQuantKVUsage:
        if not self.initialized:
            return TurboQuantKVUsage()
        assert self.dtype is not None
        assert self.head_dim is not None
        assert self.num_heads is not None

        quantized_key_bytes = 0
        if self._quantized_keys is not None:
            quantized_key_bytes = (
                _tensor_nbytes(self._quantized_keys.mse_indices)
                + _tensor_nbytes(self._quantized_keys.norms)
                + _tensor_nbytes(self._quantized_keys.qjl_signs)
                + _tensor_nbytes(self._quantized_keys.residual_norms)
            )
        quantized_value_bytes = 0
        if self._quantized_values is not None:
            quantized_value_bytes = (
                _tensor_nbytes(self._quantized_values.data)
                + _tensor_nbytes(self._quantized_values.scales)
                + _tensor_nbytes(self._quantized_values.zeros)
            )
        residual_key_bytes = _tensor_nbytes(self._residual_keys)
        residual_value_bytes = _tensor_nbytes(self._residual_values)
        quantized_bytes = quantized_key_bytes + quantized_value_bytes
        residual_bytes = residual_key_bytes + residual_value_bytes
        dense_equivalent_bytes = (
            int(self.length)
            * int(self.num_heads)
            * int(self.head_dim)
            * torch.empty((), dtype=self.dtype).element_size()
            * 2
        )
        return TurboQuantKVUsage(
            total_bytes=quantized_bytes + residual_bytes,
            dense_equivalent_bytes=dense_equivalent_bytes,
            quantized_bytes=quantized_bytes,
            residual_bytes=residual_bytes,
            quantized_tokens=self.quantized_length,
            residual_tokens=self.residual_length,
            rows=1 if self.length > 0 else 0,
        )

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

        if not self.initialized:
            self.device = key_states.device
            self.dtype = key_states.dtype
            self.head_dim = int(key_states.shape[-1])
            self.num_heads = int(key_states.shape[0])
            self.value_group_size = self.value_group_size or _resolve_value_group_size(self.head_dim)
            self._residual_keys = key_states.detach().contiguous()
            self._residual_values = value_states.detach().contiguous()
            self._length = int(key_states.shape[1])
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
            self._length += int(key_states.shape[1])

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
        quantized_keys = _quantize_turboquant_keys(keys_to_quantize, bits=self.bits, device=self.device)
        quantized_values = quantize_turboquant_values(
            values_to_quantize,
            bits=self.bits,
            group_size=self.value_group_size,
        )
        self._quantized_keys = _concat_key_states(self._quantized_keys, quantized_keys)
        self._quantized_values = _concat_value_states(self._quantized_values, quantized_values)
        self._residual_keys = residual_keys[:, overflow:, :].contiguous()
        self._residual_values = residual_values[:, overflow:, :].contiguous()

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
            key_parts.append(
                dequantize_turboquant_keys(
                    self._quantized_keys,
                    bits=self.bits,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            value_parts.append(
                dequantize_turboquant_values(
                    self._quantized_values,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
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

        grouped_shape = (self.num_heads, num_key_value_groups, int(query.shape[1]), self.head_dim)
        grouped_query = query.float().reshape(grouped_shape)
        score_parts: list[torch.Tensor] = []
        quantized_len = self.quantized_length
        residual_len = self.residual_length

        if self._quantized_keys is not None and self._quantized_values is not None and quantized_len > 0:
            score_parts.append(
                turboquant_key_attention_scores(
                    query=query,
                    state=self._quantized_keys,
                    bits=self.bits,
                    device=self.device,
                    num_key_value_groups=num_key_value_groups,
                    scaling=scaling,
                )
            )
        if self._residual_keys is not None and self._residual_values is not None and residual_len > 0:
            residual_scores = torch.matmul(
                grouped_query,
                self._residual_keys.float().unsqueeze(1).transpose(-1, -2),
            ) * scaling
            score_parts.append(residual_scores)

        if not score_parts:
            return query.new_zeros((int(query.shape[0]), 1, self.head_dim))

        attn_scores = torch.cat(score_parts, dim=-1)
        attn_probs = torch.softmax(attn_scores.float(), dim=-1)
        attn_output = query.new_zeros(grouped_shape)
        offset = 0
        if self._quantized_values is not None and quantized_len > 0:
            quantized_values = dequantize_turboquant_values(
                self._quantized_values,
                device=self.device,
                dtype=self.dtype,
            ).float()
            attn_output = attn_output + torch.matmul(
                attn_probs[..., offset : offset + quantized_len],
                quantized_values.unsqueeze(1),
            ).to(dtype=query.dtype)
            offset += quantized_len
        if self._residual_values is not None and residual_len > 0:
            attn_output = attn_output + torch.matmul(
                attn_probs[..., offset : offset + residual_len],
                self._residual_values.float().unsqueeze(1),
            ).to(dtype=query.dtype)
        return attn_output.reshape(int(query.shape[0]), 1, self.head_dim)

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
        cloned._quantized_keys = _clone_key_state(self._quantized_keys)
        cloned._quantized_values = _clone_value_state(self._quantized_values)
        cloned._residual_keys = None if self._residual_keys is None else self._residual_keys.clone()
        cloned._residual_values = None if self._residual_values is None else self._residual_values.clone()
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
