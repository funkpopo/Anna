from __future__ import annotations

from typing import Any

import numpy as np
import torch

_quantizers: dict[tuple[int, int, str], Any] = {}

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid


def turboquant_is_available() -> bool:
    try:
        from turboquant.core import TurboQuantMSE  # noqa: F401
    except ImportError:
        return False
    return True


def _require_turboquant_mse():
    try:
        from turboquant.core import TurboQuantMSE
    except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
        raise RuntimeError(
            "TurboQuant KV-cache compression requires the optional 'turboquant' dependency."
        ) from exc
    return TurboQuantMSE


def get_turboquant_quantizer(*, head_dim: int, bits: int, device: torch.device) -> Any:
    TurboQuantMSE = _require_turboquant_mse()
    key = (int(head_dim), int(bits), str(device))
    quantizer = _quantizers.get(key)
    if quantizer is None:
        quantizer = TurboQuantMSE(dim=int(head_dim), bits=int(bits), device=str(device), seed=42)
        _quantizers[key] = quantizer
    return quantizer


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
