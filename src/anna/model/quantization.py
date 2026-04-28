from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.qwen3_5_text_config import QuantizationConfig

_INT4_VALUES_PER_PACK = 8
_AUTO_ROUND_QUANT_METHODS = frozenset({"auto-round", "auto_round"})
_AUTO_ROUND_PACKING_FORMATS = frozenset({"auto_round:auto_gptq"})
_FLOAT_OVERRIDE_DATA_TYPES = frozenset({"fp", "float", "float16", "fp16", "bfloat16", "bf16"})
_REGEX_META_CHARS = frozenset("\\[](){}?*+^$|")
_XPU_INT4_MATMUL_STRATEGIES = frozenset({"auto", "torch", "dequant", "sycl"})
_XPU_INT4_LAYOUT_CACHE_VERSION = 1
_XPU_INT4_LAYOUT_NAME = "anna_xpu_int4_linear_v1"

logger = logging.getLogger(__name__)


def _empty_parameter(
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | None = None,
) -> nn.Parameter:
    dtype = dtype or torch.float32
    return nn.Parameter(torch.empty(*shape, dtype=dtype, device=device), requires_grad=False)


def _int4_shifts(*, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.arange(0, 32, 4, device=device, dtype=torch.int32)


def _dtype_cache_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _safe_cache_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "module"


def _hash_tensor(hasher: "hashlib._Hash", tensor: torch.Tensor | None) -> None:
    if tensor is None:
        hasher.update(b"<none>")
        return
    materialized = tensor.detach().to(device="cpu").contiguous()
    hasher.update(str(materialized.dtype).encode("utf-8"))
    hasher.update(json.dumps(tuple(materialized.shape)).encode("utf-8"))
    hasher.update(materialized.numpy().tobytes())


def _linear_cache_fingerprint(module: nn.Module, *, group_size: int, compute_dtype: torch.dtype) -> str:
    hasher = hashlib.sha256()
    hasher.update(type(module).__qualname__.encode("utf-8"))
    hasher.update(str(group_size).encode("utf-8"))
    hasher.update(_dtype_cache_name(compute_dtype).encode("utf-8"))
    hasher.update(_XPU_INT4_LAYOUT_NAME.encode("utf-8"))
    hasher.update(str(getattr(module, "in_features", "")).encode("utf-8"))
    hasher.update(str(getattr(module, "out_features", "")).encode("utf-8"))
    if isinstance(module, AutoRoundGPTQLinear):
        hasher.update(str(module.group_size).encode("utf-8"))
        _hash_tensor(hasher, module.qweight)
        _hash_tensor(hasher, module.qzeros)
        _hash_tensor(hasher, module.scales)
        _hash_tensor(hasher, module.bias)
    elif isinstance(module, AWQLinear):
        hasher.update(str(module.group_size).encode("utf-8"))
        _hash_tensor(hasher, module.qweight)
        _hash_tensor(hasher, module.qzeros)
        _hash_tensor(hasher, module.scales)
        _hash_tensor(hasher, module.bias)
        _hash_tensor(hasher, module.weight)
    else:
        weight, bias = _extract_dense_weight_bias(module)
        _hash_tensor(hasher, weight)
        _hash_tensor(hasher, bias)
    return hasher.hexdigest()


def _cache_file_for_linear(cache_dir: Path, module_name: str, fingerprint: str) -> Path:
    return cache_dir / f"{_safe_cache_key(module_name)}-{fingerprint[:16]}.pt"


def _load_xpu_int4_linear_from_cache(
    cache_path: Path,
    *,
    device: torch.device | str | None,
    compute_dtype: torch.dtype,
    fingerprint: str,
) -> "XPUInt4Linear | None":
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception:
        logger.warning("Ignoring unreadable XPU int4 cache file: %s", cache_path, exc_info=True)
        return None
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    if metadata.get("version") != _XPU_INT4_LAYOUT_CACHE_VERSION or metadata.get("fingerprint") != fingerprint:
        return None
    if metadata.get("layout") != _XPU_INT4_LAYOUT_NAME:
        return None
    try:
        quantized = XPUInt4Linear(
            int(metadata["in_features"]),
            int(metadata["out_features"]),
            group_size=int(metadata["group_size"]),
            bias=bool(metadata["has_bias"]),
            compute_dtype=compute_dtype,
            device=device,
            padded_in_features=int(metadata["padded_in_features"]),
        )
        with torch.no_grad():
            quantized.qweight.copy_(payload["qweight"].to(device=quantized.qweight.device))
            quantized.qscale.copy_(payload["qscale"].to(device=quantized.qscale.device))
            quantized.qzeros.copy_(payload["qzeros"].to(device=quantized.qzeros.device))
            if quantized.bias is not None:
                quantized.bias.copy_(payload["bias"].to(device=quantized.bias.device, dtype=quantized.bias.dtype))
    except Exception:
        logger.warning("Ignoring incompatible XPU int4 cache file: %s", cache_path, exc_info=True)
        return None
    logger.info("Loaded XPU int4 layout cache: %s", cache_path)
    return quantized


def _save_xpu_int4_linear_to_cache(
    cache_path: Path,
    quantized: "XPUInt4Linear",
    *,
    fingerprint: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "version": _XPU_INT4_LAYOUT_CACHE_VERSION,
            "layout": _XPU_INT4_LAYOUT_NAME,
            "fingerprint": fingerprint,
            "in_features": quantized.in_features,
            "out_features": quantized.out_features,
            "group_size": quantized.group_size,
            "padded_in_features": quantized.padded_in_features,
            "compute_dtype": _dtype_cache_name(quantized.compute_dtype),
            "has_bias": quantized.bias is not None,
        },
        "qweight": quantized.qweight.detach().to(device="cpu").contiguous(),
        "qscale": quantized.qscale.detach().to(device="cpu").contiguous(),
        "qzeros": quantized.qzeros.detach().to(device="cpu").contiguous(),
        "bias": None if quantized.bias is None else quantized.bias.detach().to(device="cpu").contiguous(),
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(cache_path)


def _unpack_int4_last_dim(packed: torch.Tensor) -> torch.Tensor:
    shifts = _int4_shifts(device=packed.device)
    return ((packed.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF).reshape(
        *packed.shape[:-1],
        packed.shape[-1] * _INT4_VALUES_PER_PACK,
    )


def _unpack_int4_first_dim(packed: torch.Tensor) -> torch.Tensor:
    shifts = _int4_shifts(device=packed.device)
    unpacked = ((packed.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF)
    unpacked = unpacked.permute(0, 2, 1).contiguous()
    return unpacked.reshape(packed.shape[0] * _INT4_VALUES_PER_PACK, packed.shape[1])


class DenseLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _empty_parameter((out_features, in_features), device=device)
        if bias:
            self.bias = _empty_parameter((out_features,), device=device)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, self.weight.to(dtype=x.dtype), bias)


class AutoRoundGPTQLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        bias: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if bits != 4:
            raise ValueError("AutoRoundGPTQLinear currently only supports 4-bit weights.")
        if group_size <= 0:
            raise ValueError("AutoRoundGPTQLinear requires a positive group_size.")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.compute_dtype = compute_dtype
        self.packed_in_features = (in_features + _INT4_VALUES_PER_PACK - 1) // _INT4_VALUES_PER_PACK
        self.packed_out_features = (out_features + _INT4_VALUES_PER_PACK - 1) // _INT4_VALUES_PER_PACK
        self.group_count = (in_features + group_size - 1) // group_size

        self.register_parameter("weight", None)
        self.register_buffer(
            "qweight",
            torch.empty((self.packed_in_features, out_features), dtype=torch.int32, device=device),
            persistent=True,
        )
        self.register_buffer(
            "qzeros",
            torch.empty((self.group_count, self.packed_out_features), dtype=torch.int32, device=device),
            persistent=True,
        )
        self.register_buffer(
            "scales",
            torch.empty((self.group_count, out_features), dtype=torch.float16, device=device),
            persistent=True,
        )
        if bias:
            self.bias = _empty_parameter((out_features,), dtype=compute_dtype, device=device)
        else:
            self.register_parameter("bias", None)

    def _unpack_qweight(self) -> torch.Tensor:
        return _unpack_int4_first_dim(self.qweight)[: self.in_features, : self.out_features]

    def _unpack_qzeros(self) -> torch.Tensor:
        # AutoRound's AutoGPTQ-compatible export stores (zero_point - 1) in the packed qzeros tensor.
        # See auto_round.export.export_to_autogptq.qlinear_triton.QuantLinear.pack().
        return _unpack_int4_last_dim(self.qzeros)[:, : self.out_features] + 1

    def _dequantize_weight(self) -> torch.Tensor:
        if self.qweight.numel() == 0 or self.qzeros.numel() == 0 or self.scales.numel() == 0:
            raise RuntimeError("AutoRoundGPTQLinear has no quantized payload.")

        weight_int = self._unpack_qweight().to(dtype=torch.float32)
        zeros = self._unpack_qzeros().to(dtype=torch.float32)
        scales = self.scales.to(dtype=torch.float32)
        group_ids = torch.arange(self.in_features, device=weight_int.device, dtype=torch.long) // self.group_size
        group_ids = torch.clamp(group_ids, max=scales.shape[0] - 1)
        expanded_scales = scales.index_select(0, group_ids)
        expanded_zeros = zeros.index_select(0, group_ids)
        dequantized = (weight_int - expanded_zeros) * expanded_scales
        return dequantized.transpose(0, 1).contiguous()

    def _to_xpu_int4_tensors(
        self,
        *,
        padded_in_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if padded_in_features < self.in_features:
            raise ValueError("AutoRound packed weight cannot shrink in_features during conversion.")

        dev = self.qweight.device
        packed_weight = _unpack_int4_first_dim(self.qweight.to(dtype=torch.int32))[
            : self.in_features, : self.out_features
        ]
        zeros_u = self._unpack_qzeros().to(dtype=torch.int32)
        scales_f = self.scales.to(dtype=torch.float32)

        padded = torch.empty((padded_in_features, self.out_features), dtype=torch.int32, device=dev)
        padded[: self.in_features].copy_(packed_weight)
        if padded_in_features > self.in_features:
            tail_group_idx = min(zeros_u.shape[0] - 1, self.in_features // self.group_size)
            fill_value = zeros_u[tail_group_idx].unsqueeze(0).expand(padded_in_features - self.in_features, -1)
            padded[self.in_features :].copy_(fill_value)

        transposed = padded.transpose(0, 1).contiguous()
        reshaped = transposed.reshape(self.out_features, padded_in_features // _INT4_VALUES_PER_PACK, _INT4_VALUES_PER_PACK)
        shifts = _int4_shifts(device=dev).view(1, 1, _INT4_VALUES_PER_PACK)
        qweight = torch.bitwise_left_shift(reshaped & 0xF, shifts).sum(dim=-1, dtype=torch.int32).contiguous()
        qzeros = zeros_u.to(dtype=torch.int8).contiguous()
        return qweight, scales_f.contiguous(), qzeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight().to(device=x.device, dtype=self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=self.compute_dtype)
        return F.linear(x.to(dtype=self.compute_dtype), weight, bias).to(dtype=x.dtype)


class AWQLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        bias: bool = False,
        compute_dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if bits != 4:
            raise ValueError("The current AWQ implementation only supports 4-bit weights.")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.compute_dtype = compute_dtype

        self.register_parameter("weight", None)
        self.register_buffer("qweight", torch.empty(0, dtype=torch.int32, device=device), persistent=True)
        self.register_buffer("qzeros", torch.empty(0, dtype=torch.int32, device=device), persistent=True)
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16, device=device), persistent=True)
        if bias:
            self.bias = _empty_parameter((out_features,), dtype=compute_dtype, device=device)
        else:
            self.register_parameter("bias", None)

    def _dequantize_awq(self) -> torch.Tensor:
        if self.qweight.numel() == 0:
            raise RuntimeError("AWQLinear has no quantized weight payload.")

        weight_int = _unpack_int4_last_dim(self.qweight)
        zeros_int = _unpack_int4_last_dim(self.qzeros) if self.qzeros.numel() else None
        scales = self.scales.to(dtype=torch.float32)

        if weight_int.shape[0] == self.in_features and weight_int.shape[1] >= self.out_features:
            weight_int = weight_int[:, : self.out_features]
            if zeros_int is not None:
                zeros_int = zeros_int[:, : self.out_features]
            group_count = scales.shape[0]
            group_ids = torch.arange(self.in_features, device=weight_int.device) // self.group_size
            group_ids = torch.clamp(group_ids, max=group_count - 1)
            expanded_scales = scales[group_ids]
            expanded_zeros = 8.0 if zeros_int is None else zeros_int[group_ids].to(torch.float32)
            dequantized = (weight_int.to(torch.float32) - expanded_zeros) * expanded_scales
            return dequantized.transpose(0, 1).contiguous()

        if weight_int.shape[0] == self.out_features and weight_int.shape[1] >= self.in_features:
            weight_int = weight_int[:, : self.in_features]
            group_count = scales.shape[0]
            group_ids = torch.arange(self.in_features, device=weight_int.device) // self.group_size
            group_ids = torch.clamp(group_ids, max=group_count - 1)
            expanded_scales = scales[group_ids].transpose(0, 1)
            if zeros_int is None:
                expanded_zeros = 8.0
            else:
                expanded_zeros = zeros_int[:, : self.in_features].to(torch.float32)
            return (weight_int.to(torch.float32) - expanded_zeros) * expanded_scales

        raise ValueError(
            f"Unsupported AWQ packed weight shape {tuple(self.qweight.shape)} for target {(self.out_features, self.in_features)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            weight = self.weight.to(dtype=self.compute_dtype)
        else:
            weight = self._dequantize_awq().to(dtype=self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(dtype=self.compute_dtype)
        return F.linear(x.to(dtype=self.compute_dtype), weight, bias).to(dtype=x.dtype)


class XPUInt4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int = 128,
        bias: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        padded_in_features: int | None = None,
    ):
        super().__init__()
        if group_size <= 0 or group_size % 32 != 0:
            raise ValueError("XPUInt4Linear requires group_size to be a positive multiple of 32.")
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        self.padded_in_features = padded_in_features or self._padded_in_features(in_features, group_size)
        group_count = self.padded_in_features // self.group_size
        packed_cols = self.padded_in_features // _INT4_VALUES_PER_PACK

        self.register_buffer("qweight", torch.empty((out_features, packed_cols), dtype=torch.int32, device=device), persistent=True)
        self.register_buffer("qscale", torch.empty((group_count, out_features), dtype=torch.float32, device=device), persistent=True)
        self.register_buffer("qzeros", torch.empty((group_count, out_features), dtype=torch.int8, device=device), persistent=True)
        if bias:
            self.bias = _empty_parameter((out_features,), dtype=compute_dtype, device=device)
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _padded_in_features(in_features: int, group_size: int) -> int:
        padded = ((in_features + group_size - 1) // group_size) * group_size
        return ((padded + _INT4_VALUES_PER_PACK - 1) // _INT4_VALUES_PER_PACK) * _INT4_VALUES_PER_PACK

    @classmethod
    def from_linear(
        cls,
        module: nn.Module,
        *,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> "XPUInt4Linear":
        if isinstance(module, AutoRoundGPTQLinear):
            resolved_group_size = int(module.group_size)
            padded_in_features = cls._padded_in_features(module.in_features, resolved_group_size)
            quantized = cls(
                module.in_features,
                module.out_features,
                group_size=resolved_group_size,
                bias=module.bias is not None,
                compute_dtype=compute_dtype or module.compute_dtype,
                device=device,
                padded_in_features=padded_in_features,
            )
            qweight, qscale, qzeros = module._to_xpu_int4_tensors(padded_in_features=padded_in_features)
            with torch.no_grad():
                quantized.qweight.copy_(qweight.to(device=quantized.qweight.device))
                quantized.qscale.copy_(qscale.to(device=quantized.qscale.device))
                quantized.qzeros.copy_(qzeros.to(device=quantized.qzeros.device))
                if module.bias is not None and quantized.bias is not None:
                    quantized.bias.copy_(module.bias.detach().to(device=quantized.bias.device, dtype=quantized.bias.dtype))
            return quantized

        weight, bias = _extract_dense_weight_bias(module)
        in_features = int(weight.shape[1])
        out_features = int(weight.shape[0])
        padded_in_features = cls._padded_in_features(in_features, group_size)
        quantized = cls(
            in_features,
            out_features,
            group_size=group_size,
            bias=bias is not None,
            compute_dtype=compute_dtype,
            device=device,
            padded_in_features=padded_in_features,
        )
        qweight, qscale, qzeros = cls._quantize_weight(weight, group_size=group_size, padded_in_features=padded_in_features)
        with torch.no_grad():
            quantized.qweight.copy_(qweight.to(device=quantized.qweight.device))
            quantized.qscale.copy_(qscale.to(device=quantized.qscale.device))
            quantized.qzeros.copy_(qzeros.to(device=quantized.qzeros.device))
            if bias is not None and quantized.bias is not None:
                quantized.bias.copy_(bias.to(device=quantized.bias.device, dtype=quantized.bias.dtype))
        return quantized

    @staticmethod
    def _quantize_weight(
        weight: torch.Tensor,
        *,
        group_size: int,
        padded_in_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = weight.device
        weight = weight.to(dtype=torch.float32, device=dev)
        out_features, in_features = weight.shape
        if padded_in_features > in_features:
            padded_weight = torch.zeros((out_features, padded_in_features), dtype=torch.float32, device=dev)
            padded_weight[:, :in_features] = weight
            weight = padded_weight

        group_count = padded_in_features // group_size
        grouped = weight.reshape(out_features, group_count, group_size)
        max_abs = grouped.abs().amax(dim=-1)
        scale = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
        q = torch.round(grouped / scale.unsqueeze(-1) + 8.0).clamp_(0.0, 15.0).to(torch.int32)
        q = q.reshape(out_features, padded_in_features)
        qzeros = torch.full((group_count, out_features), 8, dtype=torch.int8, device=dev)
        qscale = scale.transpose(0, 1).contiguous().to(dtype=torch.float32)

        q_reshaped = q.reshape(out_features, padded_in_features // _INT4_VALUES_PER_PACK, _INT4_VALUES_PER_PACK)
        shifts = _int4_shifts(device=dev).view(1, 1, _INT4_VALUES_PER_PACK)
        packed = torch.bitwise_left_shift(q_reshaped & 0xF, shifts).sum(dim=-1, dtype=torch.int32)
        return packed.contiguous(), qscale, qzeros

    def _dequantize_weight(self) -> torch.Tensor:
        packed = self.qweight.to(device="cpu", dtype=torch.int32)
        shifts = (_int4_shifts(device="cpu").view(1, 1, _INT4_VALUES_PER_PACK))
        unpacked = torch.bitwise_right_shift(packed.unsqueeze(-1), shifts) & 0xF
        unpacked = unpacked.reshape(self.out_features, self.padded_in_features).to(dtype=torch.float32)
        qscale = self.qscale.to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous()
        qzeros = self.qzeros.to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous()
        groups = unpacked.reshape(self.out_features, self.padded_in_features // self.group_size, self.group_size)
        dequantized = (groups - qzeros.unsqueeze(-1)) * qscale.unsqueeze(-1)
        return dequantized.reshape(self.out_features, self.padded_in_features)[:, : self.in_features]

    @staticmethod
    def _matmul_strategy() -> str:
        strategy = os.getenv("ANNA_XPU_INT4_MATMUL", "auto").strip().lower()
        if strategy not in _XPU_INT4_MATMUL_STRATEGIES:
            logger.warning("Ignoring unsupported ANNA_XPU_INT4_MATMUL=%r; using auto.", strategy)
            return "auto"
        return strategy

    def _forward_dequant(self, x_padded: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight().to(device=x_padded.device, dtype=self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(device=x_padded.device, dtype=self.compute_dtype)
        return F.linear(x_padded, weight, bias)

    def _forward_torch_xpu_int4(self, x_padded: torch.Tensor) -> torch.Tensor:
        op = getattr(torch.ops.aten, "_weight_int4pack_mm_with_scales_and_zeros", None)
        if op is None:
            raise RuntimeError("aten._weight_int4pack_mm_with_scales_and_zeros is unavailable")
        output = op(
            x_padded,
            self.qweight,
            self.group_size,
            self.qscale,
            self.qzeros,
        )
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output

    def _forward_sycl_int4_gemv(self, x_padded: torch.Tensor) -> torch.Tensor:
        if x_padded.shape[0] > 4:
            raise RuntimeError("Anna xpu_int4_gemv is only enabled for decode rows <= 4.")
        namespace = getattr(torch.ops, "anna", None)
        op = None if namespace is None else getattr(namespace, "xpu_int4_gemv", None)
        if op is None:
            from anna.model.fused_ops import maybe_load_gated_delta_library

            maybe_load_gated_delta_library()
            namespace = getattr(torch.ops, "anna", None)
            op = None if namespace is None else getattr(namespace, "xpu_int4_gemv", None)
        if op is None:
            raise RuntimeError(
                "Anna xpu_int4_gemv op is not registered. Build/load the custom op first, "
                "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
            )
        output = op(
            x_padded,
            self.qweight,
            self.qscale,
            self.qzeros,
            self.group_size,
            self.padded_in_features,
        )
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output

    def _should_auto_use_sycl_int4_gemv(self, x_padded: torch.Tensor) -> bool:
        if os.getenv("ANNA_XPU_AUTO_INT4_GEMV", "").strip().lower() not in {"1", "true", "yes", "on"}:
            return False
        if os.getenv("ANNA_XPU_DISABLE_AUTO_INT4_GEMV", "").strip().lower() in {"1", "true", "yes", "on"}:
            return False
        return (
            x_padded.device.type == "xpu"
            and x_padded.shape[0] == 1
            and self.in_features == 4096
            and self.out_features == 4096
            and self.padded_in_features == 4096
            and self.group_size == 128
            and self.bias is None
            and self.compute_dtype in {torch.float16, torch.bfloat16}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        uses_xpu_kernel = x_2d.device.type == "xpu" and self.qweight.device.type == "xpu"
        if uses_xpu_kernel and x_2d.shape[-1] < self.padded_in_features:
            x_padded = torch.zeros(
                (x_2d.shape[0], self.padded_in_features),
                dtype=self.compute_dtype,
                device=x_2d.device,
            )
            x_padded[:, : x_2d.shape[-1]] = x_2d.to(dtype=self.compute_dtype)
        else:
            x_padded = x_2d.to(dtype=self.compute_dtype)

        if uses_xpu_kernel:
            strategy = self._matmul_strategy()
            if strategy == "dequant":
                output = self._forward_dequant(x_padded)
            elif strategy == "sycl":
                output = self._forward_sycl_int4_gemv(x_padded)
            else:
                try:
                    if strategy == "auto" and self._should_auto_use_sycl_int4_gemv(x_padded):
                        try:
                            output = self._forward_sycl_int4_gemv(x_padded)
                        except Exception:
                            logger.warning(
                                "Falling back to PyTorch XPU int4 linear after auto sycl GEMV failure: "
                                "in_features=%s out_features=%s group_size=%s",
                                self.in_features,
                                self.out_features,
                                self.group_size,
                                exc_info=True,
                            )
                            output = self._forward_torch_xpu_int4(x_padded)
                    else:
                        output = self._forward_torch_xpu_int4(x_padded)
                except Exception:
                    if strategy == "torch":
                        raise
                    logger.warning(
                        "Falling back to dequantized XPU int4 linear: in_features=%s out_features=%s group_size=%s",
                        self.in_features,
                        self.out_features,
                        self.group_size,
                        exc_info=True,
                    )
                    output = self._forward_dequant(x_padded)
        else:
            output = self._forward_dequant(x_padded)
        return output.reshape(*original_shape, self.out_features).to(dtype=x.dtype)


@dataclass(slots=True)
class QuantizedModuleSpec:
    module_name: str
    quant_method: str


@dataclass(slots=True)
class _QuantizationLookup:
    excluded_prefixes: tuple[str, ...]
    exact_overrides: dict[str, dict[str, object]]
    normalized_candidate_overrides: dict[str, dict[str, object]]
    regex_overrides: tuple[tuple[re.Pattern[str], re.Pattern[str] | None, dict[str, object]], ...]
    block_names: tuple[str, ...]


def _normalize_exclusion(module_name: str) -> str:
    for prefix in ("model.language_model.", "model.visual.", "model."):
        if module_name.startswith(prefix):
            return module_name[len(prefix) :]
    return module_name


def _looks_like_regex(pattern: str) -> bool:
    return any(char in _REGEX_META_CHARS for char in pattern)


def _build_quantization_lookup(quantization_config: QuantizationConfig) -> _QuantizationLookup:
    exact_overrides: dict[str, dict[str, object]] = {}
    normalized_candidate_overrides: dict[str, dict[str, object]] = {}
    regex_overrides: list[tuple[re.Pattern[str], re.Pattern[str] | None, dict[str, object]]] = []

    for pattern, override in quantization_config.extra_config.items():
        exact_overrides[pattern] = override
        normalized_candidate_overrides.setdefault(_normalize_exclusion(pattern), override)
        if not _looks_like_regex(pattern):
            continue

        candidate = _normalize_exclusion(pattern)
        try:
            compiled_pattern = re.compile(pattern)
            compiled_candidate = None if candidate == pattern else re.compile(candidate)
        except re.error as exc:
            raise ValueError(f"Invalid quantization extra_config regex {pattern!r}: {exc}") from exc
        regex_overrides.append((compiled_pattern, compiled_candidate, override))

    return _QuantizationLookup(
        excluded_prefixes=tuple(_normalize_exclusion(candidate) for candidate in quantization_config.modules_to_not_convert),
        exact_overrides=exact_overrides,
        normalized_candidate_overrides=normalized_candidate_overrides,
        regex_overrides=tuple(regex_overrides),
        block_names=tuple(name for name in quantization_config.block_name_to_quantize if name),
    )


def _module_override_config_fast(
    module_name: str,
    normalized_name: str,
    lookup: _QuantizationLookup,
) -> dict[str, object] | None:
    exact_match = lookup.exact_overrides.get(module_name)
    if exact_match is not None:
        return exact_match

    normalized_match = lookup.normalized_candidate_overrides.get(normalized_name)
    if normalized_match is not None:
        return normalized_match

    for compiled_pattern, compiled_candidate, override in lookup.regex_overrides:
        if compiled_pattern.fullmatch(module_name):
            return override
        if compiled_candidate is not None and compiled_candidate.fullmatch(normalized_name):
            return override
    return None


def _should_skip_fast(
    module_name: str,
    normalized_name: str,
    lookup: _QuantizationLookup,
) -> bool:
    for candidate in lookup.excluded_prefixes:
        if normalized_name == candidate or normalized_name.startswith(candidate + "."):
            return True

    override = _module_override_config_fast(module_name, normalized_name, lookup)
    if override is None:
        return False

    override_bits = override.get("bits")
    if override_bits is not None and int(override_bits) >= 16:
        return True

    override_data_type = override.get("data_type")
    if override_data_type is not None and str(override_data_type).strip().lower() in _FLOAT_OVERRIDE_DATA_TYPES:
        return True
    return False


def _should_quantize_autoround_module_fast(module_name: str, block_names: tuple[str, ...]) -> bool:
    if not block_names:
        raise ValueError("AutoRound quantization_config is missing block_name_to_quantize.")
    return any(module_name == block_name or module_name.startswith(block_name + ".") for block_name in block_names)


def _module_override_config(module_name: str, quantization_config: QuantizationConfig) -> dict[str, object] | None:
    if not quantization_config.extra_config:
        return None
    normalized = _normalize_exclusion(module_name)
    return _module_override_config_fast(module_name, normalized, _build_quantization_lookup(quantization_config))


def _should_skip(module_name: str, quantization_config: QuantizationConfig) -> bool:
    normalized = _normalize_exclusion(module_name)
    return _should_skip_fast(module_name, normalized, _build_quantization_lookup(quantization_config))


def _should_quantize_autoround_module(module_name: str, quantization_config: QuantizationConfig) -> bool:
    return _should_quantize_autoround_module_fast(
        module_name,
        tuple(name for name in quantization_config.block_name_to_quantize if name),
    )


def _set_submodule(model: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def _extract_dense_weight_bias(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(module, AutoRoundGPTQLinear):
        weight = module._dequantize_weight()
        bias = None if module.bias is None else module.bias.detach().to(dtype=torch.float32)
        return weight.detach(), bias
    if isinstance(module, AWQLinear):
        weight = module._dequantize_awq()
        bias = None if module.bias is None else module.bias.detach().to(dtype=torch.float32)
        return weight.detach(), bias
    if isinstance(module, (DenseLinear, nn.Linear)):
        weight = module.weight.detach().to(dtype=torch.float32)
        bias = None if module.bias is None else module.bias.detach().to(dtype=torch.float32)
        return weight, bias
    if isinstance(module, XPUInt4Linear):
        weight = module._dequantize_weight()
        bias = None if module.bias is None else module.bias.detach().to(dtype=torch.float32)
        return weight, bias
    raise TypeError(f"Unsupported linear module for XPU int4 conversion: {type(module)!r}")


def estimate_xpu_int4_linear_bytes(
    in_features: int,
    out_features: int,
    *,
    group_size: int = 128,
    has_bias: bool = False,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> int:
    padded_in_features = XPUInt4Linear._padded_in_features(in_features, group_size)
    group_count = padded_in_features // group_size
    packed_weight_bytes = out_features * (padded_in_features // _INT4_VALUES_PER_PACK) * 4
    scale_bytes = group_count * out_features * torch.tensor([], dtype=torch.float32).element_size()
    zero_bytes = group_count * out_features * torch.tensor([], dtype=torch.int8).element_size()
    bias_bytes = out_features * torch.tensor([], dtype=compute_dtype).element_size() if has_bias else 0
    return packed_weight_bytes + scale_bytes + zero_bytes + bias_bytes


def estimate_module_xpu_int4_bytes(
    module: nn.Module,
    *,
    group_size: int = 128,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> int:
    total = 0
    for child in module.modules():
        if child is module:
            continue
        if isinstance(child, XPUInt4Linear):
            total += (
                child.qweight.nelement() * child.qweight.element_size()
                + child.qscale.nelement() * child.qscale.element_size()
                + child.qzeros.nelement() * child.qzeros.element_size()
            )
            if child.bias is not None:
                total += child.bias.nelement() * child.bias.element_size()
        elif isinstance(child, (AutoRoundGPTQLinear, AWQLinear, DenseLinear, nn.Linear)):
            total += estimate_xpu_int4_linear_bytes(
                child.in_features,
                child.out_features,
                group_size=group_size,
                has_bias=getattr(child, "bias", None) is not None,
                compute_dtype=compute_dtype,
            )
    return total


def convert_module_linears_to_xpu_int4(
    module: nn.Module,
    *,
    group_size: int = 128,
    compute_dtype: torch.dtype | None = None,
    device: torch.device | str = torch.device("xpu"),
    include_predicate: Callable[[str, nn.Module], bool] | None = None,
    cache_dir: str | Path | None = None,
) -> int:
    count = 0
    gc_every = 16
    resolved_cache_dir = Path(cache_dir) if cache_dir is not None else None
    env_cache_dir = os.getenv("ANNA_XPU_INT4_CACHE_DIR")
    if resolved_cache_dir is None and env_cache_dir:
        resolved_cache_dir = Path(env_cache_dir)
    for module_name, child in list(module.named_modules()):
        if not module_name:
            continue
        if isinstance(child, XPUInt4Linear):
            continue
        if not isinstance(child, (AutoRoundGPTQLinear, AWQLinear, DenseLinear, nn.Linear)):
            continue
        if include_predicate is not None and not include_predicate(module_name, child):
            continue
        resolved_group_size = int(getattr(child, "group_size", group_size))
        resolved_compute_dtype = compute_dtype or getattr(child, "compute_dtype", torch.bfloat16)
        replacement = None
        cache_path = None
        fingerprint = None
        if resolved_cache_dir is not None:
            try:
                fingerprint = _linear_cache_fingerprint(
                    child,
                    group_size=resolved_group_size,
                    compute_dtype=resolved_compute_dtype,
                )
                cache_path = _cache_file_for_linear(resolved_cache_dir, module_name, fingerprint)
                replacement = _load_xpu_int4_linear_from_cache(
                    cache_path,
                    device=device,
                    compute_dtype=resolved_compute_dtype,
                    fingerprint=fingerprint,
                )
            except Exception:
                logger.warning("Failed to probe XPU int4 cache for module %s", module_name, exc_info=True)
                replacement = None

        if replacement is None:
            replacement = XPUInt4Linear.from_linear(
                child,
                group_size=resolved_group_size,
                compute_dtype=resolved_compute_dtype,
                device=device,
            )
            if cache_path is not None and fingerprint is not None:
                try:
                    _save_xpu_int4_linear_to_cache(cache_path, replacement, fingerprint=fingerprint)
                    logger.info("Saved XPU int4 layout cache: %s", cache_path)
                except Exception:
                    logger.warning("Failed to save XPU int4 cache for module %s", module_name, exc_info=True)
        _set_submodule(module, module_name, replacement)
        count += 1
        if gc_every > 0 and count % gc_every == 0:
            gc.collect()

    gc.collect()
    return count


def replace_linear_modules_with_xpu_int4_placeholders(
    model: nn.Module,
    *,
    group_size: int = 128,
    compute_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    include_predicate: Callable[[str, nn.Module], bool] | None = None,
) -> int:
    replacements: list[tuple[str, XPUInt4Linear]] = []
    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if include_predicate is not None and not include_predicate(module_name, module):
            continue
        replacements.append(
            (
                module_name,
                XPUInt4Linear(
                    module.in_features,
                    module.out_features,
                    group_size=group_size,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype,
                    device=device,
                ),
            )
        )

    for module_name, replacement in replacements:
        _set_submodule(model, module_name, replacement)
    return len(replacements)


def replace_linear_modules(
    model: nn.Module,
    quantization_config: QuantizationConfig,
    *,
    compute_dtype: torch.dtype,
) -> list[QuantizedModuleSpec]:
    if not quantization_config.is_enabled:
        return []

    replacements: list[tuple[str, nn.Module]] = []
    specs: list[QuantizedModuleSpec] = []
    quant_method = (quantization_config.quant_method or "").strip().lower()

    if quant_method in _AUTO_ROUND_QUANT_METHODS:
        packing_format = (quantization_config.packing_format or "").strip().lower()
        if packing_format not in _AUTO_ROUND_PACKING_FORMATS:
            raise ValueError(
                f"Unsupported AutoRound packing format: {quantization_config.packing_format!r}. "
                f"Expected one of: {sorted(_AUTO_ROUND_PACKING_FORMATS)}"
            )
        if int(quantization_config.bits or 0) != 4:
            raise ValueError("AutoRound support currently requires 4-bit weights.")
        data_type = (quantization_config.data_type or "").strip().lower()
        if data_type not in {"int", "int4"}:
            raise ValueError(f"Unsupported AutoRound data_type: {quantization_config.data_type!r}.")

    lookup = _build_quantization_lookup(quantization_config)
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        normalized_name = _normalize_exclusion(module_name)
        if _should_skip_fast(module_name, normalized_name, lookup):
            continue
        if quant_method in _AUTO_ROUND_QUANT_METHODS and not _should_quantize_autoround_module_fast(module_name, lookup.block_names):
            continue
        module_device = module.weight.device

        if quant_method in _AUTO_ROUND_QUANT_METHODS:
            replacement = AutoRoundGPTQLinear(
                module.in_features,
                module.out_features,
                bits=int(quantization_config.bits or 4),
                group_size=int(quantization_config.group_size or 128),
                sym=True if quantization_config.sym is None else bool(quantization_config.sym),
                bias=module.bias is not None,
                compute_dtype=compute_dtype,
                device=module_device,
            )
        elif quant_method == "awq":
            replacement = AWQLinear(
                module.in_features,
                module.out_features,
                bits=quantization_config.bits or 4,
                group_size=quantization_config.group_size or 128,
                zero_point=quantization_config.zero_point,
                bias=module.bias is not None,
                compute_dtype=compute_dtype,
                device=module_device,
            )
        else:
            raise ValueError(f"Unsupported quantization method: {quantization_config.quant_method!r}")

        replacements.append((module_name, replacement))
        specs.append(QuantizedModuleSpec(module_name=module_name, quant_method=quant_method))

    for module_name, replacement in replacements:
        _set_submodule(model, module_name, replacement)

    return specs
