from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.qwen3_5_text_config import QuantizationConfig


def _float8_dtype():
    return getattr(torch, "float8_e4m3fn", None)


def _empty_parameter(
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    *,
    device: torch.device | str | None = None,
) -> nn.Parameter:
    dtype = dtype or torch.float32
    return nn.Parameter(torch.empty(*shape, dtype=dtype, device=device), requires_grad=False)


_AUTO_ROUND_METHODS = frozenset({"auto-round", "autoround", "auto_round", "gptq", "auto_gptq"})


def _unpack_int4_last_dim(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    return ((packed.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF).reshape(*packed.shape[:-1], packed.shape[-1] * 8)


def _unpack_int4_first_dim(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2:
        raise ValueError(f"Expected a rank-2 AutoRound packed tensor, got shape {tuple(packed.shape)}")
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32).view(1, 8, 1)
    unpacked = (packed.to(torch.int32).unsqueeze(1) >> shifts) & 0xF
    return unpacked.reshape(packed.shape[0] * 8, packed.shape[1])


def _pack_int4_last_dim(values: torch.Tensor) -> torch.Tensor:
    if values.ndim < 1:
        raise ValueError("Expected at least one dimension when packing int4 values.")
    if values.shape[-1] % 8 != 0:
        raise ValueError(f"Expected the last dimension to be divisible by 8, got shape {tuple(values.shape)}")
    packed_shape = (*values.shape[:-1], values.shape[-1] // 8, 8)
    reshaped = values.to(torch.int32).reshape(packed_shape)
    shifts = (torch.arange(8, dtype=torch.int32).view(*([1] * (reshaped.ndim - 1)), 8) * 4)
    return torch.bitwise_left_shift(reshaped & 0xF, shifts).sum(dim=-1, dtype=torch.int32).contiguous()


def _dequantize_fp8_to_cpu(module: "FP8Linear") -> torch.Tensor:
    weight = module.weight.detach().to(device=torch.device("cpu"), dtype=torch.float32)
    scale = module.weight_scale_inv.detach().to(device=torch.device("cpu"), dtype=torch.float32)
    if scale.ndim == 0:
        return weight * scale

    block_m, block_n = module.block_size or (module.out_features, module.in_features)
    rows, cols = module.out_features, module.in_features
    row_tiles = (rows + block_m - 1) // block_m
    col_tiles = (cols + block_n - 1) // block_n
    padded_rows = row_tiles * block_m
    padded_cols = col_tiles * block_n

    padded_weight = torch.zeros((padded_rows, padded_cols), device=weight.device, dtype=weight.dtype)
    padded_weight[:rows, :cols] = weight
    reshaped = padded_weight.reshape(row_tiles, block_m, col_tiles, block_n)
    expanded_scale = scale.reshape(row_tiles, col_tiles).unsqueeze(1).unsqueeze(-1)
    dequantized = (reshaped * expanded_scale).reshape(padded_rows, padded_cols)
    return dequantized[:rows, :cols]


def _dequantize_awq_to_cpu(module: "AWQLinear") -> torch.Tensor:
    if module.qweight.numel() == 0:
        raise RuntimeError("AWQLinear has no quantized weight payload.")

    qweight = module.qweight.detach().to(device=torch.device("cpu"), dtype=torch.int32)
    qzeros = module.qzeros.detach().to(device=torch.device("cpu"), dtype=torch.int32) if module.qzeros.numel() else None
    scales = module.scales.detach().to(device=torch.device("cpu"), dtype=torch.float32)
    weight_int = _unpack_int4_last_dim(qweight)
    zeros_int = _unpack_int4_last_dim(qzeros) if qzeros is not None else None

    if weight_int.shape[0] == module.in_features and weight_int.shape[1] >= module.out_features:
        weight_int = weight_int[:, : module.out_features]
        if zeros_int is not None:
            zeros_int = zeros_int[:, : module.out_features]
        group_count = scales.shape[0]
        group_ids = torch.arange(module.in_features, device=torch.device("cpu")) // module.group_size
        group_ids = torch.clamp(group_ids, max=group_count - 1)
        expanded_scales = scales[group_ids]
        expanded_zeros = 8.0 if zeros_int is None else zeros_int[group_ids].to(torch.float32)
        dequantized = (weight_int.to(torch.float32) - expanded_zeros) * expanded_scales
        return dequantized.transpose(0, 1).contiguous()

    if weight_int.shape[0] == module.out_features and weight_int.shape[1] >= module.in_features:
        weight_int = weight_int[:, : module.in_features]
        group_count = scales.shape[0]
        group_ids = torch.arange(module.in_features, device=torch.device("cpu")) // module.group_size
        group_ids = torch.clamp(group_ids, max=group_count - 1)
        expanded_scales = scales[group_ids].transpose(0, 1)
        if zeros_int is None:
            expanded_zeros = 8.0
        else:
            expanded_zeros = zeros_int[:, : module.in_features].to(torch.float32)
        return (weight_int.to(torch.float32) - expanded_zeros) * expanded_scales

    raise ValueError(
        f"Unsupported AWQ packed weight shape {tuple(module.qweight.shape)} for target {(module.out_features, module.in_features)}"
    )


def _dequantize_autoround_to_cpu(module: "AutoRoundGPTQLinear") -> torch.Tensor:
    if module.qweight.numel() == 0:
        raise RuntimeError("AutoRoundGPTQLinear has no quantized weight payload.")
    if module.group_size <= 0:
        raise ValueError("AutoRoundGPTQLinear requires a positive group_size.")

    qweight = module.qweight.detach().to(device=torch.device("cpu"), dtype=torch.int32)
    weight_int = _unpack_int4_first_dim(qweight)
    if weight_int.shape[0] < module.in_features or weight_int.shape[1] < module.out_features:
        raise ValueError(
            f"Unsupported AutoRound qweight shape {tuple(module.qweight.shape)} for target {(module.out_features, module.in_features)}"
        )
    weight_int = weight_int[: module.in_features, : module.out_features]
    scales = module.scales.detach().to(device=torch.device("cpu"), dtype=torch.float32)
    if scales.ndim != 2 or scales.shape[1] < module.out_features:
        raise ValueError(
            f"Unsupported AutoRound scales shape {tuple(module.scales.shape)} for target {(module.out_features, module.in_features)}"
        )
    group_count = scales.shape[0]
    group_ids = torch.arange(module.in_features, device=torch.device("cpu")) // module.group_size
    group_ids = torch.clamp(group_ids, max=max(0, group_count - 1))
    expanded_scales = scales[:, : module.out_features][group_ids]

    expanded_zeros: torch.Tensor | float
    if module.zero_point and module.qzeros.numel() > 0:
        qzeros = module.qzeros.detach().to(device=torch.device("cpu"), dtype=torch.int32)
        zeros_int = _unpack_int4_last_dim(qzeros)
        if zeros_int.ndim != 2 or zeros_int.shape[0] < group_count or zeros_int.shape[1] < module.out_features:
            raise ValueError(
                f"Unsupported AutoRound qzeros shape {tuple(module.qzeros.shape)} for target {(module.out_features, module.in_features)}"
            )
        expanded_zeros = zeros_int[:group_count, : module.out_features].to(dtype=torch.float32)[group_ids]
    else:
        expanded_zeros = 8.0

    dequantized = (weight_int.to(torch.float32) - expanded_zeros) * expanded_scales
    return dequantized.transpose(0, 1).contiguous()


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
        return F.linear(x, self.weight.to(dtype=x.dtype), None if self.bias is None else self.bias.to(dtype=x.dtype))


class FP8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        block_size: tuple[int, int] | None = None,
        bias: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.compute_dtype = compute_dtype
        fp8_dtype = _float8_dtype() or torch.float16
        self.weight = _empty_parameter((out_features, in_features), dtype=fp8_dtype, device=device)
        if block_size is None:
            self.weight_scale_inv = nn.Parameter(torch.ones((), dtype=torch.float32, device=device), requires_grad=False)
        else:
            scale_out = (out_features + block_size[0] - 1) // block_size[0]
            scale_in = (in_features + block_size[1] - 1) // block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.ones((scale_out, scale_in), dtype=torch.float32, device=device),
                requires_grad=False,
            )
        if bias:
            self.bias = _empty_parameter((out_features,), dtype=compute_dtype, device=device)
        else:
            self.register_parameter("bias", None)

    def _dequantize_weight(self) -> torch.Tensor:
        weight = self.weight.to(dtype=torch.float32)
        scale = self.weight_scale_inv.to(dtype=torch.float32)
        if scale.ndim == 0:
            return weight * scale

        block_m, block_n = self.block_size or (self.out_features, self.in_features)
        rows, cols = self.out_features, self.in_features
        row_tiles = (rows + block_m - 1) // block_m
        col_tiles = (cols + block_n - 1) // block_n
        padded_rows = row_tiles * block_m
        padded_cols = col_tiles * block_n

        padded_weight = torch.zeros((padded_rows, padded_cols), device=weight.device, dtype=weight.dtype)
        padded_weight[:rows, :cols] = weight
        reshaped = padded_weight.reshape(row_tiles, block_m, col_tiles, block_n)
        expanded_scale = scale.reshape(row_tiles, col_tiles).unsqueeze(1).unsqueeze(-1)
        dequantized = (reshaped * expanded_scale).reshape(padded_rows, padded_cols)
        return dequantized[:rows, :cols]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize_weight().to(dtype=self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(dtype=self.compute_dtype)
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

    @staticmethod
    def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
        return _unpack_int4_last_dim(packed)

    def _dequantize_awq(self) -> torch.Tensor:
        if self.qweight.numel() == 0:
            raise RuntimeError("AWQLinear has no quantized weight payload.")

        weight_int = self._unpack_int4(self.qweight)
        zeros_int = self._unpack_int4(self.qzeros) if self.qzeros.numel() else None
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


class AutoRoundGPTQLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        device: torch.device | str | None = None,
        bias: bool = False,
    ):
        super().__init__()
        if bits != 4:
            raise ValueError("The current AutoRound GPTQ implementation only supports 4-bit weights.")
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.compute_dtype = compute_dtype

        self.register_parameter("weight", None)
        packed_rows = (self.in_features + 7) // 8
        group_count = max(1, (self.in_features + self.group_size - 1) // self.group_size)
        packed_zero_cols = (self.out_features + 7) // 8
        self.register_buffer("qweight", torch.empty((packed_rows, self.out_features), dtype=torch.int32, device=device), persistent=True)
        self.register_buffer(
            "qzeros",
            torch.empty((group_count, packed_zero_cols), dtype=torch.int32, device=device),
            persistent=True,
        )
        self.register_buffer(
            "scales",
            torch.empty((group_count, self.out_features), dtype=torch.float16, device=device),
            persistent=True,
        )
        if bias:
            self.bias = _empty_parameter((out_features,), dtype=compute_dtype, device=device)
        else:
            self.register_parameter("bias", None)

    def _dequantize_autoround(self) -> torch.Tensor:
        if self.qweight.numel() == 0:
            raise RuntimeError("AutoRoundGPTQLinear has no quantized weight payload.")
        if self.group_size <= 0:
            raise ValueError("AutoRoundGPTQLinear requires a positive group_size.")

        weight_int = _unpack_int4_first_dim(self.qweight)
        if weight_int.shape[0] < self.in_features or weight_int.shape[1] < self.out_features:
            raise ValueError(
                f"Unsupported AutoRound qweight shape {tuple(self.qweight.shape)} for target {(self.out_features, self.in_features)}"
            )
        weight_int = weight_int[: self.in_features, : self.out_features]
        scales = self.scales.to(dtype=torch.float32)
        if scales.ndim != 2 or scales.shape[1] < self.out_features:
            raise ValueError(
                f"Unsupported AutoRound scales shape {tuple(self.scales.shape)} for target {(self.out_features, self.in_features)}"
            )
        group_count = scales.shape[0]
        group_ids = torch.arange(self.in_features, device=weight_int.device) // self.group_size
        group_ids = torch.clamp(group_ids, max=max(0, group_count - 1))
        expanded_scales = scales[:, : self.out_features][group_ids]

        expanded_zeros: torch.Tensor | float
        if self.zero_point and self.qzeros.numel() > 0:
            zeros_int = _unpack_int4_last_dim(self.qzeros)
            if zeros_int.ndim != 2 or zeros_int.shape[0] < group_count or zeros_int.shape[1] < self.out_features:
                raise ValueError(
                    f"Unsupported AutoRound qzeros shape {tuple(self.qzeros.shape)} for target {(self.out_features, self.in_features)}"
                )
            expanded_zeros = zeros_int[:group_count, : self.out_features].to(dtype=torch.float32)[group_ids]
        else:
            expanded_zeros = 8.0

        dequantized = (weight_int.to(torch.float32) - expanded_zeros) * expanded_scales
        return dequantized.transpose(0, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            weight = self.weight.to(dtype=self.compute_dtype)
        else:
            weight = self._dequantize_autoround().to(dtype=self.compute_dtype)
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
        packed_cols = self.padded_in_features // 8

        self.register_buffer("qweight", torch.empty((out_features, packed_cols), dtype=torch.int32, device=device), persistent=True)
        self.register_buffer("qscale", torch.empty((group_count, out_features), dtype=torch.float32, device=device), persistent=True)
        self.register_buffer("qzeros", torch.empty((group_count, out_features), dtype=torch.int8, device=device), persistent=True)
        if bias:
            self.bias = _empty_parameter((out_features,), dtype=compute_dtype, device=device)
        else:
            self.register_parameter("bias", None)

    def prime_device_storage_(self) -> None:
        if self.qweight.device.type != "xpu":
            return
        with torch.no_grad():
            # Touch the full backing storage so Level Zero commits the slot into
            # dedicated VRAM during runtime setup instead of on the first expert fill.
            self.qweight.zero_()
            self.qscale.fill_(1.0)
            self.qzeros.fill_(8)
            if self.bias is not None:
                self.bias.zero_()

    @staticmethod
    def _padded_in_features(in_features: int, group_size: int) -> int:
        padded = ((in_features + group_size - 1) // group_size) * group_size
        return ((padded + 7) // 8) * 8

    @classmethod
    def from_linear(
        cls,
        module: nn.Module,
        *,
        group_size: int = 128,
        compute_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> "XPUInt4Linear":
        in_features = int(module.in_features)
        out_features = int(module.out_features)
        has_bias = getattr(module, "bias", None) is not None
        padded_in_features = cls._padded_in_features(in_features, group_size)
        quantized = cls(
            in_features,
            out_features,
            group_size=group_size,
            bias=has_bias,
            compute_dtype=compute_dtype,
            device=device,
            padded_in_features=padded_in_features,
        )
        quantized.copy_from_linear(module)
        return quantized

    def copy_from_linear(self, module: nn.Module) -> None:
        if int(module.in_features) != self.in_features or int(module.out_features) != self.out_features:
            raise ValueError(
                "Source linear shape does not match the target XPUInt4Linear: "
                f"expected {(self.out_features, self.in_features)}, "
                f"got {(int(module.out_features), int(module.in_features))}"
            )
        if isinstance(module, AutoRoundGPTQLinear):
            self._copy_from_autoround(module)
            return

        weight, bias = _extract_dense_weight_bias(module)
        qweight, qscale, qzeros = self._quantize_weight(
            weight,
            group_size=self.group_size,
            padded_in_features=self.padded_in_features,
        )
        with torch.no_grad():
            self.qweight.copy_(qweight.to(dtype=self.qweight.dtype))
            self.qscale.copy_(qscale.to(dtype=self.qscale.dtype))
            self.qzeros.copy_(qzeros.to(dtype=self.qzeros.dtype))
            if bias is not None and self.bias is not None:
                self.bias.copy_(bias.to(dtype=self.bias.dtype))

    def _copy_from_autoround(self, module: AutoRoundGPTQLinear) -> None:
        if module.qweight.numel() == 0 or module.scales.numel() == 0:
            raise RuntimeError("AutoRoundGPTQLinear has no quantized weight payload.")

        group_count = self.padded_in_features // self.group_size
        unpacked_weight = _unpack_int4_first_dim(module.qweight)
        if unpacked_weight.shape[0] < self.in_features or unpacked_weight.shape[1] < self.out_features:
            raise ValueError(
                f"Unsupported AutoRound qweight shape {tuple(module.qweight.shape)} "
                f"for target {(self.out_features, self.in_features)}"
            )
        weight_int = unpacked_weight[: self.in_features, : self.out_features].transpose(0, 1).contiguous()
        padded_weight = torch.full((self.out_features, self.padded_in_features), 8, dtype=torch.int32)
        padded_weight[:, : self.in_features] = weight_int.to(dtype=torch.int32)
        qweight = _pack_int4_last_dim(padded_weight)

        scales = module.scales.to(dtype=torch.float32)
        if scales.shape[0] < group_count or scales.shape[1] < self.out_features:
            raise ValueError(
                f"Unsupported AutoRound scales shape {tuple(module.scales.shape)} "
                f"for target {(self.out_features, self.in_features)}"
            )
        qscale = scales[:group_count, : self.out_features].contiguous()

        if module.zero_point and module.qzeros.numel() > 0:
            zeros_int = _unpack_int4_last_dim(module.qzeros)
            if zeros_int.shape[0] < group_count or zeros_int.shape[1] < self.out_features:
                raise ValueError(
                    f"Unsupported AutoRound qzeros shape {tuple(module.qzeros.shape)} "
                    f"for target {(self.out_features, self.in_features)}"
                )
            qzeros = zeros_int[:group_count, : self.out_features].to(dtype=torch.int8).contiguous()
        else:
            qzeros = torch.full((group_count, self.out_features), 8, dtype=torch.int8)

        with torch.no_grad():
            self.qweight.copy_(qweight.to(dtype=self.qweight.dtype))
            self.qscale.copy_(qscale.to(dtype=self.qscale.dtype))
            self.qzeros.copy_(qzeros.to(dtype=self.qzeros.dtype))
            if module.bias is not None and self.bias is not None:
                self.bias.copy_(module.bias.detach().to(device=torch.device("cpu"), dtype=self.bias.dtype))

    @staticmethod
    def _quantize_weight(
        weight: torch.Tensor,
        *,
        group_size: int,
        padded_in_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = weight.to(dtype=torch.float32, device="cpu")
        out_features, in_features = weight.shape
        if padded_in_features > in_features:
            padded_weight = torch.zeros((out_features, padded_in_features), dtype=torch.float32)
            padded_weight[:, :in_features] = weight
            weight = padded_weight

        group_count = padded_in_features // group_size
        grouped = weight.reshape(out_features, group_count, group_size)
        max_abs = grouped.abs().amax(dim=-1)
        scale = torch.where(max_abs > 0, max_abs / 7.0, torch.ones_like(max_abs))
        q = torch.round(grouped / scale.unsqueeze(-1) + 8.0).clamp_(0.0, 15.0).to(torch.int32)
        q = q.reshape(out_features, padded_in_features)
        qzeros = torch.full((group_count, out_features), 8, dtype=torch.int8)
        qscale = scale.transpose(0, 1).contiguous().to(dtype=torch.float32)
        packed = _pack_int4_last_dim(q)
        return packed, qscale, qzeros

    def _dequantize_weight(self) -> torch.Tensor:
        packed = self.qweight.to(device="cpu", dtype=torch.int32)
        shifts = (torch.arange(8, dtype=torch.int32).view(1, 1, 8) * 4)
        unpacked = torch.bitwise_right_shift(packed.unsqueeze(-1), shifts) & 0xF
        unpacked = unpacked.reshape(self.out_features, self.padded_in_features).to(dtype=torch.float32)
        qscale = self.qscale.to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous()
        qzeros = self.qzeros.to(device="cpu", dtype=torch.float32).transpose(0, 1).contiguous()
        groups = unpacked.reshape(self.out_features, self.padded_in_features // self.group_size, self.group_size)
        dequantized = (groups - qzeros.unsqueeze(-1)) * qscale.unsqueeze(-1)
        return dequantized.reshape(self.out_features, self.padded_in_features)[:, : self.in_features]

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
            output = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
                x_padded,
                self.qweight,
                self.group_size,
                self.qscale,
                self.qzeros,
            )
            if self.bias is not None:
                output = output + self.bias.to(device=output.device, dtype=output.dtype)
        else:
            weight = self._dequantize_weight().to(device=x_padded.device, dtype=self.compute_dtype)
            bias = None if self.bias is None else self.bias.to(device=x_padded.device, dtype=self.compute_dtype)
            output = F.linear(x_padded, weight, bias)
        return output.reshape(*original_shape, self.out_features).to(dtype=x.dtype)


@dataclass(slots=True)
class QuantizedModuleSpec:
    module_name: str
    quant_method: str


def _normalize_exclusion(module_name: str) -> str:
    for prefix in ("model.language_model.", "model.visual.", "model."):
        if module_name.startswith(prefix):
            return module_name[len(prefix) :]
    return module_name


def _should_skip(module_name: str, quantization_config: QuantizationConfig) -> bool:
    if not quantization_config.modules_to_not_convert:
        return False
    normalized = _normalize_exclusion(module_name)
    for candidate in quantization_config.modules_to_not_convert:
        candidate = _normalize_exclusion(candidate)
        if normalized == candidate or normalized.startswith(candidate + "."):
            return True
    return False


def _set_submodule(model: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def _get_submodule(model: nn.Module, module_name: str) -> nn.Module:
    current = model
    for part in module_name.split("."):
        current = getattr(current, part)
    return current


def _extract_dense_weight_bias(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(module, FP8Linear):
        weight = _dequantize_fp8_to_cpu(module)
        bias = None if module.bias is None else module.bias.detach().to(device=torch.device("cpu"), dtype=torch.float32)
        return weight.detach(), bias
    if isinstance(module, AutoRoundGPTQLinear):
        weight = _dequantize_autoround_to_cpu(module)
        bias = None if module.bias is None else module.bias.detach().to(device=torch.device("cpu"), dtype=torch.float32)
        return weight.detach(), bias
    if isinstance(module, AWQLinear):
        weight = _dequantize_awq_to_cpu(module)
        bias = None if module.bias is None else module.bias.detach().to(device=torch.device("cpu"), dtype=torch.float32)
        return weight.detach(), bias
    if isinstance(module, DenseLinear) or isinstance(module, nn.Linear):
        weight = module.weight.detach().to(device=torch.device("cpu"), dtype=torch.float32)
        bias = None if module.bias is None else module.bias.detach().to(device=torch.device("cpu"), dtype=torch.float32)
        return weight, bias
    if isinstance(module, XPUInt4Linear):
        weight = module._dequantize_weight()
        bias = None if module.bias is None else module.bias.detach().to(device=torch.device("cpu"), dtype=torch.float32)
        return weight, bias
    raise TypeError(f"Unsupported linear module for XPU int4 conversion: {type(module)!r}")


def extract_linear_weight_bias_cpu(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _extract_dense_weight_bias(module)


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
    packed_weight_bytes = out_features * (padded_in_features // 8) * 4
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
        elif isinstance(child, (FP8Linear, AutoRoundGPTQLinear, AWQLinear, DenseLinear, nn.Linear)):
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
) -> int:
    candidate_module_names: list[str] = []
    for module_name, child in list(module.named_modules()):
        if not module_name:
            continue
        if isinstance(child, XPUInt4Linear):
            continue
        if not isinstance(child, (FP8Linear, AutoRoundGPTQLinear, AWQLinear, DenseLinear, nn.Linear)):
            continue
        if include_predicate is not None and not include_predicate(module_name, child):
            continue
        candidate_module_names.append(module_name)

    replacements: list[tuple[str, XPUInt4Linear]] = []
    for module_name in candidate_module_names:
        child = _get_submodule(module, module_name)
        if isinstance(child, XPUInt4Linear):
            continue
        if not isinstance(child, (FP8Linear, AutoRoundGPTQLinear, AWQLinear, DenseLinear, nn.Linear)):
            continue
        replacements.append(
            (
                module_name,
                XPUInt4Linear.from_linear(
                    child,
                    group_size=group_size,
                    compute_dtype=compute_dtype or getattr(child, "compute_dtype", torch.bfloat16),
                    device=device,
                ),
            )
        )

    for module_name, replacement in replacements:
        _set_submodule(module, module_name, replacement)
    return len(replacements)


def _normalized_quant_method(quantization_config: QuantizationConfig) -> str:
    return (quantization_config.quant_method or "").strip().lower()


def _module_within_quantized_blocks(module_name: str, quantization_config: QuantizationConfig) -> bool:
    blocks = tuple(getattr(quantization_config, "block_name_to_quantize", ()) or ())
    if not blocks:
        return True
    normalized_module_name = _normalize_exclusion(module_name)
    for block_name in blocks:
        normalized_block_name = _normalize_exclusion(block_name)
        if (
            module_name == block_name
            or module_name.startswith(block_name + ".")
            or normalized_module_name == normalized_block_name
            or normalized_module_name.startswith(normalized_block_name + ".")
        ):
            return True
    return False


def _pattern_matches_module_name(module_name: str, pattern: str) -> bool:
    if not pattern:
        return False
    if any(ch in pattern for ch in "*+?[]{}()|\\^$"):
        return re.fullmatch(pattern, module_name) is not None
    return module_name == pattern


def _module_quantization_override(module_name: str, quantization_config: QuantizationConfig) -> dict[str, object] | None:
    extra_config = getattr(quantization_config, "extra_config", {}) or {}
    if not extra_config:
        return None
    normalized_module_name = _normalize_exclusion(module_name)

    direct_match: dict[str, object] | None = None
    regex_match: dict[str, object] | None = None
    for pattern, override in extra_config.items():
        normalized_pattern = _normalize_exclusion(pattern)
        if module_name == pattern or normalized_module_name == normalized_pattern:
            direct_match = dict(override)
        elif (
            _pattern_matches_module_name(module_name, pattern)
            or _pattern_matches_module_name(normalized_module_name, normalized_pattern)
        ):
            regex_match = dict(override)
    return direct_match or regex_match


def _should_quantize_auto_round_module(module_name: str, quantization_config: QuantizationConfig) -> bool:
    if not _module_within_quantized_blocks(module_name, quantization_config):
        return False
    override = _module_quantization_override(module_name, quantization_config)
    if not override:
        return True
    bits = override.get("bits")
    if bits is not None and int(bits) >= 16:
        return False
    data_type = str(override.get("data_type", "int")).strip().lower()
    if data_type in {"fp", "float", "float16", "float32", "bfloat16", "bf16", "fp16", "fp32"}:
        return False
    return True


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
    quant_method = _normalized_quant_method(quantization_config)

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if _should_skip(module_name, quantization_config):
            continue
        module_device = module.weight.device

        if quant_method == "fp8":
            replacement = FP8Linear(
                module.in_features,
                module.out_features,
                block_size=quantization_config.weight_block_size,
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
        elif quant_method in _AUTO_ROUND_METHODS:
            if not _should_quantize_auto_round_module(module_name, quantization_config):
                continue
            replacement = AutoRoundGPTQLinear(
                module.in_features,
                module.out_features,
                bits=quantization_config.bits or 4,
                group_size=quantization_config.group_size or 128,
                zero_point=quantization_config.zero_point or bool(getattr(quantization_config, "sym", False)),
                bias=module.bias is not None,
                compute_dtype=compute_dtype,
                device=module_device,
            )
        else:
            continue

        replacements.append((module_name, replacement))
        specs.append(QuantizedModuleSpec(module_name=module_name, quant_method=quant_method))

    for module_name, replacement in replacements:
        _set_submodule(model, module_name, replacement)

    return specs
