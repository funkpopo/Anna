from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.config import QuantizationConfig


def _float8_dtype():
    return getattr(torch, "float8_e4m3fn", None)


def _empty_parameter(shape: tuple[int, ...], dtype: torch.dtype | None = None) -> nn.Parameter:
    dtype = dtype or torch.float32
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


class DenseLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _empty_parameter((out_features, in_features))
        if bias:
            self.bias = _empty_parameter((out_features,))
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
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.compute_dtype = compute_dtype
        fp8_dtype = _float8_dtype() or torch.float16
        self.weight = _empty_parameter((out_features, in_features), dtype=fp8_dtype)
        if block_size is None:
            self.weight_scale_inv = nn.Parameter(torch.ones((), dtype=torch.float32), requires_grad=False)
        else:
            scale_out = (out_features + block_size[0] - 1) // block_size[0]
            scale_in = (in_features + block_size[1] - 1) // block_size[1]
            self.weight_scale_inv = nn.Parameter(
                torch.ones((scale_out, scale_in), dtype=torch.float32),
                requires_grad=False,
            )
        if bias:
            self.bias = _empty_parameter((out_features,), dtype=compute_dtype)
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
        self.register_buffer("qweight", torch.empty(0, dtype=torch.int32), persistent=True)
        self.register_buffer("qzeros", torch.empty(0, dtype=torch.int32), persistent=True)
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16), persistent=True)
        if bias:
            self.bias = _empty_parameter((out_features,), dtype=compute_dtype)
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
        shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
        unpacked = ((packed.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF).reshape(*packed.shape[:-1], packed.shape[-1] * 8)
        return unpacked

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
    quant_method = (quantization_config.quant_method or "").lower()

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if _should_skip(module_name, quantization_config):
            continue

        if quant_method == "fp8":
            replacement = FP8Linear(
                module.in_features,
                module.out_features,
                block_size=quantization_config.weight_block_size,
                bias=module.bias is not None,
                compute_dtype=compute_dtype,
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
            )
        else:
            continue

        replacements.append((module_name, replacement))
        specs.append(QuantizedModuleSpec(module_name=module_name, quant_method=quant_method))

    for module_name, replacement in replacements:
        _set_submodule(model, module_name, replacement)

    return specs
