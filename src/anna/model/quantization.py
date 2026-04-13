from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass

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
            quantized.qweight.copy_(qweight)
            quantized.qscale.copy_(qscale)
            quantized.qzeros.copy_(qzeros)
            if bias is not None and quantized.bias is not None:
                quantized.bias.copy_(bias.to(dtype=quantized.bias.dtype))
        return quantized

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

        q_reshaped = q.reshape(out_features, padded_in_features // 8, 8)
        shifts = (torch.arange(8, dtype=torch.int32).view(1, 1, 8) * 4)
        packed = torch.bitwise_left_shift(q_reshaped & 0xF, shifts).sum(dim=-1, dtype=torch.int32)
        return packed.contiguous(), qscale, qzeros

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
        weight = module._dequantize_weight()
        bias = None if module.bias is None else module.bias.detach().to(dtype=torch.float32)
        return weight.detach(), bias
    if isinstance(module, AWQLinear):
        weight = module._dequantize_awq()
        bias = None if module.bias is None else module.bias.detach().to(dtype=torch.float32)
        return weight.detach(), bias
    if isinstance(module, DenseLinear) or isinstance(module, nn.Linear):
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
        elif isinstance(child, (FP8Linear, AWQLinear, DenseLinear, nn.Linear)):
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
        if not isinstance(child, (FP8Linear, AWQLinear, DenseLinear, nn.Linear)):
            continue
        if include_predicate is not None and not include_predicate(module_name, child):
            continue
        candidate_module_names.append(module_name)

    replacements = 0
    for module_name in candidate_module_names:
        child = _get_submodule(module, module_name)
        if isinstance(child, XPUInt4Linear):
            continue
        if not isinstance(child, (AutoRoundGPTQLinear, AWQLinear, DenseLinear, nn.Linear)):
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
        _set_submodule(module, module_name, replacement)
        replacements += 1
    return replacements


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
        else:
            continue

        replacements.append((module_name, replacement))
        specs.append(QuantizedModuleSpec(module_name=module_name, quant_method=quant_method))

    for module_name, replacement in replacements:
        _set_submodule(model, module_name, replacement)

    return specs
