from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.qwen3_5_text_config import QuantizationConfig

_INT4_VALUES_PER_PACK = 8
_AUTO_ROUND_QUANT_METHODS = frozenset({"auto-round", "auto_round"})
_AUTO_ROUND_PACKING_FORMATS = frozenset({"auto_round:auto_gptq"})
_FLOAT_OVERRIDE_DATA_TYPES = frozenset({"fp", "float", "float16", "fp16", "bfloat16", "bf16"})


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
        return _unpack_int4_last_dim(self.qzeros)[:, : self.out_features]

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

        packed_weight_cpu = _unpack_int4_first_dim(self.qweight.to(device="cpu", dtype=torch.int32))[
            : self.in_features, : self.out_features
        ]
        zeros_cpu = _unpack_int4_last_dim(self.qzeros.to(device="cpu", dtype=torch.int32))[:, : self.out_features]
        scales_cpu = self.scales.to(device="cpu", dtype=torch.float32)

        padded_weight = torch.empty((padded_in_features, self.out_features), dtype=torch.int32, device="cpu")
        padded_weight[: self.in_features].copy_(packed_weight_cpu)
        if padded_in_features > self.in_features:
            tail_group_idx = min(zeros_cpu.shape[0] - 1, self.in_features // self.group_size)
            fill_value = zeros_cpu[tail_group_idx].unsqueeze(0).expand(padded_in_features - self.in_features, -1)
            padded_weight[self.in_features :].copy_(fill_value)

        transposed = padded_weight.transpose(0, 1).contiguous()
        reshaped = transposed.reshape(self.out_features, padded_in_features // _INT4_VALUES_PER_PACK, _INT4_VALUES_PER_PACK)
        shifts = (_int4_shifts(device="cpu").view(1, 1, _INT4_VALUES_PER_PACK))
        qweight = torch.bitwise_left_shift(reshaped & 0xF, shifts).sum(dim=-1, dtype=torch.int32).contiguous()
        qzeros = zeros_cpu.to(dtype=torch.int8).contiguous()
        return qweight, scales_cpu.contiguous(), qzeros

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

        q_reshaped = q.reshape(out_features, padded_in_features // _INT4_VALUES_PER_PACK, _INT4_VALUES_PER_PACK)
        shifts = (_int4_shifts(device="cpu").view(1, 1, _INT4_VALUES_PER_PACK))
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


def _module_override_config(module_name: str, quantization_config: QuantizationConfig) -> dict[str, object] | None:
    if not quantization_config.extra_config:
        return None

    exact_match = quantization_config.extra_config.get(module_name)
    if exact_match is not None:
        return exact_match

    normalized = _normalize_exclusion(module_name)
    normalized_match = quantization_config.extra_config.get(normalized)
    if normalized_match is not None:
        return normalized_match

    for pattern, override in quantization_config.extra_config.items():
        candidate = pattern
        for prefix in ("model.language_model.", "model.visual.", "model."):
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :]
                break
        try:
            if re.fullmatch(pattern, module_name) or re.fullmatch(candidate, normalized):
                return override
        except re.error as exc:
            raise ValueError(f"Invalid quantization extra_config regex {pattern!r}: {exc}") from exc
    return None


def _should_skip(module_name: str, quantization_config: QuantizationConfig) -> bool:
    if quantization_config.modules_to_not_convert:
        normalized = _normalize_exclusion(module_name)
        for candidate in quantization_config.modules_to_not_convert:
            candidate = _normalize_exclusion(candidate)
            if normalized == candidate or normalized.startswith(candidate + "."):
                return True

    override = _module_override_config(module_name, quantization_config)
    if override is None:
        return False

    override_bits = override.get("bits")
    if override_bits is not None and int(override_bits) >= 16:
        return True

    override_data_type = override.get("data_type")
    if override_data_type is not None and str(override_data_type).strip().lower() in _FLOAT_OVERRIDE_DATA_TYPES:
        return True
    return False


def _should_quantize_autoround_module(module_name: str, quantization_config: QuantizationConfig) -> bool:
    block_names = tuple(name for name in quantization_config.block_name_to_quantize if name)
    if not block_names:
        raise ValueError("AutoRound quantization_config is missing block_name_to_quantize.")
    return any(module_name == block_name or module_name.startswith(block_name + ".") for block_name in block_names)


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
) -> int:
    replacements: list[tuple[str, XPUInt4Linear]] = []
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
        replacements.append(
            (
                module_name,
                XPUInt4Linear.from_linear(
                    child,
                    group_size=resolved_group_size,
                    compute_dtype=compute_dtype or getattr(child, "compute_dtype", torch.bfloat16),
                    device=device,
                ),
            )
        )

    for module_name, replacement in replacements:
        _set_submodule(module, module_name, replacement)
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

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if _should_skip(module_name, quantization_config):
            continue
        if quant_method in _AUTO_ROUND_QUANT_METHODS and not _should_quantize_autoround_module(module_name, quantization_config):
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
