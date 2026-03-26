from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.config import QuantizationConfig
from anna.model.linear_backend import linear_with_backend
from anna.model.onednn_custom import custom_linear_int4_weight_only


def _float8_dtype():
    return getattr(torch, "float8_e4m3fn", None)


def _empty_parameter(shape: tuple[int, ...], dtype: torch.dtype | None = None) -> nn.Parameter:
    dtype = dtype or torch.float32
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


def _flatten_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    if x.ndim <= 2:
        return x, None
    leading_shape = tuple(x.shape[:-1])
    return x.reshape(-1, x.shape[-1]), leading_shape


def _restore_last_dim(x: torch.Tensor, leading_shape: tuple[int, ...] | None) -> torch.Tensor:
    if leading_shape is None:
        return x
    return x.reshape(*leading_shape, x.shape[-1])


def _apply_activation(
    tensor: torch.Tensor,
    *,
    activation: str,
    algorithm: str | None,
) -> torch.Tensor:
    normalized = (activation or "none").lower()
    if normalized == "none":
        return tensor
    if normalized == "relu":
        return F.relu(tensor)
    if normalized == "gelu":
        approximate = "tanh" if (algorithm or "").lower() == "tanh" else "none"
        return F.gelu(tensor, approximate=approximate)
    if normalized == "swish":
        return F.silu(tensor)
    raise ValueError(f"Unsupported activation: {activation}")


def _apply_binary(
    tensor: torch.Tensor,
    other: torch.Tensor | None,
    *,
    binary: str | None,
) -> torch.Tensor:
    if other is None or binary is None:
        return tensor
    normalized = binary.lower()
    if normalized in {"add", "sum"}:
        return tensor + other
    if normalized == "mul":
        return tensor * other
    raise ValueError(f"Unsupported binary op: {binary}")


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
        return self.project_linear(x)

    def project_linear(
        self,
        x: torch.Tensor,
        *,
        activation: str = "none",
        algorithm: str | None = None,
        other: torch.Tensor | None = None,
        binary: str | None = None,
    ) -> torch.Tensor:
        weight = self.weight.to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return linear_with_backend(
            x,
            weight,
            bias,
            activation=activation,
            algorithm=algorithm,
            other=other,
            binary=binary,
        )


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
        return self.project_linear(x)

    def project_linear(
        self,
        x: torch.Tensor,
        *,
        activation: str = "none",
        algorithm: str | None = None,
        other: torch.Tensor | None = None,
        binary: str | None = None,
    ) -> torch.Tensor:
        x_compute = x.to(dtype=self.compute_dtype)
        weight = self._dequantize_weight().to(dtype=self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(dtype=self.compute_dtype)
        other_compute = None if other is None else other.to(dtype=self.compute_dtype)
        return linear_with_backend(
            x_compute,
            weight,
            bias,
            activation=activation,
            algorithm=algorithm,
            other=other_compute,
            binary=binary,
        ).to(dtype=x.dtype)


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
        self._weight_only_cache_signature: tuple[object, ...] | None = None
        self._weight_only_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @staticmethod
    def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
        shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
        unpacked = ((packed.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF).reshape(*packed.shape[:-1], packed.shape[-1] * 8)
        return unpacked

    @staticmethod
    def _pack_int4(values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] % 8 != 0:
            raise ValueError(f"INT4 packing requires the last dimension to be divisible by 8, got {values.shape[-1]}")
        shifts = torch.arange(0, 32, 4, device=values.device, dtype=torch.int32).view(*([1] * values.ndim), 8)
        reshaped = values.to(torch.int32).reshape(*values.shape[:-1], values.shape[-1] // 8, 8)
        packed = torch.sum(reshaped << shifts, dim=-1, dtype=torch.int32)
        return packed.contiguous()

    def _build_weight_only_cache_signature(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[object, ...]:
        return (
            device.type,
            device.index,
            dtype,
            tuple(self.qweight.shape),
            self.qweight.dtype,
            self.qweight.device.type,
            self.qweight.device.index,
            int(self.qweight.data_ptr()) if self.qweight.numel() else 0,
            tuple(self.qzeros.shape),
            self.qzeros.dtype,
            self.qzeros.device.type,
            self.qzeros.device.index,
            int(self.qzeros.data_ptr()) if self.qzeros.numel() else 0,
            tuple(self.scales.shape),
            self.scales.dtype,
            self.scales.device.type,
            self.scales.device.index,
            int(self.scales.data_ptr()) if self.scales.numel() else 0,
        )

    def _prepare_awq_weight_only_tensors(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.qweight.numel() == 0:
            raise RuntimeError("AWQLinear has no quantized weight payload.")
        if self.scales.numel() == 0:
            raise RuntimeError("AWQLinear has no AWQ scale payload.")
        if self.group_size <= 0:
            raise RuntimeError("AWQLinear group_size must be positive.")
        if self.in_features % 8 != 0:
            raise RuntimeError(
                f"AWQLinear XPU weight-only path requires in_features divisible by 8, got {self.in_features}"
            )
        if self.group_size % 32 != 0:
            raise RuntimeError(
                f"AWQLinear XPU weight-only path requires group_size divisible by 32, got {self.group_size}"
            )
        if self.in_features % self.group_size != 0:
            raise RuntimeError(
                "AWQLinear XPU weight-only path requires in_features divisible by group_size "
                f"(got in_features={self.in_features}, group_size={self.group_size})"
            )

        signature = self._build_weight_only_cache_signature(device=device, dtype=dtype)
        if self._weight_only_cache_signature == signature and self._weight_only_cache is not None:
            return self._weight_only_cache

        group_count = self.in_features // self.group_size

        qweight = self.qweight.to(device=device, dtype=torch.int32)
        scales = self.scales.to(device=device)
        qzeros = self.qzeros.to(device=device) if self.qzeros.numel() else self.qzeros

        if scales.ndim != 2:
            raise ValueError(f"Unsupported AWQ scale shape {tuple(scales.shape)}")
        if scales.shape == (group_count, self.out_features):
            prepared_scales = scales
        elif scales.shape == (self.out_features, group_count):
            prepared_scales = scales.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                f"Unsupported AWQ scale shape {tuple(scales.shape)} for target {(group_count, self.out_features)}"
            )
        prepared_scales = prepared_scales.to(dtype=dtype)

        if qweight.ndim != 2:
            raise ValueError(f"Unsupported AWQ packed weight rank: {qweight.ndim}")
        expected_k_packs = self.in_features // 8
        if qweight.shape[0] == self.in_features and qweight.shape[1] * 8 >= self.out_features:
            unpacked = self._unpack_int4(qweight)[:, : self.out_features]
            prepared_weight = self._pack_int4(unpacked.transpose(0, 1).contiguous())
        elif qweight.shape[0] == self.out_features and qweight.shape[1] * 8 >= self.in_features:
            prepared_weight = qweight[:, :expected_k_packs].contiguous()
        else:
            raise ValueError(
                f"Unsupported AWQ packed weight shape {tuple(qweight.shape)} for target {(self.out_features, expected_k_packs)}"
            )
        if prepared_weight.shape != (self.out_features, expected_k_packs):
            raise ValueError(
                f"AWQ packed weight repack produced shape {tuple(prepared_weight.shape)}, expected {(self.out_features, expected_k_packs)}"
            )

        if qzeros.numel() == 0:
            prepared_zeros = torch.full(
                (group_count, self.out_features),
                8,
                device=device,
                dtype=torch.int8,
            )
        else:
            if qzeros.ndim != 2:
                raise ValueError(f"Unsupported AWQ zero-point shape {tuple(qzeros.shape)}")
            if qzeros.shape[0] == group_count and qzeros.shape[1] == self.out_features:
                prepared_zeros = qzeros.to(dtype=torch.int8).contiguous()
            elif qzeros.shape[0] == group_count and qzeros.shape[1] * 8 >= self.out_features:
                prepared_zeros = self._unpack_int4(qzeros.to(dtype=torch.int32))[:, : self.out_features].to(torch.int8)
            else:
                raise ValueError(
                    f"Unsupported AWQ zero-point shape {tuple(qzeros.shape)} for target {(group_count, self.out_features)}"
                )

        self._weight_only_cache_signature = signature
        self._weight_only_cache = (prepared_weight, prepared_scales, prepared_zeros)
        return self._weight_only_cache

    def _project_awq_weight_only(
        self,
        x: torch.Tensor,
        *,
        activation: str,
        algorithm: str | None,
        other: torch.Tensor | None,
        binary: str | None,
    ) -> torch.Tensor | None:
        if x.device.type != "xpu":
            return None
        if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            return None
        try:
            packed_weight, prepared_scales, prepared_zeros = self._prepare_awq_weight_only_tensors(
                device=x.device,
                dtype=self.compute_dtype,
            )
        except (RuntimeError, ValueError):
            return None

        x_compute = x.to(dtype=self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=self.compute_dtype)
        other_compute = None if other is None else other.to(device=x.device, dtype=self.compute_dtype)
        flat_x, leading_shape = _flatten_last_dim(x_compute)
        flat_other = None
        if other_compute is not None:
            flat_other, other_shape = _flatten_last_dim(other_compute)
            if other_shape != leading_shape:
                raise ValueError("Binary linear fusion requires matching leading dimensions.")

        output = custom_linear_int4_weight_only(
            flat_x,
            packed_weight,
            prepared_scales,
            prepared_zeros,
            group_size=self.group_size,
        )
        if output is None:
            return None
        if bias is not None:
            output = output + bias
        output = _apply_activation(output, activation=activation, algorithm=algorithm)
        output = _apply_binary(output, flat_other, binary=binary)
        return _restore_last_dim(output, leading_shape).to(dtype=x.dtype)

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
        return self.project_linear(x)

    def project_linear(
        self,
        x: torch.Tensor,
        *,
        activation: str = "none",
        algorithm: str | None = None,
        other: torch.Tensor | None = None,
        binary: str | None = None,
    ) -> torch.Tensor:
        if self.weight is not None:
            weight = self.weight.to(dtype=self.compute_dtype)
        else:
            weight_only_output = self._project_awq_weight_only(
                x,
                activation=activation,
                algorithm=algorithm,
                other=other,
                binary=binary,
            )
            if weight_only_output is not None:
                return weight_only_output
            weight = self._dequantize_awq().to(dtype=self.compute_dtype)
        bias = None if self.bias is None else self.bias.to(dtype=self.compute_dtype)
        x_compute = x.to(dtype=self.compute_dtype)
        other_compute = None if other is None else other.to(dtype=self.compute_dtype)
        return linear_with_backend(
            x_compute,
            weight,
            bias,
            activation=activation,
            algorithm=algorithm,
            other=other_compute,
            binary=binary,
        ).to(dtype=x.dtype)


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
