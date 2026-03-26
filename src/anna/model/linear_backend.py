from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from anna.model.onednn_custom import custom_linear_pointwise, custom_linear_status


_SUPPORTED_ACTIVATIONS = {"none", "relu", "gelu", "swish"}
_SUPPORTED_BINARIES = {"add", "sum", "mul"}
_SINGLE_TOKEN_OUT_FEATURE_THRESHOLD = int(os.getenv("ANNA_ONEDNN_SINGLE_TOKEN_MIN_OUT_FEATURES", "8192"))


def _flatten_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    if x.ndim <= 2:
        return x, None
    leading_shape = tuple(x.shape[:-1])
    return x.reshape(-1, x.shape[-1]), leading_shape


def _restore_last_dim(x: torch.Tensor, leading_shape: tuple[int, ...] | None) -> torch.Tensor:
    if leading_shape is None:
        return x
    return x.reshape(*leading_shape, x.shape[-1])


def _normalize_activation(activation: str | None) -> str:
    normalized = (activation or "none").lower()
    if normalized not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Unsupported linear activation fusion: {activation}")
    return normalized


def _normalize_binary(binary: str | None) -> str | None:
    if binary is None:
        return None
    normalized = binary.lower()
    if normalized not in _SUPPORTED_BINARIES:
        raise ValueError(f"Unsupported linear binary fusion: {binary}")
    return normalized


def _apply_activation(
    tensor: torch.Tensor,
    *,
    activation: str,
    algorithm: str | None,
) -> torch.Tensor:
    if activation == "none":
        return tensor
    if activation == "relu":
        return F.relu(tensor)
    if activation == "gelu":
        approximate = "tanh" if (algorithm or "").lower() == "tanh" else "none"
        return F.gelu(tensor, approximate=approximate)
    if activation == "swish":
        return F.silu(tensor)
    raise ValueError(f"Unsupported activation: {activation}")


def _apply_binary(tensor: torch.Tensor, other: torch.Tensor, *, binary: str | None) -> torch.Tensor:
    if binary is None:
        return tensor
    if binary in {"add", "sum"}:
        return tensor + other
    if binary == "mul":
        return tensor * other
    raise ValueError(f"Unsupported binary op: {binary}")


def _mkldnn_linear_op():
    mkldnn_ns = getattr(torch.ops, "mkldnn", None)
    if mkldnn_ns is None or not hasattr(mkldnn_ns, "_linear_pointwise"):
        return None
    return mkldnn_ns._linear_pointwise


def _mkldnn_linear_binary_op():
    linear_op = _mkldnn_linear_op()
    if linear_op is None or not hasattr(linear_op, "binary"):
        return None
    return linear_op.binary


def onednn_linear_status() -> dict[str, object]:
    return {
        "available": _mkldnn_linear_op() is not None,
        "binary_available": _mkldnn_linear_binary_op() is not None,
        "disabled": os.getenv("ANNA_DISABLE_ONEDNN_LINEAR", "").strip() == "1",
        "single_token_min_out_features": _SINGLE_TOKEN_OUT_FEATURE_THRESHOLD,
        "custom_op": custom_linear_status(),
    }


def _should_use_onednn(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str,
    binary: str | None,
) -> bool:
    if os.getenv("ANNA_DISABLE_ONEDNN_LINEAR", "").strip() == "1":
        return False
    if x.numel() == 0 or weight.numel() == 0:
        return False
    if x.device.type != "xpu" or weight.device.type != "xpu":
        return False
    if x.dtype != weight.dtype:
        return False
    if x.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return False
    if binary is not None:
        return _mkldnn_linear_binary_op() is not None
    if activation != "none":
        return _mkldnn_linear_op() is not None

    linear_op = _mkldnn_linear_op()
    if linear_op is None:
        return False

    m = int(x.shape[0])
    n = int(weight.shape[0])
    return m > 1 or n >= _SINGLE_TOKEN_OUT_FEATURE_THRESHOLD


def linear_with_backend(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    activation: str = "none",
    algorithm: str | None = None,
    other: torch.Tensor | None = None,
    binary: str | None = None,
) -> torch.Tensor:
    activation = _normalize_activation(activation)
    binary = _normalize_binary(binary)

    flat_x, leading_shape = _flatten_last_dim(x)
    flat_other = None
    if other is not None:
        flat_other, other_shape = _flatten_last_dim(other)
        if other_shape != leading_shape:
            raise ValueError("Binary linear fusion requires matching leading dimensions.")

    use_onednn = _should_use_onednn(flat_x, weight, activation=activation, binary=binary)
    if use_onednn:
        custom_output = custom_linear_pointwise(
            flat_x,
            weight,
            bias,
            activation=activation,
            algorithm=algorithm,
            other=flat_other,
            binary=binary,
        )
        if custom_output is not None:
            return _restore_last_dim(custom_output, leading_shape)

        if flat_other is None:
            output = _mkldnn_linear_op()(flat_x, weight, bias, activation, [], (algorithm or ""))
        else:
            output = _mkldnn_linear_binary_op()(flat_x, flat_other, weight, bias, binary)
        return _restore_last_dim(output, leading_shape)

    fallback_bias = None if bias is None else bias.to(device=flat_x.device, dtype=flat_x.dtype)
    output = F.linear(flat_x, weight.to(device=flat_x.device, dtype=flat_x.dtype), fallback_bias)
    output = _apply_activation(output, activation=activation, algorithm=algorithm)
    if flat_other is not None:
        output = _apply_binary(output, flat_other.to(device=output.device, dtype=output.dtype), binary=binary)
    return _restore_last_dim(output, leading_shape)


def project_linear(
    module: nn.Module,
    x: torch.Tensor,
    *,
    activation: str = "none",
    algorithm: str | None = None,
    other: torch.Tensor | None = None,
    binary: str | None = None,
) -> torch.Tensor:
    custom_project = getattr(module, "project_linear", None)
    if callable(custom_project):
        return custom_project(
            x,
            activation=activation,
            algorithm=algorithm,
            other=other,
            binary=binary,
        )

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        output = module(x)
        output = _apply_activation(output, activation=_normalize_activation(activation), algorithm=algorithm)
        if other is not None:
            output = _apply_binary(output, other, binary=_normalize_binary(binary))
        return output

    bias = getattr(module, "bias", None)
    if bias is not None and not isinstance(bias, torch.Tensor):
        bias = None

    return linear_with_backend(
        x,
        weight,
        bias,
        activation=activation,
        algorithm=algorithm,
        other=other,
        binary=binary,
    )
