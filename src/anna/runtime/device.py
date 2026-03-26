from __future__ import annotations

from dataclasses import dataclass

import torch

from anna.mm.processor import PreparedInputs
from anna.model.ops import Qwen3DynamicCache


_DTYPE_ALIASES: dict[str, str] = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float8": "float8_e4m3fn",
    "fp8": "float8_e4m3fn",
    "float8_e4m3fn": "float8_e4m3fn",
}

_COMPUTE_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _xpu_module():
    return getattr(torch, "xpu", None)


def _float8_dtype() -> torch.dtype | None:
    return getattr(torch, "float8_e4m3fn", None)


def has_xpu() -> bool:
    xpu = _xpu_module()
    return bool(xpu is not None and xpu.is_available())


def _normalize_dtype_name(dtype: str | None) -> str:
    if not dtype:
        return "bfloat16"
    normalized = _DTYPE_ALIASES.get(dtype.lower())
    if normalized is None:
        raise ValueError(f"Unsupported dtype alias: {dtype}")
    return normalized


def _resolve_compute_dtype(dtype: str, *, fallback: str = "bfloat16") -> torch.dtype:
    normalized = _normalize_dtype_name(dtype)
    if normalized == "float8_e4m3fn":
        normalized = _normalize_dtype_name(fallback)
    resolved = _COMPUTE_DTYPES.get(normalized)
    if resolved is None:
        raise ValueError(f"Unsupported compute dtype: {dtype}")
    return resolved


@dataclass(slots=True)
class TensorMigrationPolicy:
    preprocess_device: torch.device
    execution_device: torch.device
    parameter_dtype: torch.dtype
    cache_dtype: torch.dtype
    non_blocking: bool = True
    keep_cache_on_device: bool = True


@dataclass(slots=True)
class RuntimeSafetyPolicy:
    min_free_bytes: int = 1 << 30
    reserve_margin_bytes: int = 512 << 20
    max_estimated_usage_ratio: float = 0.9
    generation_memory_safety_factor: float = 2.0


@dataclass(slots=True)
class DeviceMemoryInfo:
    free_bytes: int
    total_bytes: int
    allocated_bytes: int | None = None
    reserved_bytes: int | None = None


@dataclass(slots=True)
class DeviceContext:
    device: torch.device
    dtype: torch.dtype
    requested_device: str
    requested_dtype: str
    reported_dtype: str
    float8_available: bool
    migration_policy: TensorMigrationPolicy
    safety_policy: RuntimeSafetyPolicy

    @classmethod
    def resolve(
        cls,
        *,
        device: str = "auto",
        dtype: str = "auto",
        model_dtype: str = "bfloat16",
    ) -> "DeviceContext":
        requested_device = device
        requested_dtype = dtype

        normalized_device = device.lower()
        if normalized_device == "auto":
            normalized_device = "xpu" if has_xpu() else "cpu"

        if normalized_device == "xpu" and not has_xpu():
            raise RuntimeError("Requested device 'xpu' but torch.xpu is not available.")

        if normalized_device not in {"cpu", "xpu"}:
            raise RuntimeError(f"Unsupported device: {device}")

        resolved_device = torch.device(normalized_device)
        if dtype == "auto":
            if resolved_device.type == "cpu":
                resolved_dtype = torch.float32
                reported_dtype = "float32"
            else:
                resolved_dtype = _resolve_compute_dtype(model_dtype, fallback="bfloat16")
                reported_dtype = _normalize_dtype_name(model_dtype)
        else:
            normalized_dtype = _normalize_dtype_name(dtype)
            if normalized_dtype == "float8_e4m3fn":
                resolved_dtype = _resolve_compute_dtype(model_dtype, fallback="bfloat16")
            else:
                resolved_dtype = _resolve_compute_dtype(dtype, fallback=model_dtype)
            reported_dtype = normalized_dtype

        if resolved_device.type == "cpu" and resolved_dtype == torch.bfloat16:
            reported_dtype = "bfloat16"

        migration_policy = TensorMigrationPolicy(
            preprocess_device=torch.device("cpu"),
            execution_device=resolved_device,
            parameter_dtype=resolved_dtype,
            cache_dtype=resolved_dtype,
        )

        return cls(
            device=resolved_device,
            dtype=resolved_dtype,
            requested_device=requested_device,
            requested_dtype=requested_dtype,
            reported_dtype=reported_dtype,
            float8_available=_float8_dtype() is not None,
            migration_policy=migration_policy,
            safety_policy=RuntimeSafetyPolicy(),
        )

    def _move_tensor(
        self,
        tensor: torch.Tensor | None,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        target_dtype = dtype
        if target_dtype is None and tensor.is_floating_point():
            target_dtype = self.dtype
        if target_dtype is None:
            return tensor.to(device=self.device, non_blocking=self.migration_policy.non_blocking)
        return tensor.to(device=self.device, dtype=target_dtype, non_blocking=self.migration_policy.non_blocking)

    def move_prepared_inputs(self, prepared: PreparedInputs) -> PreparedInputs:
        return PreparedInputs(
            prompt=prepared.prompt,
            input_ids=self._move_tensor(prepared.input_ids, dtype=torch.long),
            attention_mask=self._move_tensor(prepared.attention_mask, dtype=torch.long),
            mm_token_type_ids=self._move_tensor(prepared.mm_token_type_ids, dtype=torch.int32),
            pixel_values=self._move_tensor(prepared.pixel_values, dtype=self.dtype),
            image_grid_thw=self._move_tensor(prepared.image_grid_thw, dtype=torch.long),
            pixel_values_videos=self._move_tensor(prepared.pixel_values_videos, dtype=self.dtype),
            video_grid_thw=self._move_tensor(prepared.video_grid_thw, dtype=torch.long),
        )

    def move_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._move_tensor(token_ids, dtype=torch.long)

    def move_attention_mask(self, attention_mask: torch.Tensor | None) -> torch.Tensor | None:
        return self._move_tensor(attention_mask, dtype=torch.long)

    def move_cache(self, cache: Qwen3DynamicCache | None) -> Qwen3DynamicCache | None:
        if cache is None:
            return None
        return cache.to(device=self.device, dtype=self.migration_policy.cache_dtype)

    def element_size(self, dtype: torch.dtype | None = None) -> int:
        dtype = dtype or self.dtype
        return torch.empty((), dtype=dtype).element_size()

    def get_memory_info(self) -> DeviceMemoryInfo | None:
        if self.device.type != "xpu":
            return None
        xpu = _xpu_module()
        if xpu is None:
            return None

        free_bytes = total_bytes = allocated_bytes = reserved_bytes = None
        if hasattr(xpu, "mem_get_info"):
            free_bytes, total_bytes = xpu.mem_get_info()
        if hasattr(xpu, "memory_allocated"):
            allocated_bytes = int(xpu.memory_allocated(self.device))
        if hasattr(xpu, "memory_reserved"):
            reserved_bytes = int(xpu.memory_reserved(self.device))
        if free_bytes is None or total_bytes is None:
            return None
        return DeviceMemoryInfo(
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            allocated_bytes=allocated_bytes,
            reserved_bytes=reserved_bytes,
        )

    @staticmethod
    def classify_runtime_error(exc: BaseException) -> str:
        message = str(exc).lower()
        if "out of memory" in message or "memory" in message and "allocation" in message:
            return "out_of_memory"
        if "device lost" in message or "ur_result_error_device_lost" in message:
            return "device_lost"
        if "out of resources" in message:
            return "out_of_resources"
        return "runtime_error"

    def should_recover(self, exc: BaseException) -> bool:
        category = self.classify_runtime_error(exc)
        return category in {"out_of_memory", "device_lost", "out_of_resources"}

    def recover(self) -> None:
        self.synchronize()
        if self.device.type != "xpu":
            return
        xpu = _xpu_module()
        if xpu is None:
            return
        empty_cache = getattr(xpu, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
        synchronize = getattr(xpu, "synchronize", None)
        if callable(synchronize):
            synchronize()

    def synchronize(self) -> None:
        if self.device.type == "xpu":
            xpu = _xpu_module()
            if xpu is not None:
                xpu.synchronize()
