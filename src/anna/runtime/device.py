from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from anna.mm.prepared_inputs import PreparedInputsT, replace_prepared_inputs
from anna.model.ops import Qwen3DynamicCache


_DTYPE_ALIASES: dict[str, str] = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}

_COMPUTE_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

logger = logging.getLogger(__name__)


def _xpu_module():
    return getattr(torch, "xpu", None)


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


def _resolve_compute_dtype(dtype: str) -> torch.dtype:
    normalized = _normalize_dtype_name(dtype)
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
class XPUDeviceInfo:
    device_index: int
    name: str
    total_memory: int | None = None
    free_memory: int | None = None
    allocated_memory: int | None = None
    reserved_memory: int | None = None
    runtime: str = "torch.xpu"
    device_type: str | None = None
    is_arc_alchemist: bool = False
    is_acm_g10: bool = False
    is_arc_a770_or_a750: bool = False

    def as_log_fields(self) -> dict[str, object]:
        return {
            "device_index": self.device_index,
            "name": self.name,
            "total_memory": self.total_memory,
            "free_memory": self.free_memory,
            "allocated_memory": self.allocated_memory,
            "reserved_memory": self.reserved_memory,
            "runtime": self.runtime,
            "device_type": self.device_type,
            "is_arc_alchemist": self.is_arc_alchemist,
            "is_acm_g10": self.is_acm_g10,
            "is_arc_a770_or_a750": self.is_arc_a770_or_a750,
        }


def _call_xpu_getter(name: str, *args):
    xpu = _xpu_module()
    if xpu is None:
        return None
    getter = getattr(xpu, name, None)
    if not callable(getter):
        return None
    try:
        return getter(*args)
    except Exception:
        logger.debug("torch.xpu.%s failed", name, exc_info=True)
        return None


def _current_xpu_index(device: torch.device | None = None) -> int:
    if device is not None and device.index is not None:
        return int(device.index)
    current = _call_xpu_getter("current_device")
    if current is None:
        return 0
    return int(current)


def inspect_xpu_device(device: torch.device | None = None) -> XPUDeviceInfo | None:
    if not has_xpu():
        return None

    index = _current_xpu_index(device)
    name = _call_xpu_getter("get_device_name", index)
    if name is None:
        properties = _call_xpu_getter("get_device_properties", index)
        name = getattr(properties, "name", None) if properties is not None else None
    name = str(name or f"xpu:{index}")
    lowered_name = name.lower()

    probe_device = torch.device("xpu", index)
    free_memory = total_memory = allocated_memory = reserved_memory = None
    xpu = _xpu_module()
    if xpu is not None and hasattr(xpu, "mem_get_info"):
        try:
            free_memory, total_memory = xpu.mem_get_info()
        except Exception:
            logger.debug("torch.xpu.mem_get_info failed", exc_info=True)
    if xpu is not None and hasattr(xpu, "memory_allocated"):
        try:
            allocated_memory = int(xpu.memory_allocated(probe_device))
        except Exception:
            logger.debug("torch.xpu.memory_allocated failed", exc_info=True)
    if xpu is not None and hasattr(xpu, "memory_reserved"):
        try:
            reserved_memory = int(xpu.memory_reserved(probe_device))
        except Exception:
            logger.debug("torch.xpu.memory_reserved failed", exc_info=True)

    is_arc = "arc" in lowered_name
    is_a770_or_a750 = "a770" in lowered_name or "a750" in lowered_name
    is_acm_g10 = is_a770_or_a750 or "acm-g10" in lowered_name or "acm_g10" in lowered_name
    return XPUDeviceInfo(
        device_index=index,
        name=name,
        total_memory=None if total_memory is None else int(total_memory),
        free_memory=None if free_memory is None else int(free_memory),
        allocated_memory=allocated_memory,
        reserved_memory=reserved_memory,
        device_type="arc" if is_arc else None,
        is_arc_alchemist=is_arc,
        is_acm_g10=is_acm_g10,
        is_arc_a770_or_a750=is_a770_or_a750,
    )


def configure_xpu_environment(*, device_index: int | None = None, set_selector: bool = False) -> dict[str, str]:
    configured: dict[str, str] = {}
    defaults = {
        "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS": "1",
        "ZES_ENABLE_SYSMAN": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
        configured[key] = os.environ[key]
    if set_selector and device_index is not None:
        os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{int(device_index)}"
        configured["ONEAPI_DEVICE_SELECTOR"] = os.environ["ONEAPI_DEVICE_SELECTOR"]
    elif "ONEAPI_DEVICE_SELECTOR" in os.environ:
        configured["ONEAPI_DEVICE_SELECTOR"] = os.environ["ONEAPI_DEVICE_SELECTOR"]
    return configured


@dataclass(slots=True)
class DeviceContext:
    device: torch.device
    dtype: torch.dtype
    requested_device: str
    requested_dtype: str
    reported_dtype: str
    migration_policy: TensorMigrationPolicy
    safety_policy: RuntimeSafetyPolicy
    xpu_info: XPUDeviceInfo | None = None

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
                resolved_dtype = _resolve_compute_dtype(model_dtype)
                reported_dtype = _normalize_dtype_name(model_dtype)
        else:
            normalized_dtype = _normalize_dtype_name(dtype)
            resolved_dtype = _resolve_compute_dtype(dtype)
            reported_dtype = normalized_dtype

        if resolved_device.type == "cpu" and resolved_dtype == torch.bfloat16:
            reported_dtype = "bfloat16"

        migration_policy = TensorMigrationPolicy(
            preprocess_device=torch.device("cpu"),
            execution_device=resolved_device,
            parameter_dtype=resolved_dtype,
            cache_dtype=resolved_dtype,
        )

        xpu_info = inspect_xpu_device(resolved_device) if resolved_device.type == "xpu" else None
        if xpu_info is not None:
            logger.info("Resolved XPU device info: %s", xpu_info.as_log_fields())

        return cls(
            device=resolved_device,
            dtype=resolved_dtype,
            requested_device=requested_device,
            requested_dtype=requested_dtype,
            reported_dtype=reported_dtype,
            migration_policy=migration_policy,
            safety_policy=RuntimeSafetyPolicy(),
            xpu_info=xpu_info,
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
        if tensor.device == self.device and (target_dtype is None or tensor.dtype == target_dtype):
            return tensor
        if target_dtype is None:
            return tensor.to(device=self.device, non_blocking=self.migration_policy.non_blocking)
        return tensor.to(device=self.device, dtype=target_dtype, non_blocking=self.migration_policy.non_blocking)

    def move_prepared_inputs(self, prepared: PreparedInputsT) -> PreparedInputsT:
        input_ids = self._move_tensor(prepared.input_ids, dtype=torch.long)
        attention_mask = self._move_tensor(prepared.attention_mask, dtype=torch.long)
        mm_token_type_ids = self._move_tensor(prepared.mm_token_type_ids, dtype=torch.int32)
        pixel_values = self._move_tensor(prepared.pixel_values, dtype=self.dtype)
        image_position_ids = self._move_tensor(prepared.image_position_ids, dtype=torch.long)
        image_grid_thw = self._move_tensor(prepared.image_grid_thw, dtype=torch.long)
        pixel_values_videos = self._move_tensor(prepared.pixel_values_videos, dtype=self.dtype)
        video_position_ids = self._move_tensor(prepared.video_position_ids, dtype=torch.long)
        video_grid_thw = self._move_tensor(prepared.video_grid_thw, dtype=torch.long)
        input_features = self._move_tensor(prepared.input_features, dtype=self.dtype)
        input_features_mask = self._move_tensor(prepared.input_features_mask, dtype=torch.bool)
        if (
            input_ids is prepared.input_ids
            and attention_mask is prepared.attention_mask
            and mm_token_type_ids is prepared.mm_token_type_ids
            and pixel_values is prepared.pixel_values
            and image_position_ids is prepared.image_position_ids
            and image_grid_thw is prepared.image_grid_thw
            and pixel_values_videos is prepared.pixel_values_videos
            and video_position_ids is prepared.video_position_ids
            and video_grid_thw is prepared.video_grid_thw
            and input_features is prepared.input_features
            and input_features_mask is prepared.input_features_mask
        ):
            return prepared
        return replace_prepared_inputs(
            prepared,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_position_ids=video_position_ids,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            input_features_mask=input_features_mask,
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

    def get_memory_stats(self) -> dict[str, int | float] | None:
        if self.device.type != "xpu":
            return None
        xpu = _xpu_module()
        if xpu is None or not hasattr(xpu, "memory_stats"):
            return None

        memory_stats = getattr(xpu, "memory_stats")
        try:
            stats = memory_stats(self.device)
        except TypeError:
            stats = memory_stats()
        if not isinstance(stats, dict):
            return None
        return stats

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

    def release_unused_memory(self) -> None:
        if self.device.type != "xpu":
            return
        xpu = _xpu_module()
        if xpu is None:
            return
        empty_cache = getattr(xpu, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()

    def synchronize(self) -> None:
        if self.device.type == "xpu":
            xpu = _xpu_module()
            if xpu is not None:
                xpu.synchronize()
