from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from anna.core.gguf_model import has_gguf_model
from anna.core.native import (
    SafetensorsShardPlan,
    inspect_safetensors_load_plan,
    inspect_safetensors_manifest,
    quantize_safetensors_linear_int4_batch,
)
from anna.model.qwen3_5_text_config import Qwen3_5TextModelConfig
from anna.model.qwen3_5_text_model import Qwen3_5TextForConditionalGeneration
from anna.model.quantization import (
    XPUInt4Linear,
    _release_cpu_memory_caches,
    replace_linear_modules,
    replace_linear_modules_with_xpu_int4_placeholders,
)
from anna.weights.safetensors_device import safetensors_pt_device_str

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WeightLoadReport:
    loaded: int
    skipped: int
    quantized_replacements: int = 0


def load_qwen3_5_text_model_config(model_dir: str | Path) -> Qwen3_5TextModelConfig:
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.exists():
        if has_gguf_model(model_path):
            from anna.weights.gguf_support import load_qwen3_5_text_model_config_from_gguf

            return load_qwen3_5_text_model_config_from_gguf(model_path)
        raise FileNotFoundError(f"Missing config file: {config_path}")
    return Qwen3_5TextModelConfig.from_model_dir(model_path)


def _iter_weight_files(model_dir: Path) -> list[Path]:
    return inspect_safetensors_manifest(model_dir)[0]


def estimate_qwen3_5_text_model_weight_bytes(model_dir: str | Path) -> int:
    model_path = Path(model_dir)
    if has_gguf_model(model_path):
        from anna.weights.gguf_support import estimate_qwen3_5_text_model_weight_bytes_from_gguf

        return estimate_qwen3_5_text_model_weight_bytes_from_gguf(model_path)
    return inspect_safetensors_manifest(model_path)[1]


def _load_plan(model_dir: Path) -> tuple[list[SafetensorsShardPlan], int]:
    return inspect_safetensors_load_plan(model_dir)


def build_qwen3_5_text_model(
    config: Qwen3_5TextModelConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    int4_placeholder_predicate: Callable[[str, torch.nn.Module], bool] | None = None,
) -> tuple[Qwen3_5TextForConditionalGeneration, int]:
    default_dtype = torch.get_default_dtype()
    build_dtype = dtype if dtype in {torch.float16, torch.bfloat16, torch.float32} else torch.float32
    try:
        torch.set_default_dtype(build_dtype)
        logger.info("Constructing Qwen3.5 model skeleton on meta device.")
        with torch.device("meta"):
            model = Qwen3_5TextForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(default_dtype)

    logger.info("Replacing Qwen3.5 linear layers with quantized placeholders.")
    quantized_specs = replace_linear_modules(model, config.quantization_config, compute_dtype=dtype)
    direct_int4_placeholders = 0
    if int4_placeholder_predicate is not None:
        logger.info("Replacing selected Qwen3.5 linear layers with direct XPU-int4 placeholders.")
        direct_int4_placeholders = replace_linear_modules_with_xpu_int4_placeholders(
            model,
            compute_dtype=dtype,
            device=torch.device("meta"),
            include_predicate=int4_placeholder_predicate,
        )
    if not hasattr(model, "to_empty"):
        raise RuntimeError("The current torch build does not support nn.Module.to_empty, which is required for low-memory model loading.")
    logger.info("Allocating empty Qwen3.5 tensors on %s.", device)
    model.to_empty(device=device)
    logger.info(
        "Constructed Qwen3.5 model skeleton: quantized_placeholders=%s target_device=%s",
        len(quantized_specs) + direct_int4_placeholders,
        device,
    )
    return model, len(quantized_specs) + direct_int4_placeholders


def load_qwen3_5_text_model_weights(model: Qwen3_5TextForConditionalGeneration, model_dir: str | Path) -> WeightLoadReport:
    model_path = Path(model_dir)
    if has_gguf_model(model_path):
        from anna.weights.gguf_support import load_qwen3_5_text_model_weights_from_gguf

        return load_qwen3_5_text_model_weights_from_gguf(model, model_path)
    tensor_targets = {name: tensor for name, tensor in model.named_parameters()}
    tensor_targets.update({name: tensor for name, tensor in model.named_buffers()})
    module_targets = dict(model.named_modules())
    loaded = 0
    skipped = 0
    load_plan, total_bytes = _load_plan(model_path)
    total_shards = len(load_plan)
    loaded_bytes = 0
    st_device = safetensors_pt_device_str(tensor_targets)

    logger.info(
        "Loading Qwen3.5 weights from %s shard(s), total=%s bytes, model_dir=%s, safetensors_device=%s",
        total_shards,
        total_bytes,
        model_path,
        st_device,
    )

    for shard_idx, shard_plan in enumerate(load_plan, start=1):
        weight_file = shard_plan.path
        shard_size = shard_plan.size_bytes
        tensor_entries = {entry.name: entry for entry in shard_plan.tensors}
        logger.info(
            "Loading Qwen3.5 weight shard %s/%s: %s (%s bytes)",
            shard_idx,
            total_shards,
            weight_file.name,
            shard_size,
        )
        with safe_open(str(weight_file), framework="pt", device=st_device) as handle:
            direct_int4_requests = []
            direct_int4_keys: set[str] = set()
            for key in shard_plan.keys:
                if key in tensor_targets or not key.endswith(".weight"):
                    continue
                module_name = key[: -len(".weight")]
                module = module_targets.get(module_name)
                if not isinstance(module, XPUInt4Linear):
                    continue
                tensor_entry = tensor_entries[key]
                if tuple(tensor_entry.shape) != (module.out_features, module.in_features):
                    raise ValueError(
                        f"Shape mismatch for {key}: expected {(module.out_features, module.in_features)}, got {tuple(tensor_entry.shape)}"
                    )
                direct_int4_requests.append(
                    (module_name, tensor_entry, module.group_size, module.padded_in_features)
                )
                direct_int4_keys.add(key)
            direct_int4_payloads = {}
            if direct_int4_requests:
                for (
                    module_name,
                    qweight_bytes,
                    qscale_bytes,
                    qzeros_bytes,
                    out_features,
                    padded_in_features,
                    group_size,
                ) in quantize_safetensors_linear_int4_batch(
                    shard_path=weight_file,
                    header_len=shard_plan.header_len,
                    requests=direct_int4_requests,
                ):
                    direct_int4_payloads[str(module_name)] = (
                        qweight_bytes,
                        qscale_bytes,
                        qzeros_bytes,
                        int(out_features),
                        int(padded_in_features),
                        int(group_size),
                    )

            for key in shard_plan.keys:
                if key not in tensor_targets:
                    if key.endswith(".weight"):
                        module_name = key[: -len(".weight")]
                        module = module_targets.get(module_name)
                        if isinstance(module, XPUInt4Linear) and key in direct_int4_keys:
                            qweight_bytes, qscale_bytes, qzeros_bytes, out_features, padded_in_features, group_size = (
                                direct_int4_payloads[module_name]
                            )
                            qweight = torch.frombuffer(bytearray(qweight_bytes), dtype=torch.int32).reshape(
                                out_features, padded_in_features // 8
                            )
                            qscale = torch.frombuffer(bytearray(qscale_bytes), dtype=torch.float32).reshape(
                                padded_in_features // group_size, out_features
                            )
                            qzeros = torch.frombuffer(bytearray(qzeros_bytes), dtype=torch.int8).reshape(
                                padded_in_features // group_size, out_features
                            )
                            with torch.no_grad():
                                module.qweight.copy_(qweight.to(device=module.qweight.device))
                                module.qscale.copy_(qscale.to(device=module.qscale.device))
                                module.qzeros.copy_(qzeros.to(device=module.qzeros.device))
                            del qweight, qscale, qzeros
                            loaded += 1
                            continue
                    skipped += 1
                    continue

                source = handle.get_tensor(key)
                target = tensor_targets[key]
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"Shape mismatch for {key}: expected {tuple(target.shape)}, got {tuple(source.shape)}"
                    )

                with torch.no_grad():
                    if source.device == target.device and source.dtype == target.dtype:
                        target.copy_(source)
                    else:
                        target.copy_(source.to(device=target.device, dtype=target.dtype))
                del source
                loaded += 1
        loaded_bytes += shard_size
        _release_cpu_memory_caches()
        logger.info(
            "Loaded Qwen3.5 weight shard %s/%s: %s (cumulative_bytes=%s/%s, tensors_loaded=%s, tensors_skipped=%s)",
            shard_idx,
            total_shards,
            weight_file.name,
            loaded_bytes,
            total_bytes,
            loaded,
            skipped,
        )

    model.tie_weights()
    del tensor_targets
    del module_targets
    _release_cpu_memory_caches()
    return WeightLoadReport(loaded=loaded, skipped=skipped)
