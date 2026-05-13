from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SafetensorsTensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int


@dataclass(frozen=True, slots=True)
class SafetensorsShardPlan:
    path: Path
    size_bytes: int
    header_len: int
    tensors: tuple[SafetensorsTensorEntry, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.tensors)


def _rust_module():
    from anna import _rust

    return _rust


def inspect_safetensors_manifest(model_dir: str | Path) -> tuple[list[Path], int]:
    """Return safetensors shard paths and total bytes via the required Rust extension."""

    _rust = _rust_module()
    files, total_bytes = _rust.inspect_safetensors_manifest(str(Path(model_dir)))
    return [Path(path) for path in files], int(total_bytes)


def inspect_safetensors_load_plan(model_dir: str | Path) -> tuple[list[SafetensorsShardPlan], int]:
    """Return a Rust-built per-shard tensor loading plan."""

    _rust = _rust_module()
    raw_plans, total_bytes = _rust.inspect_safetensors_load_plan(str(Path(model_dir)))
    plans = [
        SafetensorsShardPlan(
            path=Path(path),
            size_bytes=int(size_bytes),
            header_len=int(header_len),
            tensors=tuple(
                SafetensorsTensorEntry(
                    name=str(name),
                    dtype=str(dtype),
                    shape=tuple(int(dim) for dim in shape),
                    data_start=int(data_start),
                    data_end=int(data_end),
                )
                for name, dtype, shape, data_start, data_end in entries
            ),
        )
        for path, size_bytes, header_len, entries in raw_plans
    ]
    return plans, int(total_bytes)


def quantize_safetensors_linear_int4(
    *,
    shard_path: str | Path,
    header_len: int,
    entry: SafetensorsTensorEntry,
    group_size: int,
    padded_in_features: int,
):
    """Quantize a safetensors dense linear tensor through the required Rust extension."""

    _rust = _rust_module()
    if len(entry.shape) != 2:
        raise ValueError(f"Expected a 2D linear weight tensor, got shape={entry.shape}")
    out_features, in_features = entry.shape
    return _rust.quantize_safetensors_linear_int4(
        str(Path(shard_path)),
        int(header_len),
        int(entry.data_start),
        int(entry.data_end),
        entry.dtype,
        int(out_features),
        int(in_features),
        int(group_size),
        int(padded_in_features),
    )


def quantize_safetensors_linear_int4_batch(
    *,
    shard_path: str | Path,
    header_len: int,
    requests: list[tuple[str, SafetensorsTensorEntry, int, int]],
):
    """Batch-quantize linear tensors from one safetensors shard via Rust."""

    _rust = _rust_module()
    specs = []
    for module_name, entry, group_size, padded_in_features in requests:
        if len(entry.shape) != 2:
            raise ValueError(f"Expected a 2D linear weight tensor for {module_name}, got shape={entry.shape}")
        out_features, in_features = entry.shape
        specs.append(
            (
                str(module_name),
                int(entry.data_start),
                int(entry.data_end),
                entry.dtype,
                int(out_features),
                int(in_features),
                int(group_size),
                int(padded_in_features),
            )
        )
    return _rust.quantize_safetensors_linear_int4_batch(str(Path(shard_path)), int(header_len), specs)
