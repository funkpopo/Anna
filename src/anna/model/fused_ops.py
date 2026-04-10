from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_LOAD_LOCK = threading.Lock()
_LOADED_LIBRARIES: set[str] = set()
_LOAD_FAILURES: set[str] = set()
_DLL_DIRECTORY_HANDLES: list[object] = []
_CONFIGURED_DLL_PATHS: set[str] = set()


def _default_library_candidates() -> list[str]:
    repo_root = Path(__file__).resolve().parents[3]
    build_dir = repo_root / ".build" / "anna_gated_delta_fused"
    candidates: list[Path] = [
        build_dir / "anna_gated_delta_fused.pyd",
        build_dir / "anna_gated_delta_fused.dll",
        build_dir / "anna_gated_delta_fused.so",
    ]
    return [str(candidate) for candidate in candidates if candidate.exists()]


def _configure_windows_runtime_paths(library_path: Path) -> None:
    if os.name != "nt":
        return

    runtime_candidates = [library_path.parent]
    manifest_path = library_path.parent / "runtime_paths.txt"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            runtime_candidates.append(Path(line.strip()))

    for candidate in runtime_candidates:
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if resolved in _CONFIGURED_DLL_PATHS:
            continue
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(resolved))
            except FileNotFoundError:
                logger.debug("Skipping missing DLL directory %s", resolved)
                continue
        existing_path = os.environ.get("PATH", "")
        entries = existing_path.split(os.pathsep) if existing_path else []
        if resolved not in entries:
            os.environ["PATH"] = resolved + (os.pathsep + existing_path if existing_path else "")
        _CONFIGURED_DLL_PATHS.add(resolved)


def _gated_delta_op():
    namespace = getattr(torch.ops, "anna", None)
    if namespace is None:
        return None
    return getattr(namespace, "gated_delta_fused", None)


def _gqa_decode_op():
    namespace = getattr(torch.ops, "anna", None)
    if namespace is None:
        return None
    return getattr(namespace, "gqa_decode_fused", None)


def _paged_gqa_decode_op():
    namespace = getattr(torch.ops, "anna", None)
    if namespace is None:
        return None
    return getattr(namespace, "paged_gqa_decode_fused", None)


def _moe_router_op():
    namespace = getattr(torch.ops, "anna", None)
    if namespace is None:
        return None
    return getattr(namespace, "moe_router_fused", None)


def _rmsnorm_op():
    namespace = getattr(torch.ops, "anna", None)
    if namespace is None:
        return None
    return getattr(namespace, "rmsnorm_fused", None)


def _rmsnorm_ex_op():
    namespace = getattr(torch.ops, "anna", None)
    if namespace is None:
        return None
    return getattr(namespace, "rmsnorm_fused_ex", None)


def _qk_norm_rotary_op():
    namespace = getattr(torch.ops, "anna", None)
    if namespace is None:
        return None
    return getattr(namespace, "qk_norm_rotary_fused", None)


def _qk_norm_rotary_ex_op():
    namespace = getattr(torch.ops, "anna", None)
    if namespace is None:
        return None
    return getattr(namespace, "qk_norm_rotary_fused_ex", None)


def _causal_conv1d_op():
    namespace = getattr(torch.ops, "anna", None)
    if namespace is None:
        return None
    return getattr(namespace, "causal_conv1d_fused", None)


def gqa_decode_fused_is_available() -> bool:
    return _gqa_decode_op() is not None


def paged_gqa_decode_fused_is_available() -> bool:
    return _paged_gqa_decode_op() is not None


def moe_router_fused_is_available() -> bool:
    return _moe_router_op() is not None


def rmsnorm_fused_is_available() -> bool:
    return _rmsnorm_op() is not None


def qk_norm_rotary_fused_is_available() -> bool:
    return _qk_norm_rotary_op() is not None


def rmsnorm_fused_ex_is_available() -> bool:
    return _rmsnorm_ex_op() is not None


def gated_delta_fused_is_available() -> bool:
    return _gated_delta_op() is not None


def causal_conv1d_fused_is_available() -> bool:
    return _causal_conv1d_op() is not None


def qk_norm_rotary_fused_ex_is_available() -> bool:
    return _qk_norm_rotary_ex_op() is not None


def maybe_load_gated_delta_library(path: str | os.PathLike[str] | None = None) -> bool:
    candidates: list[str] = []
    if path is not None:
        candidates.append(os.fspath(path))
    env_candidate = os.getenv("ANNA_GATED_DELTA_OP_LIB")
    if env_candidate:
        candidates.append(env_candidate)
    candidates.extend(_default_library_candidates())
    if not candidates:
        return gated_delta_fused_is_available()

    for candidate in candidates:
        resolved_path = Path(candidate).expanduser()
        resolved = str(resolved_path)
        if resolved in _LOADED_LIBRARIES:
            return True
        if resolved in _LOAD_FAILURES:
            continue

        with _LOAD_LOCK:
            if resolved in _LOADED_LIBRARIES:
                return True
            if resolved in _LOAD_FAILURES:
                continue
            try:
                _configure_windows_runtime_paths(resolved_path)
                torch.ops.load_library(resolved)
            except Exception:
                _LOAD_FAILURES.add(resolved)
                logger.exception("Failed to load Anna fused-op library from %s", resolved)
                continue
            _LOADED_LIBRARIES.add(resolved)
            logger.info("Loaded Anna fused-op library from %s", resolved)
            return True
    return gated_delta_fused_is_available()


def run_gqa_decode_fused(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    visible_lengths: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    op = _gqa_decode_op()
    if op is None:
        maybe_load_gated_delta_library()
        op = _gqa_decode_op()
    if op is None:
        raise RuntimeError(
            "Anna gqa_decode_fused op is not registered. Build/load the custom op first, "
            "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
        )
    return op(query, key, value, visible_lengths, float(scaling))


def run_paged_gqa_decode_fused(
    *,
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    page_table: torch.Tensor,
    visible_lengths: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    op = _paged_gqa_decode_op()
    if op is None:
        maybe_load_gated_delta_library()
        op = _paged_gqa_decode_op()
    if op is None:
        raise RuntimeError(
            "Anna paged_gqa_decode_fused op is not registered. Build/load the custom op first, "
            "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
        )
    return op(query, key_pages, value_pages, page_table, visible_lengths, float(scaling))


def run_moe_router_fused(
    *,
    router_logits: torch.Tensor,
    top_k: int,
    normalize_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op = _moe_router_op()
    if op is None:
        maybe_load_gated_delta_library()
        op = _moe_router_op()
    if op is None:
        raise RuntimeError(
            "Anna moe_router_fused op is not registered. Build/load the custom op first, "
            "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
        )
    return op(router_logits, int(top_k), bool(normalize_topk_prob))


def run_rmsnorm_fused(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    op = _rmsnorm_op()
    if op is None:
        maybe_load_gated_delta_library()
        op = _rmsnorm_op()
    if op is None:
        raise RuntimeError(
            "Anna rmsnorm_fused op is not registered. Build/load the custom op first, "
            "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
        )
    return op(input, weight, float(eps))


def run_rmsnorm_fused_ex(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_unit_offset: bool,
) -> torch.Tensor:
    op = _rmsnorm_ex_op()
    if op is None:
        maybe_load_gated_delta_library()
        op = _rmsnorm_ex_op()
    if op is None:
        raise RuntimeError(
            "Anna rmsnorm_fused_ex op is not registered. Build/load the custom op first, "
            "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
        )
    return op(input, weight, float(eps), bool(add_unit_offset))


def run_qk_norm_rotary_fused(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_norm_eps: float,
    key_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    op = _qk_norm_rotary_op()
    if op is None:
        maybe_load_gated_delta_library()
        op = _qk_norm_rotary_op()
    if op is None:
        raise RuntimeError(
            "Anna qk_norm_rotary_fused op is not registered. Build/load the custom op first, "
            "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
        )
    return op(
        query,
        key,
        query_norm_weight,
        key_norm_weight,
        cos,
        sin,
        float(query_norm_eps),
        float(key_norm_eps),
    )


def run_qk_norm_rotary_fused_ex(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_norm_eps: float,
    key_norm_eps: float,
    add_unit_offset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    op = _qk_norm_rotary_ex_op()
    if op is None:
        maybe_load_gated_delta_library()
        op = _qk_norm_rotary_ex_op()
    if op is None:
        raise RuntimeError(
            "Anna qk_norm_rotary_fused_ex op is not registered. Build/load the custom op first, "
            "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
        )
    return op(
        query,
        key,
        query_norm_weight,
        key_norm_weight,
        cos,
        sin,
        float(query_norm_eps),
        float(key_norm_eps),
        bool(add_unit_offset),
    )


def run_causal_conv1d_fused(
    *,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    op = _causal_conv1d_op()
    if op is None:
        maybe_load_gated_delta_library()
        op = _causal_conv1d_op()
    if op is None:
        raise RuntimeError(
            "Anna causal_conv1d_fused op is not registered. Build/load the custom op first, "
            "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
        )
    if bias is None:
        return op(hidden_states, conv_state, weight)
    return op(hidden_states, conv_state, weight, bias)


def run_gated_delta_fused(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    state_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    op = _gated_delta_op()
    if op is None:
        maybe_load_gated_delta_library()
        op = _gated_delta_op()
    if op is None:
        raise RuntimeError(
            "Anna Gated DeltaNet fused op is not registered. Build/load the custom op first, "
            "or set ANNA_GATED_DELTA_OP_LIB to the compiled library path."
        )
    return op(
        query,
        key,
        value,
        g,
        beta,
        z,
        norm_weight,
        float(norm_eps),
        initial_state,
        bool(output_final_state),
        state_buffer,
    )
