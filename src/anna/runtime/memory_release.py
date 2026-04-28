from __future__ import annotations

import gc
import logging
import sys

import torch

logger = logging.getLogger(__name__)


def release_cpu_memory_caches() -> None:
    for _ in range(2):
        gc.collect()
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            logger.debug("Failed to trim libc malloc arena.", exc_info=True)
    elif sys.platform == "win32":
        try:
            import ctypes

            try:
                ctypes.CDLL("msvcrt")._heapmin()
            except Exception:
                logger.debug("Failed to minimize CRT heap.", exc_info=True)
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            logger.debug("Failed to trim Windows working set.", exc_info=True)


def release_conversion_artifacts(device: torch.device) -> None:
    release_cpu_memory_caches()
    if device.type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    release_cpu_memory_caches()
