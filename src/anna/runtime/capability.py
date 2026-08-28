"""Intel XPU device capability abstraction.

Phase 0 of the XPU-engine transition (docs/xpu-engine-transition.md): replace
device-name string matching ("a770"/"a750") with a structured capability model so
kernel strategy tables, decode policies, and quantization presets can be selected
per GPU architecture generation instead of per exact device name.

Detection is PyTorch-only: ``torch.xpu.get_device_properties`` plus the reported
device name. Unknown devices degrade gracefully to a conservative generic profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DeviceCapability:
    """Static per-generation capability profile for an Intel GPU."""

    device_index: int
    name: str
    arch_family: str  # e.g. "alchemist", "battlemage", "xe2_igpu", "pvc", "generic"
    arch_generation: str  # e.g. "xe-hpg", "xe2-hpg", "xe-lpg", "pvc"
    supports_xmx: bool  # XMX matrix engines (bf16/int8 DPAS)
    supports_dp4a: bool  # int8 dot-product on ALU pipes
    eu_count: int | None
    subgroup_sizes: tuple[int, ...]
    # Decode-strategy tier used by kernel strategy tables (ops decode defaults):
    # "high" = desktop dGPU class, "mid" = iGPU/laptop, "low" = conservative fallback.
    performance_tier: str
    notes: tuple[str, ...] = field(default=())

    def as_log_fields(self) -> dict[str, Any]:
        return {
            "arch_family": self.arch_family,
            "arch_generation": self.arch_generation,
            "supports_xmx": self.supports_xmx,
            "supports_dp4a": self.supports_dp4a,
            "eu_count": self.eu_count,
            "performance_tier": self.performance_tier,
        }


# Known-generation profiles. ``name_pattern`` is matched (case-insensitive) against
# the torch.xpu device name. Order matters: first match wins.
_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "name_pattern": re.compile(r"arc.*a770", re.IGNORECASE),
        "arch_family": "alchemist",
        "arch_generation": "xe-hpg",
        "supports_xmx": True,
        "supports_dp4a": True,
        "eu_count": 512,
        "subgroup_sizes": (8, 16, 32),
        "performance_tier": "high",
        "notes": ("acm-g10",),
    },
    {
        "name_pattern": re.compile(r"arc.*a7[35]0", re.IGNORECASE),
        "arch_family": "alchemist",
        "arch_generation": "xe-hpg",
        "supports_xmx": True,
        "supports_dp4a": True,
        "eu_count": 448,
        "subgroup_sizes": (8, 16, 32),
        "performance_tier": "high",
        "notes": ("acm-g11",),
    },
    {
        "name_pattern": re.compile(r"arc.*a(580|380|310)", re.IGNORECASE),
        "arch_family": "alchemist",
        "arch_generation": "xe-hpg",
        "supports_xmx": True,
        "supports_dp4a": True,
        "eu_count": 384,
        "subgroup_sizes": (8, 16, 32),
        "performance_tier": "mid",
    },
    {
        "name_pattern": re.compile(r"arc.*(b580|b570)", re.IGNORECASE),
        "arch_family": "battlemage",
        "arch_generation": "xe2-hpg",
        "supports_xmx": True,
        "supports_dp4a": True,
        "eu_count": 160,  # B580; B570 slightly lower but same family tier
        "subgroup_sizes": (8, 16, 32),
        "performance_tier": "high",
    },
    {
        "name_pattern": re.compile(r"arc.*pro.*a\d+", re.IGNORECASE),
        "arch_family": "alchemist-pro",
        "arch_generation": "xe-hpg",
        "supports_xmx": True,
        "supports_dp4a": True,
        "eu_count": None,
        "subgroup_sizes": (8, 16, 32),
        "performance_tier": "mid",
    },
    {
        "name_pattern": re.compile(r"data center gpu max", re.IGNORECASE),
        "arch_family": "pvc",
        "arch_generation": "pvc",
        "supports_xmx": True,
        "supports_dp4a": True,
        "eu_count": None,
        "subgroup_sizes": (16, 32),
        "performance_tier": "high",
    },
    {
        "name_pattern": re.compile(r"data center gpu flex", re.IGNORECASE),
        "arch_family": "ats",
        "arch_generation": "xe-hpc",
        "supports_xmx": True,
        "supports_dp4a": True,
        "eu_count": None,
        "subgroup_sizes": (16, 32),
        "performance_tier": "mid",
    },
    {
        # Lunar Lake / Arrow Lake iGPUs report as "Intel(R) Graphics" or similar;
        # driver strings may mention the platform. Treat as Xe2 iGPU tier.
        "name_pattern": re.compile(r"(lunar lake|arrow lake|panther lake)", re.IGNORECASE),
        "arch_family": "xe2_igpu",
        "arch_generation": "xe-lpg",
        "supports_xmx": True,
        "supports_dp4a": True,
        "eu_count": None,
        "subgroup_sizes": (8, 16),
        "performance_tier": "mid",
    },
)

_GENERIC_PROFILE = {
    "arch_family": "generic",
    "arch_generation": "unknown",
    "supports_xmx": True,  # conservative: modern drivers on Xe-class GPUs have XMX
    "supports_dp4a": True,
    "eu_count": None,
    "subgroup_sizes": (8, 16, 32),
    "performance_tier": "low",
}


def resolve_capability(device_index: int, device_name: str | None) -> DeviceCapability:
    """Resolve a :class:`DeviceCapability` for the given device.

    Unknown names fall back to a conservative generic profile rather than failing,
    so new Intel GPUs work out of the box with default strategies.
    """

    name = str(device_name or f"xpu:{device_index}")
    for profile in _PROFILES:
        if profile["name_pattern"].search(name):
            return DeviceCapability(
                device_index=device_index,
                name=name,
                arch_family=profile["arch_family"],
                arch_generation=profile["arch_generation"],
                supports_xmx=profile["supports_xmx"],
                supports_dp4a=profile["supports_dp4a"],
                eu_count=profile["eu_count"],
                subgroup_sizes=profile["subgroup_sizes"],
                performance_tier=profile["performance_tier"],
                notes=tuple(profile.get("notes", ())),
            )
    return DeviceCapability(device_index=device_index, name=name, **_GENERIC_PROFILE)


def capability_for_torch_device(device: Any) -> DeviceCapability | None:
    """Best-effort capability resolution from a ``torch.device`` (``xpu`` only)."""

    xpu = getattr(torch_module(), "xpu", None)
    if xpu is None or not getattr(xpu, "is_available", lambda: False)():
        return None
    index = 0 if device is None or device.index is None else int(device.index)
    name = None
    try:
        name = xpu.get_device_name(index)
    except Exception:
        properties = getattr(xpu, "get_device_properties", None)
        if callable(properties):
            try:
                name = getattr(properties(index), "name", None)
            except Exception:
                name = None
    return resolve_capability(index, name)


def torch_module():
    import torch

    return torch
