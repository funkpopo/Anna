from __future__ import annotations

import logging

from anna.runtime.device import configure_xpu_environment

logger = logging.getLogger(__name__)


def add_xpu_environment_args(parser) -> None:
    parser.add_argument(
        "--xpu-device-index",
        type=int,
        default=None,
        help="Select a specific Intel XPU through ONEAPI_DEVICE_SELECTOR=level_zero:<index>.",
    )
    parser.add_argument(
        "--no-xpu-env-defaults",
        action="store_true",
        help="Do not set Anna's recommended Level Zero environment defaults before XPU startup.",
    )


def configure_cli_xpu_environment(*, device: str, xpu_device_index: int | None, no_xpu_env_defaults: bool) -> None:
    if no_xpu_env_defaults:
        return
    if device.lower() not in {"auto", "xpu"} and xpu_device_index is None:
        return
    configured = configure_xpu_environment(
        device_index=xpu_device_index,
        set_selector=xpu_device_index is not None,
    )
    logger.info("Configured XPU environment: %s", configured)
