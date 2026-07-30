"""GPU layer: device workgroup / buffer limit helpers."""

from __future__ import annotations

from typing import Any


def device_limit(device_or_limits: Any, name: str, default: int | float) -> int | float:
    """Read a WebGPU limit from a device or limits mapping."""
    if isinstance(device_or_limits, dict):
        limits = device_or_limits
    elif hasattr(device_or_limits, "limits"):
        limits = device_or_limits.limits
    else:
        limits = device_or_limits

    snake = name
    kebab = name.replace("_", "-")
    if isinstance(limits, dict):
        for key in (snake, kebab):
            if key in limits:
                return limits[key]
    for key in (snake, kebab):
        if hasattr(limits, key):
            return getattr(limits, key)
        try:
            return limits[key]
        except (KeyError, TypeError):
            pass
    return default


def max_compute_workgroups_per_dimension(device: Any) -> int:
    return int(device_limit(device, "max_compute_workgroups_per_dimension", 65535))
