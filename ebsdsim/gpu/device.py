"""WebGPU adapter/device singleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import wgpu


@dataclass
class GpuContext:
    adapter: Any
    device: Any
    queue: Any


_context: GpuContext | None = None


def get_device(*, force: bool = False) -> GpuContext:
    """Return the shared GPU adapter/device/queue, creating it on first use."""
    global _context
    if _context is not None and not force:
        return _context
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No WebGPU adapter available")
    device = adapter.request_device_sync()
    _context = GpuContext(adapter=adapter, device=device, queue=device.queue)
    return _context


def require_gpu() -> GpuContext:
    """Return the shared GPU context or raise if WebGPU is unavailable."""
    return get_device()


def sync_device(device: Any) -> None:
    """Block until submitted GPU work on *device* has finished."""
    queue = getattr(device, "queue", None)
    if queue is not None:
        done = getattr(queue, "on_submitted_work_done_sync", None)
        if callable(done):
            done()
            return
    poll_wait = getattr(device, "_poll_wait", None)
    if callable(poll_wait):
        poll_wait()
        return
    poll_fn = getattr(device, "poll", None)
    if callable(poll_fn):
        while poll_fn():
            pass
