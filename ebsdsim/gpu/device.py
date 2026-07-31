"""GPU layer: WebGPU adapter/device singleton."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

import wgpu


@dataclass
class GpuContext:
    adapter: Any
    device: Any
    queue: Any


def _patch_darwin_poller(device: Any) -> None:
    """Work around the wgpu-hal Metal blocking-poll deadlock (gfx-rs/wgpu#9531).

    wgpu-py's poll thread waits with ``wgpuDevicePoll(block=True)``, whose
    Metal implementation spin-polls ``MTLCommandBuffer.status()`` and can fail
    to ever observe completion for command buffers longer than ~hundreds of
    ms — the GPU work finishes but the wait never returns, so map/sync
    promises hang forever. Non-blocking polls use the reliable fence-value
    path, so on macOS we make the poll thread always poll non-blocking (with
    a short sleep in place of the block). No-op if wgpu-py's internals differ.
    """
    if sys.platform != "darwin":
        return
    try:
        poller = device._poller
        orig = poller._poll_func
    except AttributeError:
        return

    def pumped_poll(block: bool) -> None:
        orig(False)
        if block:
            time.sleep(0.002)

    poller._poll_func = pumped_poll


_context: GpuContext | None = None
_context_features: frozenset[str] | None = None


def get_device(
    *,
    force: bool = False,
    required_features: Iterable[str] | None = None,
) -> GpuContext:
    """Return the shared GPU adapter/device/queue, creating it on first use.

    When ``required_features`` is set, the device is requested with those
    WebGPU features (e.g. ``shader-f16`` for the Smith iterative path). A cached
    device without the features is replaced if ``force`` or if the cache was
    created with a different feature set.
    """
    global _context, _context_features
    want = frozenset(required_features or ())
    if _context is not None and not force:
        if want <= (_context_features or frozenset()):
            return _context
        # Cached device lacks newly required features — reopen.
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No WebGPU adapter available")
    if want:
        missing = [f for f in want if f not in adapter.features]
        if missing:
            raise RuntimeError(
                f"WebGPU adapter missing required features: {missing}. "
                "The default Smith iterative solver needs shader-f16; "
                "pass solver='lu_smith' to use the dense LU–Smith path."
            )
        device = adapter.request_device_sync(required_features=list(want))
    else:
        device = adapter.request_device_sync()
    _patch_darwin_poller(device)
    _context = GpuContext(adapter=adapter, device=device, queue=device.queue)
    _context_features = want
    return _context


def require_gpu(*, required_features: Iterable[str] | None = None) -> GpuContext:
    """Return the shared GPU context or raise if WebGPU is unavailable."""
    return get_device(required_features=required_features)


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
        # Fallback for backends lacking on_submitted_work_done_sync: CPU spin-poll.
        while poll_fn():
            pass
