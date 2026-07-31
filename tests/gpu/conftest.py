"""GPU test fixtures."""

import sys

import pytest


@pytest.fixture(autouse=True)
def _fresh_device_on_metal() -> None:
    """Force a fresh WebGPU device before each test on macOS.

    wgpu-native's Metal backend (pre-v30) accumulates corrupted completion
    state across repeated pipeline/buffer create-destroy cycles on one
    device, causing reads to intermittently return zeros. A fresh device
    starts clean; other platforms keep the shared cached device.
    """
    if sys.platform == "darwin":
        from ebsdsim.gpu.device import get_device

        try:
            get_device(force=True, required_features=("shader-f16",))
        except RuntimeError:
            pass  # no adapter; tests will skip via require_gpu
