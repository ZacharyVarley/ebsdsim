"""GPU test fixtures."""

import sys

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Allow GPU tests two reruns on macOS.

    A freshly created Metal device is *usually* clean but not always —
    the same pre-v30 wgpu-native corruption occasionally zeroes reads even
    on a new device. Reruns re-draw a fresh device via the fixture below,
    making a corrupt draw on every attempt vanishingly unlikely.
    """
    if sys.platform != "darwin":
        return
    flaky = pytest.mark.flaky(reruns=2)
    for item in items:
        item.add_marker(flaky)


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
