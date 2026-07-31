"""GPU test fixtures."""

import sys

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Allow GPU tests two reruns on macOS.

    An aborted overlong Metal command buffer can leave the cached device
    unusable for the rest of the process. Reruns re-draw a fresh device via
    the fixture below so a later attempt can proceed.
    """
    if sys.platform != "darwin":
        return
    flaky = pytest.mark.flaky(reruns=2)
    for item in items:
        item.add_marker(flaky)


@pytest.fixture(autouse=True)
def _fresh_device_on_metal() -> None:
    """Force a fresh WebGPU device before each test on macOS.

    An aborted Metal command buffer can leave the shared cached device
    unusable. Starting each test on a fresh device isolates that failure
    mode from later tests; other platforms keep the shared cached device.
    """
    if sys.platform == "darwin":
        from ebsdsim.gpu.device import get_device

        try:
            get_device(force=True, required_features=("shader-f16",))
        except RuntimeError:
            pass  # no adapter; tests will skip via require_gpu
