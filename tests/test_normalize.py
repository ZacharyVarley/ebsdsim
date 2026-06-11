"""Display scaling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ebsdsim.normalize import scale_fs_channel


def test_scale_fs_channel_minmax():
    vals = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    out = scale_fs_channel(vals, "minmax")
    assert np.allclose(out, [0.0, 0.5, 1.0])


def test_scale_fs_channel_robust_percentiles():
    vals = np.linspace(0.0, 100.0, 101, dtype=np.float32)
    out = scale_fs_channel(vals, "robust", robust_p_low=0.1, robust_p_high=0.9)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6
    assert out[10] == pytest.approx(0.0, abs=1e-5)
    assert out[90] == pytest.approx(1.0, abs=1e-5)


def test_scale_fs_channel_raw():
    vals = np.array([1.5, 2.5], dtype=np.float32)
    out = scale_fs_channel(vals, None)
    assert np.allclose(out, vals)
