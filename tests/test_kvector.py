"""K-vector convention tests (must not use dsm.T)."""

from __future__ import annotations

import math

import numpy as np

from ebsdsim.kgrid import build_lambert_k_grid, transform_k_grid_to_reciprocal


def test_cubic_k_scaling():
    a = 4.0
    mlambda = 0.025
    dsm = np.array([a, 0, 0, 0, a, 0, 0, 0, a], dtype=np.float64)
    grid = build_lambert_k_grid(2, fundamental_only=False)
    k = transform_k_grid_to_reciprocal(grid, dsm, mlambda)
    scale = a / mlambda
    for i in range(0, grid.khat.size, 3):
        assert abs(k[i] - grid.khat[i] * scale) < 1e-3
        assert abs(k[i + 1] - grid.khat[i + 1] * scale) < 1e-3
        assert abs(k[i + 2] - grid.khat[i + 2] * scale) < 1e-3


def test_hexagonal_differs_from_transpose_bug():
    """Non-cubic cell: khat @ dsm must differ from khat @ dsm.T."""
    a = 3.0
    c = 5.0
    gamma = math.radians(120.0)
    dsm = np.array(
        [
            a,
            0.0,
            0.0,
            a * math.cos(gamma),
            a * math.sin(gamma),
            0.0,
            0.0,
            0.0,
            c,
        ],
        dtype=np.float64,
    )
    khat = np.array([0.1, 0.2, 0.97], dtype=np.float64)
    khat /= np.linalg.norm(khat)
    mlambda = 0.02
    k_correct = (khat @ dsm.reshape(3, 3)) / mlambda
    k_wrong = (khat @ dsm.reshape(3, 3).T) / mlambda
    assert not np.allclose(k_correct, k_wrong, atol=1e-6)

    grid = build_lambert_k_grid(1, fundamental_only=False)
    k_gpu = transform_k_grid_to_reciprocal(grid, dsm, mlambda)
    # First direction should match row-vector @ dsm convention.
    expected = (grid.khat[0:3] @ dsm.reshape(3, 3)) / mlambda
    assert np.allclose(k_gpu[0:3], expected, atol=1e-5)
    wrong = (grid.khat[0:3] @ dsm.reshape(3, 3).T) / mlambda
    assert not np.allclose(k_gpu[0:3], wrong, atol=1e-5)
