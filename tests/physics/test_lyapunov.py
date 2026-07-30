"""Unit tests for batched CPU Lyapunov solvers."""

from __future__ import annotations

import numpy as np
from ebsdsim.physics.lyapunov import (
    build_excitation_e0,
    hermitian_sqrt_factor,
    solve_lyapunov_eig_batched,
)


def test_lyapunov_eig_recovers_known_solution():
    n = 4
    a = np.zeros((n, n), dtype=np.complex128)
    a[np.arange(n), np.arange(n)] = -np.arange(1, n + 1, dtype=np.float64)
    e0 = np.zeros(n, dtype=np.complex128)
    e0[0] = 1.0
    x = solve_lyapunov_eig_batched(a[None], e0[None])[0]
    residual = a @ x + x @ a.conj().T + np.outer(e0, e0.conj())
    assert np.linalg.norm(residual, ord="fro") < 1e-8 * np.linalg.norm(x, ord="fro")
    w = hermitian_sqrt_factor(x[None])[0]
    approx = w @ w.conj().T
    assert np.linalg.norm(approx - x, ord="fro") < 1e-8 * np.linalg.norm(x, ord="fro")


def test_build_excitation_e0_batch():
    idx = np.array([[1, 0, 2], [3, 4, 0]], dtype=np.uint32)
    e0 = build_excitation_e0(idx)
    assert e0[0, 1] == 1.0
    assert e0[1, 2] == 1.0
    assert np.count_nonzero(e0) == 2
