"""Exact CPU Lyapunov solvers for the dynamical strong-beam system."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def c64_flat_to_complex(flat: NDArray[np.floating]) -> NDArray[np.complex128]:
    pairs = np.asarray(flat, dtype=np.float32).reshape(-1, 2)
    return pairs[:, 0].astype(np.float64) + 1j * pairs[:, 1].astype(np.float64)


def read_c64_buffer(buf: Any, n_values: int) -> NDArray[np.complex128]:
    """Read ``n_values`` interleaved float32 complex pairs from a GPU buffer.

    ``StorageBuffer.read_as`` expects a **byte** length; each complex64 is 8 bytes.
    """
    return c64_flat_to_complex(buf.read_as(np.float32, size=8 * int(n_values)))


def complex_to_c64_flat(z: NDArray[np.complexfloating]) -> NDArray[np.float32]:
    z = np.asarray(z)
    out = np.empty(z.size * 2, dtype=np.float32)
    out[0::2] = z.real.astype(np.float32, copy=False)
    out[1::2] = z.imag.astype(np.float32, copy=False)
    return out


def assemble_geff_matrix(
    v_aa: NDArray[np.complex128],
    d_a: NDArray[np.complex128],
    sigma: NDArray[np.complex128] | None,
) -> NDArray[np.complex128]:
    """Build the strong-beam matrix G from Bethe sub-blocks (batched)."""
    g = np.array(v_aa, dtype=np.complex128, copy=True)
    n = g.shape[-1]
    di = np.arange(n)
    g[:, di, di] += d_a
    if sigma is not None:
        g -= sigma
    return g


def build_excitation_e0(idx_a: NDArray[np.integer]) -> NDArray[np.complex128]:
    """Unit excitation on the direct (000) beam index per batch row."""
    idx = np.asarray(idx_a, dtype=np.int64)
    batch, n = idx.shape
    e0 = np.zeros((batch, n), dtype=np.complex128)
    inc = np.argmax(idx == 0, axis=1)
    e0[np.arange(batch), inc] = 1.0
    return e0


def solve_lyapunov_eig_batched(
    a: NDArray[np.complex128],
    e0: NDArray[np.complex128],
    mu_shift: float = 0.0,
    denom_eps: float = 1e-12,
) -> NDArray[np.complex128]:
    """Solve ``A X + X A^H + e0 e0^H = 0`` via batched ``numpy.linalg.eig``.

    Parameters
    ----------
    a :
        Strong-beam matrices, shape ``(..., M, M)``.
    e0 :
        Excitation vectors, shape ``(..., M)``.
    mu_shift :
        Absorption shift applied as ``A - (mu_shift / 2) I`` before the solve.
    """
    a = np.asarray(a, dtype=np.complex128)
    e0 = np.asarray(e0, dtype=np.complex128)
    n = a.shape[-1]
    eye = np.eye(n, dtype=np.complex128)
    a_shift = a - 0.5 * float(mu_shift) * eye
    q = np.einsum("...i,...j->...ij", e0, e0.conj())
    vals, v = np.linalg.eig(a_shift)
    vinv = np.linalg.inv(v)
    qtilde = vinv @ q @ np.conj(np.swapaxes(vinv, -1, -2))
    denom = vals[..., :, np.newaxis] + np.conj(vals[..., np.newaxis, :])
    safe = np.where(np.abs(denom) > denom_eps, denom, denom_eps + 0j)
    p = -qtilde / safe
    x = v @ p @ np.conj(np.swapaxes(v, -1, -2))
    return 0.5 * (x + x.conj().swapaxes(-1, -2))


def hermitian_sqrt_factor(x: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return ``W`` with ``X ≈ W @ W^H`` using a batched Hermitian eigendecomposition."""
    evals, u = np.linalg.eigh(x)
    evals = np.clip(evals.real, 0.0, None)
    return u * np.sqrt(evals)[..., np.newaxis, :]
