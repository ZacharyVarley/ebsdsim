"""LU solve + Smith loop + exact CPU Lyapunov glue (GPU dynamical layer).

Free functions take the kernels façade (device/queue/pipelines/LU) as their first argument.
"""
from __future__ import annotations

import numpy as np

from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.gpu.device import sync_device
from ebsdsim.gpu.dynamical.workspace import (
    FixedRankWorkspace,
)
from ebsdsim.physics.lyapunov import (
    assemble_geff_matrix,
    build_excitation_e0,
    complex_to_c64_flat,
    hermitian_sqrt_factor,
    read_c64_buffer,
    solve_lyapunov_eig_batched,
)


def _solve_c64(
    kernels,
    lu: StorageBuffer,
    pivots: StorageBuffer,
    rhs: StorageBuffer,
    out: StorageBuffer,
    *,
    batch_count: int,
    n: int,
) -> None:
    kernels.lu_kernels.lu_solve_complex64_batched(lu, pivots, rhs, out, batch_count, n)


def run_exact_lyapunov_cpu(
    kernels,
    ws: FixedRankWorkspace,
    *,
    batch_count: int,
    n: int,
    use_sigma: bool,
    mu_shift: float = 0.0,
) -> None:
    """Full-rank Lyapunov solve on CPU via batched ``numpy.linalg.eig``.

    Reads assembled Bethe blocks back from GPU workspace buffers. The default
    Smith path never touches ``ws.v_aa`` on the host â€” it stays on-device
    through ``assemble_geff_q`` and ``run_fixed_smith_loop``.
    """
    sync_device(kernels.device)
    v_aa = read_c64_buffer(ws.v_aa, batch_count * n * n).reshape(batch_count, n, n)
    d_a = read_c64_buffer(ws.d_a, batch_count * n).reshape(batch_count, n)
    sigma = None
    if use_sigma:
        sigma = read_c64_buffer(ws.sigma, batch_count * n * n).reshape(batch_count, n, n)
    idx_a = ws.idx_a.read_as(np.uint32, size=4 * batch_count * n).reshape(batch_count, n)
    g = assemble_geff_matrix(v_aa, d_a, sigma)
    e0 = build_excitation_e0(idx_a)
    x = solve_lyapunov_eig_batched(g, e0, mu_shift=mu_shift)
    w = hermitian_sqrt_factor(x)
    ws.w_stack.write(complex_to_c64_flat(w.reshape(batch_count * n * n)))


def run_fixed_smith_loop(
    kernels,
    ws: FixedRankWorkspace,
    *,
    batch_count: int,
    n: int,
    rank: int,
) -> None:
    kernels.lu_kernels.lu_factor_complex64_batched(ws.q_i_minus_a, ws.pivots, batch_count, n)
    kernels._solve_c64(ws.q_i_minus_a, ws.pivots, ws.e0, ws.w_a, batch_count=batch_count, n=n)
    kernels.pack_w(ws.w_a, ws, batch_count=batch_count, n=n, rank=rank, iter=0, apply_scale=True, q_offset=3)

    current = ws.w_a
    next_buf = ws.w_b
    for iter_idx in range(1, rank):
        kernels.gemv_smith_rhs(ws.q_i_plus_a, current, ws.rhs, batch_count=batch_count, n=n)
        kernels._solve_c64(ws.q_i_minus_a, ws.pivots, ws.rhs, next_buf, batch_count=batch_count, n=n)
        kernels.pack_w(
            next_buf,
            ws,
            batch_count=batch_count,
            n=n,
            rank=rank,
            iter=iter_idx,
            apply_scale=False,
            q_offset=0,
        )
        current, next_buf = next_buf, current

