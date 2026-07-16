"""Compare GPU LU solve vs NumPy for one k-direction."""

from __future__ import annotations

import importlib.resources

import numpy as np

from ebsdsim.gpu import EBSDDynamicalKernels, require_gpu
from ebsdsim.gpu.device import sync_device
from ebsdsim.gpu.dynamical import FixedRankChunkDescriptor
from ebsdsim.integrate import compute_mu_eff
from ebsdsim.kgrid import build_pg_k_grid, transform_pg_k_grid_to_reciprocal
from ebsdsim.lookup import BuildLookupOptions, build_diff_lookup
from ebsdsim.sgh import prepare_site_sgh_tables
from ebsdsim.structure import build_cell_from_cif_path, metric_to_float32


def c64_to_complex(flat: np.ndarray) -> np.ndarray:
    pairs = flat.reshape(-1, 2)
    return pairs[:, 0] + 1j * pairs[:, 1]


def complex_to_c64(z: np.ndarray) -> np.ndarray:
    out = np.zeros(z.size * 2, dtype=np.float32)
    out[0::2] = z.real.astype(np.float32)
    out[1::2] = z.imag.astype(np.float32)
    return out


def main() -> None:
    row = 146
    ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    cell = build_cell_from_cif_path(ni)
    ctx = require_gpu()
    lookup = build_diff_lookup(cell, BuildLookupOptions(voltage_kv=19.5, dmin=0.05))
    pg = build_pg_k_grid(cell.pg_num, 250)
    sgh = prepare_site_sgh_tables(cell, 0.05)
    kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
    metric = kernels.create_metric_buffer(metric_to_float32(cell))
    kvecs = transform_pg_k_grid_to_reciprocal(pg, cell.direct_structure_matrix, lookup.mlambda)
    persistent = kernels.create_persistent_buffers(
        hkl=lookup.hkl,
        hkl_hash=lookup.hkl_hash,
        diff_table=lookup.diff_table,
        coupling=lookup.coupling,
        reflection_dbdiff=lookup.reflection_dbdiff,
        sgh_tables=sgh.tables,
    )
    prescan = kernels.prescan_bethe_beam_counts_gpu(
        kvecs,
        persistent,
        metric,
        batch_count=kvecs.size // 3,
        n_g=lookup.hkl.size // 3,
        bethe_c_cutoff=200,
        dbdiff_sg_cutoff=1.0,
        bethe_c_strong=20,
        bethe_c_weak=40,
    )
    n = max(1, prescan.n_strong)
    n_weak = max(0, prescan.n_weak)
    mu = compute_mu_eff(0.05, lookup.mlambda, lookup.diag_imag, "bloch")
    kv = kvecs[row * 3 : (row + 1) * 3]
    ws = kernels.create_workspace(
        FixedRankChunkDescriptor(1, lookup.hkl.size // 3, n, n_weak, 20, lookup.table_size, sgh.n_sites, debug=True)
    )
    kbuf = kernels.create_k_chunk_buffer(kv)
    try:
        score_kw = dict(
            batch_count=1,
            n_g=lookup.hkl.size // 3,
            bethe_c_cutoff=200,
            dbdiff_sg_cutoff=1.0,
            bethe_c_strong=20,
            bethe_c_weak=40,
        )
        kernels.excitation_score(kbuf, persistent, metric, ws, **score_kw)
        kernels.top_k(ws, batch_count=1, n_g=lookup.hkl.size // 3, n_strong=n, n_weak=n_weak)
        kernels.lookup_submatrix(
            ws.idx_a, ws.idx_a, persistent.hkl_hash, persistent.diff_table, ws.v_aa,
            batch_count=1, n_rows=n, n_cols=n, table_size=lookup.table_size,
            offset=lookup.offset, prefactor=lookup.prefactor, zero_diagonal=True,
        )
        kernels.gather_diagonal(
            ws.sg, ws.idx_a, ws.d_a, batch_count=1, n_g=lookup.hkl.size // 3, n=n,
            mode="bloch", diag_imag=lookup.diag_imag, mlambda=lookup.mlambda,
        )
        if n_weak > 0:
            kernels.lookup_submatrix(
                ws.idx_a, ws.idx_w, persistent.hkl_hash, persistent.diff_table, ws.v_sw,
                batch_count=1, n_rows=n, n_cols=n_weak, table_size=lookup.table_size,
                offset=lookup.offset, prefactor=lookup.prefactor,
            )
            kernels.gather_diagonal(
                ws.sg, ws.idx_w, ws.d_w, batch_count=1, n_g=lookup.hkl.size // 3, n=n_weak,
                mode="bethe", diag_imag=lookup.diag_imag, mlambda=lookup.mlambda,
            )
            kernels.build_sigma_with_bethe_gemm(ws, batch_count=1, n_a=n, n_w=n_weak)
        kernels.assemble_geff_q(ws, batch_count=1, n=n, use_sigma=n_weak > 0, mu_shift=mu)
        sync_device(ctx.device)

        q_vals = ws.q_values.read_as(np.float32, size=16)
        qm = c64_to_complex(ws.q_i_minus_a.read_as(np.float32, size=n * n * 8)).reshape(n, n)
        qp = c64_to_complex(ws.q_i_plus_a.read_as(np.float32, size=n * n * 8)).reshape(n, n)
        e0 = c64_to_complex(ws.e0.read_as(np.float32, size=n * 8))
        print(f"q={q_vals[:4]}")
        print(f"||Qm||={np.linalg.norm(qm):.6e} cond~{np.linalg.cond(qm):.6e}")

        # NumPy reference solve on copy before LU overwrites qm buffer
        w_np = np.linalg.solve(qm.copy(), e0)
        scale = q_vals[3]  # sqrt(2*q)
        w_np_scaled = w_np * scale

        # GPU LU path (single factorization, two solves — matches smith loop)
        kernels.lu_kernels.lu_factor_complex64_batched(ws.q_i_minus_a, ws.pivots, 1, n)
        kernels.lu_kernels.lu_solve_complex64_batched(ws.q_i_minus_a, ws.pivots, ws.e0, ws.w_a, 1, n)
        sync_device(ctx.device)
        w_gpu = c64_to_complex(ws.w_a.read_as(np.float32, size=n * 8))
        w_gpu_scaled = w_gpu * scale

        err = np.max(np.abs(w_np - w_gpu))
        err_s = np.max(np.abs(w_np_scaled - w_gpu_scaled))
        print(f"pre-scale solve max|np-gpu|={err:.6e}")
        print(f"post-scale max|np-gpu|={err_s:.6e}")

        # Apply pack_w scale in-place on GPU buffer (iter 0)
        ws.w_a.write(complex_to_c64(w_gpu_scaled))
        rhs_np = qp @ w_np_scaled
        w1_np = np.linalg.solve(qm, rhs_np)

        kernels.gemv_smith_rhs(ws.q_i_plus_a, ws.w_a, ws.rhs, batch_count=1, n=n)
        kernels.lu_kernels.lu_solve_complex64_batched(ws.q_i_minus_a, ws.pivots, ws.rhs, ws.w_b, 1, n)
        sync_device(ctx.device)
        w1_gpu = c64_to_complex(ws.w_b.read_as(np.float32, size=n * 8))
        rhs_gpu = c64_to_complex(ws.rhs.read_as(np.float32, size=n * 8))
        print(f"smith rhs max|np-gpu|={np.max(np.abs(rhs_np - rhs_gpu)):.6e}")
        print(f"smith w1 max|np-gpu|={np.max(np.abs(w1_np - w1_gpu)):.6e}")
        print(f"|w1_np|={np.linalg.norm(w1_np):.6e} |w1_gpu|={np.linalg.norm(w1_gpu):.6e}")

        # Full Smith loop in NumPy (rank=20)
        w_cur = w_np_scaled.copy()
        for it in range(1, 20):
            w_cur = np.linalg.solve(qm, qp @ w_cur)
            if it in (1, 2, 3, 4, 5, 8, 10, 14, 19):
                print(f"  numpy iter {it:2d} |w|={np.linalg.norm(w_cur):.6e} finite={np.all(np.isfinite(w_cur))}")
    finally:
        kbuf.destroy()


if __name__ == "__main__":
    main()
