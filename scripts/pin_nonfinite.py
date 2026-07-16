"""Pin down first non-finite source in full-grid dynamical solve."""

from __future__ import annotations

import importlib.resources

import numpy as np

from ebsdsim.gpu import EBSDDynamicalKernels, require_gpu
from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.integrate import compute_mu_eff
from ebsdsim.kgrid import build_pg_k_grid, chunk_k_vectors, transform_pg_k_grid_to_reciprocal
from ebsdsim.lookup import BuildLookupOptions, build_diff_lookup
from ebsdsim.runner import plan_safe_chunk_size
from ebsdsim.sgh import prepare_site_sgh_tables
from ebsdsim.structure import build_cell_from_cif_path


def main() -> None:
    ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    cell = build_cell_from_cif_path(ni)
    ctx = require_gpu()
    hw = 250
    rank = 20
    chunk_size = 256
    voltage_kv = 19.5
    lookup = build_diff_lookup(cell, BuildLookupOptions(voltage_kv=voltage_kv, dmin=0.05))
    pg = build_pg_k_grid(cell.pg_num, hw)
    sgh = prepare_site_sgh_tables(cell, 0.05)
    kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
    metric = kernels.create_metric_buffer(
        __import__("ebsdsim.structure", fromlist=["metric_to_float32"]).metric_to_float32(cell)
    )
    kvecs = transform_pg_k_grid_to_reciprocal(pg, cell.direct_structure_matrix, lookup.mlambda)
    n_k = kvecs.size // 3
    n_g = lookup.hkl.size // 3
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
        batch_count=n_k,
        n_g=n_g,
        bethe_c_cutoff=200,
        dbdiff_sg_cutoff=1.0,
        bethe_c_strong=20,
        bethe_c_weak=40,
    )
    n_strong = max(1, prescan.n_strong)
    n_weak = max(0, prescan.n_weak)
    print(f"n_k={n_k} n_strong={n_strong} n_weak={n_weak}")

    from ebsdsim.gpu.dynamical import FixedRankChunkDescriptor

    effective = plan_safe_chunk_size(
        chunk_size,
        ctx.device.limits,
        n_g=n_g,
        n_strong=n_strong,
        n_weak=n_weak,
        rank=rank,
        n_sites=sgh.n_sites,
    )
    ws = kernels.create_workspace(
        FixedRankChunkDescriptor(
            batch_count=effective,
            n_g=n_g,
            n_strong=n_strong,
            n_weak=n_weak,
            rank=rank,
            table_size=lookup.table_size,
            n_sites=sgh.n_sites,
        )
    )
    mu_eff = compute_mu_eff(0.05, lookup.mlambda, lookup.diag_imag, "bloch")
    local_idx = np.arange(n_k, dtype=np.uint32)
    chunk0 = next(iter(chunk_k_vectors(kvecs, local_idx, effective)))
    rows = chunk0.kvecs.size // 3
    kbuf = kernels.create_k_chunk_buffer(chunk0.kvecs)
    try:
        kernels.run_fixed_rank_bethe_chunk(
            kbuf,
            metric,
            persistent,
            ws,
            batch_count=rows,
            n_g=n_g,
            n_strong=n_strong,
            n_weak=n_weak,
            rank=rank,
            table_size=lookup.table_size,
            offset=lookup.offset,
            prefactor=lookup.prefactor,
            bethe_c_cutoff=200,
            dbdiff_sg_cutoff=1.0,
            bethe_c_strong=20,
            bethe_c_weak=40,
            mode="bloch",
            diag_imag=lookup.diag_imag,
            mlambda=lookup.mlambda,
            mu_shift=mu_eff,
            amplitude=1.0,
            n_sites=sgh.n_sites,
        )
        from ebsdsim.gpu.device import sync_device

        sync_device(ctx.device)
        ints = ws.intensities.read_as(np.float32, size=rows * sgh.n_sites * 4)
        bad = ~np.isfinite(ints)
        print(f"chunk0 rows={rows} intensities bad={int(bad.sum())}/{ints.size} max={float(np.nanmax(ints))}")
        if bad.any():
            idx = int(np.argmax(bad))
            print(f"first bad index {idx} value {ints[idx]}")
        q = ws.q_i_minus_a.read_as(np.float32, size=rows * n_strong * n_strong * 8)
        print(f"q_i_minus_a finite={np.all(np.isfinite(q))} max={float(np.max(np.abs(q)))}")
        w = ws.w_stack.read_as(np.float32, size=rows * n_strong * rank * 8)
        print(f"w_stack finite={np.all(np.isfinite(w))} max={float(np.max(np.abs(w)))}")
    finally:
        kbuf.destroy()


if __name__ == "__main__":
    main()
