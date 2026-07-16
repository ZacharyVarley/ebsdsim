"""Find Smith rank where row 146 goes non-finite."""

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
    n_strong, n_weak = max(1, prescan.n_strong), max(0, prescan.n_weak)
    mu = compute_mu_eff(0.05, lookup.mlambda, lookup.diag_imag, "bloch")
    kv = kvecs[row * 3 : (row + 1) * 3]
    print(f"row={row} k={kv} n_strong={n_strong}")

    for rank in [1, 2, 3, 4, 5, 8, 10, 15, 20]:
        ws = kernels.create_workspace(
            FixedRankChunkDescriptor(1, lookup.hkl.size // 3, n_strong, n_weak, rank, lookup.table_size, sgh.n_sites)
        )
        kbuf = kernels.create_k_chunk_buffer(kv)
        try:
            kernels.run_fixed_rank_bethe_chunk(
                kbuf,
                metric,
                persistent,
                ws,
                batch_count=1,
                n_g=lookup.hkl.size // 3,
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
                mu_shift=mu,
                amplitude=1.0,
                n_sites=sgh.n_sites,
            )
            sync_device(ctx.device)
            val = float(ws.intensities.read_as(np.float32, size=4)[0])
            ok = np.isfinite(val)
            print(f"  rank={rank:2d} intensity={val:.6e} finite={ok}")
        finally:
            kbuf.destroy()


if __name__ == "__main__":
    main()
