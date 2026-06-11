"""Compare batch vs single-row dynamical solves for chunk 0."""

from __future__ import annotations

import importlib.resources

import numpy as np

from ebsdsim.cif import parse_cif_crystal
from ebsdsim.gpu import EBSDDynamicalKernels, require_gpu
from ebsdsim.gpu.device import sync_device
from ebsdsim.integrate import compute_mu_eff
from ebsdsim.kgrid import build_pg_k_grid, chunk_k_vectors, transform_pg_k_grid_to_reciprocal
from ebsdsim.lookup import BuildLookupOptions, build_diff_lookup
from ebsdsim.runner import plan_safe_chunk_size
from ebsdsim.sgh import prepare_site_sgh_tables
from ebsdsim.structure import build_cell_from_cif, metric_to_float32


def _setup():
    ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    cell = build_cell_from_cif(parse_cif_crystal(ni.read_text(encoding="utf-8")))
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
        kvecs, persistent, metric, batch_count=kvecs.size // 3, n_g=lookup.hkl.size // 3,
        bethe_c_cutoff=200, dbdiff_sg_cutoff=1.0, bethe_c_strong=20, bethe_c_weak=40,
    )
    n_strong, n_weak = max(1, prescan.n_strong), max(0, prescan.n_weak)
    from ebsdsim.gpu.dynamical import FixedRankChunkDescriptor

    ws1 = kernels.create_workspace(
        FixedRankChunkDescriptor(1, lookup.hkl.size // 3, n_strong, n_weak, 20, lookup.table_size, sgh.n_sites)
    )
    ws256 = kernels.create_workspace(
        FixedRankChunkDescriptor(
            plan_safe_chunk_size(256, ctx.device.limits, n_g=lookup.hkl.size // 3,
                                 n_strong=n_strong, n_weak=n_weak, rank=20, n_sites=sgh.n_sites),
            lookup.hkl.size // 3, n_strong, n_weak, 20, lookup.table_size, sgh.n_sites,
        )
    )
    mu = compute_mu_eff(0.05, lookup.mlambda, lookup.diag_imag, "bloch")
    chunk0 = next(iter(chunk_k_vectors(kvecs, np.arange(kvecs.size // 3, dtype=np.uint32), 256)))
    return kernels, metric, persistent, ws1, ws256, lookup, sgh, mu, chunk0, n_strong, n_weak


def run_rows(kernels, metric, persistent, ws, lookup, sgh, mu, kvecs, n_strong, n_weak, batch_count):
    kbuf = kernels.create_k_chunk_buffer(kvecs)
    try:
        kernels.run_fixed_rank_bethe_chunk(
            kbuf, metric, persistent, ws, batch_count=batch_count,
            n_g=lookup.hkl.size // 3, n_strong=n_strong, n_weak=n_weak, rank=20,
            table_size=lookup.table_size, offset=lookup.offset, prefactor=lookup.prefactor,
            bethe_c_cutoff=200, dbdiff_sg_cutoff=1.0, bethe_c_strong=20, bethe_c_weak=40,
            mode="bloch", diag_imag=lookup.diag_imag, mlambda=lookup.mlambda, mu_shift=mu,
            amplitude=1.0, n_sites=sgh.n_sites,
        )
        sync_device(kernels.device)
        rows = kvecs.size // 3
        ints = ws.intensities.read_as(np.float32, size=rows * sgh.n_sites * 4)
        return ints, int(np.sum(~np.isfinite(ints)))
    finally:
        kbuf.destroy()


def main():
    kernels, metric, persistent, ws1, ws256, lookup, sgh, mu, chunk0, ns, nw = _setup()
    batched, bad_b = run_rows(kernels, metric, persistent, ws256, lookup, sgh, mu, chunk0.kvecs, ns, nw, chunk0.kvecs.size // 3)
    print(f"batched256 bad={bad_b}")
    bad_rows = []
    singles = []
    for i in range(chunk0.kvecs.size // 3):
        kv = chunk0.kvecs[i * 3 : (i + 1) * 3]
        ints, bad = run_rows(kernels, metric, persistent, ws1, lookup, sgh, mu, kv, ns, nw, 1)
        singles.append(float(ints[0]))
        if bad:
            bad_rows.append(i)
    print(f"single-row bad rows: {bad_rows}")
    if bad_b:
        batched_rows = batched.reshape(-1, sgh.n_sites)[:, 0]
        for i in range(len(batched_rows)):
            if not np.isfinite(batched_rows[i]):
                print(f"  batched bad row {i} single={singles[i]}")


if __name__ == "__main__":
    main()
