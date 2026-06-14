"""Profile CPU work before the first voltage GPU hot loop (GaN 501 defaults)."""

from __future__ import annotations

import cProfile
import importlib.resources
import pstats
import time
from io import StringIO

import numpy as np

from ebsdsim.cif import parse_cif_crystal
from ebsdsim.gpu.device import require_gpu
from ebsdsim.gpu.dynamical import EBSDDynamicalKernels, FixedRankChunkDescriptor
from ebsdsim.integrate import next_active_voltage_kv, surrogate_to_multi_voltage_mc, trim_multi_voltage_mc_by_coverage
from ebsdsim.kgrid import build_pg_k_grid, transform_pg_k_grid_to_reciprocal
from ebsdsim.lookup import (
    BuildLookupOptions,
    build_diff_lookup_from_geometry,
    prepare_diff_lookup_geometry,
)
from ebsdsim.runner import RunOneVoltageDeps, make_metric_buffer, plan_safe_chunk_size
from ebsdsim.sgh import prepare_site_sgh_tables
from ebsdsim.structure import build_cell_from_cif
from ebsdsim.surrogate import infer_direct_exp_from_cell_rebinned


def _timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"  {label}: {time.perf_counter() - t0:.3f}s")
    return out


def main() -> None:
    gan = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif")
    cell = build_cell_from_cif(parse_cif_crystal(gan.read_text(encoding="utf-8")))
    dmin = 0.05
    mode = "bloch"
    hw = 250
    rank = 20
    chunk_size = 256

    print("GaN startup profile (dmin=0.05, halfw=250)")

    direct = _timed(
        "surrogate MC",
        lambda: infer_direct_exp_from_cell_rebinned(
            cell=cell, sigma_deg=70.0, beam_kv=20.0, energy_binwidth_keV=1.0, n_energy_bins=20
        ),
    )
    mc = surrogate_to_multi_voltage_mc(direct, 20.0)
    mc_for_bins = trim_multi_voltage_mc_by_coverage(mc, 1.0)
    first_vkv = next_active_voltage_kv(mc_for_bins, -1)
    assert first_vkv is not None

    pg_grid = _timed("build_pg_k_grid", lambda: build_pg_k_grid(cell.pg_num, hw))
    sgh = _timed("prepare_site_sgh_tables", lambda: prepare_site_sgh_tables(cell, dmin))
    lookup_geometry = _timed(
        "prepare_diff_lookup_geometry", lambda: prepare_diff_lookup_geometry(cell, dmin)
    )

    lookup = _timed(
        "build_diff_lookup_from_geometry (first bin)",
        lambda: build_diff_lookup_from_geometry(
            lookup_geometry, BuildLookupOptions(voltage_kv=first_vkv, dmin=dmin, mode=mode)
        ),
    )

    ctx = require_gpu()
    kernels = _timed("EBSDDynamicalKernels init", lambda: EBSDDynamicalKernels(ctx.device, ctx.queue))
    metric = _timed("make_metric_buffer", lambda: make_metric_buffer(kernels, cell))

    n_k = pg_grid.khat.size // 3
    n_g = lookup.hkl.size // 3
    kvecs = _timed(
        "transform_pg_k_grid_to_reciprocal",
        lambda: transform_pg_k_grid_to_reciprocal(pg_grid, cell.direct_structure_matrix, lookup.mlambda),
    )

    from ebsdsim.runner import ReusablePersistent

    persistent = _timed(
        "create_persistent_buffers",
        lambda: ReusablePersistent.create(kernels, lookup, sgh.tables),
    )

    prescan = _timed(
        "prescan_bethe_beam_counts_gpu",
        lambda: kernels.prescan_bethe_beam_counts_gpu(
            kvecs,
            persistent.buffers,
            metric,
            batch_count=n_k,
            n_g=n_g,
        ),
    )

    n_strong = max(1, prescan.n_strong)
    n_weak = max(0, prescan.n_weak)
    effective_chunk = plan_safe_chunk_size(
        chunk_size, kernels.device.limits, n_g=n_g, n_strong=n_strong, n_weak=n_weak, rank=rank, n_sites=sgh.n_sites
    )
    _timed(
        "create_workspace",
        lambda: kernels.create_workspace(
            FixedRankChunkDescriptor(
                batch_count=effective_chunk,
                n_g=n_g,
                n_strong=n_strong,
                n_weak=n_weak,
                rank=rank,
                table_size=lookup.table_size,
                n_sites=sgh.n_sites,
            )
        ),
    )

    print("\nDetailed cProfile: prepare_diff_lookup_geometry + first lookup build")
    pr = cProfile.Profile()
    pr.enable()
    geom = prepare_diff_lookup_geometry(cell, dmin)
    build_diff_lookup_from_geometry(geom, BuildLookupOptions(voltage_kv=first_vkv, dmin=dmin, mode=mode))
    pr.disable()
    stream = StringIO()
    stats = pstats.Stats(pr, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(40)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
