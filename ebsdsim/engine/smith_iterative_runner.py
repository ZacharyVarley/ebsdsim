"""Engine layer: Smith iterative voltage-integrated orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np

from ebsdsim.crystal.simcell import SimCell
from ebsdsim.energy.mc import MultiVoltageMC
from ebsdsim.engine.integrate import (
    MasterPatternIntegratedResult,
    active_voltage_bins,
    trim_multi_voltage_mc_by_coverage,
)
from ebsdsim.engine.progress import MasterPatternProgress
from ebsdsim.gpu.dynamical.smith_iterative import (
    auto_tile_k,
    load_smith_iterative_shader,
    run_bins_dynamical,
)
from ebsdsim.gpu.dynamical.smith_iterative_prep import (
    VoltageBin,
    activate_voltage_bin,
    open_smith_iterative_prep,
)
from ebsdsim.gpu.pipelines import load_wgsl


def _voltage_bins_from_mc(mc: MultiVoltageMC) -> list[VoltageBin]:
    return [
        VoltageBin(
            voltage_kv=b.voltage_kv,
            next_voltage_kv=b.next_voltage_kv,
            beta=b.beta,
            amplitude=b.amplitude,
            energy_weight=b.energy_weight,
            bin_index=b.bin_index,
        )
        for b in active_voltage_bins(mc)
    ]


def run_smith_iterative_voltage_integrated(
    *,
    cell: SimCell,
    mc: MultiVoltageMC,
    halfw: int,
    dmin: float,
    voltage_kv: float,
    bethe_c_strong: float,
    bethe_c_weak: float,
    bethe_c_cutoff: float,
    dbdiff_sg_cutoff: float,
    kernels: Any,
    progress: MasterPatternProgress | None = None,
    queue_depth: int = 4,
    marginal_coverage: float = 1.0,
    relative_image_stop: float = 0.01,
    max_bins_run: int | None = None,
    max_chunks: int | None = None,
) -> tuple[MasterPatternIntegratedResult, dict[str, Any]]:
    """Run multi-bin Smith iterative integrate; return FS intensities + dyn stats."""
    mc_trim = trim_multi_voltage_mc_by_coverage(mc, marginal_coverage)
    bins = _voltage_bins_from_mc(mc_trim)
    if max_bins_run is not None:
        bins = bins[: max(0, int(max_bins_run))]
    if not bins:
        raise RuntimeError("no active voltage bins for Smith iterative path")

    prep, bins = open_smith_iterative_prep(
        "cell",
        halfw=halfw,
        voltage_kv=voltage_kv,
        dmin=dmin,
        bethe_c_strong=bethe_c_strong,
        bethe_c_weak=bethe_c_weak,
        bethe_c_cutoff=bethe_c_cutoff,
        dbdiff_sg_cutoff=dbdiff_sg_cutoff,
        cell=cell,
        reuse_kernels=kernels,
        bins=bins,
    )

    code_topk = load_wgsl("dynamical/topk_radix_exact.wgsl")
    code_slim = load_wgsl("dynamical/smith_iterative_slim_q.wgsl")
    code_inten = load_wgsl("dynamical/intensity_fused_exact.wgsl")
    first = activate_voltage_bin(prep, bins[0])
    code_smith_iterative, smith_iterative_mode = load_smith_iterative_shader(int(first["n_strong"]))
    tile = auto_tile_k(
        int(first["n_g"]), int(first["n_strong"]), int(first["n_weak"]), prep.n_sites
    )
    if progress is not None and progress.enabled:
        print(
            f"[ebsdsim]   Smith iterative BiCGSTAB  mode={smith_iterative_mode}  "
            f"k_tile={tile}  bins={len(bins)}",
            flush=True,
        )

    pg_grid = prep.pg
    try:
        dyn = run_bins_dynamical(
            prep,
            bins,
            chunk=tile,
            code_topk=code_topk,
            code_slim=code_slim,
            code_smith_iterative=code_smith_iterative,
            code_inten=code_inten,
            first_act=first,
            queue_depth=queue_depth,
            relative_image_stop=relative_image_stop,
            max_chunks=max_chunks,
        )
    finally:
        prep.close()

    intensities = np.asarray(dyn["intensities"], dtype=np.float32)
    n_k, n_sites = int(intensities.shape[0]), int(intensities.shape[1])
    bins_run = int(dyn.get("n_bins", len(bins)))
    bin_stack = np.asarray(dyn["bin_intensities"], dtype=np.float32)
    if bin_stack.ndim != 3 or bin_stack.shape != (bins_run, n_k, n_sites):
        raise RuntimeError(
            f"smith_iterative bin_intensities shape {bin_stack.shape} != "
            f"({bins_run}, {n_k}, {n_sites})"
        )
    bin_patterns = list(np.ascontiguousarray(bin_stack.reshape(bins_run, -1)))
    result = MasterPatternIntegratedResult(
        integrated=intensities.reshape(-1),
        n_k=n_k,
        n_sites=n_sites,
        bin_patterns=bin_patterns,
        n_bins_run=bins_run,
        stopped_by_relative_change=bool(dyn.get("stopped_by_relative_change", False)),
        last_relative_change=float(dyn.get("last_relative_change", float("inf"))),
        bin_voltages_kv=[b.voltage_kv for b in bins[:bins_run]],
        bin_weights=[b.energy_weight for b in bins[:bins_run]],
        bin_indices=[b.bin_index for b in bins[:bins_run]],
    )
    meta = {
        "smith_iterative_mode": smith_iterative_mode,
        "fail_k": int(dyn.get("fail_k", 0)),
        "dyn_wall_s": float(dyn.get("dyn_wall_s", 0.0)),
        "k_per_s": float(dyn.get("k_per_s", 0.0)),
        "k_solved": int(dyn.get("k_solved", 0)),
        "pg_grid": pg_grid,
    }
    return result, meta
