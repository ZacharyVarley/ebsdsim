"""Engine layer: full master-pattern orchestration (MC -> dynamical -> Lambert)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from ebsdsim.crystal.pointgroup import CENTROSYMMETRIC_PG, folding_symbol
from ebsdsim.crystal.simcell import SimCell
from ebsdsim.energy.surrogate import infer_direct_exp_from_cell_rebinned
from ebsdsim.energy.weights import site_weights_from_cell
from ebsdsim.engine.integrate import (
    RunMasterPatternIntegratedOptions,
    next_active_voltage_kv,
    run_master_pattern_voltage_integrated,
    surrogate_to_multi_voltage_mc,
    trim_multi_voltage_mc_by_coverage,
)
from ebsdsim.engine.metadata import cell_metadata
from ebsdsim.engine.params import SimParams, resolve_solver_choice
from ebsdsim.engine.progress import MasterPatternProgress
from ebsdsim.engine.results import MasterPattern, stack_bins
from ebsdsim.engine.runner import RunOneVoltageDeps, run_one_voltage
from ebsdsim.gpu import EBSDDynamicalKernels, require_gpu, run_monte_carlo_gpu
from ebsdsim.gpu.resident import make_metric_buffer
from ebsdsim.lambert.kgrid import build_pg_k_grid
from ebsdsim.physics.lookup import LookupPrefetcher, prepare_diff_lookup_geometry
from ebsdsim.physics.mode import MasterPatternMode
from ebsdsim.physics.site_tables import prepare_site_sgh_tables

_CIF_CELL_META_KEYS = (
    "setting",
    "origin_choice",
    "transformed",
    "setting_describe",
    "setting_note",
    "P",
    "p",
    "rhombohedral_input",
)


def _build_mc(cell: SimCell, params: SimParams):
    n_energy_bins = max(1, int(params.voltage_kv / params.energy_binwidth_keV))
    if params.mc_backend == "surrogate":
        direct = infer_direct_exp_from_cell_rebinned(
            cell=cell,
            sigma_deg=params.sigma_deg,
            beam_kv=params.voltage_kv,
            energy_binwidth_keV=params.energy_binwidth_keV,
            n_energy_bins=n_energy_bins,
        )
        return surrogate_to_multi_voltage_mc(direct, params.voltage_kv), "surrogate"
    if params.mc_backend == "gpu":
        mc = run_monte_carlo_gpu(
            cell,
            voltage_kv=params.voltage_kv,
            energy_binwidth_kev=params.energy_binwidth_keV,
            n_trajectories=None if params.mc_auto_stop else params.n_trajectories,
            sigma_deg=params.sigma_deg,
            omega_deg=params.omega_deg,
            auto_stop=params.mc_auto_stop,
            relative_tol=params.mc_relative_tol,
            min_trajectories=params.mc_min_trajectories,
            max_trajectories=params.mc_max_trajectories,
        )
        return mc, "gpu_fly_first"
    raise ValueError(f"unknown mc_backend: {params.mc_backend!r}")


def _merge_cell_metadata(cell: SimCell, structure_meta: dict[str, Any] | None) -> dict[str, Any]:
    cell_meta = dict(cell_metadata(cell))
    if structure_meta is not None:
        used = structure_meta.get("cell") or {}
        for key in _CIF_CELL_META_KEYS:
            if key in used:
                cell_meta[key] = used[key]
    return cell_meta


def build_run_metadata(
    *,
    cell: SimCell,
    params: SimParams,
    source: str,
    mode: MasterPatternMode,
    mc,
    mc_backend_label: str,
    integrated_result,
    pg_symbol: str,
    is_centro: bool,
    needs_southern: bool,
    cell_meta: dict[str, Any],
    solver_label: str,
    rank: int,
    exact_slow_cpu: bool,
    structure_meta: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build save/run metadata; ``extras`` carry solver-specific keys."""
    grid_size = 1 + 2 * params.halfw
    metadata: dict[str, Any] = {
        "format": "ebsdsim-master-pattern",
        "format_version": 1,
        "source": source,
        "voltage_kv": float(params.voltage_kv),
        "grid_size": int(grid_size),
        "halfw": int(params.halfw),
        "dmin": float(params.dmin),
        "energy_binwidth_keV": float(params.energy_binwidth_keV),
        "mode": mode,
        "marginal_coverage": float(params.marginal_coverage),
        "solver": solver_label,
        "rank": int(rank),
        "exact_slow_cpu": bool(exact_slow_cpu),
        "verbosity": int(params.verbosity),
        "bethe_c_strong": float(params.bethe_c_strong),
        "bethe_c_weak": float(params.bethe_c_weak),
        "bethe_c_cutoff": float(params.bethe_c_cutoff),
        "dbdiff_sg_cutoff": float(params.dbdiff_sg_cutoff),
        "relative_image_stop": float(params.relative_image_stop),
        "mc_backend": mc_backend_label,
        "sigma_deg": float(params.sigma_deg),
        "omega_deg": float(params.omega_deg),
        "n_mc_bins": int(mc.voltages_kv.size),
        "n_bins_run": int(integrated_result.n_bins_run),
        "stopped_by_relative_change": bool(integrated_result.stopped_by_relative_change),
        "last_relative_change": float(integrated_result.last_relative_change),
        "mc_n_trajectories": int(getattr(mc, "n_trajectories", 0)),
        "mc_converged": bool(getattr(mc, "converged", False)),
        "mc_relative_tol": float(params.mc_relative_tol),
        "mc_last_relative_change": float(getattr(mc, "last_relative_change", float("inf"))),
        "pg_num": int(cell.pg_num),
        "pg_symbol": pg_symbol,
        "is_centrosymmetric": bool(is_centro),
        "needs_southern_hemisphere": needs_southern,
        "n_k": int(integrated_result.n_k),
        "n_sites": int(integrated_result.n_sites),
        "cell": cell_meta,
    }
    if extras:
        metadata.update(extras)
    if structure_meta is not None:
        metadata["symmetry_provenance"] = structure_meta.get("symmetry_provenance")
        metadata["cif_input"] = structure_meta.get("cif_input")
    return metadata


def _finish_master_pattern(
    *,
    cell: SimCell,
    params: SimParams,
    source: str,
    mode: MasterPatternMode,
    mc,
    mc_backend_label: str,
    integrated_result,
    pg_grid,
    data: NDArray[np.float32],
    axes: dict[str, Any],
    kij: NDArray[np.int32],
    solver_label: str,
    rank: int,
    exact_slow_cpu: bool,
    structure_meta: dict[str, Any] | None,
    extras: dict[str, Any] | None = None,
) -> MasterPattern:
    if pg_grid.symbol:
        pg_symbol = pg_grid.symbol
    elif cell.space_group is None:
        raise ValueError(
            "PgKGrid.symbol is empty and cell.space_group is missing; cannot "
            "recover an oriented folding symbol (never guess from pg_num alone)."
        )
    else:
        pg_symbol = folding_symbol(int(cell.pg_num), int(cell.space_group))
    khat = pg_grid.khat.reshape(-1, 3).astype(np.float32, copy=False)
    is_centro = int(cell.pg_num) in CENTROSYMMETRIC_PG
    needs_southern = bool(np.any(khat[:, 2] < -1e-9))
    e_int = axes["energy_integrated_index"]
    s_int = axes["site_integrated_index"]
    pattern = np.ascontiguousarray(data[e_int, s_int, 0])
    cell_meta = _merge_cell_metadata(cell, structure_meta)
    metadata = build_run_metadata(
        cell=cell,
        params=params,
        source=source,
        mode=mode,
        mc=mc,
        mc_backend_label=mc_backend_label,
        integrated_result=integrated_result,
        pg_symbol=pg_symbol,
        is_centro=is_centro,
        needs_southern=needs_southern,
        cell_meta=cell_meta,
        solver_label=solver_label,
        rank=rank,
        exact_slow_cpu=exact_slow_cpu,
        structure_meta=structure_meta,
        extras=extras,
    )
    return MasterPattern(
        pattern=pattern,
        integrated=integrated_result.integrated,
        n_k=integrated_result.n_k,
        n_sites=integrated_result.n_sites,
        metadata=metadata,
        bin_patterns=list(integrated_result.bin_patterns),
        bin_voltages_kv=list(integrated_result.bin_voltages_kv),
        bin_weights=list(integrated_result.bin_weights),
        kij=kij,
        khat=khat,
        pg_num=int(cell.pg_num),
        pg_symbol=pg_symbol,
        data=data,
        axes=axes,
    )


def _run_smith_iterative(
    *,
    cell: SimCell,
    params: SimParams,
    mc,
    mc_backend_label: str,
    progress: MasterPatternProgress,
    source: str,
    mode: MasterPatternMode,
    structure_meta: dict[str, Any] | None,
    max_bins_run: int | None = None,
    max_chunks: int | None = None,
) -> MasterPattern:
    # Lazy: smith_iterative path needs shader-f16 + GPU raster modules.
    from ebsdsim.engine.smith_iterative_runner import run_smith_iterative_voltage_integrated
    from ebsdsim.gpu.raster import GpuLambertRasterizer, build_master_pattern_data_gpu

    grid_size = 1 + 2 * params.halfw
    ctx = require_gpu(required_features=("shader-f16",))
    kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
    progress.run_banner(
        mc_backend=mc_backend_label,
        n_bins=int(mc.voltages_kv.size),
        n_k=0,
    )
    try:
        integrated_result, smith_iterative_meta = run_smith_iterative_voltage_integrated(
            cell=cell,
            mc=mc,
            halfw=params.halfw,
            dmin=params.dmin,
            voltage_kv=params.voltage_kv,
            bethe_c_strong=params.bethe_c_strong,
            bethe_c_weak=params.bethe_c_weak,
            bethe_c_cutoff=params.bethe_c_cutoff,
            dbdiff_sg_cutoff=params.dbdiff_sg_cutoff,
            kernels=kernels,
            progress=progress,
            marginal_coverage=params.marginal_coverage,
            relative_image_stop=params.relative_image_stop,
            max_bins_run=max_bins_run,
            max_chunks=max_chunks,
        )
        pg_grid = smith_iterative_meta["pg_grid"]
        n_k = int(integrated_result.n_k)
        site_weights = site_weights_from_cell(cell)
        kij = pg_grid.kij.reshape(-1, 3).astype(np.int32, copy=False)
        rasterizer = GpuLambertRasterizer(
            ctx.device, ctx.queue, pg_grid, pipelines=kernels.pipelines
        )
        try:
            data, axes = build_master_pattern_data_gpu(
                rasterizer,
                integrated_fs=np.asarray(integrated_result.integrated, dtype=np.float32).reshape(
                    n_k, integrated_result.n_sites
                ),
                bin_fs=stack_bins(list(integrated_result.bin_patterns), n_k, integrated_result.n_sites),
                kij=kij,
                hw=params.halfw,
                side=grid_size,
                needs_southern_hemisphere=bool(np.any(pg_grid.khat.reshape(-1, 3)[:, 2] < -1e-9)),
                site_weights=site_weights,
            )
        finally:
            rasterizer.destroy()
    finally:
        kernels.destroy()

    extras = {
        "smith_iterative_mode": smith_iterative_meta.get("smith_iterative_mode"),
        "fail_k": smith_iterative_meta.get("fail_k"),
        "k_per_s": smith_iterative_meta.get("k_per_s"),
        "dyn_wall_s": smith_iterative_meta.get("dyn_wall_s"),
        "k_solved": smith_iterative_meta.get("k_solved"),
    }
    return _finish_master_pattern(
        cell=cell,
        params=params,
        source=source,
        mode=mode,
        mc=mc,
        mc_backend_label=mc_backend_label,
        integrated_result=integrated_result,
        pg_grid=pg_grid,
        data=data,
        axes=axes,
        kij=kij,
        solver_label="smith_iterative",
        rank=16,
        exact_slow_cpu=False,
        structure_meta=structure_meta,
        extras=extras,
    )


def _run_lu(
    *,
    cell: SimCell,
    params: SimParams,
    mc,
    mc_backend_label: str,
    progress: MasterPatternProgress,
    source: str,
    mode: MasterPatternMode,
    structure_meta: dict[str, Any] | None,
    max_bins_run: int | None,
    bin_callback: Callable[[int, int, float, float, float], None] | None,
    max_chunks: int | None = None,
) -> MasterPattern:
    # Lazy: LU path pulls GPU raster + lookup only when selected.
    from ebsdsim.gpu.raster import GpuLambertRasterizer, build_master_pattern_data_gpu

    grid_size = 1 + 2 * params.halfw
    ctx = require_gpu()
    mc_for_bins = trim_multi_voltage_mc_by_coverage(mc, params.marginal_coverage)
    first_vkv = next_active_voltage_kv(mc_for_bins, -1)
    lookup_geometry = prepare_diff_lookup_geometry(cell, params.dmin)
    lookup_prefetcher = LookupPrefetcher(lookup_geometry, params.dmin, mode)
    if first_vkv is not None:
        lookup_prefetcher.prefetch(first_vkv)
    pg_grid = build_pg_k_grid(cell.pg_num, params.halfw, cell.space_group)
    n_k = pg_grid.khat.size // 3
    progress.run_banner(
        mc_backend=mc_backend_label,
        n_bins=int(mc_for_bins.voltages_kv.size),
        n_k=n_k,
    )
    sgh = prepare_site_sgh_tables(cell, params.dmin)
    kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
    metric = make_metric_buffer(kernels, cell)
    deps = RunOneVoltageDeps(
        cell=cell,
        pg_grid=pg_grid,
        sgh=sgh,
        kernels=kernels,
        metric=metric,
        chunk_size=params.chunk_size,
        rank=params.rank,
        exact_slow_cpu=params.exact_slow_cpu,
        verbosity=params.verbosity,
        progress=progress,
        dmin=params.dmin,
        bethe_c_strong=params.bethe_c_strong,
        bethe_c_weak=params.bethe_c_weak,
        bethe_c_cutoff=params.bethe_c_cutoff,
        dbdiff_sg_cutoff=params.dbdiff_sg_cutoff,
        lookup_geometry=lookup_geometry,
        lookup_prefetcher=lookup_prefetcher,
        max_chunks=max_chunks,
    )

    site_weights = site_weights_from_cell(cell)
    kij = pg_grid.kij.reshape(-1, 3).astype(np.int32, copy=False)
    rasterizer = GpuLambertRasterizer(
        ctx.device, ctx.queue, pg_grid, pipelines=kernels.pipelines
    )

    def on_bin_integrated(integrated_flat: NDArray[np.float32], n_k_i: int, n_sites: int) -> None:
        rasterizer.rasterize_fs_values(
            integrated_flat,
            n_k_i,
            n_sites,
            kij,
            params.halfw,
            site_weights=site_weights,
            readback=False,
        )

    try:
        integrated_result = run_master_pattern_voltage_integrated(
            RunMasterPatternIntegratedOptions(
                mc=mc,
                run_one_voltage=lambda per_ctx: run_one_voltage(per_ctx, deps),
                mode=mode,
                marginal_coverage=params.marginal_coverage,
                relative_image_stop=params.relative_image_stop,
                on_bin_integrated=on_bin_integrated,
                max_bins_run=max_bins_run,
                bin_callback=bin_callback or (progress.on_bin_start if progress.enabled else None),
                on_bin_complete=progress.on_bin_complete if progress.enabled else None,
            )
        )
        if progress.enabled:
            if integrated_result.stopped_by_relative_change:
                progress.integration_stopped(
                    last_relative_change=integrated_result.last_relative_change,
                    n_bins_run=integrated_result.n_bins_run,
                )
            print(
                f"[ebsdsim] master pattern complete  "
                f"{integrated_result.n_bins_run} bins integrated  "
                f"{n_k} k-pts  {integrated_result.n_sites} site(s)",
                flush=True,
            )
    finally:
        lookup_prefetcher.close()
        if deps.reusable_persistent is not None:
            deps.reusable_persistent.destroy()
        metric.destroy()

    n_k = int(integrated_result.n_k)
    n_sites = int(integrated_result.n_sites)
    khat = pg_grid.khat.reshape(-1, 3).astype(np.float32, copy=False)
    needs_southern = bool(np.any(khat[:, 2] < -1e-9))
    try:
        data, axes = build_master_pattern_data_gpu(
            rasterizer,
            integrated_fs=np.asarray(integrated_result.integrated, dtype=np.float32).reshape(
                n_k, n_sites
            ),
            bin_fs=stack_bins(list(integrated_result.bin_patterns), n_k, n_sites),
            kij=kij,
            hw=params.halfw,
            side=grid_size,
            needs_southern_hemisphere=needs_southern,
            site_weights=site_weights,
        )
    finally:
        rasterizer.destroy()

    return _finish_master_pattern(
        cell=cell,
        params=params,
        source=source,
        mode=mode,
        mc=mc,
        mc_backend_label=mc_backend_label,
        integrated_result=integrated_result,
        pg_grid=pg_grid,
        data=data,
        axes=axes,
        kij=kij,
        solver_label="lu_smith",
        rank=int(params.rank),
        exact_slow_cpu=bool(params.exact_slow_cpu),
        structure_meta=structure_meta,
        extras={
            "dyn_wall_s": float(deps.dyn_wall_s),
            "k_solved": int(deps.k_solved),
            "k_per_s": (
                float(deps.k_solved) / deps.dyn_wall_s if deps.dyn_wall_s > 0 else 0.0
            ),
        },
    )


def run_master_pattern(
    cell: SimCell,
    params: SimParams,
    *,
    source: str,
    mode: MasterPatternMode = "bloch",
    max_bins_run: int | None = None,
    bin_callback: Callable[[int, int, float, float, float], None] | None = None,
    structure_meta: dict[str, Any] | None = None,
    structure_log: str | None = None,
    max_chunks: int | None = None,
) -> MasterPattern:
    """Run MC + dynamical solve + Lambert post-process for one simulation cell."""
    choice = resolve_solver_choice(params)
    progress = MasterPatternProgress(
        verbosity=params.verbosity,
        source=source,
        halfw=params.halfw,
        dmin=params.dmin,
        exact_slow_cpu=params.exact_slow_cpu,
        rank=params.rank,
        chunk_size=params.chunk_size,
        solver=choice,
    )
    if structure_log and params.verbosity >= 1:
        print(f"[ebsdsim] {structure_log}", flush=True)

    mc, mc_backend_label = _build_mc(cell, params)

    if choice == "smith_iterative":
        return _run_smith_iterative(
            cell=cell,
            params=params,
            mc=mc,
            mc_backend_label=mc_backend_label,
            progress=progress,
            source=source,
            mode=mode,
            structure_meta=structure_meta,
            max_bins_run=max_bins_run,
            max_chunks=max_chunks,
        )
    return _run_lu(
        cell=cell,
        params=params,
        mc=mc,
        mc_backend_label=mc_backend_label,
        progress=progress,
        source=source,
        mode=mode,
        structure_meta=structure_meta,
        max_bins_run=max_bins_run,
        bin_callback=bin_callback,
        max_chunks=max_chunks,
    )
