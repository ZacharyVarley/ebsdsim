"""Public API for EBSD master-pattern generation."""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from ebsdsim.cif import parse_cif_crystal
from ebsdsim.gpu import EBSDDynamicalKernels, require_gpu, run_monte_carlo_gpu
from ebsdsim.integrate import (
    PerVoltageContext,
    RunMasterPatternIntegratedOptions,
    run_master_pattern_voltage_integrated,
    surrogate_to_multi_voltage_mc,
)
from ebsdsim.surrogate import infer_direct_exp_from_cell_rebinned
from ebsdsim.kgrid import build_pg_k_grid
from ebsdsim.material import build_cell_from_material
from ebsdsim.runner import RunOneVoltageDeps, make_metric_buffer, run_one_voltage
from ebsdsim.sgh import prepare_site_sgh_tables
from ebsdsim.structure import build_cell_from_cif
from ebsdsim.types import MasterPatternMode


@dataclass(frozen=True)
class Atom:
    """Crystallographic site in fractional coordinates (direct lattice)."""

    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_iso: float | None = None  # Å²; default room-temperature fallback


@dataclass(frozen=True)
class Cell:
    """Unit-cell lattice parameters (Å and degrees)."""

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    space_group: int | str = 1


@dataclass
class Material:
    """Crystal specification for master-pattern simulation."""

    cell: Cell
    atoms: list[Atom]
    name: str = ""

    def to_simulation_cell(self):
        return build_cell_from_material(
            a=self.cell.a,
            b=self.cell.b,
            c=self.cell.c,
            alpha=self.cell.alpha,
            beta=self.cell.beta,
            gamma=self.cell.gamma,
            space_group=self.cell.space_group,
            atoms=self.atoms,
        )


@dataclass
class MasterPattern:
    """Rasterized master pattern and integration metadata.

    Attributes
    ----------
    pattern:
        North-hemisphere Lambert raster ``(side, side)`` with
        ``side = 1 + 2 * halfw``. Equal to ``data[energy_int, site_int, 0]``.
    data:
        Dense Lambert tensor ``(E, S, H, side, side)`` built on
        construction. ``E`` is ``1 + n_energy_bins`` (index 0 integrated) when
        there is more than one bin, else ``1``; ``S`` is ``1 + n_sites``
        (index 0 site-integrated) when there is more than one site, else ``1``;
        ``H`` is ``2`` when the southern hemisphere is needed, else ``1``.
        See ``axes`` for the index layout.
    integrated:
        Energy-integrated fundamental-sector intensities, flattened
        ``(n_k * n_sites,)`` in row-major ``(direction, site)`` order.
    bin_patterns:
        Per-energy-bin fundamental-sector intensities (the intermediates),
        each flattened ``(n_k * n_sites,)``. Stored unweighted.
    kij / khat:
        Fundamental-sector pixel indices ``(n_k, 3)`` as ``(i, j, sign)`` and
        unit directions ``(n_k, 3)``.
    """

    pattern: NDArray[np.float32]
    integrated: NDArray[np.float32]
    n_k: int
    n_sites: int
    metadata: dict[str, Any] = field(default_factory=dict)
    bin_patterns: list[NDArray[np.float32]] = field(default_factory=list)
    bin_voltages_kv: list[float] = field(default_factory=list)
    bin_weights: list[float] = field(default_factory=list)
    kij: NDArray[np.int32] | None = None
    khat: NDArray[np.float32] | None = None
    pg_num: int | None = None
    data: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0,), np.float32))
    axes: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Write this master pattern (with intermediates) to a compressed ``.npz``.

        See :func:`ebsdsim.save_master_pattern` for the on-disk format.
        """
        from ebsdsim.save import save_master_pattern

        return save_master_pattern(self, path)


def _validate_halfw(halfw: int) -> int:
    halfw_i = int(halfw)
    if halfw_i < 1:
        raise ValueError("halfw must be >= 1")
    return halfw_i


def _resolve_cif_path(path: str | Path) -> Path:
    """Resolve a filesystem path or bundled preset name (e.g. ``\"GaN.cif\"``)."""
    p = Path(path)
    if p.is_file():
        return p
    stem = p.stem if p.suffix else str(path)
    for name in (f"{stem}.cif", p.name):
        bundled = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs", name)
        if bundled.is_file():
            return Path(str(bundled))
    raise FileNotFoundError(f"CIF not found: {path}")


def master_pattern(
    material: Material,
    *,
    voltage_kv: float = 20.0,
    halfw: int = 250,
    dmin: float = 0.05,
    energy_binwidth_keV: float = 1.0,
    n_trajectories: int = 1_048_576,
    sigma_deg: float = 70.0,
    omega_deg: float = 0.0,
    rank: int = 20,
    chunk_size: int = 256,
    marginal_coverage: float = 1.0,
    relative_image_stop: float = 0.01,
    mc_backend: Literal["surrogate", "gpu"] = "surrogate",
    bethe_c_strong: float = 20.0,
    bethe_c_weak: float = 40.0,
    bethe_c_cutoff: float = 200.0,
    dbdiff_sg_cutoff: float = 1.0,
    mc_auto_stop: bool = True,
    mc_relative_tol: float = 0.01,
    mc_min_trajectories: int = 1_048_576,
    mc_max_trajectories: int = 16_777_216,
) -> MasterPattern:
    """Generate an EBSD master pattern from a manual material specification."""
    cell = material.to_simulation_cell()
    if cell.pg_num is None:
        raise ValueError("Could not resolve point group; provide a valid space_group.")
    return _run_master_pattern(
        cell=cell,
        voltage_kv=voltage_kv,
        halfw=_validate_halfw(halfw),
        dmin=dmin,
        energy_binwidth_keV=energy_binwidth_keV,
        n_trajectories=n_trajectories,
        sigma_deg=sigma_deg,
        omega_deg=omega_deg,
        rank=rank,
        chunk_size=chunk_size,
        mode="bloch",
        marginal_coverage=marginal_coverage,
        relative_image_stop=relative_image_stop,
        mc_backend=mc_backend,
        bethe_c_strong=bethe_c_strong,
        bethe_c_weak=bethe_c_weak,
        bethe_c_cutoff=bethe_c_cutoff,
        dbdiff_sg_cutoff=dbdiff_sg_cutoff,
        mc_auto_stop=mc_auto_stop,
        mc_relative_tol=mc_relative_tol,
        mc_min_trajectories=mc_min_trajectories,
        mc_max_trajectories=mc_max_trajectories,
        source=material.name or "material",
    )


def master_pattern_from_cif(
    path: str | Path,
    *,
    voltage_kv: float = 20.0,
    halfw: int = 250,
    dmin: float = 0.05,
    energy_binwidth_keV: float = 1.0,
    n_trajectories: int = 1_048_576,
    sigma_deg: float = 70.0,
    omega_deg: float = 0.0,
    rank: int = 20,
    chunk_size: int = 256,
    marginal_coverage: float = 1.0,
    relative_image_stop: float = 0.01,
    mc_backend: Literal["surrogate", "gpu"] = "surrogate",
    bethe_c_strong: float = 20.0,
    bethe_c_weak: float = 40.0,
    bethe_c_cutoff: float = 200.0,
    dbdiff_sg_cutoff: float = 1.0,
    mc_auto_stop: bool = True,
    mc_relative_tol: float = 0.01,
    mc_min_trajectories: int = 1_048_576,
    mc_max_trajectories: int = 16_777_216,
) -> MasterPattern:
    """Generate an EBSD master pattern from a CIF file or bundled preset name."""
    cif_path = _resolve_cif_path(path)
    crystal = parse_cif_crystal(cif_path.read_text(encoding="utf-8", errors="replace"))
    cell = build_cell_from_cif(crystal)
    if cell.pg_num is None:
        raise ValueError("Could not resolve point group from CIF; include _space_group_IT_number.")
    return _run_master_pattern(
        cell=cell,
        voltage_kv=voltage_kv,
        halfw=_validate_halfw(halfw),
        dmin=dmin,
        energy_binwidth_keV=energy_binwidth_keV,
        n_trajectories=n_trajectories,
        sigma_deg=sigma_deg,
        omega_deg=omega_deg,
        rank=rank,
        chunk_size=chunk_size,
        mode="bloch",
        marginal_coverage=marginal_coverage,
        relative_image_stop=relative_image_stop,
        mc_backend=mc_backend,
        bethe_c_strong=bethe_c_strong,
        bethe_c_weak=bethe_c_weak,
        bethe_c_cutoff=bethe_c_cutoff,
        dbdiff_sg_cutoff=dbdiff_sg_cutoff,
        mc_auto_stop=mc_auto_stop,
        mc_relative_tol=mc_relative_tol,
        mc_min_trajectories=mc_min_trajectories,
        mc_max_trajectories=mc_max_trajectories,
        source=str(cif_path),
    )


def _run_master_pattern(
    *,
    cell,
    voltage_kv: float,
    halfw: int,
    dmin: float,
    energy_binwidth_keV: float,
    n_trajectories: int,
    sigma_deg: float,
    omega_deg: float,
    rank: int,
    chunk_size: int,
    mode: MasterPatternMode,
    marginal_coverage: float,
    relative_image_stop: float,
    mc_backend: Literal["surrogate", "gpu"],
    source: str,
    bethe_c_strong: float = 20.0,
    bethe_c_weak: float = 40.0,
    bethe_c_cutoff: float = 200.0,
    dbdiff_sg_cutoff: float = 1.0,
    mc_auto_stop: bool = True,
    mc_relative_tol: float = 0.01,
    mc_min_trajectories: int = 1_048_576,
    mc_max_trajectories: int = 16_777_216,
    max_bins_run: int | None = None,
    bin_callback: Callable[[int, int, float], None] | None = None,
) -> MasterPattern:
    ctx = require_gpu()
    grid_size = 1 + 2 * halfw
    n_energy_bins = max(1, int(voltage_kv / energy_binwidth_keV))

    if mc_backend == "surrogate":
        direct = infer_direct_exp_from_cell_rebinned(
            cell=cell,
            sigma_deg=sigma_deg,
            beam_kv=voltage_kv,
            energy_binwidth_keV=energy_binwidth_keV,
            n_energy_bins=n_energy_bins,
        )
        mc = surrogate_to_multi_voltage_mc(direct, voltage_kv)
        mc_backend_label = "surrogate"
    elif mc_backend == "gpu":
        mc = run_monte_carlo_gpu(
            cell,
            voltage_kv=voltage_kv,
            energy_binwidth_kev=energy_binwidth_keV,
            n_trajectories=None if mc_auto_stop else n_trajectories,
            sigma_deg=sigma_deg,
            omega_deg=omega_deg,
            auto_stop=mc_auto_stop,
            relative_tol=mc_relative_tol,
            min_trajectories=mc_min_trajectories,
            max_trajectories=mc_max_trajectories,
        )
        mc_backend_label = "gpu_fly_first"
    else:
        raise ValueError(f"unknown mc_backend: {mc_backend!r}")

    from concurrent.futures import ThreadPoolExecutor

    from ebsdsim.integrate import next_active_voltage_kv, trim_multi_voltage_mc_by_coverage
    from ebsdsim.lookup import LookupPrefetcher, prepare_diff_lookup_geometry

    mc_for_bins = trim_multi_voltage_mc_by_coverage(mc, marginal_coverage)
    first_vkv = next_active_voltage_kv(mc_for_bins, -1)
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_pg = pool.submit(build_pg_k_grid, cell.pg_num, halfw)
        f_sgh = pool.submit(prepare_site_sgh_tables, cell, dmin)
        f_geom = pool.submit(prepare_diff_lookup_geometry, cell, dmin)
        lookup_geometry = f_geom.result()
        lookup_prefetcher = LookupPrefetcher(lookup_geometry, dmin, mode)
        if first_vkv is not None:
            lookup_prefetcher.prefetch(first_vkv)
        pg_grid = f_pg.result()
        sgh = f_sgh.result()
    kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
    metric = make_metric_buffer(kernels, cell)
    deps = RunOneVoltageDeps(
        cell=cell,
        pg_grid=pg_grid,
        sgh=sgh,
        kernels=kernels,
        metric=metric,
        chunk_size=chunk_size,
        rank=rank,
        dmin=dmin,
        bethe_c_strong=bethe_c_strong,
        bethe_c_weak=bethe_c_weak,
        bethe_c_cutoff=bethe_c_cutoff,
        dbdiff_sg_cutoff=dbdiff_sg_cutoff,
        lookup_geometry=lookup_geometry,
        lookup_prefetcher=lookup_prefetcher,
    )

    from ebsdsim.gpu.rasterize import GpuLambertRasterizer, build_master_pattern_data_gpu

    kij = pg_grid.kij.reshape(-1, 3).astype(np.int32, copy=False)
    rasterizer = GpuLambertRasterizer(
        ctx.device, ctx.queue, pg_grid, pipelines=kernels.pipelines
    )

    def on_bin_integrated(integrated_flat: NDArray[np.float32], n_k: int, n_sites: int) -> None:
        rasterizer.rasterize_fs_values(
            integrated_flat,
            n_k,
            n_sites,
            kij,
            halfw,
            readback=False,
        )

    def run_bin(per_ctx: PerVoltageContext):
        return run_one_voltage(per_ctx, deps)

    try:
        integrated_result = run_master_pattern_voltage_integrated(
            RunMasterPatternIntegratedOptions(
                mc=mc,
                run_one_voltage=run_bin,
                mode=mode,
                marginal_coverage=marginal_coverage,
                relative_image_stop=relative_image_stop,
                on_bin_integrated=on_bin_integrated,
                max_bins_run=max_bins_run,
                bin_callback=bin_callback,
            )
        )
    finally:
        lookup_prefetcher.close()
        if deps.reusable_persistent is not None:
            deps.reusable_persistent.destroy()
        metric.destroy()

    from ebsdsim.save import _stack_bins, cell_metadata
    from ebsdsim.pg_ops import CENTROSYMMETRIC_PG, pg_num_to_symbol

    khat = pg_grid.khat.reshape(-1, 3).astype(np.float32, copy=False)
    is_centro = int(cell.pg_num) in CENTROSYMMETRIC_PG
    needs_southern = bool(np.any(khat[:, 2] < -1e-9))

    n_k = int(integrated_result.n_k)
    n_sites = int(integrated_result.n_sites)
    try:
        data, axes = build_master_pattern_data_gpu(
            rasterizer,
            integrated_fs=np.asarray(integrated_result.integrated, dtype=np.float32).reshape(
                n_k, n_sites
            ),
            bin_fs=_stack_bins(list(integrated_result.bin_patterns), n_k, n_sites),
            kij=kij,
            hw=halfw,
            side=grid_size,
            needs_southern_hemisphere=needs_southern,
        )
    finally:
        rasterizer.destroy()
    e_int = axes["energy_integrated_index"]
    s_int = axes["site_integrated_index"]
    pattern = np.ascontiguousarray(data[e_int, s_int, 0])

    metadata: dict[str, Any] = {
        "format": "ebsdsim-master-pattern",
        "format_version": 1,
        "source": source,
        "voltage_kv": float(voltage_kv),
        "grid_size": int(grid_size),
        "halfw": int(halfw),
        "dmin": float(dmin),
        "energy_binwidth_keV": float(energy_binwidth_keV),
        "mode": mode,
        "marginal_coverage": float(marginal_coverage),
        # Dynamical-solver (Bethe / Smith) parameters — any change here
        # materially changes the simulated pattern.
        "rank": int(rank),
        "bethe_c_strong": float(bethe_c_strong),
        "bethe_c_weak": float(bethe_c_weak),
        "bethe_c_cutoff": float(bethe_c_cutoff),
        "dbdiff_sg_cutoff": float(dbdiff_sg_cutoff),
        "relative_image_stop": float(relative_image_stop),
        # Monte Carlo / energy model.
        "mc_backend": mc_backend_label,
        "sigma_deg": float(sigma_deg),
        "omega_deg": float(omega_deg),
        "n_mc_bins": int(mc.voltages_kv.size),
        "n_bins_run": int(integrated_result.n_bins_run),
        "stopped_by_relative_change": bool(integrated_result.stopped_by_relative_change),
        "last_relative_change": float(integrated_result.last_relative_change),
        "mc_n_trajectories": int(getattr(mc, "n_trajectories", 0)),
        "mc_converged": bool(getattr(mc, "converged", False)),
        "mc_relative_tol": float(mc_relative_tol),
        "mc_last_relative_change": float(getattr(mc, "last_relative_change", float("inf"))),
        # Geometry / point group.
        "pg_num": int(cell.pg_num),
        "pg_symbol": pg_num_to_symbol(int(cell.pg_num)),
        "is_centrosymmetric": bool(is_centro),
        "needs_southern_hemisphere": needs_southern,
        "n_k": int(integrated_result.n_k),
        "n_sites": int(integrated_result.n_sites),
        # Full crystallographic cell, including per-site isotropic
        # Debye–Waller factors (these are frequently estimated and do change
        # the outgoing master pattern, so they are recorded verbatim).
        "cell": cell_metadata(cell),
    }

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
        data=data,
        axes=axes,
    )
