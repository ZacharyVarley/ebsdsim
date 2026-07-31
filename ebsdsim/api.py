"""Public API for EBSD master-pattern generation.

Ångström ``Cell`` / ``Atom`` in; conversion to nm ``SimCell`` happens once in
:mod:`ebsdsim.crystal.build`. Default dynamical solver is Smith iterative.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ebsdsim.crystal.build import build_cell_from_structure
from ebsdsim.crystal.cif import load_structure
from ebsdsim.crystal.material import Atom, Cell, Material
from ebsdsim.crystal.pointgroup import fs_normals, point_group_operators, resolve_oriented_symbol
from ebsdsim.energy.weights import site_weights_from_meta_cell
from ebsdsim.engine.master_pattern import run_master_pattern
from ebsdsim.engine.params import SimParams
from ebsdsim.engine.results import MasterPattern, stack_bins
from ebsdsim.lambert.display import NormalizeMode

# ``Atom`` / ``Cell`` / ``Material`` are defined in :mod:`ebsdsim.crystal.material`
# and re-exported here as the public spelling.
__all__ = [
    "Atom",
    "Cell",
    "Material",
    "MasterPattern",
    "master_pattern",
    "master_pattern_from_cif",
]


def _master_pattern_save(self: MasterPattern, path: str | Path) -> Path:
    """Write this master pattern (with intermediates) to a compressed ``.npz``.

    See :func:`ebsdsim.save_master_pattern` for the on-disk format.
    """
    from ebsdsim.io.save import save_master_pattern

    return save_master_pattern(self, path)


def _master_pattern_lambert_data(
    self: MasterPattern,
    *,
    normalize: NormalizeMode | None = None,
    robust_p_low: float = 0.01,
    robust_p_high: float = 0.99,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Expand raw FS intensities to Lambert ``(E, S, H, side, side)``.

    Parameters
    ----------
    normalize : {"minmax", "robust"} or None, optional
        Display scaling mode. ``None`` returns raw intensities unchanged.
    robust_p_low, robust_p_high : float, optional
        Percentile window used when ``normalize="robust"``.

    Returns
    -------
    data : ndarray of float32
        Lambert tensor; see :attr:`data` for axis semantics.
    axes : dict
        Index maps for energy, site, and hemisphere axes.
    """
    if normalize is None:
        return self.data, self.axes
    if self.kij is None or self.pg_num is None:
        raise ValueError("MasterPattern is missing fundamental-sector grid data.")
    from ebsdsim.io.load import build_master_pattern_data

    n_k, n_sites = int(self.n_k), int(self.n_sites)
    cell_meta = self.metadata.get("cell") or {}
    space_group = self.metadata.get("space_group", cell_meta.get("space_group"))
    symbol = resolve_oriented_symbol(
        int(self.pg_num),
        pg_symbol=self.pg_symbol or self.metadata.get("pg_symbol"),
        space_group=int(space_group) if space_group is not None else None,
    )
    halfw = int(self.metadata["halfw"])
    side = int(self.metadata["grid_size"])
    site_weights = site_weights_from_meta_cell(self.metadata.get("cell", {}))
    return build_master_pattern_data(
        integrated_fs=np.asarray(self.integrated, dtype=np.float32).reshape(n_k, n_sites),
        bin_fs=stack_bins(list(self.bin_patterns), n_k, n_sites),
        kij=self.kij,
        pg_operators=point_group_operators(symbol).reshape(-1, 3, 3),
        fs_normals=fs_normals(symbol).reshape(-1, 3),
        hw=halfw,
        side=side,
        needs_southern_hemisphere=bool(self.metadata.get("needs_southern_hemisphere", False)),
        site_weights=site_weights,
        normalize=normalize,
        robust_p_low=robust_p_low,
        robust_p_high=robust_p_high,
    )


# Attach I/O helpers on the engine dataclass without engine importing io.
MasterPattern.save = _master_pattern_save  # type: ignore[method-assign]
MasterPattern.lambert_data = _master_pattern_lambert_data  # type: ignore[method-assign]


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
    solver: Literal["smith_iterative", "lu_smith"] = "smith_iterative",
    rank: int = 16,
    exact_slow_cpu: bool = False,
    verbosity: int = 0,
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
    """Generate an EBSD master pattern from a manual material specification.

    Requires a WebGPU-capable GPU. Returns raw dynamical intensities; call
    :meth:`MasterPattern.lambert_data` for display scaling.

    Parameters
    ----------
    material : Material
        Crystal structure and composition.
    voltage_kv : float
        Beam energy in kV (default ``20``).
    halfw : int
        Lambert half-width; side is ``1 + 2 * halfw`` (default ``250``).
    dmin : float
        Minimum interplanar spacing in nm (default ``0.05``).
    energy_binwidth_keV : float
        Monte Carlo energy-bin width in keV (default ``1``).
    n_trajectories : int
        MC trajectories per bin when ``mc_auto_stop=False``.
    sigma_deg, omega_deg : float
        Specimen tilt and azimuth (degrees).
    solver : {"smith_iterative", "lu_smith"}
        Dynamical backend (default ``smith_iterative``).
    rank : int
        Smith / Lyapunov rank (default 16; ``smith_iterative`` currently
        supports only 16, ``lu_smith`` accepts higher).
    exact_slow_cpu : bool
        Full-rank CPU Lyapunov instead of GPU Smith.
    verbosity : {0, 1, 2}
        Progress reporting level.
    chunk_size : int
        GPU batch size for the multi-beam solve.
    marginal_coverage : float
        Fraction of the MC energy distribution to integrate.
    relative_image_stop : float
        Early-stop threshold on relative image change.
    mc_backend : {"surrogate", "gpu"}
        Energy model (default ``surrogate``).
    bethe_c_strong, bethe_c_weak, bethe_c_cutoff : float
        Bethe perturbation cutoffs.
    dbdiff_sg_cutoff : float
        Structure-factor inclusion threshold.
    mc_auto_stop : bool
        Adaptive Monte Carlo trajectory count.
    mc_relative_tol : float
        MC convergence tolerance.
    mc_min_trajectories, mc_max_trajectories : int
        Bounds on adaptive MC trajectory count.

    Returns
    -------
    MasterPattern
        Lambert-rasterized result with per-bin intermediates and metadata.

    See also
    --------
    :doc:`parameters` for detailed parameter notes.
    """
    cell = material.to_simulation_cell()
    if cell.pg_num is None:
        raise ValueError("Could not resolve point group; provide a valid space_group.")
    params = SimParams(
        voltage_kv=voltage_kv,
        halfw=halfw,
        dmin=dmin,
        energy_binwidth_keV=energy_binwidth_keV,
        n_trajectories=n_trajectories,
        sigma_deg=sigma_deg,
        omega_deg=omega_deg,
        solver=solver,
        rank=rank,
        exact_slow_cpu=exact_slow_cpu,
        verbosity=verbosity,
        chunk_size=chunk_size,
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
    )
    return run_master_pattern(cell, params, source=material.name or "material")


def master_pattern_from_cif(path: str | Path, **kwargs: Any) -> MasterPattern:
    """Generate an EBSD master pattern from a CIF file or bundled preset.

    ``path`` may be a filesystem path or a bundled preset name such as
    ``"GaN.cif"`` or ``"Ni.cif"``. The CIF is standardized to the International
    Tables setting on load (dual-origin groups use origin choice 2).

    Accepts the same keyword arguments as :func:`master_pattern`.
    See :doc:`parameters` for the full parameter reference.

    Parameters
    ----------
    path : str or Path
        CIF file path or preset stem/name.
    **kwargs
        Forwarded to :func:`master_pattern` / :class:`~ebsdsim.engine.params.SimParams`.

    Returns
    -------
    MasterPattern
        Lambert-rasterized result with per-bin intermediates and metadata.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not resolve to a file or bundled preset.
    ValueError
        If the CIF is symmetry-inconsistent or the point group cannot be resolved.
    """
    cif_path = _resolve_cif_path(path)
    structure = load_structure(cif_path)
    cell = build_cell_from_structure(structure)
    if cell.pg_num is None:
        raise ValueError("Could not resolve point group from CIF; include _space_group_IT_number.")
    params = SimParams(**kwargs)
    return run_master_pattern(
        cell,
        params,
        source=str(cif_path),
        structure_meta=structure.metadata(),
        structure_log=structure.log(),
    )
