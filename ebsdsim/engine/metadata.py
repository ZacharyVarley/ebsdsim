"""Engine layer: simulation-cell metadata for run records / NPZ."""

from __future__ import annotations

from typing import Any

import numpy as np

from ebsdsim.crystal.elements import element_symbol
from ebsdsim.crystal.pointgroup import folding_symbol
from ebsdsim.crystal.simcell import SimCell

_NM_TO_ANGSTROM = 10.0
_NM_SQ_TO_ANGSTROM_SQ = 100.0


def _oriented_pg_symbol(cell: SimCell) -> str | None:
    if not cell.pg_num:
        return None
    if cell.space_group is None:
        raise ValueError(
            "SimCell.space_group is required to record an oriented pg_symbol; "
            "bare point-group numbers are orientation-ambiguous for some families."
        )
    return folding_symbol(int(cell.pg_num), int(cell.space_group))


def cell_metadata(cell: SimCell) -> dict[str, Any]:
    """Serialize a :class:`~ebsdsim.crystal.simcell.SimCell` to a JSON-friendly dict.

    Per-site isotropic Debye–Waller factors are recorded in both Å² and nm²
    because they are commonly estimated and materially affect the pattern.
    """
    atom_types = np.asarray(cell.atom_types).reshape(-1)
    n_sites = int(atom_types.size)
    atom_data = np.asarray(cell.atom_data, dtype=np.float64).reshape(n_sites, 5)
    mult = np.asarray(cell.multiplicities).reshape(-1)
    sites: list[dict[str, Any]] = []
    for i in range(n_sites):
        z = int(atom_types[i])
        b_iso_nm2 = float(atom_data[i, 4])
        sites.append(
            {
                "index": i,
                "atomic_number": z,
                "symbol": element_symbol(z),
                "fract": [float(atom_data[i, 0]), float(atom_data[i, 1]), float(atom_data[i, 2])],
                "occupancy": float(atom_data[i, 3]),
                "b_iso_angstrom_sq": b_iso_nm2 * _NM_SQ_TO_ANGSTROM_SQ,
                "b_iso_nm_sq": b_iso_nm2,
                "multiplicity": int(mult[i]) if i < mult.size else None,
            }
        )
    return {
        "a_angstrom": float(cell.a) * _NM_TO_ANGSTROM,
        "b_angstrom": float(cell.b) * _NM_TO_ANGSTROM,
        "c_angstrom": float(cell.c) * _NM_TO_ANGSTROM,
        "alpha_deg": float(cell.alpha),
        "beta_deg": float(cell.beta),
        "gamma_deg": float(cell.gamma),
        "volume_angstrom3": float(cell.volume) * (_NM_TO_ANGSTROM**3),
        "density_g_cm3": float(cell.density),
        "lattice_centering": cell.lattice_centering,
        "space_group": cell.space_group,
        "pg_num": cell.pg_num,
        "pg_symbol": _oriented_pg_symbol(cell),
        "average_atomic_number": float(cell.average_atomic_number),
        "average_atomic_weight": float(cell.average_atomic_weight),
        "n_sites": n_sites,
        "sites": sites,
    }
