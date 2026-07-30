"""Internal SimCell (lengths in nm; B_iso in nm²).

Public :class:`~ebsdsim.crystal.material.Atom` / :class:`~ebsdsim.crystal.material.Cell`
are in ångströms. Conversion to nm happens once in :mod:`ebsdsim.crystal.build`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ANGSTROM_TO_NM = 0.1
NM_TO_ANGSTROM = 10.0
ANGSTROM_SQ_TO_NM_SQ = 0.01

LatticeCentering = Literal["P", "F", "I", "A", "B", "C", "R"]


@dataclass
class SimCell:
    """Unit cell in internal units (nm; B_iso in nm²)."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: float
    direct_structure_matrix: NDArray[np.float64]
    reciprocal_metric: NDArray[np.float64]
    atom_types: NDArray[np.int32]
    atom_data: NDArray[np.float64]
    positions: list[list[tuple[float, float, float]]]
    multiplicities: NDArray[np.int32]
    density: float
    average_atomic_number: float
    average_atomic_weight: float
    lattice_centering: LatticeCentering
    space_group: int | None = None
    pg_num: int | None = None
