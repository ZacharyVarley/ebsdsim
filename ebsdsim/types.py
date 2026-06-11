"""Shared dataclasses for crystal structures and master-pattern results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

LatticeCentering = Literal["P", "F", "I", "A", "B", "C", "R"]
MasterPatternMode = Literal["bloch", "structure"]


@dataclass(frozen=True)
class Atom:
    label: str
    symbol: str
    atomic_number: int
    fract: tuple[float, float, float]
    occupancy: float
    b_iso: float  # nm²


@dataclass
class Cell:
    """Unit cell in internal units (nm, nm² for B_iso)."""

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


@dataclass
class Material:
    """Crystal cell plus optional simulation metadata."""

    cell: Cell
    name: str = ""
    source: str = ""


@dataclass
class MultiVoltageMC:
    binsize_energy_keV: float
    voltages_kv: NDArray[np.float64]
    energy_weights: NDArray[np.float64]
    amplitudes: NDArray[np.float64]
    betas: NDArray[np.float64]
    n_trajectories: int = 0
    converged: bool = False
    n_convergence_checks: int = 0
    last_relative_change: float = float("inf")


@dataclass
class MasterPatternResult:
    integrated: NDArray[np.float32]
    n_k: int
    n_sites: int
    bin_patterns: list[NDArray[np.float32]] = field(default_factory=list)
