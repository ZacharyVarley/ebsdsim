"""Energy layer: MultiVoltageMC view (keV, dimensionless weights)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ebsdsim.energy.binning import dynamical_voltages_kv

if TYPE_CHECKING:
    from ebsdsim.energy.surrogate import SurrogateDirectExp


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


def surrogate_to_multi_voltage_mc(de: SurrogateDirectExp, beam_kv: float) -> MultiVoltageMC:
    """Map surrogate depth/energy marginals to a :class:`MultiVoltageMC`."""
    n = de.amplitudes.size
    centers = de.energy_centers_keV
    if centers.size >= 2:
        d_e = float(centers[1] - centers[0])
    elif centers.size == 1:
        d_e = float(centers[0]) * 2.0
    else:
        d_e = 1.0
    voltages = dynamical_voltages_kv(beam_kv, n, d_e)
    return MultiVoltageMC(
        binsize_energy_keV=d_e,
        voltages_kv=voltages,
        energy_weights=de.energy_weights.astype(np.float64, copy=True),
        amplitudes=de.amplitudes.astype(np.float64, copy=True),
        betas=de.betas.astype(np.float64, copy=True),
    )
