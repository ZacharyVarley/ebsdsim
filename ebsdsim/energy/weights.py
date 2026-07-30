"""Energy layer: site/energy marginal weights (dimensionless)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ebsdsim.crystal.simcell import SimCell


def site_weights_from_cell(cell: SimCell) -> NDArray[np.float32]:
    """Normalized site weights from occupancy × symmetry multiplicity."""
    n = int(cell.atom_types.size)
    atom_data = np.asarray(cell.atom_data, dtype=np.float64).reshape(n, 5)
    mult = np.asarray(cell.multiplicities, dtype=np.float64).reshape(n)
    w = atom_data[:, 3] * mult
    total = float(w.sum())
    if total <= 0:
        w = np.ones(n, dtype=np.float64)
        total = float(n)
    return (w / total).astype(np.float32)


def site_weights_from_meta_cell(meta_cell: dict[str, Any]) -> NDArray[np.float32] | None:
    """Reconstruct site weights from ``meta_json`` ``cell`` block."""
    sites = meta_cell.get("sites")
    if not sites:
        return None
    w = np.array(
        [
            float(s.get("occupancy", 1.0)) * float(s.get("multiplicity") or 1)
            for s in sites
        ],
        dtype=np.float64,
    )
    total = float(w.sum())
    if total <= 0:
        return None
    return (w / total).astype(np.float32)


def reduce_over_sites(
    values_fs: NDArray[np.floating],
    site_weights: NDArray[np.floating] | None,
) -> NDArray[np.float32]:
    """Collapse ``(n_k, n_sites)`` to one value per direction."""
    arr = np.asarray(values_fs, dtype=np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] == 1:
        return arr[:, 0].astype(np.float32, copy=False)
    if site_weights is None:
        return arr.mean(axis=1, dtype=np.float32)
    w = np.asarray(site_weights, dtype=np.float64).reshape(-1)
    if w.size != arr.shape[1]:
        return arr.mean(axis=1, dtype=np.float32)
    w = w / w.sum()
    return (arr.astype(np.float64) * w).sum(axis=1).astype(np.float32)
