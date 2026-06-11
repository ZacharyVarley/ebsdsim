"""Per-site SGH (structure-factor × DWF) lookup tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ebsdsim.lookup import (
    _stack_site_positions,
    generate_reflections,
    hkl_limits,
    reciprocal_length_batch,
)
from ebsdsim.types import Cell

PI = np.pi


@dataclass
class SghTableData:
    tables: NDArray[np.float32]
    n_sites: int
    table_size: int


def _prepend_zero(hkl: NDArray[np.int32]) -> NDArray[np.int32]:
    out = np.zeros(hkl.size + 3, dtype=np.int32)
    out[3:] = hkl
    return out


def prepare_site_sgh_tables(cell: Cell, dmin: float) -> SghTableData:
    hkl_diff = _prepend_zero(generate_reflections(cell, dmin, True))
    hmax, kmax, lmax = hkl_limits(cell, dmin)
    stride_k = 4 * lmax + 1
    stride_h = (4 * kmax + 1) * stride_k
    table_size = (4 * hmax + 1) * stride_h
    n_sites = cell.atom_types.size
    tables = np.zeros(n_sites * table_size * 2, dtype=np.float32)

    h = hkl_diff.reshape(-1, 3).astype(np.float64, copy=False)
    hash_idx = (
        (h[:, 0] + 2 * hmax) * stride_h + (h[:, 1] + 2 * kmax) * stride_k + (h[:, 2] + 2 * lmax)
    ).astype(np.intp)
    g = reciprocal_length_batch(hkl_diff, cell.reciprocal_metric)
    s_sq = 0.25 * g * g
    site_pos, _ = _stack_site_positions(cell)
    ang = 2 * PI * (
        h[:, None, None, 0] * site_pos[None, :, :, 0]
        + h[:, None, None, 1] * site_pos[None, :, :, 1]
        + h[:, None, None, 2] * site_pos[None, :, :, 2]
    )
    phase_re = np.cos(ang).sum(axis=2)
    phase_im = np.sin(ang).sum(axis=2)

    for site in range(n_sites):
        z = int(cell.atom_types[site])
        occ = cell.atom_data[site, 3] or 1.0
        b_iso = cell.atom_data[site, 4] or 0.0
        znsq = z * z * occ
        dwf = znsq * np.exp(-b_iso * s_sq)
        out = (site * table_size + hash_idx) * 2
        tables[out] = (dwf * phase_re[:, site]).astype(np.float32)
        tables[out + 1] = (dwf * phase_im[:, site]).astype(np.float32)

    return SghTableData(tables=tables, n_sites=n_sites, table_size=table_size)
