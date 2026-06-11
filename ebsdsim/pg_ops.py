"""Point-group operators and fundamental-sector geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ebsdsim._pg_ops_data import FS_NORMALS, PG_NUM_TO_SYMBOL, PG_OPERATORS

CENTROSYMMETRIC_PG: frozenset[int] = frozenset({2, 5, 8, 11, 15, 17, 20, 23, 27, 29, 32})


def pg_num_to_symbol(pg_num: int) -> str:
    if pg_num < 1 or pg_num > len(PG_NUM_TO_SYMBOL):
        raise ValueError(f"pg_num {pg_num} out of range")
    return PG_NUM_TO_SYMBOL[pg_num - 1]


def _sym_key(symbol: str) -> str:
    return symbol.replace(" ", "")


def point_group_operators(symbol: str) -> NDArray[np.float64]:
    key = _sym_key(symbol)
    data = PG_OPERATORS.get(key)
    if data is None:
        raise ValueError(f"Unknown PG symbol '{symbol}'")
    return data.copy()


def fs_normals(symbol: str) -> NDArray[np.float64]:
    key = _sym_key(symbol)
    data = FS_NORMALS.get(key)
    if data is None:
        raise ValueError(f"No FS normals for PG symbol '{symbol}'")
    return data.copy()


def in_fundamental_sector_vec(
    kx: float,
    ky: float,
    kz: float,
    normals: NDArray[np.float64],
    eps: float,
) -> bool:
    m = normals.size // 3
    for i in range(m):
        o = i * 3
        margin = kx * normals[o] + ky * normals[o + 1] + kz * normals[o + 2]
        if margin < -eps:
            return False
    return True


def orbit_fs_representative(
    kx: float,
    ky: float,
    kz: float,
    ops: NDArray[np.float64],
    normals: NDArray[np.float64],
    eps: float,
    out: NDArray[np.float64],
) -> None:
    g = ops.size // 9
    m = normals.size // 3
    best_idx = 0
    best_margin = float("-inf")
    best_found = False
    for gi in range(g):
        o = gi * 9
        ox = ops[o + 0] * kx + ops[o + 1] * ky + ops[o + 2] * kz
        oy = ops[o + 3] * kx + ops[o + 4] * ky + ops[o + 5] * kz
        oz = ops[o + 6] * kx + ops[o + 7] * ky + ops[o + 8] * kz
        min_margin = float("inf")
        for mi in range(m):
            no = mi * 3
            margin = ox * normals[no] + oy * normals[no + 1] + oz * normals[no + 2]
            min_margin = min(min_margin, margin)
        if m == 0:
            min_margin = 0.0
        in_fs = min_margin >= -eps
        if in_fs and min_margin > best_margin:
            best_margin = min_margin
            best_idx = gi
            best_found = True
        elif not best_found and min_margin > best_margin:
            best_margin = min_margin
            best_idx = gi
    o = best_idx * 9
    out[0] = ops[o + 0] * kx + ops[o + 1] * ky + ops[o + 2] * kz
    out[1] = ops[o + 3] * kx + ops[o + 4] * ky + ops[o + 5] * kz
    out[2] = ops[o + 6] * kx + ops[o + 7] * ky + ops[o + 8] * kz
