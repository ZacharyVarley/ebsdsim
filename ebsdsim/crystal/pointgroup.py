"""Crystal layer: point-group operators and FS geometry."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ebsdsim.crystal._generated.pg_ops import FS_NORMALS, PG_NUM_TO_SYMBOL, PG_OPERATORS

CENTROSYMMETRIC_PG: frozenset[int] = frozenset({2, 5, 8, 11, 15, 17, 20, 23, 27, 29, 32})

# Generated short symbols '2' / 'm' / '2/m' are unique-c. IT cells and Hall
# tables in this package are unique-b, so remap operators (and FS cuts) to the
# unique-b settings already present in PG_OPERATORS.
_MONOCLINIC_UNIQUE_B_OPS: dict[str, str] = {
    "2": "121",
    "m": "1m1",
    "2/m": "12/m1",
}
_MONOCLINIC_UNIQUE_B_FS: dict[str, NDArray[np.float64]] = {
    # 2 || b: half-sphere cut by the yz plane (x >= 0); 2_y maps NH <-> SH.
    "2": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    # m ⊥ b: keep y >= 0.
    "m": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    # 2/m unique-b {E, 2_y, i, m_y}: z >= 0 and y >= 0. The y-cut is required
    # because m_y flips only y; an x-cut would let m_y double-cover the sector
    # (leaving half the sphere uncovered when the group fills the rest).
    "2/m": np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64),
}

# A few point groups have two IT-standard orientations that belong to DIFFERENT
# space groups (the secondary/tertiary symmetry directions differ by a rotation
# about the main axis; that rotation is not a lattice symmetry, so the two are
# genuinely distinct space groups, not a free setting choice):
#   32   -> 321 (2 || a)  vs  312 (2 rotated 30 deg)
#   3m   -> 3m1 (m ⊥ a)   vs  31m
#   -3m  -> -3m1          vs  -31m
#   -42m -> -42m          vs  -4m2 (mirrors/2-folds swapped, 45 deg)
#   -6m2 -> -6m2          vs  -62m (30 deg)
# The number→symbol table stores only one spelling per number (the first of each
# pair). Space groups that need the *other* orientation are remapped here
# (verified for all 230 groups by scripts/verify_pg_ops.py):
_SG_ORIENTED_SYMBOL: dict[int, str] = {
    **dict.fromkeys((149, 151, 153), "312"),
    **dict.fromkeys((157, 159), "31m"),
    **dict.fromkeys((162, 163), "-31m"),
    **dict.fromkeys((115, 116, 117, 118, 119, 120), "-4m2"),
    **dict.fromkeys((189, 190), "-62m"),
}
# Point-group numbers that admit more than one oriented spelling. A bare-number
# lookup is orientation-ambiguous for these; callers must pass a space group.
ORIENTATION_AMBIGUOUS_PG: frozenset[int] = frozenset({14, 18, 19, 20, 26})
# Fundamental sectors for the rotated settings (absent from the generated table).
# Verified with scripts/verify_pg_ops.py to tile the sphere exactly once.
_ORIENTED_FS: dict[str, NDArray[np.float64]] = {
    "312": np.array([0.0, 0.0, 1.0, -0.5, 0.8660254038, 0.0,
                     0.5, 0.8660254038, 0.0], dtype=np.float64),
    "31m": np.array([0.0, 1.0, 0.0, 0.8660254038, -0.5, 0.0], dtype=np.float64),
    "-31m": np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                      0.8660254038, -0.5, 0.0], dtype=np.float64),
    "-4m2": np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                     dtype=np.float64),
    "-62m": np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.8660254038, -0.5, 0.0],
                     dtype=np.float64),
}


def _default_pg_symbol(pg_num: int) -> str:
    """Internal number→default-spelling table lookup (orientation-blind)."""
    if pg_num < 1 or pg_num > len(PG_NUM_TO_SYMBOL):
        raise ValueError(f"pg_num {pg_num} out of range")
    return PG_NUM_TO_SYMBOL[pg_num - 1]


def folding_symbol(pg_num: int, space_group: int) -> str:
    """Point-group symbol used for Lambert folding, orientation included.

    ``space_group`` is required: several point-group numbers occur in two
    IT orientations (e.g. trigonal 321-vs-312, tetragonal -42m-vs-4m2), and a
    bare-number lookup silently picks one.
    """
    override = _SG_ORIENTED_SYMBOL.get(int(space_group))
    if override is not None:
        return override
    return _default_pg_symbol(pg_num)


def resolve_oriented_symbol(
    pg_num: int,
    *,
    pg_symbol: str | None = None,
    space_group: int | None = None,
) -> str:
    """Resolve an oriented folding symbol from stored metadata fields.

    Prefers an explicit ``pg_symbol``, then derives via ``space_group``. For
    orientation-ambiguous point groups, a missing space group raises rather
    than guessing.
    """
    if pg_symbol:
        return str(pg_symbol)
    if space_group is not None:
        return folding_symbol(int(pg_num), int(space_group))
    if int(pg_num) in ORIENTATION_AMBIGUOUS_PG:
        raise ValueError(
            f"Point group {pg_num} has orientation-ambiguous spellings "
            f"(e.g. 321 vs 312); cannot recover the folding symbol without a "
            f"space_group or stored pg_symbol. Re-run the simulation or load a "
            f"file that records space_group / pg_symbol."
        )
    return _default_pg_symbol(int(pg_num))


def _sym_key(symbol: str) -> str:
    return symbol.replace(" ", "")


def point_group_operators(symbol: str) -> NDArray[np.float64]:
    key = _sym_key(symbol)
    key = _MONOCLINIC_UNIQUE_B_OPS.get(key, key)
    data = PG_OPERATORS.get(key)
    if data is None:
        raise ValueError(f"Unknown PG symbol '{symbol}'")
    return data.copy()


def fs_normals(symbol: str) -> NDArray[np.float64]:
    key = _sym_key(symbol)
    if key in _MONOCLINIC_UNIQUE_B_FS:
        return _MONOCLINIC_UNIQUE_B_FS[key].copy()
    if key in _ORIENTED_FS:
        return _ORIENTED_FS[key].copy()
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
