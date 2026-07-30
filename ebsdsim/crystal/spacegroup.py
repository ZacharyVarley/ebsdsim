"""Crystal layer: space-group operators and orbit expansion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ebsdsim.crystal._generated.sg_point_groups import SG_PG_FOR_SG
from ebsdsim.crystal.pointgroup import folding_symbol, point_group_operators
from ebsdsim.crystal.reader import STD_HALL, hall_ops

Vec3 = tuple[float, float, float]

_DEN = 24.0


def require_space_group(sg: int) -> int:
    """Validate an International Tables space-group number in ``[1, 230]``."""
    if not isinstance(sg, int) or sg < 1 or sg > 230:
        raise ValueError(f"space_group must be in [1, 230], got {sg}")
    return sg


def pg_from_sg(sg: int) -> int:
    require_space_group(sg)
    return int(SG_PG_FOR_SG[sg - 1])


def ops_from_hall(sg: int) -> NDArray[np.float64]:
    """IT-standard Hall operators as ``(n_op, 12)`` float row-major ``W | t``.

    Translations are ``w / 24`` (Hall denominator). Dual-origin groups use
    origin choice 2.
    """
    require_space_group(sg)
    ops = hall_ops(STD_HALL[sg])
    w_rot = np.stack([op.W for op in ops], axis=0).astype(np.float64)
    t = np.stack([op.w for op in ops], axis=0).astype(np.float64) / _DEN
    mats = np.concatenate([w_rot, t[:, :, None]], axis=2)
    return mats.reshape(len(ops), 12)


def cartesian_point_group_from_sg(
    sg: int, direct_structure_matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Crystal point group in the Cartesian frame of the given cell.

    Conjugates the IT-standard space-group rotation parts by the direct
    structure matrix (columns = lattice vectors in Cartesian), then dedups.
    Returns ``(m, 3, 3)`` rotation matrices (proper and improper).
    """
    flat = ops_from_hall(sg).reshape(-1, 3, 4)
    w_rot = flat[:, :, :3]
    m = np.asarray(direct_structure_matrix, dtype=np.float64).reshape(3, 3)
    m_inv = np.linalg.inv(m)
    r_cart = np.einsum("ij,njk,kl->nil", m, w_rot, m_inv)
    return _dedup_matrices(r_cart)


def _matrix_key_set(mats: NDArray[np.float64], decimals: int = 3) -> set[bytes]:
    rounded = np.round(np.asarray(mats, dtype=np.float64), decimals) + 0.0
    return {row.tobytes() for row in rounded.reshape(-1, 9)}


def verify_folding_matches_crystal(
    space_group: int, direct_structure_matrix: NDArray[np.float64]
) -> str:
    """Check the Lambert folding point group matches the crystal's real symmetry.

    Derives the crystal's Cartesian point group from its IT-standard space-group
    operators and compares it, as a set of matrices, to the operators the folding
    path selects (``folding_symbol`` -> ``point_group_operators``). Raises
    ``ValueError`` on any mismatch. This is a verification helper used by the test
    suite to prove the selection is correct for all 230 space groups; it is not
    called per simulation. Returns the validated folding symbol.
    """
    pg_num = pg_from_sg(space_group)
    symbol = folding_symbol(pg_num, space_group)
    table = point_group_operators(symbol).reshape(-1, 3, 3)
    crystal = cartesian_point_group_from_sg(space_group, direct_structure_matrix)
    table_set = _matrix_key_set(table)
    crystal_set = _matrix_key_set(crystal)
    if table_set != crystal_set:
        n_missing = len(crystal_set - table_set)
        n_extra = len(table_set - crystal_set)
        raise ValueError(
            f"Folding point group '{symbol}' (pg {pg_num}) does not match the "
            f"Cartesian symmetry of space group {space_group}: "
            f"{n_missing} crystal op(s) absent from the folding set, "
            f"{n_extra} folding op(s) absent from the crystal. This indicates a "
            f"point-group orientation/convention mismatch."
        )
    return symbol


def _dedup_matrices(mats: NDArray[np.float64], tol: float = 1e-4) -> NDArray[np.float64]:
    mats = np.asarray(mats, dtype=np.float64)
    if mats.shape[0] == 0:
        return mats
    dist = np.abs(mats[:, None] - mats[None]).sum(axis=(2, 3))
    first_match = np.argmax(dist < tol, axis=1)
    keep = first_match == np.arange(mats.shape[0])
    return mats[keep]


def _wrap_frac_array(xyz: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    """Vectorized fractional wrap into ``[0, 1)`` with near-zero / near-one snap."""
    out = np.asarray(xyz, dtype=np.float64).copy()
    out[np.abs(out) < eps] = 0.0
    out = np.mod(out, 1.0)
    out[np.abs(out - 1.0) < eps] = 0.0
    out[np.abs(out) < eps] = 0.0
    return out


def expand_orbit_with_ops(
    ops: NDArray[np.float64],
    pos: Vec3 | NDArray[np.float64],
    eps: float = 1e-4,
) -> list[Vec3]:
    """Expand one site under packed ``(n_op, 12)`` or flat ``(n_op*12,)`` ops.

    Distinct images are returned in a **canonical** order independent of the
    operator enumeration: lexicographic by the integer grid
    ``round(coord / eps)``, with a float ``(x, y, z)`` lex tie-break so the
    representative of each duplicate key is unique. Callers that expand one
    input atom at a time keep per-atom grouping; only the order *within* each
    atom's orbit is fixed.
    """
    packed = np.asarray(ops, dtype=np.float64).reshape(-1, 12)
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    w = packed[:, [0, 1, 2, 4, 5, 6, 8, 9, 10]].reshape(-1, 3, 3)
    t = packed[:, [3, 7, 11]]
    xyz = _wrap_frac_array(np.einsum("nij,j->ni", w, p) + t, eps)
    if xyz.size == 0:
        return []
    keys = np.round(xyz / eps).astype(np.int64)
    # Canonical within-orbit order independent of operator enumeration:
    # sort by integer grid key (x,y,z), then by float (x,y,z) as a stable
    # tie-break so the representative of each duplicate key is unique.
    order = np.lexsort(
        (xyz[:, 2], xyz[:, 1], xyz[:, 0], keys[:, 2], keys[:, 1], keys[:, 0])
    )
    sorted_xyz = xyz[order]
    sorted_keys = keys[order]
    if sorted_keys.shape[0] == 1:
        uniq = sorted_xyz
    else:
        boundary = np.empty(sorted_keys.shape[0], dtype=bool)
        boundary[0] = True
        boundary[1:] = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
        uniq = sorted_xyz[boundary]
    return [(float(r[0]), float(r[1]), float(r[2])) for r in uniq]
