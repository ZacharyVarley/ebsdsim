"""Space-group operators and orbit expansion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ebsdsim._sg_ops_data import SG_OP_DATA, SG_OP_OFFSETS, SG_PG_FOR_SG

Vec3 = tuple[float, float, float]


def sg_operators(sg: int) -> NDArray[np.float64]:
    if sg < 1 or sg > 230:
        raise ValueError(f"Space group {sg} out of range")
    beg = int(SG_OP_OFFSETS[sg - 1])
    end = int(SG_OP_OFFSETS[sg])
    return SG_OP_DATA[beg * 12 : end * 12].copy()


def pg_from_sg(sg: int) -> int:
    if sg < 1 or sg > 230:
        raise ValueError(f"Space group {sg} out of range")
    return int(SG_PG_FOR_SG[sg - 1])


def _wrap_mod1(v: float, eps: float) -> float:
    x = v
    if abs(x) < eps:
        x = 0.0
    x = ((x % 1.0) + 1.0) % 1.0
    if abs(x - 1.0) < eps:
        x = 0.0
    if abs(x) < eps:
        x = 0.0
    return x


def expand_sg_orbit(sg: int, pos: Vec3, eps: float = 1e-4) -> list[Vec3]:
    ops = sg_operators(sg)
    n_op = ops.size // 12
    out: list[Vec3] = []
    for k in range(n_op):
        o = k * 12
        x = _wrap_mod1(
            ops[o + 0] * pos[0] + ops[o + 1] * pos[1] + ops[o + 2] * pos[2] + ops[o + 3],
            eps,
        )
        y = _wrap_mod1(
            ops[o + 4] * pos[0] + ops[o + 5] * pos[1] + ops[o + 6] * pos[2] + ops[o + 7],
            eps,
        )
        z = _wrap_mod1(
            ops[o + 8] * pos[0] + ops[o + 9] * pos[1] + ops[o + 10] * pos[2] + ops[o + 11],
            eps,
        )
        dup = any(
            abs(p[0] - x) < eps and abs(p[1] - y) < eps and abs(p[2] - z) < eps for p in out
        )
        if not dup:
            out.append((x, y, z))
    return out
