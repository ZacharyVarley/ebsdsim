"""Build simulation cells from parsed CIF crystals."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ebsdsim.cif import CIFCrystal
from ebsdsim.elements import atomic_weight
from ebsdsim.spacegroup import expand_sg_orbit, pg_from_sg
from ebsdsim.types import Cell, LatticeCentering

Vec3 = tuple[float, float, float]

_DEG = math.pi / 180
_ANGSTROM_TO_NM = 0.1
_ANGSTROM_SQ_TO_NM_SQ = 0.01
_UNIQUE_POSITION_EPS = 1e-4

_SG_CENTERING: list[LatticeCentering] = ["P"] * 231


def _init_sg_centering() -> None:
    def set_centering(centering: LatticeCentering, nums: list[int]) -> None:
        for n in nums:
            _SG_CENTERING[n] = centering

    set_centering("C", [5, 8, 9, 12, 15])
    set_centering("C", [20, 21, 35, 36, 37, 38, 39, 40, 41, 63, 64, 65, 66, 67, 68])
    set_centering("F", [22, 42, 43, 69, 70])
    set_centering("I", [23, 24, 44, 45, 46, 71, 72, 73, 74])
    set_centering("I", [79, 80, 82, 87, 88, 97, 98, 107, 108, 109, 110, 119, 120, 121, 122, 139, 140, 141, 142])
    set_centering("R", [146, 148, 155, 160, 161, 166, 167])
    set_centering("F", [196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228])
    set_centering("I", [197, 199, 204, 206, 211, 214, 217, 220, 229, 230])


_init_sg_centering()


def centering_from_sg_num(sg_num: int | None) -> LatticeCentering | None:
    if sg_num is None or sg_num < 1 or sg_num > 230:
        return None
    return _SG_CENTERING[sg_num]


def infer_centering(hm_symbol: str | None, sg_num: int | None = None) -> LatticeCentering:
    first = (hm_symbol or "").strip().strip("'\"")[:1].upper()
    if first in ("F", "I", "A", "B", "C", "R"):
        return first  # type: ignore[return-value]
    if first == "P":
        return "P"
    from_sg = centering_from_sg_num(sg_num)
    if from_sg:
        return from_sg
    return "P"


def _inv3(m: NDArray[np.float64]) -> NDArray[np.float64]:
    a = m.reshape(3, 3)
    det = np.linalg.det(a)
    if abs(det) <= 1e-20:
        raise ValueError("Cell metric is singular")
    return np.linalg.inv(a).ravel()


def _mod1(value: float) -> float:
    wrapped = ((value % 1.0) + 1.0) % 1.0
    return 0.0 if abs(wrapped) < 1e-6 else wrapped


def _parse_number(raw: str) -> float:
    if "/" in raw:
        num, den = raw.split("/", 1)
        return float(num) / float(den)
    return float(raw)


def _eval_sym_op_component(expr: str, pos: Vec3) -> float:
    cleaned = expr.replace(" ", "").replace("-", "+-")
    out = 0.0
    for term in cleaned.split("+"):
        if not term:
            continue
        if term == "x":
            out += pos[0]
        elif term == "-x":
            out -= pos[0]
        elif term == "y":
            out += pos[1]
        elif term == "-y":
            out -= pos[1]
        elif term == "z":
            out += pos[2]
        elif term == "-z":
            out -= pos[2]
        else:
            out += _parse_number(term)
    return _mod1(out)


def _apply_sym_op(op: str, pos: Vec3) -> Vec3:
    parts = op.strip("'\"").split(",")
    if len(parts) != 3:
        raise ValueError(f"Unsupported CIF symmetry operator '{op}'")
    return (
        _eval_sym_op_component(parts[0], pos),
        _eval_sym_op_component(parts[1], pos),
        _eval_sym_op_component(parts[2], pos),
    )


def _unique_positions(positions: list[Vec3]) -> list[Vec3]:
    out: list[Vec3] = []
    for pos in positions:
        if not any(
            abs(p[0] - pos[0]) < _UNIQUE_POSITION_EPS
            and abs(p[1] - pos[1]) < _UNIQUE_POSITION_EPS
            and abs(p[2] - pos[2]) < _UNIQUE_POSITION_EPS
            for p in out
        ):
            out.append(pos)
    return out


def build_cell_from_cif(crystal: CIFCrystal) -> Cell:
    a = crystal.a * _ANGSTROM_TO_NM
    b = crystal.b * _ANGSTROM_TO_NM
    c = crystal.c * _ANGSTROM_TO_NM
    alpha = crystal.alpha * _DEG
    beta = crystal.beta * _DEG
    gamma = crystal.gamma * _DEG
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    direct = np.array([
        a * a, a * b * cg, a * c * cb,
        a * b * cg, b * b, b * c * ca,
        a * c * cb, b * c * ca, c * c,
    ], dtype=np.float64)
    volume = a * b * c * math.sqrt(max(1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg, 0))
    sg = math.sin(gamma)
    direct_structure_matrix = np.array([
        a, b * cg, c * cb,
        0, b * sg, -c * (cb * cg - ca) / sg,
        0, 0, volume / (a * b * sg),
    ], dtype=np.float64)

    atom_types = np.array([site.atomic_number for site in crystal.atom_sites], dtype=np.int32)
    atom_data = np.zeros((len(crystal.atom_sites), 5), dtype=np.float64)
    positions: list[list[Vec3]] = []
    use_sg_ops = isinstance(crystal.space_group, int) and 1 <= crystal.space_group <= 230

    for i, site in enumerate(crystal.atom_sites):
        atom_data[i, 0:3] = site.fract
        atom_data[i, 3] = site.occupancy
        atom_data[i, 4] = site.b_iso * _ANGSTROM_SQ_TO_NM_SQ
        if use_sg_ops:
            orbit = expand_sg_orbit(crystal.space_group, site.fract)
        else:
            ops = crystal.sym_ops if crystal.sym_ops else ["x,y,z"]
            orbit = _unique_positions([_apply_sym_op(op, site.fract) for op in ops])
        positions.append(orbit)

    multiplicities = np.array([len(p) for p in positions], dtype=np.int32)

    m_sum = 0.0
    z_num = 0.0
    a_num = 0.0
    a_mass_sum = 0.0
    for i in range(atom_types.size):
        occ = atom_data[i, 3]
        mult = multiplicities[i]
        aw = atomic_weight(int(atom_types[i]))
        m_sum += mult
        z_num += atom_types[i] * occ * mult
        a_num += aw * occ * mult
        a_mass_sum += aw * occ * mult

    average_atomic_number = z_num / (atom_types.size * m_sum) if m_sum > 0 else 0.0
    average_atomic_weight = a_num / (atom_types.size * m_sum) if m_sum > 0 else 0.0
    density = a_mass_sum / (volume * 6.02214076e2)

    pg_num = (
        pg_from_sg(crystal.space_group)
        if isinstance(crystal.space_group, int) and 1 <= crystal.space_group <= 230
        else None
    )

    return Cell(
        a=a,
        b=b,
        c=c,
        alpha=crystal.alpha,
        beta=crystal.beta,
        gamma=crystal.gamma,
        volume=volume,
        direct_structure_matrix=direct_structure_matrix,
        reciprocal_metric=_inv3(direct),
        atom_types=atom_types,
        atom_data=atom_data,
        positions=positions,
        multiplicities=multiplicities,
        space_group=crystal.space_group,
        pg_num=pg_num,
        density=density,
        average_atomic_number=average_atomic_number,
        average_atomic_weight=average_atomic_weight,
        lattice_centering=infer_centering(crystal.hm_symbol, crystal.space_group),
    )


def metric_to_float32(cell: Cell) -> NDArray[np.float32]:
    return cell.reciprocal_metric.astype(np.float32)
