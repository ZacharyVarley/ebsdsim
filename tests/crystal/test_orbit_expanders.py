"""Orbit expansion: canonical order, group invariants, spglib oracle."""

from __future__ import annotations

import numpy as np
import pytest
from ebsdsim.crystal.material import Atom, Cell, Material
from ebsdsim.crystal.pointgroup import folding_symbol, point_group_operators
from ebsdsim.crystal.reader import STD_HALL, hall_ops
from ebsdsim.crystal.reader.sym import IDENTITY, close_group, lattice_letter, opset
from ebsdsim.crystal.spacegroup import expand_orbit_with_ops, ops_from_hall, pg_from_sg

_EPS = 1e-4
_CENTERING_MULT = {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "F": 4, "R": 3}

# Independent IT Wyckoff multiplicity anchors (not derived from this Hall path).
_IT_WYCKOFF_ANCHORS = (
    (14, (0.0, 0.0, 0.0), 2),  # P2_1/c 2a
    (62, (0.0, 0.0, 0.0), 4),  # Pnma 4a
    (166, (0.0, 0.0, 0.0), 3),  # R-3m (hex) 3a
    (186, (0.0, 0.0, 0.0), 2),  # P6_3mc 2a
    (194, (0.0, 0.0, 0.0), 2),  # P6_3/mmc 2a
    (194, (1 / 3, 2 / 3, 0.25), 2),  # P6_3/mmc 2c
    (225, (0.0, 0.0, 0.0), 4),  # Fm-3m 4a
    (225, (0.25, 0.25, 0.25), 8),  # Fm-3m 8c
    (227, (0.125, 0.125, 0.125), 8),  # Fd-3m 8a (legacy SG_OP_DATA wrongly gave 16)
    (229, (0.0, 0.0, 0.0), 2),  # Im-3m 2a
)


def _orbit_keys(orbit: list[tuple[float, float, float]]) -> set[tuple[int, int, int]]:
    arr = np.asarray(orbit, dtype=np.float64).reshape(-1, 3)
    if arr.size == 0:
        return set()
    keyed = np.round(arr / _EPS).astype(np.int64)
    return {tuple(int(v) for v in row) for row in keyed}


def _expected_group_order(sg: int) -> int:
    """IT general-position multiplicity = |point group| × centering translations."""
    letter = lattice_letter(hall_ops(STD_HALL[sg]))
    n_pg = point_group_operators(folding_symbol(pg_from_sg(sg), sg)).size // 9
    return int(n_pg * _CENTERING_MULT[letter])


def test_diamond_fd3m_227_eighth_site_multiplicity_is_8():
    """SG 227 diamond 8a site (1/8,1/8,1/8): Hall/IT multiplicity is 8, not 16."""
    material = Material(
        cell=Cell(a=5.43, b=5.43, c=5.43, space_group=227),
        atoms=[Atom("C", 0.125, 0.125, 0.125)],
    )
    cell = material.to_simulation_cell()
    assert int(cell.multiplicities[0]) == 8
    assert len(cell.positions[0]) == 8
    hall = expand_orbit_with_ops(ops_from_hall(227), (0.125, 0.125, 0.125))
    assert len(hall) == 8


@pytest.mark.parametrize("sg,pos,expected", _IT_WYCKOFF_ANCHORS)
def test_it_wyckoff_multiplicity_anchors(sg: int, pos: tuple[float, float, float], expected: int):
    """Hand-checked IT multiplicities — independent of this codebase's expander."""
    n = len(expand_orbit_with_ops(ops_from_hall(sg), pos))
    assert n == expected, f"SG {sg} at {pos}: expected {expected}, got {n}"


def test_hall_operator_group_invariants_all_230():
    """Every SG: identity, closure, IT order, general-position orbit == |G|.

    Closure / identity use the exact integer Hall ``Op`` algebra (1/24 lattice),
    not float packed matrices — hexagonal √3 floats would false-fail otherwise.
    Expected |G| is |point group| × centering (IT), independent of orbit code.
    """
    gen = (0.1234, 0.5678, 0.9012)
    failures: list[str] = []
    for sg in range(1, 231):
        ops_int = hall_ops(STD_HALL[sg])
        n = len(ops_int)
        expect = _expected_group_order(sg)
        if n != expect:
            failures.append(f"SG {sg}: |G|={n} != expected {expect}")
            continue
        if IDENTITY not in ops_int and not any(op == IDENTITY for op in ops_int):
            # hall_ops always includes identity via close_group; still assert.
            failures.append(f"SG {sg}: missing identity")
            continue
        closed = close_group(ops_int)
        if opset(closed) != opset(ops_int) or len(closed) != n:
            failures.append(f"SG {sg}: Hall ops not a closed group of order {n}")
            continue
        packed = ops_from_hall(sg)
        if packed.shape[0] != n:
            failures.append(f"SG {sg}: ops_from_hall length {packed.shape[0]} != {n}")
            continue
        orbit = expand_orbit_with_ops(packed, gen)
        if len(orbit) != n:
            failures.append(f"SG {sg}: general orbit {len(orbit)} != |G| {n}")
    assert not failures, "\n".join(failures[:20])


def test_canonical_orbit_order_independent_of_op_permutation():
    """Lexicographic site order must not depend on operator listing order."""
    rng = np.random.default_rng(0)
    positions = (
        (0.12, 0.34, 0.56),
        (0.0, 0.0, 0.0),
        (0.25, 0.25, 0.25),
        (1 / 3, 2 / 3, 0.1),
    )
    for sg in (14, 62, 166, 186, 194, 216, 225, 227, 229):
        ops = ops_from_hall(sg)
        for pos in positions:
            base = expand_orbit_with_ops(ops, pos)
            for _ in range(3):
                perm = rng.permutation(ops.shape[0])
                shuffled = expand_orbit_with_ops(ops[perm], pos)
                assert shuffled == base, f"SG {sg} pos={pos}: order changed under op shuffle"
                assert _orbit_keys(shuffled) == _orbit_keys(base)


def test_canonical_sort_preserves_orbit_sets_all_230():
    """Canonical order must not change multiplicities or site sets for any SG."""
    rng = np.random.default_rng(1)
    seeds = np.array(
        [
            [0.1234, 0.5678, 0.9012],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.125, 0.125, 0.125],
            [1 / 3, 2 / 3, 0.0],
        ],
        dtype=np.float64,
    )
    failures: list[str] = []
    for sg in range(1, 231):
        ops = ops_from_hall(sg)
        # Reference set from a single expansion (post-canonical).
        ref_sets = [
            _orbit_keys(expand_orbit_with_ops(ops, (float(s[0]), float(s[1]), float(s[2]))))
            for s in seeds
        ]
        for _ in range(2):
            ops_shuf = ops[rng.permutation(ops.shape[0])]
            for seed, ref in zip(seeds, ref_sets):
                pos = (float(seed[0]), float(seed[1]), float(seed[2]))
                got = expand_orbit_with_ops(ops_shuf, pos)
                keys = _orbit_keys(got)
                if len(got) != len(ref) or keys != ref:
                    failures.append(
                        f"SG {sg} pos={pos}: mult {len(got)}/{len(ref)} or set mismatch"
                    )
                    break
            else:
                continue
            break
    assert not failures, "\n".join(failures[:20])


def test_material_and_structure_paths_still_share_expander():
    """Smoke: Material and Structure builders still produce identical cells."""
    from ebsdsim.crystal.build import build_cell_from_structure
    from ebsdsim.crystal.material import structure_from_material

    for sg in (14, 186, 225, 227):
        material = Material(
            cell=Cell(a=5.0, b=5.0, c=5.0, space_group=sg),
            atoms=[
                Atom("Si", 0.125, 0.125, 0.125, occupancy=0.9, b_iso=0.4),
                Atom("C", 0.0, 0.5, 0.25, occupancy=1.0, b_iso=0.3),
            ],
            name=f"sg{sg}",
        )
        from_material = material.to_simulation_cell()
        structure = structure_from_material(
            a=5.0,
            b=5.0,
            c=5.0,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            space_group=sg,
            atoms=material.atoms,
            name=material.name,
        )
        from_structure = build_cell_from_structure(structure)
        np.testing.assert_array_equal(from_material.multiplicities, from_structure.multiplicities)
        assert from_material.positions == from_structure.positions


def _spglib_ops_keys(sym) -> set[tuple]:
    from ebsdsim.crystal.reader.sym import DEN

    keys = set()
    for R, t in zip(sym["rotations"], sym["translations"]):
        tt = tuple(int(v) for v in (np.round(np.asarray(t) * DEN).astype(np.int64) % DEN))
        rr = tuple(int(v) for v in np.asarray(R, dtype=np.int64).ravel())
        keys.add((rr, tt))
    return keys


def _spglib_hall_number(sg: int) -> int:
    """Match ebsdsim STD_HALL setting: exact Hall symbol, else identical op set."""
    spglib = pytest.importorskip("spglib")
    target = STD_HALL[sg]
    ours = opset(hall_ops(target))
    symbol_hit = None
    set_hit = None
    for h in range(1, 531):
        info = spglib.get_spacegroup_type(h)
        number = int(getattr(info, "number", info["number"]))
        if number != sg:
            continue
        hall = getattr(info, "hall_symbol", None) or info["hall_symbol"]
        if hall == target:
            symbol_hit = int(h)
            break
        sym = spglib.get_symmetry_from_database(h)
        if _spglib_ops_keys(sym) == ours:
            set_hit = int(h)
    if symbol_hit is not None:
        return symbol_hit
    if set_hit is not None:
        return set_hit
    raise AssertionError(f"No spglib hall_number for SG {sg} Hall {target!r}")


def _spglib_orbit(sg: int, pos: tuple[float, float, float]) -> set[tuple[int, int, int]]:
    spglib = pytest.importorskip("spglib")
    hn = _spglib_hall_number(sg)
    sym = spglib.get_symmetry_from_database(hn)
    rot = np.asarray(sym["rotations"], dtype=np.float64)
    trans = np.asarray(sym["translations"], dtype=np.float64)
    p = np.asarray(pos, dtype=np.float64)
    xyz = np.einsum("nij,j->ni", rot, p) + trans
    xyz = np.mod(xyz, 1.0)
    near1 = np.abs(xyz - 1.0) < _EPS
    xyz = np.where(near1, 0.0, xyz)
    xyz = np.where(np.abs(xyz) < _EPS, 0.0, xyz)
    keys = np.round(xyz / _EPS).astype(np.int64)
    return {tuple(int(v) for v in row) for row in keys}


def test_hall_orbits_match_spglib_all_230():
    """Independent oracle: Hall orbits match spglib ops for matching Hall setting."""
    pytest.importorskip("spglib")
    seeds = (
        (0.1234, 0.5678, 0.9012),  # general
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.5, 0.5),
        (0.25, 0.25, 0.25),
        (0.125, 0.125, 0.125),
        (1 / 3, 2 / 3, 0.0),
    )
    failures: list[str] = []
    for sg in range(1, 231):
        ops = ops_from_hall(sg)
        for pos in seeds:
            ours = _orbit_keys(expand_orbit_with_ops(ops, pos))
            theirs = _spglib_orbit(sg, pos)
            if ours != theirs:
                failures.append(
                    f"SG {sg} pos={pos}: mult ours={len(ours)} spglib={len(theirs)} "
                    f"symmetric_diff={len(ours ^ theirs)}"
                )
                break
    assert not failures, "\n".join(failures[:25])
