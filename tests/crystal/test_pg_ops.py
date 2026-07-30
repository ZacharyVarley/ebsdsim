"""Point-group operator convention tests."""

from __future__ import annotations

import numpy as np
import pytest
from ebsdsim.crystal._generated.pg_ops import PG_NUM_TO_SYMBOL
from ebsdsim.crystal.pointgroup import fs_normals, point_group_operators


def _is_closed_group(mats: np.ndarray, tol: float = 1e-6) -> bool:
    n = mats.shape[0]
    prod = np.einsum("aij,bjk->abik", mats, mats).reshape(-1, 3, 3)
    diff = np.abs(prod[:, None, :, :] - mats[None, :, :, :]).sum(axis=(2, 3))
    matched = (diff < tol).any(axis=1).all()
    has_identity = (np.abs(mats - np.eye(3)).sum(axis=(1, 2)) < tol).any()
    d = np.abs(mats[:, None] - mats[None]).sum(axis=(2, 3))
    n_unique = int((np.argmax(d < tol, axis=1) == np.arange(n)).sum())
    return bool(matched and has_identity and n_unique == n)


def _orbit_occupancy(mats: np.ndarray, normals: np.ndarray, n_samp: int, eps: float):
    rng = np.random.default_rng(1234)
    v = rng.normal(size=(n_samp, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    orb = np.einsum("gij,nj->ngi", mats, v)
    if normals.size:
        nrm = normals.reshape(-1, 3)
        inmask = (np.einsum("ngi,mi->ngm", orb, nrm) >= -eps).all(axis=2)
    else:
        inmask = np.ones(orb.shape[:2], dtype=bool)
    return inmask.sum(axis=1)


@pytest.mark.parametrize("pg", range(1, 33))
def test_production_pairing_is_valid_fundamental_domain(pg: int):
    """ops(pg) + fs_normals(pg) must tile the sphere exactly once.

    Guards against both broken FS cuts (gaps/overlaps) and non-group op sets.
    This is the check that catches monoclinic-2/m style errors for every group.
    """
    sym = PG_NUM_TO_SYMBOL[pg - 1]
    mats = point_group_operators(sym).reshape(-1, 3, 3)
    normals = fs_normals(sym)
    assert _is_closed_group(mats), f"pg {pg} ({sym}) operators are not a closed group"
    cnt = _orbit_occupancy(mats, normals, n_samp=40000, eps=1e-6)
    assert cnt.min() >= 1, f"pg {pg} ({sym}) fundamental sector has gaps"
    assert abs(float(cnt.mean()) - 1.0) < 1e-3, (
        f"pg {pg} ({sym}) sector over/under-covers the sphere "
        f"(mean orbit occupancy {cnt.mean():.4f})"
    )


def test_all_pg_nums_have_symbol_ops_and_fs():
    for pg in range(1, len(PG_NUM_TO_SYMBOL) + 1):
        sym = PG_NUM_TO_SYMBOL[pg - 1]
        assert point_group_operators(sym).size % 9 == 0
        fs_normals(sym)  # must not raise


def test_monoclinic_2m_fs_has_y_cut():
    """Unique-b 2/m sector must include a y-cut (m_y flips only y)."""
    normals = fs_normals("2/m").reshape(-1, 3)
    assert any(np.allclose(n, [0.0, 1.0, 0.0]) for n in normals)
    # an x-only cut would silently double-cover; make sure that regression stays fixed
    assert not (
        any(np.allclose(n, [1.0, 0.0, 0.0]) for n in normals)
        and not any(np.allclose(n, [0.0, 1.0, 0.0]) for n in normals)
    )


def _system(sg: int) -> str:
    if sg <= 2:
        return "tri"
    if sg <= 15:
        return "mono"
    if sg <= 74:
        return "ortho"
    if sg <= 142:
        return "tetra"
    if sg <= 167:
        return "trig"
    if sg <= 194:
        return "hex"
    return "cubic"


# Several valid metrics per crystal system: the folding point group must match
# the crystal for any cell shape, not just one representative.
_SYSTEM_PARAMS = {
    "tri": [(5.0, 6.0, 7.0, 80.0, 85.0, 95.0),
            (4.2, 9.1, 6.3, 71.0, 101.0, 113.0),
            (8.0, 3.5, 5.5, 88.0, 78.0, 99.0)],
    "mono": [(5.0, 6.0, 7.0, 90.0, 97.0, 90.0),
             (9.1, 4.3, 6.7, 90.0, 112.0, 90.0),
             (3.9, 8.8, 5.1, 90.0, 90.5, 90.0)],
    "ortho": [(5.0, 6.0, 7.0, 90.0, 90.0, 90.0),
              (9.1, 4.3, 6.7, 90.0, 90.0, 90.0),
              (3.9, 8.8, 12.1, 90.0, 90.0, 90.0)],
    "tetra": [(5.0, 5.0, 7.0, 90.0, 90.0, 90.0),
              (4.2, 4.2, 3.1, 90.0, 90.0, 90.0),
              (8.0, 8.0, 12.5, 90.0, 90.0, 90.0)],
    "trig": [(5.0, 5.0, 7.0, 90.0, 90.0, 120.0),
             (4.2, 4.2, 11.1, 90.0, 90.0, 120.0),
             (7.3, 7.3, 5.0, 90.0, 90.0, 120.0)],
    "hex": [(5.0, 5.0, 7.0, 90.0, 90.0, 120.0),
            (4.2, 4.2, 3.1, 90.0, 90.0, 120.0),
            (9.0, 9.0, 15.2, 90.0, 90.0, 120.0)],
    "cubic": [(5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
              (3.2, 3.2, 3.2, 90.0, 90.0, 90.0),
              (11.7, 11.7, 11.7, 90.0, 90.0, 90.0)],
}


def test_folding_symbol_matches_crystal_for_all_space_groups():
    """The folding point group must equal each crystal's real Cartesian symmetry.

    Exhaustive static check over all 230 space groups and several cell metrics per
    crystal system. Because this holds for every space group in the IT-standard
    setting ebsdsim simulates in (and origin choice cannot change rotation parts),
    the selection is correct without any per-simulation verification.
    """
    from ebsdsim.crystal.build import _lattice_arrays
    from ebsdsim.crystal.spacegroup import verify_folding_matches_crystal

    failures = []
    for sg in range(1, 231):
        for params in _SYSTEM_PARAMS[_system(sg)]:
            *_, dsm = _lattice_arrays(*params)
            try:
                verify_folding_matches_crystal(sg, dsm)
            except ValueError as exc:  # pragma: no cover - failure path
                failures.append((sg, params, str(exc)))
    assert not failures, f"folding/crystal mismatch for {[f[0] for f in failures]}"


def test_resolve_oriented_symbol_refuses_ambiguous_guess():
    from ebsdsim.crystal.pointgroup import resolve_oriented_symbol

    assert resolve_oriented_symbol(18, space_group=149) == "312"
    assert resolve_oriented_symbol(18, space_group=150) == "32"
    assert resolve_oriented_symbol(18, pg_symbol="312") == "312"
    with pytest.raises(ValueError, match="orientation-ambiguous"):
        resolve_oriented_symbol(18)
    # Unambiguous PG may fall back to the number table.
    assert resolve_oriented_symbol(32) == "m-3m"


def test_monoclinic_2_is_unique_b():
    """Short symbol '2' must be 2 || b (IT), not 2 || c."""
    ops = point_group_operators("2").reshape(-1, 3, 3)
    assert ops.shape[0] == 2
    two_y = np.diag([-1.0, 1.0, -1.0])
    assert any(np.allclose(op, two_y) for op in ops)
    assert not any(np.allclose(op, np.diag([-1.0, -1.0, 1.0])) for op in ops)
    # FS cut is the yz plane so 2_y pairs NH with SH instead of spinning NH.
    normals = fs_normals("2").reshape(-1, 3)
    assert normals.shape == (1, 3)
    assert np.allclose(normals[0], [1.0, 0.0, 0.0])


def test_monoclinic_m_and_2m_unique_b():
    assert np.allclose(
        point_group_operators("m").reshape(-1, 3, 3)[0],
        np.diag([1.0, -1.0, 1.0]),
    )
    ops = point_group_operators("2/m").reshape(-1, 3, 3)
    assert any(np.allclose(op, np.diag([-1.0, 1.0, -1.0])) for op in ops)
    assert any(np.allclose(op, np.diag([1.0, -1.0, 1.0])) for op in ops)
