"""First-principles verification of PG operators and fundamental sectors.

Two independent checks:

1. Internal consistency (per symbol in PG_OPERATORS):
   - operators are a closed group (order n),
   - the FS_NORMALS half-space cuts define an exact fundamental domain for that
     group acting on the sphere (every orbit lands in the FS exactly once, and
     solid_angle(FS) * n == 4*pi).

2. Convention match (per pg_num 1..32 as production selects it):
   - the symbol chosen by pg_num_to_symbol + the monoclinic remap yields ops of
     the expected order and Laue class.

Run: python -m scripts.verify_pg_ops
"""

from __future__ import annotations

import numpy as np

from ebsdsim._pg_ops_data import FS_NORMALS, PG_NUM_TO_SYMBOL, PG_OPERATORS
from ebsdsim.pg_ops import fs_normals, point_group_operators

RNG = np.random.default_rng(0)
EPS = 1e-9


def _mats(flat: np.ndarray) -> np.ndarray:
    return flat.reshape(-1, 3, 3)


def _is_closed_group(mats: np.ndarray, tol: float = 1e-6) -> tuple[bool, int]:
    n = mats.shape[0]
    prod = np.einsum("aij,bjk->abik", mats, mats).reshape(-1, 3, 3)
    # every product must equal some existing element
    diff = np.abs(prod[:, None, :, :] - mats[None, :, :, :]).sum(axis=(2, 3))
    matched = (diff < tol).any(axis=1)
    has_identity = (np.abs(mats - np.eye(3)).sum(axis=(1, 2)) < tol).any()
    unique = _count_unique(mats, tol)
    return bool(matched.all() and has_identity and unique == n), n


def _count_unique(mats: np.ndarray, tol: float = 1e-6) -> int:
    n = mats.shape[0]
    d = np.abs(mats[:, None] - mats[None]).sum(axis=(2, 3))
    seen = np.zeros(n, dtype=bool)
    count = 0
    for i in range(n):
        if seen[i]:
            continue
        count += 1
        seen |= d[i] < tol
    return count


def _in_fs(V: np.ndarray, normals: np.ndarray, eps: float) -> np.ndarray:
    if normals.size == 0:
        return np.ones(V.shape[0], dtype=bool)
    N = normals.reshape(-1, 3)
    return (V @ N.T >= -eps).all(axis=1)


def _sample_sphere(n: int) -> np.ndarray:
    v = RNG.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def verify_pairing(
    label: str, mats: np.ndarray, normals: np.ndarray,
    n_samp: int = 300000, eps: float = 1e-6,
) -> dict:
    closed, n = _is_closed_group(mats)
    V = _sample_sphere(n_samp)
    orb = np.einsum("gij,nj->ngi", mats, V)
    if normals.size:
        innorm = normals.reshape(-1, 3)
        inmask = (np.einsum("ngi,mi->ngm", orb, innorm) >= -eps).all(axis=2)
    else:
        inmask = np.ones(orb.shape[:2], dtype=bool)
    cnt = inmask.sum(axis=1)
    return {
        "label": label,
        "n": n,
        "closed": closed,
        "min_orbit_in_fs": int(cnt.min()),
        "max_orbit_in_fs": int(cnt.max()),
        "mean_orbit_in_fs": float(cnt.mean()),
        "gaps": int((cnt == 0).sum()),
    }


def main() -> None:
    print("=== Production pairing per pg_num: ops(pg) + FS(pg) form a valid "
          "fundamental domain of the sphere? ===")
    print(f"{'pg':>3} {'symbol':8} {'n':>3} {'closed':>6} {'gaps':>5} "
          f"{'minFS':>5} {'maxFS':>5} {'meanFS':>7}  verdict")
    bad = []
    for pg in range(1, len(PG_NUM_TO_SYMBOL) + 1):
        sym = PG_NUM_TO_SYMBOL[pg - 1]
        mats = _mats(point_group_operators(sym))
        normals = fs_normals(sym)
        r = verify_pairing(f"pg{pg}:{sym}", mats, normals)
        ok = (
            r["closed"]
            and r["gaps"] == 0
            and r["min_orbit_in_fs"] >= 1
            and abs(r["mean_orbit_in_fs"] - 1.0) < 1e-3
        )
        if not ok:
            bad.append(f"{pg}:{sym}")
        print(f"{pg:3d} {sym:8} {r['n']:3d} {str(r['closed']):>6} "
              f"{r['gaps']:5d} {r['min_orbit_in_fs']:5d} "
              f"{r['max_orbit_in_fs']:5d} {r['mean_orbit_in_fs']:7.4f}  "
              f"{'OK' if ok else 'FAIL'}")

    if bad:
        print(f"\nPRODUCTION FAILURES: {bad}")
    else:
        print("\nAll 32 production pairings are valid fundamental domains "
              "(mean orbit occupancy = 1.0000, no gaps, closed groups).")

    print("\n=== Convention match: folding PG vs real crystal symmetry, "
          "all 230 space groups ===")
    from ebsdsim.spacegroup import verify_folding_matches_crystal
    from ebsdsim.structure import _lattice_arrays

    def system(sg: int) -> str:
        for hi, name in ((2, "tri"), (15, "mono"), (74, "ortho"), (142, "tetra"),
                         (167, "trig"), (194, "hex"), (230, "cubic")):
            if sg <= hi:
                return name
        return "cubic"

    params = {
        "tri": (5.0, 6.0, 7.0, 80.0, 85.0, 95.0),
        "mono": (5.0, 6.0, 7.0, 90.0, 97.0, 90.0),
        "ortho": (5.0, 6.0, 7.0, 90.0, 90.0, 90.0),
        "tetra": (5.0, 5.0, 7.0, 90.0, 90.0, 90.0),
        "trig": (5.0, 5.0, 7.0, 90.0, 90.0, 120.0),
        "hex": (5.0, 5.0, 7.0, 90.0, 90.0, 120.0),
        "cubic": (5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
    }
    sg_fail = []
    for sg in range(1, 231):
        *_, dsm = _lattice_arrays(*params[system(sg)])
        try:
            verify_folding_matches_crystal(sg, dsm)
        except ValueError as exc:
            sg_fail.append((sg, str(exc)[:100]))
    if sg_fail:
        print(f"CONVENTION FAILURES ({len(sg_fail)}):")
        for sg, msg in sg_fail:
            print(f"  SG {sg}: {msg}")
    else:
        print("All 230 space groups: folding PG matches crystal Cartesian PG.")


if __name__ == "__main__":
    main()
