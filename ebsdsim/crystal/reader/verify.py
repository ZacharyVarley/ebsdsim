"""
Verification harness.  Run this once, in a dev environment, whenever you touch
tables.STD_HALL.  It is not imported at runtime.

    python -m ebsdsim.crystal.reader.verify

Checks, in increasing order of strength:

  1. every STD_HALL entry parses and closes to a finite group
  2. the inferred Bravais letter matches the H-M name in tables.SGNAME
  3. the derived glide/screw letters are consistent with SGNAME
     (monoclinic + orthorhombic, i.e. 72 of the 230 -- and the 72 where a
     transcription slip is most likely)
  4. round trip: scramble each standard group by a known (P, p) and confirm
     setting.find_setting recovers a basis change that reproduces it
  5. OPTIONAL cross-check against spglib and/or cctbx if importable

Checks 1-4 need nothing but numpy.  Check 5 is the only genuinely independent
test of the table's *content* -- 1-4 can all pass on a table that is
self-consistent but wrong.  Install spglib once in a scratch venv, run this,
and then never depend on it again.
"""
from __future__ import annotations

import sys

import numpy as np

from .setting import find_setting, transform_ops
from .sym import (
    DEN,
    I3,
    SGNAME,
    STD_HALL,
    Op,
    SymmetryError,
    close_group,
    hall_ops,
    lattice_letter,
    opset,
)


def std_ops(n):
    return hall_ops(STD_HALL[n])


# ---------------------------------------------------------------------------
# symbol derivation
# ---------------------------------------------------------------------------

def _glide_letter(t):
    t = tuple(int(v) % DEN for v in t)
    nz = [i for i, v in enumerate(t) if v]
    if not nz:
        return "m"
    if all(t[i] == DEN // 2 for i in nz):
        return "abc"[nz[0]] if len(nz) == 1 else "n"
    if all(t[i] in (DEN // 4, 3 * DEN // 4) for i in nz):
        return "d"
    return "?"


def _mirror_letters(ops, axis):
    """All glide letters realised by mirrors perpendicular to `axis` (0=a,1=b,2=c)."""
    W = np.diag([1, 1, 1]).astype(np.int64)
    W[axis, axis] = -1
    out = set()
    for o in ops:
        if np.array_equal(o.W, W):
            out.add(_glide_letter(o.intrinsic()))
    return out


def _rotation_letters(ops, axis):
    """All 2 / 2_1 symbols realised by 2-folds along `axis`."""
    W = np.diag([-1, -1, -1]).astype(np.int64)
    W[axis, axis] = 1
    out = set()
    for o in ops:
        if np.array_equal(o.W, W):
            t = o.intrinsic()
            out.add("21" if int(t[axis]) % DEN else "2")
    return out


def derived_ortho_positions(ops):
    """
    For each of the three axes, the set of H-M symbols IT could legitimately
    have put in that position.

    A set, not a single letter: in a centred group the same reflection appears
    with several different glide vectors (in Cmcm the plane perpendicular to a
    is simultaneously m and b), and IT's choice among them is conventional, not
    algorithmic.  Ibca is the clean counterexample to any simple priority rule.
    So we check membership, which still catches the errors that matter --
    Pbcm vs Pbcn, a wrong screw axis, a dropped glide.
    """
    pos = []
    for axis in range(3):
        letters = _mirror_letters(ops, axis)
        if not letters:
            letters = _rotation_letters(ops, axis)
        if not letters:
            letters = {"1"}
        pos.append(letters)
    return pos


def check_ortho(n):
    ops = std_ops(n)
    name = SGNAME[n - 1].split()
    want = name[1:]  # Bravais letter name[0] unused; ortho check is positional only
    if len(want) != 3:
        return "unexpected name format %r" % SGNAME[n - 1]
    got = derived_ortho_positions(ops)
    for i in range(3):
        if want[i] not in got[i]:
            return "position %d: name says %r, ops allow %s" % (i + 1, want[i], sorted(got[i]))
    return None


def check_mono(n):
    """Monoclinic, unique axis b: check the element(s) along/perpendicular to b."""
    ops = std_ops(n)
    sym = SGNAME[n - 1].split()[1]  # '2', '21', 'm', 'c', '2/m', '21/c', ...
    rot, _, mir = sym.partition("/")
    if rot in ("m", "c", "a", "b", "n"):
        rot, mir = None, sym
    got_rot = _rotation_letters(ops, 1)
    got_mir = _mirror_letters(ops, 1)
    if rot and rot not in got_rot:
        return "rotation: name says %r, ops allow %s" % (rot, sorted(got_rot))
    if mir and mir not in got_mir:
        return "mirror: name says %r, ops allow %s" % (mir, sorted(got_mir))
    return None


# ---------------------------------------------------------------------------
# round trip
# ---------------------------------------------------------------------------

_SCRAMBLE = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64),
    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int64),   # cab
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int64),   # bca
    np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.int64),   # c -> a+c
]


def check_roundtrip(n, rng):
    ops = std_ops(n)
    for P in _SCRAMBLE:
        # scrambling with a non-symmetry-preserving P is only meaningful if the
        # result is still a group in the new basis -- it always is; the question
        # is whether we can find our way back.
        p = rng.integers(0, DEN, size=3) if rng is not None else np.zeros(3, dtype=np.int64)
        Pinv = np.round(np.linalg.inv(P.astype(float))).astype(np.int64)
        try:
            scrambled = transform_ops(ops, Pinv, np.zeros(3, dtype=np.int64))
            scrambled = [Op(o.W, o.w + (I3 - o.W) @ p) for o in scrambled]
            scrambled = close_group(scrambled)
        except SymmetryError as e:
            return "scramble failed: %s" % e
        try:
            st = find_setting(scrambled, n, std_ops)
        except SymmetryError as e:
            return "P=%s: %s" % (P.tolist(), e)
        got = opset(transform_ops(scrambled, st.P, st.p))
        if got != opset(ops):
            return "P=%s: recovered setting does not reproduce the standard group" % P.tolist()
    return None


# ---------------------------------------------------------------------------
# optional external cross-check
# ---------------------------------------------------------------------------

def check_spglib():
    try:
        import spglib  # optional dependency; verify CLI only
    except ImportError:
        return None, "spglib not installed -- skipping the only independent check of table CONTENT"
    bad = []
    for n in range(1, 231):
        ours = opset(std_ops(n))
        # spglib's hall_number for the standard setting of group n
        hn = None
        for h in range(1, 531):
            t = spglib.get_spacegroup_type(h)
            if t["number"] == n:
                hn = h
                break
        sym = spglib.get_symmetry_from_database(hn)
        theirs = set()
        for R, t in zip(sym["rotations"], sym["translations"]):
            tt = np.round(np.asarray(t) * DEN).astype(np.int64) % DEN
            theirs.add((tuple(int(v) for v in np.asarray(R).ravel()), tuple(int(v) for v in tt)))
        if ours != theirs:
            bad.append((n, SGNAME[n - 1], STD_HALL[n],
                        "differs from spglib hall_number %s (%s)" % (hn, spglib.get_spacegroup_type(hn)["hall_symbol"])))
    return bad, None


# ---------------------------------------------------------------------------

def main(argv=None):
    argv = argv or sys.argv[1:]
    quick = "--quick" in argv
    rng = np.random.default_rng(0)
    fails = []

    print("1/2. expanding all 230 standard Hall symbols ...")
    for n in range(1, 231):
        try:
            ops = std_ops(n)
        except SymmetryError as e:
            fails.append((n, "expand", str(e)))
            continue
        lat = lattice_letter(ops)
        want = SGNAME[n - 1].split()[0]
        if lat != want:
            fails.append((n, "lattice", "%s != %s" % (lat, want)))
    print("     %d failures" % len([f for f in fails if f[1] in ("expand", "lattice")]))

    print("3. deriving H-M symbols from ops for the 72 monoclinic + orthorhombic groups ...")
    nsym = 0
    for n in range(3, 75):
        f = check_mono(n) if n <= 15 else check_ortho(n)
        if f:
            fails.append((n, "symbol", f))
        nsym += 1
    print("     checked %d, %d failures" % (nsym, len([f for f in fails if f[1] == "symbol"])))

    if not quick:
        print("4. round-tripping every group through find_setting ...")
        for n in range(1, 231):
            f = check_roundtrip(n, rng)
            if f:
                fails.append((n, "roundtrip", f))
        print("     %d failures" % len([f for f in fails if f[1] == "roundtrip"]))

    print("5. cross-checking against spglib (independent) ...")
    bad, msg = check_spglib()
    if msg:
        print("     " + msg)
    else:
        print("     %d differences" % len(bad))
        for b in bad:
            fails.append((b[0], "spglib", "%s %s: %s" % (b[1], b[2], b[3])))

    print()
    if fails:
        print("FAILURES (%d):" % len(fails))
        for n, kind, msg in fails:
            print("  SG %3d %-9s %-11s %s" % (n, SGNAME[n - 1], kind, msg))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
