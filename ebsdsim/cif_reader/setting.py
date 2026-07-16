"""
Map CIF operators onto the IT standard setting, and recover ops from sites.

  * find_setting / Setting / transform_cell / transform_coords
  * sites-tree descent (Hall-table subgroups)
  * H-M last-resort salvage (hm_to_ops)
"""
from __future__ import annotations

import itertools
import os as _os
import re
from collections import defaultdict

import numpy as np

from .sym import (
    DEN, I3, Op, SymmetryError,
    ORIGIN1_HALL, RHOMB_HALL, SGNAME, STD_HALL, crystal_system,
    cached_hall_ops, clear_hall_cache, close_group, close_packed,
    fingerprint, hall_ops as _hall_expand, lattice_letter, opset, pack,
    store_hall_ops, unpack, _WBASE, _WMAX,
)

# =============================================================================
# Change-of-basis search
# =============================================================================

# 1/24-lattice grid of candidate origin shifts.  The denominator matters:
# origin-choice shifts include 1/8 (Fddd, Fd-3m) and 1/12 (trigonal, cubic),
# so a 1/12 or 1/4 grid would silently miss real settings.
_P_GRID = np.array(list(itertools.product(range(DEN), repeat=3)), dtype=np.int64)

# Hexagonal <- rhombohedral (obverse).  Columns are a_hex, b_hex, c_hex in the
# rhombohedral basis.
P_HEX_FROM_RHOMB = np.array([[1, 0, 1],
                             [-1, 1, 1],
                             [0, -1, 1]], dtype=np.int64)


def _candidate_P():
    """
    All integer 3x3 matrices with entries in {-1,0,1} and det = +1.

    That set covers every IT setting: orthorhombic axis permutations, the three
    monoclinic cell choices, unique-axis changes, and R obverse/reverse.  It
    also covers the mild non-standard cells people actually deposit (a' = a+b
    and friends).

    det is restricted to +1, not +/-1, on purpose.  A det = -1 basis change is
    a mirror, and applying one would map an enantiomorphic group onto its
    partner -- turning P4_1 into P4_3 and producing a plausible-looking master
    pattern of the wrong crystal.  If the CIF really is left-handed we fail
    instead of guessing.
    """
    out = []
    for entries in itertools.product((-1, 0, 1), repeat=9):
        M = np.array(entries, dtype=np.int64).reshape(3, 3)
        if round(np.linalg.det(M)) == 1:
            out.append(M)
    # try the identity first so an already-standard CIF is left untouched, then
    # prefer sparse/simple matrices
    out.sort(key=lambda M: (not np.array_equal(M, I3), int(np.abs(M).sum())))
    return out


_CANDIDATES = None
_CAND_STACK = None


def candidates():
    global _CANDIDATES
    if _CANDIDATES is None:
        _CANDIDATES = _candidate_P()
    return _CANDIDATES


def _candidate_stack():
    """(Ps, Pinvs) as (N,3,3) arrays, built once."""
    global _CAND_STACK
    if _CAND_STACK is None:
        Ps = np.stack(candidates())
        Pinvs = np.stack([np.round(np.linalg.inv(P.astype(float))).astype(np.int64) for P in Ps])
        _CAND_STACK = (Ps, Pinvs)
    return _CAND_STACK


def _inv_int(P):
    """Exact inverse of a unimodular integer matrix (det = +1 => inverse is integer)."""
    P = np.asarray(P, dtype=np.int64)
    d = int(round(np.linalg.det(P)))
    if d != 1:
        raise SymmetryError("change of basis has det=%d, expected +1" % d)
    Pinv = np.round(np.linalg.inv(P.astype(float))).astype(np.int64)
    if not np.array_equal(Pinv @ P, I3):
        raise SymmetryError("change of basis is not unimodular")
    return Pinv


def rhomb_to_hex(ops):
    """
    Convert operators given on primitive rhombohedral axes to hexagonal axes.

    The basis change has det = 3, so it is not unimodular and cannot go through
    the ordinary search: the hexagonal cell holds three lattice points and the
    operator count triples.  We apply the fixed matrix with exact rational
    arithmetic, re-attach the R centring, and close.
    """
    P = P_HEX_FROM_RHOMB
    adj = np.round(np.linalg.inv(P.astype(float)) * 3.0).astype(np.int64)  # 3 * P^-1
    Ws = np.stack([o.W for o in ops])
    ws = np.stack([o.w for o in ops])
    num = np.einsum("ij,njk,kl->nil", adj, Ws, P)
    if np.any(num % 3):
        raise SymmetryError("operator set is not compatible with rhombohedral axes")
    W2 = num // 3
    tnum = np.einsum("ij,nj->ni", adj, ws)
    if np.any(tnum % 3):
        raise SymmetryError(
            "translations do not land on the 1/%d lattice in hexagonal axes" % DEN)
    out = [Op(W2[i], tnum[i] // 3) for i in range(len(ops))]
    out.append(Op(I3, (16, 8, 8)))
    out.append(Op(I3, (8, 16, 16)))
    return close_group(out)


def transform_op(op, P, Pinv, p):
    W2 = Pinv @ op.W @ P
    w2 = Pinv @ (op.W @ p + op.w - p)
    return Op(W2, w2)


def transform_ops(ops, P, p):
    Pinv = _inv_int(P)
    p = np.asarray(p, dtype=np.int64)
    return [transform_op(o, P, Pinv, p) for o in ops]


# ---------------------------------------------------------------------------
# stage A: rotation parts
# ---------------------------------------------------------------------------

def _rot_code(W):
    """Map a 3x3 int rotation to a single int64 (same packing as encode_rows W)."""
    flat = np.asarray(W, dtype=np.int64).ravel()
    code = np.int64(0)
    for v in flat:
        code = code * _WBASE + (int(v) + _WMAX)
    return int(code)


def _rot_codes_rows(flat):
    """(n, 9) int array -> (n,) codes."""
    code = np.zeros(flat.shape[0], dtype=np.int64)
    for i in range(9):
        code = code * _WBASE + (flat[:, i] + _WMAX)
    return code


def _matching_P(ops_cif, ops_std):
    """
    Candidate P for which {P^-1 W P} == {W_std} as multisets.

    Filters progressively rather than testing each P against the whole group:
    membership of a single conjugated W in the target set is already a strong
    necessary condition, and two or three of them cut thousands of candidates
    down to a handful.  The full multiset equality is then checked only on the
    survivors.
    """
    target_codes = {_rot_code(o.W) for o in ops_std}
    target_ms = sorted(_rot_code(o.W) for o in ops_std)
    ident = _rot_code(I3)

    Ps, Pinvs = _candidate_stack()
    idx = np.arange(Ps.shape[0])

    distinct = []
    seen = set()
    for o in ops_cif:
        k = _rot_code(o.W)
        if k not in seen and k != ident:
            seen.add(k)
            distinct.append(o.W)

    targ = np.array(sorted(target_codes), dtype=np.int64)
    for W in distinct:
        if idx.size == 0:
            break
        Wt = np.einsum("nij,jk,nkl->nil", Pinvs[idx], W, Ps[idx])
        codes = _rot_codes_rows(Wt.reshape(len(idx), 9))
        pos = np.searchsorted(targ, codes)
        pos = np.clip(pos, 0, len(targ) - 1)
        keep = targ[pos] == codes
        idx = idx[keep]
        if idx.size <= 64:
            break

    good = []
    Wc = np.stack([o.W for o in ops_cif])
    for i in idx:
        Wt = np.einsum("ij,njk,kl->nil", Pinvs[i], Wc, Ps[i])
        got = sorted(int(c) for c in _rot_codes_rows(Wt.reshape(len(ops_cif), 9)))
        if got == target_ms:
            good.append(Ps[i])
    return good


# ---------------------------------------------------------------------------
# stage B: origin shift
# ---------------------------------------------------------------------------

def _encode(Wlist, wlist, windex):
    """Pack (W, w) into a single int per operator, so op sets compare by sorting."""
    npts = wlist[0].shape[0]
    codes = np.empty((npts, len(Wlist)), dtype=np.int64)
    for i, key in enumerate(Wlist):
        t = wlist[i]  # (Np, 3)
        codes[:, i] = windex[key] * (DEN ** 3) + t[:, 0] * DEN * DEN + t[:, 1] * DEN + t[:, 2]
    codes.sort(axis=1)
    return codes


def _find_p(ops_cif, ops_std, P):
    """
    Solve for the origin shift p, given P.

    For each op, w' = P^-1 (W p + w - p) = P^-1 ((W - I) p + w).  Sweep the
    whole 1/24 grid at once with numpy and compare the resulting operator sets.
    """
    Pinv = _inv_int(P)
    std_keys = opset(ops_std)

    # canonical index for every distinct rotation part appearing in the target
    windex = {}
    for k in sorted({tuple(int(v) for v in o.W.ravel()) for o in ops_std}):
        windex[k] = len(windex)

    Wlist = []
    wlist = []
    for o in ops_cif:
        W2 = Pinv @ o.W @ P
        key = tuple(int(v) for v in W2.ravel())
        if key not in windex:
            return None  # stage A should have prevented this
        Wlist.append(key)
        # (Np, 3) translation part for every candidate p
        A = Pinv @ (o.W - I3)
        t = (_P_GRID @ A.T + (Pinv @ o.w)) % DEN
        wlist.append(t)

    codes = _encode(Wlist, wlist, windex)

    std_codes = np.sort(np.array(
        [windex[tuple(int(v) for v in o.W.ravel())] * (DEN ** 3)
         + int(o.w[0]) * DEN * DEN + int(o.w[1]) * DEN + int(o.w[2])
         for o in ops_std], dtype=np.int64))

    if codes.shape[1] != std_codes.shape[0]:
        return None

    hit = np.nonzero((codes == std_codes[None, :]).all(axis=1))[0]
    if hit.size == 0:
        return None

    # prefer p = 0, then the smallest shift
    ps = _P_GRID[hit]
    order = np.lexsort((ps[:, 2], ps[:, 1], ps[:, 0], np.abs(ps).sum(axis=1)))
    p = ps[order[0]]

    # paranoia: reconstruct and compare exactly
    got = opset(transform_ops(ops_cif, P, p))
    if got != std_keys:
        return None
    return p


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

class Setting:
    rhombohedral_input = False

    def __init__(self, number, P, p, note=""):
        self.number = int(number)
        self.P = np.asarray(P, dtype=np.int64)
        self.p = np.asarray(p, dtype=np.int64)
        self.note = note

    @property
    def is_standard(self):
        return np.array_equal(self.P, I3) and not np.any(self.p)

    def describe(self):
        from fractions import Fraction
        cols = []
        for j in range(3):
            terms = []
            for i, v in enumerate("abc"):
                c = int(self.P[i, j])
                if c:
                    terms.append(("+" if c > 0 else "-") + (v if abs(c) == 1 else "%d%s" % (abs(c), v)))
            cols.append("".join(terms).lstrip("+"))
        shift = ",".join(str(Fraction(int(v), DEN)) for v in self.p)
        return "(a',b',c') = (%s | %s | %s);  origin shift = (%s)" % (cols[0], cols[1], cols[2], shift)

    def __repr__(self):
        return "<Setting SG%d %s>" % (self.number, "standard" if self.is_standard else self.describe())


# ---------------------------------------------------------------------------
# precomputed settings table
#
# A space group in a given setting IS a particular set of operators.  So the
# question "which setting is this CIF in" is a dict lookup keyed on that set --
# not a search.  The search below still exists, but only for cells nobody has
# tabulated; every setting that actually appears in the COD is in here.
# ---------------------------------------------------------------------------

_TABLE = None
_TABLE_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_settings.npz")


def _load_table():
    """fingerprint(bytes) -> (n, P, p).  Loaded from allow_pickle=False .npz."""
    try:
        with np.load(_TABLE_PATH, allow_pickle=False) as z:
            fp_concat = np.asarray(z["fp_concat"], dtype=np.uint8)
            fp_off = np.asarray(z["fp_off"], dtype=np.int64)
            sg = np.asarray(z["sg"], dtype=np.int16)
            P = np.asarray(z["P"], dtype=np.int64)
            p = np.asarray(z["p"], dtype=np.int64)
        n = int(sg.shape[0])
        tab = {}
        for i in range(n):
            fp = fp_concat[int(fp_off[i]):int(fp_off[i + 1])].tobytes()
            tab[fp] = (int(sg[i]), P[i].copy(), p[i].copy())
        return tab
    except Exception:
        return None


def _save_table(tab):
    """Write settings table as numeric .npz (no pickle)."""
    try:
        items = list(tab.items())
        n = len(items)
        fps = [k for k, _ in items]
        lengths = np.fromiter((len(fp) for fp in fps), dtype=np.int64, count=n)
        fp_off = np.empty(n + 1, dtype=np.int64)
        fp_off[0] = 0
        np.cumsum(lengths, out=fp_off[1:])
        fp_concat = np.empty(int(fp_off[-1]), dtype=np.uint8)
        sg = np.empty(n, dtype=np.int16)
        P = np.empty((n, 3, 3), dtype=np.int64)
        p = np.empty((n, 3), dtype=np.int64)
        for i, (fp, (number, Pi, pi)) in enumerate(items):
            fp_concat[int(fp_off[i]):int(fp_off[i + 1])] = np.frombuffer(fp, dtype=np.uint8)
            sg[i] = number
            P[i] = Pi
            p[i] = pi
        tmp = _TABLE_PATH[:-4] + ".tmp.npz"
        np.savez_compressed(
            tmp,
            fp_concat=fp_concat,
            fp_off=fp_off,
            sg=sg,
            P=P,
            p=p,
        )
        _os.replace(tmp, _TABLE_PATH)
    except OSError:
        pass  # read-only install: just rebuild each process


# Generic cells, one per crystal system, in that system's standard metric.
# Deliberately un-round so no basis change can satisfy a metric constraint by
# arithmetic coincidence.
_GENERIC_CELL = {
    "triclinic":    (5.1, 6.2, 7.3, 83.1, 97.4, 104.7),
    "monoclinic":   (5.1, 6.2, 7.3, 90.0, 100.4, 90.0),
    "orthorhombic": (5.1, 6.2, 7.3, 90.0, 90.0, 90.0),
    "tetragonal":   (5.1, 5.1, 7.3, 90.0, 90.0, 90.0),
    "trigonal":     (5.1, 5.1, 7.3, 90.0, 90.0, 120.0),
    "hexagonal":    (5.1, 5.1, 7.3, 90.0, 90.0, 120.0),
    "cubic":        (5.1, 5.1, 5.1, 90.0, 90.0, 90.0),
}

_ATOL = 1e-6


def _metric_ok(sysname, cell):
    """
    Could a CIF plausibly be deposited with this cell, for this crystal system?

    This is the whole point of the table.  A basis change is only a real
    "setting" if the resulting cell still looks like the system it belongs to --
    a tetragonal crystal is deposited with a tetragonal cell, always.  Sheared
    bases are mathematically valid conjugations but nobody deposits them, so
    they belong in the search fallback, not the cache.
    """
    a, b, c, al, be, ga = cell
    eq = lambda x, y: abs(x - y) < 1e-4
    ang90 = lambda x: abs(x - 90.0) < _ATOL
    if sysname == "triclinic":
        return True
    if sysname == "monoclinic":
        # unique axis a, b or c: at most one angle may be oblique
        return sum(1 for x in (al, be, ga) if not ang90(x)) <= 1
    if sysname == "orthorhombic":
        return ang90(al) and ang90(be) and ang90(ga)
    if sysname == "tetragonal":
        return eq(a, b) and ang90(al) and ang90(be) and ang90(ga)
    if sysname in ("trigonal", "hexagonal"):
        return eq(a, b) and ang90(al) and ang90(be) and abs(ga - 120.0) < 1e-4
    if sysname == "cubic":
        return eq(a, b) and eq(b, c) and ang90(al) and ang90(be) and ang90(ga)
    return False


_LEGAL_P = None


def _metric_legal_P():
    """
    system -> [P]: the basis changes that keep the cell in its standard metric.

    Depends only on the crystal system, not the group, so this is seven filters
    rather than 230.
    """
    global _LEGAL_P
    if _LEGAL_P is None:
        _LEGAL_P = {}
        for sysname, cell in _GENERIC_CELL.items():
            _LEGAL_P[sysname] = [P for P in candidates()
                                 if _metric_ok(sysname, transform_cell(cell, P))]
    return _LEGAL_P


def _compose(P1, p1, P2, p2):
    """Apply (P1, p1) then (P2, p2).  x -> (P1 P2)^-1 (x - p1 - P1 p2)."""
    return P1 @ P2, p1 + P1 @ p2


def _settings_table():
    """
    fingerprint(ops) -> (number, P, p), where transform_ops(ops, P, p) == std.

    Built once and cached to disk next to the module as ``_settings.npz``
    (``allow_pickle=False``); rebuilding takes a few seconds, loading a few ms.
    Regenerate with ``python -m ebsdsim.cif_reader.setting --rebuild``.

    Covers, for every one of the 230 groups: the standard setting, every
    metric-legal basis change of it, origin choice 2 -> 1 where a second origin
    exists, and every metric-legal basis change of THAT.  Anything outside this
    -- a sheared or supercell basis, an origin nobody tabulated -- still loads
    correctly via the search in find_setting, just slower.
    """
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    _TABLE = _load_table()
    if _TABLE is not None:
        return _TABLE

    t = {}
    zero = np.zeros(3, dtype=np.int64)
    legal = _metric_legal_P()

    for n in range(1, 231):
        std = hall_ops(STD_HALL[n])
        std_fp = fingerprint(pack(std))

        # base settings: (ops, P, p) triples already known to map onto standard
        bases = [(std, I3, zero)]
        if n in ORIGIN1_HALL:
            o1 = hall_ops(ORIGIN1_HALL[n])
            p1 = _find_p(o1, std, I3)
            if p1 is not None:
                bases.append((o1, I3, p1))

        for ops0, P0, p0 in bases:
            for Q in legal[crystal_system(n)]:
                Qinv = _inv_int(Q)
                ops = transform_ops(ops0, Q, zero)
                # (Q^-1, 0) undoes the basis change, then (P0, p0) reaches standard
                P, p = _compose(Qinv, zero, P0, p0)
                fp = fingerprint(pack(ops))
                if fp in t:
                    continue
                # one-time build, so verify rather than trust the algebra
                if fingerprint(pack(transform_ops(ops, P, p % DEN))) != std_fp:
                    raise SymmetryError(
                        "table build: SG %d, Q=%s does not invert correctly" % (n, Q.tolist()))
                t[fp] = (n, P, p % DEN)

    _TABLE = t
    _save_table(t)
    return t


def find_setting(ops_cif, number, std_ops_for):
    """
    Determine (P, p) taking `ops_cif` to the standard setting of space group
    `number`.  `std_ops_for(n)` supplies the standard operator list.

    Returns a Setting.  Raises SymmetryError if no det=+1 change of basis
    reproduces the standard group -- which means the CIF's operations and its
    claimed IT number genuinely disagree, and the right response is to stop.
    """
    note = ""

    # Fast path 1: the operator set as listed is already a known setting.  Most
    # CIFs list the complete coset decomposition, so this hits without even
    # closing the group.
    for attempt in (ops_cif, None):
        if attempt is None:
            packed = close_packed(ops_cif)
            ops_cif = unpack(packed)
            attempt_fp = fingerprint(packed)
        else:
            attempt_fp = fingerprint(pack(attempt))
        hit = _settings_table().get(attempt_fp)
        if hit is not None:
            n, P, p = hit
            if number is not None and int(number) != n:
                raise SymmetryError(
                    "the CIF's operators are space group %d but it declares IT number %s"
                    % (n, number))
            st = Setting(n, P, p, note)
            st.rhombohedral_input = False
            return st

    if number is None:
        raise SymmetryError(
            "the CIF declares no IT number and its %d operators are not a tabulated "
            "setting of any of the 230 space groups. Add _space_group_IT_number, or a "
            "Hall symbol, or check the symop loop." % len(ops_cif))
    number = int(number)
    ops_std = std_ops_for(number)

    # Rhombohedral primitive setting: the operator count is exactly 1/3 of the
    # hexagonal one, so no unimodular P can bridge them.  Convert first, then
    # hit the settings table again before falling back to search.
    rhomb = False
    if len(ops_cif) * 3 == len(ops_std):
        ops_cif = rhomb_to_hex(ops_cif)
        rhomb = True
        note = "converted from rhombohedral to hexagonal axes"
        hit = _settings_table().get(fingerprint(pack(ops_cif)))
        if hit is not None:
            n, P, p = hit
            if int(number) != n:
                raise SymmetryError(
                    "the CIF's operators are space group %d but it declares IT number %s"
                    % (n, number))
            st = Setting(n, P, p, note)
            st.rhombohedral_input = True
            return st

    if len(ops_cif) != len(ops_std):
        raise SymmetryError(
            "CIF group has %d operators but space group %d has %d. Either the IT number "
            "is wrong or the symop list is not a group."
            % (len(ops_cif), number, len(ops_std))
        )

    Ps = _matching_P(ops_cif, ops_std)
    if not Ps:
        raise SymmetryError(
            "no det=+1 change of basis maps the CIF's rotation parts onto space group %d. "
            "The CIF's operations and its claimed IT number disagree." % number
        )

    for P in Ps:
        p = _find_p(ops_cif, ops_std, P)
        if p is not None:
            st = Setting(number, P, p, note)
            st.rhombohedral_input = rhomb
            return st

    raise SymmetryError(
        "found %d candidate basis changes for space group %d but none admits an origin "
        "shift on the 1/%d lattice that reproduces the standard operators."
        % (len(Ps), number, DEN)
    )


# ---------------------------------------------------------------------------
# applying the setting to cell + atoms
# ---------------------------------------------------------------------------

def transform_cell(cell, P):
    """
    cell = (a, b, c, alpha, beta, gamma) in Angstrom / degrees.
    Returns the transformed cell parameters.
    """
    a, b, c, al, be, ga = [float(v) for v in cell]
    ca, cb, cg = (np.cos(np.radians(x)) for x in (al, be, ga))
    # metric tensor
    G = np.array([
        [a * a, a * b * cg, a * c * cb],
        [a * b * cg, b * b, b * c * ca],
        [a * c * cb, b * c * ca, c * c],
    ])
    P = np.asarray(P, dtype=float)
    G2 = P.T @ G @ P
    a2, b2, c2 = (np.sqrt(max(G2[i, i], 0.0)) for i in range(3))

    def ang(i, j, li, lj):
        v = G2[i, j] / (li * lj)
        return float(np.degrees(np.arccos(np.clip(v, -1.0, 1.0))))

    return (float(a2), float(b2), float(c2),
            ang(1, 2, b2, c2), ang(0, 2, a2, c2), ang(0, 1, a2, b2))


def transform_coords(x, P, p):
    """x' = P^-1 (x - p), with p given in 1/DEN units."""
    Pinv = np.linalg.inv(np.asarray(P, dtype=float))
    pv = np.asarray(p, dtype=float) / DEN
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return ((x - pv) @ Pinv.T) % 1.0


def warm_settings_cache():
    """Load the fingerprint→(P,p) settings table into memory."""
    _settings_table()


# =============================================================================
# Sites-tree descent
# =============================================================================

DEFAULT_EPS = 1e-5

_TREE_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_descent.npz")

# Packed Hall ops from .npz / build; Op lists live in the Hall memo.
_PACKED_OPS = {}              # symbol -> (n, 12) int64
_ORDERS = None
_TOPS = None
_DAG = None


def hall_ops(symbol):
    """Hall expansion: packed .npz cache when present, else expand (memoized)."""
    hit = cached_hall_ops(symbol)
    if hit is not None:
        return hit
    packed = _PACKED_OPS.get(symbol)
    if packed is not None:
        ops = unpack(packed)
        store_hall_ops(symbol, ops)
        return ops
    ops = _hall_expand(symbol)
    _PACKED_OPS.setdefault(symbol, pack(ops))
    return ops


# ---------------------------------------------------------------------------
# site predicate (vectorized over operators)
# ---------------------------------------------------------------------------

def _min_image_dist(mapped, pts):
    """mapped (M,3), pts (N,3) -> (M,) nearest minimum-image distances."""
    d = mapped[:, None, :] - pts[None, :, :]
    d = (d + 0.5) % 1.0 - 0.5
    return np.linalg.norm(d, axis=2).min(axis=1)


def _ops_as_arrays(ops):
    """list[Op] or (n,12) packed -> (W (n,3,3) float, w (n,3) float)."""
    if isinstance(ops, np.ndarray):
        arr = np.asarray(ops, dtype=np.int64)
    else:
        arr = pack(ops)
    W = arr[:, :9].reshape(-1, 3, 3).astype(float)
    w = arr[:, 9:].astype(float) / DEN
    return W, w


def _unique_frac(pts, eps):
    """Greedy unique under minimum-image distance (vectorized pairwise cull)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    m = pts.shape[0]
    if m <= 1:
        return pts
    d = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm((d + 0.5) % 1.0 - 0.5, axis=2)
    keep = np.ones(m, dtype=bool)
    for i in range(m):
        if keep[i]:
            keep[i + 1:] &= d[i, i + 1:] > eps
    return pts[keep]


def sites_invariant(ops, species, coords, eps=DEFAULT_EPS):
    """
    True iff every operator maps each species set onto itself within `eps`
    (fractional coordinates, minimum-image).

    `ops` may be a list of Op or a packed (n, 12) int64 array from the cache.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    species = np.asarray(species, dtype=object).reshape(-1)
    if len(coords) != len(species):
        raise ValueError("species and coords length mismatch")
    if not len(coords):
        return True

    W, w = _ops_as_arrays(ops)
    for sp in np.unique(species):
        pts = coords[species == sp]
        mapped = np.einsum("ni,kji->nkj", pts, W) + w[None, :, :]
        mapped = np.mod(mapped, 1.0)
        flat = mapped.reshape(-1, 3)
        if np.any(_min_image_dist(flat, pts) > eps):
            return False
    return True


def expand_orbits(ops, species, coords, eps=DEFAULT_EPS):
    """Apply ops to (species, coords) and unique within eps (species-wise)."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    species = np.asarray(species, dtype=object).reshape(-1)
    W, w = _ops_as_arrays(ops)
    out_sp, chunks = [], []
    for sp in np.unique(species):
        pts = coords[species == sp]
        mapped = np.einsum("ni,kji->nkj", pts, W) + w[None, :, :]
        uniq = _unique_frac(np.mod(mapped, 1.0).reshape(-1, 3), eps)
        out_sp.extend([sp] * len(uniq))
        chunks.append(uniq)
    if not chunks:
        return [], np.empty((0, 3), dtype=float)
    return out_sp, np.vstack(chunks)


def fold_asymmetric(ops, species, coords, eps=DEFAULT_EPS):
    """
    Reduce a full orbit set to one representative per orbit under `ops`.

    Returns (species, coords, indices) indexing the input rows.
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    species = np.asarray(species, dtype=object).reshape(-1)
    W, w = _ops_as_arrays(ops)
    kept_mask = np.zeros(len(coords), dtype=bool)

    for sp in np.unique(species):
        idx = np.flatnonzero(species == sp)
        pts = coords[idx]
        local = []
        for li in range(len(idx)):
            if not local:
                local.append(li)
                continue
            Y = pts[local]
            mapped = np.einsum("ki,nji->knj", Y, W) + w[None, :, :]
            mapped = np.mod(mapped, 1.0).reshape(-1, 3)
            if float(_min_image_dist(pts[li:li + 1], mapped)[0]) > eps:
                local.append(li)
        kept_mask[idx[local]] = True

    kept_idx = np.flatnonzero(kept_mask).tolist()
    return species[kept_idx].tolist(), coords[kept_idx], kept_idx


# ---------------------------------------------------------------------------
# metric -> crystal family
# ---------------------------------------------------------------------------

_ATOL = 1e-3


def metric_family(cell):
    """
    Crystal family implied by the cell metric (not the IT number).

    Returns one of: triclinic, monoclinic, orthorhombic, tetragonal,
    hexagonal_family, rhombohedral, cubic.
    """
    a, b, c, al, be, ga = (float(v) for v in cell)
    eq = lambda x, y: abs(x - y) < _ATOL * max(1.0, abs(y))
    ang90 = lambda x: abs(x - 90.0) < _ATOL
    ang120 = lambda x: abs(x - 120.0) < _ATOL

    if eq(a, b) and eq(b, c) and ang90(al) and ang90(be) and ang90(ga):
        return "cubic"
    if eq(a, b) and ang90(al) and ang90(be) and ang120(ga):
        return "hexagonal_family"
    if eq(a, b) and eq(b, c) and eq(al, be) and eq(be, ga) and not ang90(al):
        return "rhombohedral"
    if eq(a, b) and ang90(al) and ang90(be) and ang90(ga):
        return "tetragonal"
    if ang90(al) and ang90(be) and ang90(ga):
        return "orthorhombic"
    n_oblique = sum(1 for x in (al, be, ga) if not ang90(x))
    if n_oblique <= 1:
        return "monoclinic"
    return "triclinic"


# ---------------------------------------------------------------------------
# tops + DAG + packed Hall ops cache
# ---------------------------------------------------------------------------

def _all_hall_symbols():
    syms = set(STD_HALL.values())
    syms.update(ORIGIN1_HALL.values())
    syms.update(RHOMB_HALL.values())
    return sorted(syms)


def _build_tree_and_ops():
    """Expand every tabulated Hall symbol once; build DAG from orders."""
    packed = {}
    for sym in _all_hall_symbols():
        ops = _hall_expand(sym)
        packed[sym] = pack(ops)
        store_hall_ops(sym, ops)

    orders = {n: packed[STD_HALL[n]].shape[0] for n in range(1, 231)}
    by_sys = defaultdict(list)
    for n in range(1, 231):
        by_sys[crystal_system(n)].append(n)

    families = {
        "triclinic": by_sys["triclinic"],
        "monoclinic": by_sys["monoclinic"],
        "orthorhombic": by_sys["orthorhombic"],
        "tetragonal": by_sys["tetragonal"],
        "hexagonal_family": by_sys["trigonal"] + by_sys["hexagonal"],
        "rhombohedral": [n for n in by_sys["trigonal"] if n in RHOMB_HALL],
        "cubic": by_sys["cubic"],
    }

    tops = {}
    dag = {n: [] for n in range(1, 231)}

    for fam, nums in families.items():
        if not nums:
            continue
        max_o = max(orders[n] for n in nums)
        tops[fam] = sorted(n for n in nums if orders[n] == max_o)

        for n in nums:
            on = orders[n]
            cands = [m for m in nums if orders[m] < on and on % orders[m] == 0]
            children = []
            for m in cands:
                om = orders[m]
                if any(om < orders[k] < on and on % orders[k] == 0 and orders[k] % om == 0
                       for k in cands):
                    continue
                children.append(m)
            children.sort(key=lambda m: (-orders[m], m))
            seen = set(dag[n])
            for m in children:
                if m not in seen:
                    dag[n].append(m)
                    seen.add(m)

    return orders, tops, dag, packed


def _load_tree():
    """orders, tops, dag, packed — from allow_pickle=False .npz."""
    try:
        with np.load(_TREE_PATH, allow_pickle=False) as z:
            orders_arr = np.asarray(z["orders"], dtype=np.int32)
            dag_off = np.asarray(z["dag_off"], dtype=np.int64)
            dag_child = np.asarray(z["dag_child"], dtype=np.int16)
            hall_symbols = [str(s) for s in np.asarray(z["hall_symbols"])]
            ops_off = np.asarray(z["ops_off"], dtype=np.int64)
            ops_flat = np.asarray(z["ops_flat"], dtype=np.int64)
            top_families = [str(s) for s in np.asarray(z["top_families"])]
            top_off = np.asarray(z["top_off"], dtype=np.int64)
            top_sg = np.asarray(z["top_sg"], dtype=np.int16)

        orders = {i + 1: int(orders_arr[i]) for i in range(230)}
        dag = {}
        for i in range(230):
            a, b = int(dag_off[i]), int(dag_off[i + 1])
            dag[i + 1] = [int(x) for x in dag_child[a:b]]
        packed = {}
        for i, sym in enumerate(hall_symbols):
            a, b = int(ops_off[i]), int(ops_off[i + 1])
            packed[sym] = ops_flat[a:b].copy()
        tops = {}
        for i, fam in enumerate(top_families):
            a, b = int(top_off[i]), int(top_off[i + 1])
            tops[fam] = [int(x) for x in top_sg[a:b]]
        return orders, tops, dag, packed
    except Exception:
        return None


def _save_tree(orders, tops, dag, packed):
    """Write descent DAG + packed Hall ops as numeric .npz (no pickle)."""
    try:
        orders_arr = np.array([orders[i] for i in range(1, 231)], dtype=np.int32)
        child_lists = [list(dag.get(i, [])) for i in range(1, 231)]
        dag_off = np.zeros(231, dtype=np.int64)
        for i, ch in enumerate(child_lists):
            dag_off[i + 1] = dag_off[i] + len(ch)
        dag_child = np.fromiter(
            (c for ch in child_lists for c in ch),
            dtype=np.int16,
            count=int(dag_off[-1]),
        )

        hall_symbols = list(packed.keys())
        ops_rows = [np.asarray(packed[s], dtype=np.int64).reshape(-1, 12) for s in hall_symbols]
        ops_off = np.zeros(len(hall_symbols) + 1, dtype=np.int64)
        for i, rows in enumerate(ops_rows):
            ops_off[i + 1] = ops_off[i] + rows.shape[0]
        ops_flat = (
            np.vstack(ops_rows) if ops_rows else np.empty((0, 12), dtype=np.int64)
        )

        top_families = list(tops.keys())
        top_lists = [list(tops[f]) for f in top_families]
        top_off = np.zeros(len(top_families) + 1, dtype=np.int64)
        for i, lst in enumerate(top_lists):
            top_off[i + 1] = top_off[i] + len(lst)
        top_sg = np.fromiter(
            (g for lst in top_lists for g in lst),
            dtype=np.int16,
            count=int(top_off[-1]),
        )

        tmp = _TREE_PATH[:-4] + ".tmp.npz"
        np.savez_compressed(
            tmp,
            orders=orders_arr,
            dag_off=dag_off,
            dag_child=dag_child,
            hall_symbols=np.asarray(hall_symbols),
            ops_off=ops_off,
            ops_flat=ops_flat,
            top_families=np.asarray(top_families),
            top_off=top_off,
            top_sg=top_sg,
        )
        _os.replace(tmp, _TREE_PATH)
    except OSError:
        pass


def _ensure_tree():
    """Load DAG + packed Hall ops from .npz, or build and cache once."""
    global _ORDERS, _TOPS, _DAG, _PACKED_OPS
    if _DAG is not None and _PACKED_OPS:
        return

    loaded = _load_tree()
    if loaded is not None:
        orders, tops, dag, packed = loaded
        _ORDERS, _TOPS, _DAG = orders, tops, dag
        _PACKED_OPS.update(packed)
        return

    orders, tops, dag, packed = _build_tree_and_ops()
    _ORDERS, _TOPS, _DAG = orders, tops, dag
    _PACKED_OPS.update(packed)
    _save_tree(orders, tops, dag, packed)


def warm_descent_cache(prefill_hall=False):
    """
    Force-load the descent .npz (DAG + packed ops).

    `prefill_hall=True` also unpacks every symbol into Op lists up front.
    """
    _ensure_tree()
    if not prefill_hall:
        return
    for sym in list(_PACKED_OPS):
        hall_ops(sym)


def tops_for_family(family):
    _ensure_tree()
    return list(_TOPS.get(family, [1]))


def subgroup_children(n):
    _ensure_tree()
    return list(_DAG.get(n, []))


# ---------------------------------------------------------------------------
# tree descent
# ---------------------------------------------------------------------------

def _ops_variants_packed(n, family):
    """Yield (label, packed_array) for IT number n."""
    _ensure_tree()
    yield "std", _PACKED_OPS[STD_HALL[n]]
    if n in ORIGIN1_HALL:
        yield "origin1", _PACKED_OPS[ORIGIN1_HALL[n]]
    if family == "rhombohedral" and n in RHOMB_HALL:
        yield "rhomb", _PACKED_OPS[RHOMB_HALL[n]]


def recover_ops_from_sites(cell, species, coords, eps=DEFAULT_EPS):
    """
    Descend the SG ops tree until the maximal Hall group leaving sites
    invariant is found.  Returns that operator list (exact integers).
    """
    _ensure_tree()
    family = metric_family(cell)
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    species = np.asarray(species, dtype=object).reshape(-1)

    if family == "rhombohedral":
        start = tops_for_family("rhombohedral") or tops_for_family("hexagonal_family")
    else:
        start = tops_for_family(family)

    best_n = None
    best_packed = None
    best_order = -1
    stack = list(start)
    visited = set()

    while stack:
        n = stack.pop()
        if n in visited:
            continue
        visited.add(n)

        hit = None
        for _label, packed in _ops_variants_packed(n, family):
            if sites_invariant(packed, species, coords, eps=eps):
                hit = packed
                break

        if hit is not None:
            order = int(hit.shape[0])
            if order > best_order or (order == best_order and (best_n is None or n < best_n)):
                best_n, best_packed, best_order = n, hit, order
            continue

        for m in subgroup_children(n):
            if m not in visited:
                stack.append(m)

    if best_packed is None:
        for n in (2, 1):
            packed = _PACKED_OPS[STD_HALL[n]]
            if sites_invariant(packed, species, coords, eps=eps):
                best_n, best_packed = n, packed
                break

    if best_packed is None:
        raise SymmetryError(
            "site tree descent found no Hall group leaving the atom positions invariant "
            "(family=%s, eps=%g)" % (family, eps)
        )
    return unpack(best_packed)


def ops_are_trivial(ops):
    """True if ops are empty or only the identity (P1 deposition)."""
    if not ops:
        return True
    if len(ops) == 1:
        o = ops[0]
        return bool(np.array_equal(o.W, np.eye(3, dtype=np.int64)) and not np.any(o.w))
    return len(close_group(ops)) == 1


def _reset_tree_for_rebuild():
    global _ORDERS, _TOPS, _DAG, _PACKED_OPS
    _ORDERS = _TOPS = _DAG = None
    _PACKED_OPS = {}
    clear_hall_cache()


def _reset_settings_for_rebuild():
    global _TABLE
    _TABLE = None


if __name__ == "__main__":
    import sys
    import time

    if "--rebuild" in sys.argv:
        for path in (_TABLE_PATH, _TREE_PATH):
            try:
                _os.remove(path)
            except OSError:
                pass
        _reset_settings_for_rebuild()
        _reset_tree_for_rebuild()

    t0 = time.time()
    tab = _settings_table()
    print("%d settings in %.2fs -> %s" % (len(tab), time.time() - t0, _TABLE_PATH))
    t0 = time.time()
    _ensure_tree()
    nbytes = sum(a.nbytes for a in _PACKED_OPS.values())
    print(
        "descent cache: %d groups, %d Hall symbols (%d KB packed ops) in %.2fs -> %s"
        % (len(_DAG), len(_PACKED_OPS), nbytes // 1024, time.time() - t0, _TREE_PATH)
    )


# =============================================================================
# H-M salvage (last resort)
# =============================================================================

# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"(-?[1-6](?:[1-5])?|[mabcnde])(?:/(-?[1-6]|[mabcnd]))?")


def _strip_suffix(s):
    """Pull off :H / :R / :1 / :2 / 'hexagonal axes' style qualifiers."""
    s = s.strip()
    origin = None
    axes = None
    m = re.search(r":\s*([HR12])\s*$", s, re.I)
    if m:
        q = m.group(1).upper()
        if q in "HR":
            axes = q
        else:
            origin = int(q)
        s = s[: m.start()].strip()
    m = re.search(r"\b(hexagonal|rhombohedral)\s+axes\b", s, re.I)
    if m:
        axes = m.group(1)[0].upper()
        s = (s[: m.start()] + s[m.end():]).strip()
    return s, axes, origin


def normalize(sym):
    """
    'C 1 2 1' / 'C2' / 'P2(1)/n' -> (lattice, (p1, p2, p3), axes, origin)

    Positions are '1' where the symbol carries no element.  Short symbols are
    expanded to three positions using the crystal system implied by the
    tokens; a short monoclinic symbol means unique axis b, per convention.
    """
    s = str(sym).strip().strip("'\"").strip()
    s, axes, origin = _strip_suffix(s)
    s = s.replace("(", "").replace(")", "")     # P2(1)/n -> P21/n
    s = re.sub(r"\s+", " ", s)
    if not s:
        raise SymmetryError("empty H-M symbol")

    lat = s[0].upper()
    if lat not in "PABCIRSTF":
        raise SymmetryError("H-M symbol %r does not start with a lattice letter" % sym)
    rest = s[1:].strip()

    toks = []
    pos = 0
    while pos < len(rest):
        if rest[pos] in " _":
            pos += 1
            continue
        m = _TOKEN.match(rest, pos)
        if not m:
            raise SymmetryError("cannot tokenise %r at %r" % (sym, rest[pos:]))
        toks.append(m.group(0))
        pos = m.end()

    if not toks:
        raise SymmetryError("H-M symbol %r has no symmetry positions" % sym)

    # -1 / 1 are whole symbols in the triclinic case
    if toks == ["1"] or toks == ["-1"]:
        return lat, tuple(toks), axes, origin

    if len(toks) == 1:
        # short monoclinic: unique axis b by convention
        toks = ["1", toks[0], "1"]
    elif len(toks) == 2:
        # short trigonal/hexagonal (R-3c, P6/m) or cubic (P23, P2_1_3)
        if lat == "R" or (lat in "PFI" and toks[0].lstrip("-")[:1] in "2346"
                          and toks[1].lstrip("-")[:1] in "3mabcd"):
            pass  # leave as two positions
        else:
            raise SymmetryError("H-M symbol %r has 2 positions; expected 1 or 3" % sym)
    elif len(toks) > 3:
        raise SymmetryError("H-M symbol %r has %d positions; expected 3" % (sym, len(toks)))

    # accept both the old 'P m 3 m' and the modern 'P m -3 m'
    toks = ["-3" if t == "3" and len(toks) == 3 and toks.index(t) == 1 and lat in "PFI"
            and len(set(toks)) <= 3 and any(x in ("m", "n", "d", "a") for x in toks)
            else t for t in toks]

    return lat, tuple(toks), axes, origin


def _canon(sym):
    lat, toks, axes, origin = normalize(sym)
    return (lat, toks)


# ---------------------------------------------------------------------------
# standard names
# ---------------------------------------------------------------------------

_STD_INDEX = None


def _std_index():
    global _STD_INDEX
    if _STD_INDEX is None:
        idx = {}
        for n, name in enumerate(SGNAME, start=1):
            variants = [name]
            # Cubic short forms write 'm 3 m'; IT uses 'm -3 m'.  Do NOT apply
            # this to R/trigonal names (R 3 c must not become R -3 c).
            if name[0] in "PFI" and " 3 " in name:
                variants.append(name.replace(" 3 ", " -3 "))
            if name[0] in "PFI" and name.endswith(" 3"):
                variants.append(name[:-2] + " -3")
            for variant in variants:
                try:
                    k = _canon(variant)
                except SymmetryError:
                    continue
                idx.setdefault(k, n)
        _STD_INDEX = idx
    return _STD_INDEX


# ---------------------------------------------------------------------------
# setting search
# ---------------------------------------------------------------------------

def _derived_positions(ops):
    from .verify import derived_ortho_positions
    return derived_ortho_positions(ops)


def _mono_ortho_search(lat, toks):
    """
    Find (number, ops) whose derived symbol matches (lat, toks), over every
    unimodular basis change of every monoclinic/orthorhombic group.
    """
    hits = []
    for n in range(3, 75):
        std = hall_ops(STD_HALL[n])
        for P in candidates():
            try:
                ops = transform_ops(std, P, (0, 0, 0))
            except SymmetryError:
                continue
            if lattice_letter(ops) != lat:
                continue
            pos = _derived_positions(ops)
            ok = True
            for i in range(3):
                want = toks[i]
                if want == "1":
                    if pos[i] != {"1"}:
                        ok = False
                        break
                elif want not in pos[i]:
                    ok = False
                    break
            if ok:
                hits.append((n, ops, P))
                break   # first (simplest) P for this group is enough
    return hits


def hm_to_ops(sym, it_number=None):
    """
    Resolve an H-M string to (number, ops, note).

    `it_number`, if given, is used to disambiguate and is cross-checked.
    Raises SymmetryError when the string cannot be pinned down -- at which
    point the honest answer is "this CIF needs its symop loop".
    """
    lat, toks, axes, origin = normalize(sym)
    key = (lat, toks)

    n = _std_index().get(key)
    note = ""
    if n is not None:
        if it_number is not None and int(it_number) != n:
            raise SymmetryError(
                "H-M symbol %r resolves to space group %d but the CIF declares "
                "_space_group_IT_number %s" % (sym, n, it_number))
        if axes == "R":
            if n not in RHOMB_HALL:
                raise SymmetryError("space group %d has no rhombohedral setting" % n)
            return n, hall_ops(RHOMB_HALL[n]), "rhombohedral axes"
        if origin == 1:
            if n not in ORIGIN1_HALL:
                raise SymmetryError("space group %d has only one origin choice" % n)
            return n, hall_ops(ORIGIN1_HALL[n]), "origin choice 1"
        return n, hall_ops(STD_HALL[n]), note

    # not a standard name -- try alternative monoclinic / orthorhombic settings
    hits = _mono_ortho_search(lat, toks)
    if it_number is not None:
        hits = [h for h in hits if h[0] == int(it_number)]
    if len(hits) == 1:
        n, ops, P = hits[0]
        return n, ops, "non-standard setting recognised from H-M symbol; P=%s" % P.tolist()
    if len(hits) > 1:
        raise SymmetryError(
            "H-M symbol %r is ambiguous: it matches space groups %s. Supply "
            "_space_group_IT_number, or better, the symop loop."
            % (sym, sorted({h[0] for h in hits})))
    raise SymmetryError(
        "cannot resolve H-M symbol %r to a space group. This CIF needs a "
        "_space_group_symop_operation_xyz loop or a Hall symbol." % sym)

