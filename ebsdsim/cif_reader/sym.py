"""
Space-group operators, Hall expansion, and IT reference tables.

  * Op algebra (exact 1/24 translations)
  * Hall symbol -> closed operator list (memoized)
  * STD_HALL / ORIGIN1_HALL / RHOMB_HALL / SGNAME
"""
from __future__ import annotations

import re
from fractions import Fraction

import numpy as np

# =============================================================================
# Operator algebra
# =============================================================================


DEN = 24
I3 = np.eye(3, dtype=np.int64)


class SymmetryError(Exception):
    """Raised when a CIF's symmetry information is absent or self-inconsistent."""


class Op:
    """A Seitz operator (W, w): x -> W x + w/DEN."""

    __slots__ = ("W", "w")

    def __init__(self, W, w):
        self.W = np.asarray(W, dtype=np.int64).reshape(3, 3)
        self.w = np.asarray(w, dtype=np.int64).reshape(3) % DEN

    # -- hashing / equality -------------------------------------------------
    def key(self):
        return (tuple(int(v) for v in self.W.ravel()), tuple(int(v) for v in self.w))

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        return isinstance(other, Op) and self.key() == other.key()

    def __repr__(self):
        return "Op(%s)" % self.xyz()

    # -- algebra ------------------------------------------------------------
    def __mul__(self, other):
        """self ∘ other  (apply `other` first)."""
        return Op(self.W @ other.W, self.W @ other.w + self.w)

    def inverse(self):
        Winv = np.round(np.linalg.inv(self.W)).astype(np.int64)
        if not np.array_equal(Winv @ self.W, I3):
            raise SymmetryError("non-invertible rotation part")
        return Op(Winv, -(Winv @ self.w))

    def det(self):
        return int(round(np.linalg.det(self.W)))

    def order(self):
        """Order of the rotation part (1..6)."""
        M = I3.copy()
        for k in range(1, 7):
            M = M @ self.W
            if np.array_equal(M, I3):
                return k
        raise SymmetryError("rotation part is not crystallographic")

    def intrinsic(self):
        """
        Intrinsic (screw / glide) part of the translation, in 1/DEN units.

            t_i = (1/k) * sum_{j=0}^{k-1} W^j w      with k = order(W)

        This is the part of w that cannot be removed by an origin shift; it is
        what distinguishes m from a c-glide, or 2 from 2_1.
        """
        k = self.order()
        acc = np.zeros(3, dtype=np.int64)
        M = I3.copy()
        for _ in range(k):
            acc += M @ self.w
            M = M @ self.W
        if np.any(acc % k):
            # can only happen if w is not on the 1/DEN lattice consistently
            raise SymmetryError("intrinsic translation is not on the 1/%d lattice" % DEN)
        return (acc // k) % DEN

    def is_translation(self):
        return np.array_equal(self.W, I3)

    def xyz(self):
        """Render back to a CIF-style 'x,y,z' string (for logging)."""
        out = []
        for i in range(3):
            parts = []
            for j, v in enumerate("xyz"):
                c = int(self.W[i, j])
                if c == 0:
                    continue
                if c == 1:
                    parts.append("+" + v)
                elif c == -1:
                    parts.append("-" + v)
                else:
                    parts.append("%+d%s" % (c, v))
            t = Fraction(int(self.w[i]), DEN)
            if t:
                parts.append("+%s" % t)
            s = "".join(parts).lstrip("+")
            out.append(s if s else "0")
        return ",".join(out)


IDENTITY = Op(I3, (0, 0, 0))


# ---------------------------------------------------------------------------
# parsing 'x,y,z' strings
# ---------------------------------------------------------------------------

_TERM = re.compile(r"[+-]?[^+-]+")


def _parse_component(s):
    """'1/2+x-y' -> ([1, -1, 0], Fraction(1, 2))"""
    s = s.strip().lower().replace(" ", "")
    if not s:
        raise SymmetryError("empty component in symmetry operation")
    row = [0, 0, 0]
    t = Fraction(0)
    for m in _TERM.finditer(s):
        tok = m.group(0)
        sign = 1
        if tok[0] == "+":
            tok = tok[1:]
        elif tok[0] == "-":
            sign = -1
            tok = tok[1:]
        var = None
        for v in "xyz":
            if v in tok:
                var = v
                tok = tok.replace(v, "")
                break
        tok = tok.strip("*").strip()
        if tok in ("", "+"):
            coef = Fraction(1)
        else:
            try:
                coef = Fraction(tok)
            except (ValueError, ZeroDivisionError):
                raise SymmetryError("cannot parse term %r in %r" % (m.group(0), s))
        if var is not None:
            if coef.denominator != 1:
                raise SymmetryError("non-integer coefficient on %s in %r" % (var, s))
            row["xyz".index(var)] += sign * int(coef)
        else:
            t += sign * coef
    return row, t


def op_from_xyz(s):
    """Parse a CIF symop string into an Op. Tolerant of spaces, '*', decimals."""
    s = s.strip().strip("'\"")
    parts = s.split(",")
    if len(parts) != 3:
        raise SymmetryError("expected 3 comma-separated components, got %r" % s)
    W = np.zeros((3, 3), dtype=np.int64)
    w = np.zeros(3, dtype=np.int64)
    for i, p in enumerate(parts):
        row, t = _parse_component(p)
        W[i, :] = row
        num = t * DEN
        if num.denominator != 1:
            raise SymmetryError("translation %s in %r is not a multiple of 1/%d" % (t, s, DEN))
        w[i] = int(num)
    return Op(W, w)


# ---------------------------------------------------------------------------
# group closure
# ---------------------------------------------------------------------------

def pack(ops):
    """Ops -> (n, 12) int64 array: 9 rotation entries then 3 translations."""
    if isinstance(ops, np.ndarray):
        return np.asarray(ops, dtype=np.int64)
    if not ops:
        return np.empty((0, 12), dtype=np.int64)
    return np.concatenate(
        [np.stack([o.W.ravel() for o in ops]),
         np.stack([o.w for o in ops])],
        axis=1,
    )


def unpack(arr):
    arr = np.asarray(arr, dtype=np.int64)
    return [Op(row[:9].reshape(3, 3), row[9:]) for row in arr]


def fingerprint(arr):
    """
    Hashable canonical key for a set of operators.

    Equal groups <=> equal fingerprints, so identifying a setting reduces to a
    dict lookup instead of a search.
    """
    arr = np.asarray(arr, dtype=np.int64)
    try:
        return np.sort(encode_rows(arr)).tobytes()
    except _Unpackable:
        return arr[np.lexsort(arr.T[::-1])].tobytes()


_IDENT_ROW = np.concatenate([I3.ravel(), np.zeros(3, dtype=np.int64)])

# Rotation entries of a crystallographic operator in a lattice basis are small;
# +/-4 is generous.  Packing a whole operator into one int64 turns dedup into a
# 1-D np.unique, which is several times faster than np.unique(axis=0)'s lexsort
# over 36k rows.
# 20 is chosen so the packed code still fits an int64:
#   base^9 * DEN^3 = 41^9 * 13824 ~ 4.5e18 < 2^63 ~ 9.2e18.
# Operators of a real CIF have entries in {-1,0,1}, but conjugating a hexagonal
# group by a sheared basis can push them past a tighter bound, and a crash there
# would be a crash on a merely-unusual file rather than an invalid one.
_WMAX = 20
_WBASE = 2 * _WMAX + 1


class _Unpackable(Exception):
    pass


def encode_rows(arr):
    """(n, 12) packed ops -> (n,) int64 codes. Injective."""
    arr = np.asarray(arr, dtype=np.int64)
    W = arr[:, :9]
    if np.abs(W).max(initial=0) > _WMAX:
        raise _Unpackable("rotation entry exceeds +/-%d" % _WMAX)
    code = np.zeros(arr.shape[0], dtype=np.int64)
    for i in range(9):
        code = code * _WBASE + (W[:, i] + _WMAX)
    return ((code * DEN + arr[:, 9]) * DEN + arr[:, 10]) * DEN + arr[:, 11]


def _unique_rows(arr):
    try:
        _, idx = np.unique(encode_rows(arr), return_index=True)
    except _Unpackable:
        return np.unique(arr, axis=0)   # slower, but never wrong
    return arr[idx]


def close_array(arr, max_order=1536):
    """
    Close a packed operator set under composition, vectorised.

    The obvious implementation -- a Python BFS calling Op.__mul__ -- costs ~70k
    interpreter round trips and tiny numpy matmuls for a group like Fd-3m, and
    dominates the whole load.  Composing all pairs at once and deduplicating
    with np.unique does the same work in a handful of numpy calls.
    """
    cur = np.vstack([np.asarray(arr, dtype=np.int64), _IDENT_ROW[None, :]])
    cur[:, 9:] %= DEN
    cur = _unique_rows(cur)
    while True:
        n = cur.shape[0]
        W = cur[:, :9].reshape(n, 3, 3)
        w = cur[:, 9:]
        nW = np.einsum("aij,bjk->abik", W, W).reshape(n * n, 9)
        nw = (np.einsum("aij,bj->abi", W, w) + w[:, None, :]).reshape(n * n, 3) % DEN
        merged = _unique_rows(np.vstack([cur, np.concatenate([nW, nw], axis=1)]))
        if merged.shape[0] == n:
            return merged
        if merged.shape[0] > max_order:
            raise SymmetryError(
                "operator set does not close (>%d elements) -- the CIF's symmetry "
                "operations are probably not a group" % max_order)
        cur = merged


def close_packed(ops, max_order=1536):
    """Close operators; return packed (n, 12) int64 without building Op list."""
    if isinstance(ops, np.ndarray):
        arr = np.asarray(ops, dtype=np.int64)
    elif not ops:
        arr = _IDENT_ROW[None, :].copy()
    else:
        arr = pack(ops)
    return close_array(arr, max_order)


def close_group(ops, max_order=1536):
    """
    Close a set of operators under composition modulo integer translations.

    Necessary because CIFs are inconsistent about what they list: some give the
    full coset decomposition, some give generators only, some give the ops of
    the primitive subgroup and declare centering separately.  Closing is cheap
    and makes all three cases converge to the same answer.
    """
    return unpack(close_packed(ops, max_order))


def opset(ops):
    return frozenset(o.key() for o in ops)


def centering_vectors(ops):
    """The pure-translation coset representatives, in 1/DEN units."""
    return sorted(tuple(int(v) for v in o.w) for o in ops if o.is_translation())


def lattice_letter(ops):
    """Infer the Bravais centering letter from the pure translations."""
    cv = set(centering_vectors(ops))
    cv.discard((0, 0, 0))
    h = 12  # 1/2
    t1, t2 = 8, 16  # 1/3, 2/3
    table = {
        frozenset(): "P",
        frozenset({(0, h, h)}): "A",
        frozenset({(h, 0, h)}): "B",
        frozenset({(h, h, 0)}): "C",
        frozenset({(h, h, h)}): "I",
        frozenset({(t2, t1, t1), (t1, t2, t2)}): "R",
        frozenset({(0, h, h), (h, 0, h), (h, h, 0)}): "F",
    }
    return table.get(frozenset(cv))


# =============================================================================
# Hall symbols
# =============================================================================


# ---------------------------------------------------------------------------
# lattice centring translations, in 1/24
# ---------------------------------------------------------------------------
_H = DEN // 2          # 1/2  = 12
_T1 = DEN // 3         # 1/3  =  8
_T2 = 2 * DEN // 3     # 2/3  = 16

LATTICE_TRANSLATIONS = {
    "P": [],
    "A": [(0, _H, _H)],
    "B": [(_H, 0, _H)],
    "C": [(_H, _H, 0)],
    "I": [(_H, _H, _H)],
    "R": [(_T2, _T1, _T1), (_T1, _T2, _T2)],
    "S": [(_T1, _T1, _T2), (_T2, _T2, _T1)],
    "T": [(_T1, _T2, _T1), (_T2, _T1, _T2)],
    "F": [(0, _H, _H), (_H, 0, _H), (_H, _H, 0)],
}

# ---------------------------------------------------------------------------
# rotation matrices
# ---------------------------------------------------------------------------
_ROT = {
    ("x", 1): ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ("x", 2): ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
    ("x", 3): ((1, 0, 0), (0, 0, -1), (0, 1, -1)),
    ("x", 4): ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    ("x", 6): ((1, 0, 0), (0, 1, -1), (0, 1, 0)),
    ("y", 1): ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ("y", 2): ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    ("y", 3): ((-1, 0, 1), (0, 1, 0), (-1, 0, 0)),
    ("y", 4): ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    ("y", 6): ((0, 0, 1), (0, 1, 0), (-1, 0, 1)),
    ("z", 1): ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ("z", 2): ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ("z", 3): ((0, -1, 0), (1, -1, 0), (0, 0, 1)),
    ("z", 4): ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ("z", 6): ((1, -1, 0), (1, 0, 0), (0, 0, 1)),
}

# 2-folds on the ' and " diagonal axes, keyed by the *preceding* rotation axis
_PRIME = {
    ("x", "'"): ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
    ("y", "'"): ((0, 0, -1), (0, -1, 0), (-1, 0, 0)),
    ("z", "'"): ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
    ("x", '"'): ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ("y", '"'): ((0, 0, 1), (0, -1, 0), (1, 0, 0)),
    ("z", '"'): ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
}

# 3-fold along [111]
_STAR3 = ((0, 0, 1), (1, 0, 0), (0, 1, 0))

# ---------------------------------------------------------------------------
# translation letters, in 1/24
# ---------------------------------------------------------------------------
_Q = DEN // 4  # 1/4 = 6
TRANS_LETTERS = {
    "a": (_H, 0, 0),
    "b": (0, _H, 0),
    "c": (0, 0, _H),
    "n": (_H, _H, _H),
    "u": (_Q, 0, 0),
    "v": (0, _Q, 0),
    "w": (0, 0, _Q),
    "d": (_Q, _Q, _Q),
}

_AXIS_VEC = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}

_ROTTOK = re.compile(r"^(-?)([12346])([xyz'\"*]?)([abcnuvwd1-6]*)$")


def _default_axis(index, order, prev_order):
    """Hall's implicit-axis rules."""
    if order == 1:
        return "z"                       # irrelevant, W = +/- I
    if index == 0:
        return "z"
    if order == 3:
        return "*"                       # e.g. the trailing '3' of cubic symbols
    if order == 2:
        if prev_order in (2, 4):
            return "x"
        if prev_order in (3, 6):
            return "'"
    return "z"


def _screw(axis, k, order):
    if axis not in _AXIS_VEC:
        raise SymmetryError("screw digit on non-principal axis %r" % axis)
    num = DEN * k
    if num % order:
        raise SymmetryError("screw %d_%d is not on the 1/%d lattice" % (order, k, DEN))
    step = num // order
    return np.array(_AXIS_VEC[axis], dtype=np.int64) * step


def _unquote(s):
    """
    Strip CIF quoting -- but only a MATCHED pair.

    A blanket .strip("'\"") eats the trailing double-prime of Hall symbols like
    R 3 2", silently demoting 2" to 2' and yielding the wrong trigonal group.
    """
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    return s.strip()


def hall_generators(symbol):
    """Parse a Hall symbol into (generator Ops, lattice letter, centrosymmetric)."""
    s = " ".join(_unquote(str(symbol)).split())
    if not s:
        raise SymmetryError("empty Hall symbol")

    # trailing origin shift, in twelfths
    shift = np.zeros(3, dtype=np.int64)
    m = re.search(r"\(([^()]*)\)\s*$", s)
    if m:
        parts = m.group(1).replace(",", " ").split()
        if len(parts) != 3:
            raise SymmetryError("bad origin shift in Hall symbol %r" % symbol)
        try:
            shift = np.array([int(p) for p in parts], dtype=np.int64) * (DEN // 12)
        except ValueError:
            raise SymmetryError("bad origin shift in Hall symbol %r" % symbol)
        s = s[: m.start()].strip()

    toks = s.split()
    lat = toks[0]
    centro = lat.startswith("-")
    if centro:
        lat = lat[1:]
    lat = lat.upper()
    if lat not in LATTICE_TRANSLATIONS:
        raise SymmetryError("unknown lattice symbol %r in Hall symbol %r" % (lat, symbol))

    gens = []
    prev_order = None
    prev_axis = None
    for i, tok in enumerate(toks[1:]):
        mm = _ROTTOK.match(tok)
        if not mm:
            raise SymmetryError("cannot parse rotation token %r in Hall symbol %r" % (tok, symbol))
        sign, N, axis, trans = mm.groups()
        N = int(N)
        if not axis:
            axis = _default_axis(i, N, prev_order)

        if axis in ("'", '"'):
            ref = prev_axis if prev_axis in _AXIS_VEC else "z"
            if N != 2:
                raise SymmetryError("order %d on a diagonal axis in %r" % (N, symbol))
            W = np.array(_PRIME[(ref, axis)], dtype=np.int64)
            eff_axis = axis
        elif axis == "*":
            if N != 3:
                raise SymmetryError("order %d on the * axis in %r" % (N, symbol))
            W = np.array(_STAR3, dtype=np.int64)
            eff_axis = "*"
        else:
            W = np.array(_ROT[(axis, N)], dtype=np.int64)
            eff_axis = axis

        if sign == "-":
            W = -W

        w = np.zeros(3, dtype=np.int64)
        for ch in trans:
            if ch.isdigit():
                w = w + _screw(eff_axis, int(ch), N)
            else:
                w = w + np.array(TRANS_LETTERS[ch], dtype=np.int64)

        gens.append(Op(W, w))
        prev_order = N
        prev_axis = eff_axis

    if centro:
        gens.append(Op(-I3, (0, 0, 0)))
    for t in LATTICE_TRANSLATIONS[lat]:
        gens.append(Op(I3, t))

    # origin shift:  (W, w) -> (W, w + (I - W) v)
    if np.any(shift):
        gens = [Op(g.W, g.w + (I3 - g.W) @ shift) for g in gens]

    return gens, lat, centro


_HALL_CACHE = {}


def hall_ops(symbol):
    """Hall symbol -> the full, closed operator list (process-memoized)."""
    hit = _HALL_CACHE.get(symbol)
    if hit is not None:
        return hit
    gens, _, _ = hall_generators(symbol)
    ops = close_group(gens)
    _HALL_CACHE[symbol] = ops
    return ops


def cached_hall_ops(symbol):
    """Return memoized ops for `symbol`, or None."""
    return _HALL_CACHE.get(symbol)


def store_hall_ops(symbol, ops):
    """Seed the Hall memo (e.g. from packed descent .npz)."""
    _HALL_CACHE[symbol] = ops


def clear_hall_cache():
    _HALL_CACHE.clear()


# =============================================================================
# IT tables
# =============================================================================

STD_HALL = {
    # ---- triclinic -------------------------------------------------------
    1: "P 1",
    2: "-P 1",
    # ---- monoclinic (unique axis b, cell choice 1) ------------------------
    3: "P 2y",
    4: "P 2yb",
    5: "C 2y",
    6: "P -2y",
    7: "P -2yc",
    8: "C -2y",
    9: "C -2yc",
    10: "-P 2y",
    11: "-P 2yb",
    12: "-C 2y",
    13: "-P 2yc",
    14: "-P 2ybc",
    15: "-C 2yc",
    # ---- orthorhombic ----------------------------------------------------
    16: "P 2 2",
    17: "P 2c 2",
    18: "P 2 2ab",
    19: "P 2ac 2ab",
    20: "C 2c 2",
    21: "C 2 2",
    22: "F 2 2",
    23: "I 2 2",
    24: "I 2b 2c",
    25: "P 2 -2",
    26: "P 2c -2",
    27: "P 2 -2c",
    28: "P 2 -2a",
    29: "P 2c -2ac",
    30: "P 2 -2bc",
    31: "P 2ac -2",
    32: "P 2 -2ab",
    33: "P 2c -2n",
    34: "P 2 -2n",
    35: "C 2 -2",
    36: "C 2c -2",
    37: "C 2 -2c",
    38: "A 2 -2",
    39: "A 2 -2b",
    40: "A 2 -2a",
    41: "A 2 -2ab",
    42: "F 2 -2",
    43: "F 2 -2d",
    44: "I 2 -2",
    45: "I 2 -2c",
    46: "I 2 -2a",
    47: "-P 2 2",
    48: "-P 2ab 2bc",     # origin choice 2
    49: "-P 2 2c",
    50: "-P 2ab 2b",      # origin choice 2
    51: "-P 2a 2a",
    52: "-P 2a 2bc",
    53: "-P 2ac 2",
    54: "-P 2a 2ac",
    55: "-P 2 2ab",
    56: "-P 2ab 2ac",
    57: "-P 2c 2b",
    58: "-P 2 2n",
    59: "-P 2ab 2a",      # origin choice 2
    60: "-P 2n 2ab",
    61: "-P 2ac 2ab",
    62: "-P 2ac 2n",
    63: "-C 2c 2",
    64: "-C 2ac 2",
    65: "-C 2 2",
    66: "-C 2 2c",
    67: "-C 2a 2",
    68: "-C 2a 2ac",      # origin choice 2
    69: "-F 2 2",
    70: "-F 2uv 2vw",     # origin choice 2
    71: "-I 2 2",
    72: "-I 2 2c",
    73: "-I 2b 2c",
    74: "-I 2b 2",
    # ---- tetragonal ------------------------------------------------------
    75: "P 4",
    76: "P 4w",
    77: "P 4c",
    78: "P 4cw",
    79: "I 4",
    80: "I 4bw",
    81: "P -4",
    82: "I -4",
    83: "-P 4",
    84: "-P 4c",
    85: "-P 4a",          # origin choice 2
    86: "-P 4bc",         # origin choice 2
    87: "-I 4",
    88: "-I 4ad",         # origin choice 2
    89: "P 4 2",
    90: "P 4ab 2ab",
    91: "P 4w 2c",
    92: "P 4abw 2nw",
    93: "P 4c 2",
    94: "P 4n 2n",
    95: "P 4cw 2c",
    96: "P 4nw 2abw",
    97: "I 4 2",
    98: "I 4bw 2bw",
    99: "P 4 -2",
    100: "P 4 -2ab",
    101: "P 4c -2c",
    102: "P 4n -2n",
    103: "P 4 -2c",
    104: "P 4 -2n",
    105: "P 4c -2",
    106: "P 4c -2ab",
    107: "I 4 -2",
    108: "I 4 -2c",
    109: "I 4bw -2",
    110: "I 4bw -2c",
    111: "P -4 2",
    112: "P -4 2c",
    113: "P -4 2ab",
    114: "P -4 2n",
    115: "P -4 -2",
    116: "P -4 -2c",
    117: "P -4 -2ab",
    118: "P -4 -2n",
    119: "I -4 -2",
    120: "I -4 -2c",
    121: "I -4 2",
    122: "I -4 2bw",
    123: "-P 4 2",
    124: "-P 4 2c",
    125: "-P 4a 2b",      # origin choice 2
    126: "-P 4a 2bc",     # origin choice 2
    127: "-P 4 2ab",
    128: "-P 4 2n",
    129: "-P 4a 2a",      # origin choice 2
    130: "-P 4a 2ac",     # origin choice 2
    131: "-P 4c 2",
    132: "-P 4c 2c",
    133: "-P 4ac 2b",     # origin choice 2
    134: "-P 4ac 2bc",    # origin choice 2
    135: "-P 4c 2ab",
    136: "-P 4n 2n",
    137: "-P 4ac 2a",     # origin choice 2
    138: "-P 4ac 2ac",    # origin choice 2
    139: "-I 4 2",
    140: "-I 4 2c",
    141: "-I 4bd 2",      # origin choice 2
    142: "-I 4bd 2c",     # origin choice 2
    # ---- trigonal (R groups on hexagonal axes) ---------------------------
    143: "P 3",
    144: "P 31",
    145: "P 32",
    146: "R 3",
    147: "-P 3",
    148: "-R 3",
    149: "P 3 2",
    150: "P 3 2\"",
    151: "P 31 2 (0 0 4)",
    152: "P 31 2\"",
    153: "P 32 2 (0 0 2)",
    154: "P 32 2\"",
    155: "R 3 2\"",
    156: "P 3 -2\"",
    157: "P 3 -2",
    158: "P 3 -2\"c",
    159: "P 3 -2c",
    160: "R 3 -2\"",
    161: "R 3 -2\"c",
    162: "-P 3 2",
    163: "-P 3 2c",
    164: "-P 3 2\"",
    165: "-P 3 2\"c",
    166: "-R 3 2\"",
    167: "-R 3 2\"c",
    # ---- hexagonal -------------------------------------------------------
    168: "P 6",
    169: "P 61",
    170: "P 65",
    171: "P 62",
    172: "P 64",
    173: "P 6c",
    174: "P -6",
    175: "-P 6",
    176: "-P 6c",
    177: "P 6 2",
    178: "P 61 2 (0 0 5)",
    179: "P 65 2 (0 0 1)",
    180: "P 62 2c (0 0 1)",
    181: "P 64 2c (0 0 5)",
    182: "P 6c 2c",
    183: "P 6 -2",
    184: "P 6 -2c",
    185: "P 6c -2",
    186: "P 6c -2c",
    187: "P -6 2",
    188: "P -6c 2",
    189: "P -6 -2",
    190: "P -6c -2c",
    191: "-P 6 2",
    192: "-P 6 2c",
    193: "-P 6c 2",
    194: "-P 6c 2c",
    # ---- cubic -----------------------------------------------------------
    195: "P 2 2 3",
    196: "F 2 2 3",
    197: "I 2 2 3",
    198: "P 2ac 2ab 3",
    199: "I 2b 2c 3",
    200: "-P 2 2 3",
    201: "-P 2ab 2bc 3",   # origin choice 2
    202: "-F 2 2 3",
    203: "-F 2uv 2vw 3",   # origin choice 2
    204: "-I 2 2 3",
    205: "-P 2ac 2ab 3",
    206: "-I 2b 2c 3",
    207: "P 4 2 3",
    208: "P 4n 2 3",
    209: "F 4 2 3",
    210: "F 4d 2 3",
    211: "I 4 2 3",
    212: "P 4acd 2ab 3",
    213: "P 4bd 2ab 3",
    214: "I 4bd 2c 3",
    215: "P -4 2 3",
    216: "F -4 2 3",
    217: "I -4 2 3",
    218: "P -4n 2 3",
    219: "F -4a 2 3",
    220: "I -4bd 2c 3",
    221: "-P 4 2 3",
    222: "-P 4a 2bc 3",    # origin choice 2
    223: "-P 4n 2 3",
    224: "-P 4bc 2bc 3",   # origin choice 2
    225: "-F 4 2 3",
    226: "-F 4a 2 3",
    227: "-F 4vw 2vw 3",   # origin choice 2
    228: "-F 4ud 2vw 3",   # origin choice 2
    229: "-I 4 2 3",
    230: "-I 4bd 2c 3",
}

# Rhombohedral (primitive) settings of the seven R groups.  Accepted on input;
# ebsdsim works in the hexagonal setting, so these are converted on the way in.
RHOMB_HALL = {
    146: "P 3*",
    148: "-P 3*",
    155: "P 3* 2",
    160: "P 3* -2",
    161: "P 3* -2n",
    166: "-P 3* 2",
    167: "-P 3* 2n",
}

# Alternative origin choice 1, for the 24 two-origin groups.  Used only to
# recognise and re-origin an incoming CIF that declares origin choice 1.
ORIGIN1_HALL = {
    48: "P 2 2 -1n",
    50: "P 2 2ab -1ab",
    59: "P 2 2ab -1ab",
    68: "C 2 2 -1ac",
    70: "F 2 2 -1d",
    85: "P 4ab -1ab",
    86: "P 4n -1n",
    88: "I 4bw -1bw",
    125: "P 4 2 -1ab",
    126: "P 4 2 -1n",
    129: "P 4ab 2ab -1ab",
    130: "P 4ab 2n -1ab",
    133: "P 4c 2 -1ab",
    134: "P 4c 2 -1n",
    137: "P 4ac 2b -1ab",
    138: "P 4ac 2ac -1ac",
    141: "I 4bw 2bw -1bw",
    142: "I 4bw 2aw -1abw",
    201: "P 2 2 3 -1n",
    203: "F 2 2 3 -1d",
    222: "P 4 2 3 -1n",
    224: "P 4n 2 3 -1n",
    227: "F 4d 2 3 -1d",
    228: "F 4d 2 3 -1ad",
}

# Crystal-system boundaries, by IT number.
SYSTEM_RANGES = [
    ("triclinic", 1, 2),
    ("monoclinic", 3, 15),
    ("orthorhombic", 16, 74),
    ("tetragonal", 75, 142),
    ("trigonal", 143, 167),
    ("hexagonal", 168, 194),
    ("cubic", 195, 230),
]


def crystal_system(n):
    for name, lo, hi in SYSTEM_RANGES:
        if lo <= n <= hi:
            return name
    raise ValueError("space group number out of range: %r" % (n,))


# ---------------------------------------------------------------------------
# Short Hermann-Mauguin names, IT numbers 1..230.
# Transcribed verbatim from De Graef's SYM_SGname (first 230 entries; the
# trailing 7 rhombohedral-setting entries are dropped).  Used as reference
# data by verify.py and by the H-M string fallback in hm.py.
# ---------------------------------------------------------------------------
SGNAME = [
    "P 1", "P -1",
    "P 2", "P 21", "C 2", "P m", "P c", "C m", "C c", "P 2/m",
    "P 21/m", "C 2/m", "P 2/c", "P 21/c", "C 2/c",
    "P 2 2 2", "P 2 2 21", "P 21 21 2", "P 21 21 21",
    "C 2 2 21", "C 2 2 2", "F 2 2 2", "I 2 2 2",
    "I 21 21 21", "P m m 2", "P m c 21", "P c c 2",
    "P m a 2", "P c a 21", "P n c 2", "P m n 21",
    "P b a 2", "P n a 21", "P n n 2", "C m m 2",
    "C m c 21", "C c c 2", "A m m 2", "A b m 2",
    "A m a 2", "A b a 2", "F m m 2", "F d d 2",
    "I m m 2", "I b a 2", "I m a 2", "P m m m",
    "P n n n", "P c c m", "P b a n", "P m m a",
    "P n n a", "P m n a", "P c c a", "P b a m",
    "P c c n", "P b c m", "P n n m", "P m m n",
    "P b c n", "P b c a", "P n m a", "C m c m",
    "C m c a", "C m m m", "C c c m", "C m m a",
    "C c c a", "F m m m", "F d d d", "I m m m",
    "I b a m", "I b c a", "I m m a",
    "P 4", "P 41", "P 42", "P 43",
    "I 4", "I 41", "P -4", "I -4",
    "P 4/m", "P 42/m", "P 4/n", "P 42/n",
    "I 4/m", "I 41/a", "P 4 2 2", "P 4 21 2",
    "P 41 2 2", "P 41 21 2", "P 42 2 2", "P 42 21 2",
    "P 43 2 2", "P 43 21 2", "I 4 2 2", "I 41 2 2",
    "P 4 m m", "P 4 b m", "P 42 c m", "P 42 n m",
    "P 4 c c", "P 4 n c", "P 42 m c", "P 42 b c",
    "I 4 m m", "I 4 c m", "I 41 m d", "I 41 c d",
    "P -4 2 m", "P -4 2 c", "P -4 21 m", "P -4 21 c",
    "P -4 m 2", "P -4 c 2", "P -4 b 2", "P -4 n 2",
    "I -4 m 2", "I -4 c 2", "I -4 2 m", "I -4 2 d",
    "P 4/m m m", "P 4/m c c", "P 4/n b m", "P 4/n n c",
    "P 4/m b m", "P 4/m n c", "P 4/n m m", "P 4/n c c",
    "P 42/m m c", "P 42/m c m", "P 42/n b c", "P 42/n n m",
    "P 42/m b c", "P 42/m n m", "P 42/n m c", "P 42/n c m",
    "I 4/m m m", "I 4/m c m", "I 41/a m d", "I 41/a c d",
    "P 3", "P 31", "P 32", "R 3",
    "P -3", "R -3", "P 3 1 2", "P 3 2 1",
    "P 31 1 2", "P 31 2 1", "P 32 1 2", "P 32 2 1",
    "R 3 2", "P 3 m 1", "P 3 1 m", "P 3 c 1",
    "P 3 1 c", "R 3 m", "R 3 c", "P -3 1 m",
    "P -3 1 c", "P -3 m 1", "P -3 c 1", "R -3 m",
    "R -3 c",
    "P 6", "P 61", "P 65", "P 62",
    "P 64", "P 63", "P -6", "P 6/m",
    "P 63/m", "P 6 2 2", "P 61 2 2", "P 65 2 2",
    "P 62 2 2", "P 64 2 2", "P 63 2 2", "P 6 m m",
    "P 6 c c", "P 63 c m", "P 63 m c", "P -6 m 2",
    "P -6 c 2", "P -6 2 m", "P -6 2 c", "P 6/m m m",
    "P 6/m c c", "P 63/m c m", "P 63/m m c",
    "P 2 3", "F 2 3", "I 2 3", "P 21 3",
    "I 21 3", "P m 3", "P n 3", "F m 3",
    "F d 3", "I m 3", "P a 3", "I a 3",
    "P 4 3 2", "P 42 3 2", "F 4 3 2", "F 41 3 2",
    "I 4 3 2", "P 43 3 2", "P 41 3 2", "I 41 3 2",
    "P -4 3 m", "F -4 3 m", "I -4 3 m", "P -4 3 n",
    "F -4 3 c", "I -4 3 d", "P m 3 m", "P n 3 n",
    "P m 3 n", "P n 3 m", "F m 3 m", "F m 3 c",
    "F d 3 m", "F d 3 c", "I m 3 m", "I a 3 d",
]
assert len(SGNAME) == 230

# The six orthorhombic axis settings, from De Graef's extendedOrthsettings.
# Each entry is the change-of-basis matrix P whose COLUMNS are the new basis
# vectors expressed in the old (abc) basis, i.e. (a',b',c') = (a,b,c) P.
ORTHO_SETTINGS = {
    "abc": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "ba-c": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    "cab": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "-cba": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "bca": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "a-cb": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
}

