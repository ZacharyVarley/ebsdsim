"""
CIF parse + Structure load into the IT standard setting.
"""
from __future__ import annotations

import re

import numpy as np

from .setting import (
    DEFAULT_EPS,
    P_HEX_FROM_RHOMB,
    expand_orbits,
    find_setting,
    fold_asymmetric,
    hall_ops,
    hm_to_ops,
    metric_family,
    ops_are_trivial,
    recover_ops_from_sites,
    sites_invariant,
    transform_cell,
    transform_coords,
)
from .sym import (
    ORIGIN1_HALL, RHOMB_HALL, STD_HALL, SymmetryError, crystal_system,
    close_group, close_packed, fingerprint, op_from_xyz,
)

# =============================================================================
# Minimal CIF extractor
# =============================================================================

_WANTED = frozenset({
    "_cell_length_a", "_cell_length_b", "_cell_length_c",
    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
    "_space_group_it_number", "_symmetry_int_tables_number",
    "_space_group_name_hall", "_symmetry_space_group_name_hall",
    "_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m",
    "_space_group_name_h-m_ref", "_space_group_name_h-m_full",
    "_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz",
    "_space_group_symop.operation_xyz",
    "_atom_site_label", "_atom_site_type_symbol",
    "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
    "_atom_site_occupancy",
    "_atom_site_u_iso_or_equiv", "_atom_site_b_iso_or_equiv",
})


class CifBlock(dict):
    """Lowercase tag -> str (single) or list[str] (loop column)."""

    def get_str(self, *names):
        for n in names:
            v = self.get(n.lower())
            if v is None:
                continue
            if isinstance(v, list):
                if not v:
                    continue
                v = v[0]
            s = str(v).strip()
            if s in ("", "?", "."):
                continue
            return s
        return None

    def get_loop(self, *names):
        for n in names:
            v = self.get(n.lower())
            if v is None:
                continue
            return list(v) if isinstance(v, list) else [v]
        return None


def read_cif_file(path):
    """Return {block_name: CifBlock} for every data_ block."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return read_cif_text(fh.read())


def read_cif_text(text):
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    s = text
    n = len(s)
    i = 0
    blocks = {}

    while True:
        i = _skip_ws(s, i, n)
        if i >= n:
            break
        if _kw(s, i, n, "data_"):
            i += 5
            j = i
            while j < n and s[j] not in " \t\n#":
                j += 1
            name = s[i:j]
            i = j
            blk, i = _parse_block(s, i, n)
            blocks[name] = blk
            continue
        if s[i] == "_":
            i = _skip_tag(s, i, n)
            i = _skip_value(s, i, n)
        elif _kw(s, i, n, "loop_"):
            i += 5
            i = _skip_loop(s, i, n)
        elif _kw(s, i, n, "save_"):
            i = _skip_save(s, i + 5, n)
        elif _kw(s, i, n, "global_"):
            i += 7
        else:
            i = _skip_value(s, i, n)

    if not blocks:
        raise SymmetryError("no data_ block found in CIF")
    return blocks


def _parse_block(s, i, n):
    blk = CifBlock()
    while i < n:
        i = _skip_ws(s, i, n)
        if i >= n:
            break
        c = s[i]
        if c == "_":
            tag, i = _read_tag(s, i, n)
            if tag in _WANTED:
                val, i = _read_value(s, i, n)
                blk[tag] = val
            else:
                i = _skip_value(s, i, n)
            continue
        if c in "lL" and _kw(s, i, n, "loop_"):
            i += 5
            i = _parse_or_skip_loop(s, i, n, blk)
            continue
        # next data_ / save_ / global_ ends this block
        if c in "dD" and _kw(s, i, n, "data_"):
            break
        if c in "sS" and _kw(s, i, n, "save_"):
            break
        if c in "gG" and _kw(s, i, n, "global_"):
            break
        i = _skip_value(s, i, n)
    return blk, i


def _parse_or_skip_loop(s, i, n, blk):
    tags = []
    while i < n:
        i = _skip_ws(s, i, n)
        if i >= n or s[i] != "_":
            break
        tag, i = _read_tag(s, i, n)
        tags.append(tag)
    if not tags:
        return i

    wanted = [(k, t) for k, t in enumerate(tags) if t in _WANTED]
    ncol = len(tags)
    if not wanted:
        return _skip_loop_values(s, i, n)

    cols = {t: [] for _, t in wanted}
    row = [None] * ncol
    col = 0
    while i < n:
        i = _skip_ws(s, i, n)
        if i >= n:
            break
        c = s[i]
        if c == "_" or (c in "lLdDgGsS" and (
                _kw(s, i, n, "loop_") or _kw(s, i, n, "data_")
                or _kw(s, i, n, "save_") or _kw(s, i, n, "global_"))):
            break
        val, i = _read_value(s, i, n)
        row[col] = val
        col += 1
        if col == ncol:
            for k, t in wanted:
                cols[t].append(row[k])
            col = 0
    if col:
        while col < ncol:
            row[col] = ""
            col += 1
        for k, t in wanted:
            cols[t].append(row[k])
    for t, colv in cols.items():
        blk[t] = colv
    return i


def _skip_loop(s, i, n):
    while i < n:
        i = _skip_ws(s, i, n)
        if i >= n or s[i] != "_":
            break
        i = _skip_tag(s, i, n)
    return _skip_loop_values(s, i, n)


def _skip_loop_values(s, i, n):
    while i < n:
        i = _skip_ws(s, i, n)
        if i >= n:
            break
        c = s[i]
        if c == "_" or (c in "lLdDgGsS" and (
                _kw(s, i, n, "loop_") or _kw(s, i, n, "data_")
                or _kw(s, i, n, "save_") or _kw(s, i, n, "global_"))):
            break
        i = _skip_value(s, i, n)
    return i


def _skip_save(s, i, n):
    """Skip a save_ frame body; leading 'save_' already consumed."""
    while i < n:
        i = _skip_ws(s, i, n)
        if i >= n:
            break
        if _kw(s, i, n, "save_"):
            # bare save_ ends frame (name may follow for nested — treat as end)
            j = i + 5
            if j >= n or s[j] in " \t\n#":
                return j
            # save_name — still inside; skip name and continue
            while j < n and s[j] not in " \t\n#":
                j += 1
            i = j
            continue
        if s[i] == "_":
            i = _skip_tag(s, i, n)
            i = _skip_value(s, i, n)
        elif _kw(s, i, n, "loop_"):
            i = _skip_loop(s, i + 5, n)
        elif _kw(s, i, n, "data_"):
            return i
        else:
            i = _skip_value(s, i, n)
    return i


# ---- primitives -----------------------------------------------------------

def _skip_ws(s, i, n):
    while i < n:
        c = s[i]
        if c in " \t\n":
            i += 1
            continue
        if c == "#":
            i += 1
            while i < n and s[i] != "\n":
                i += 1
            continue
        break
    return i


def _kw(s, i, n, kw):
    """True if s[i:] starts with keyword kw (ASCII case-insensitive)."""
    k = len(kw)
    if i + k > n:
        return False
    for j in range(k):
        a = s[i + j]
        b = kw[j]
        if a != b and a.lower() != b:
            return False
    if kw in ("data_", "save_"):
        return True
    return i + k == n or s[i + k] in " \t\n#"


def _read_tag(s, i, n):
    # s[i] == '_'
    j = i + 1
    while j < n and s[j] not in " \t\n#":
        j += 1
    tag = s[i:j].lower()
    return tag, j


def _skip_tag(s, i, n):
    j = i + 1
    while j < n and s[j] not in " \t\n#":
        j += 1
    return j


def _read_value(s, i, n):
    i = _skip_ws(s, i, n)
    if i >= n:
        return "", i
    c = s[i]
    if c == ";" and (i == 0 or s[i - 1] == "\n"):
        i += 1
        if i < n and s[i] == "\n":
            i += 1
        start = i
        while i < n:
            if s[i] == "\n" and i + 1 < n and s[i + 1] == ";":
                return s[start:i], i + 2
            i += 1
        return s[start:], n
    if c in "'\"":
        q = c
        i += 1
        start = i
        while i < n:
            if s[i] == q and (i + 1 >= n or s[i + 1] in " \t\n"):
                return s[start:i], i + 1
            i += 1
        return s[start:], n
    j = i
    while j < n and s[j] not in " \t\n":
        j += 1
    return s[i:j], j


def _skip_value(s, i, n):
    i = _skip_ws(s, i, n)
    if i >= n:
        return i
    c = s[i]
    if c == ";" and (i == 0 or s[i - 1] == "\n"):
        i += 1
        while i < n:
            if s[i] == "\n" and i + 1 < n and s[i + 1] == ";":
                return i + 2
            i += 1
        return n
    if c in "'\"":
        q = c
        i += 1
        while i < n:
            if s[i] == q and (i + 1 >= n or s[i + 1] in " \t\n"):
                return i + 1
            i += 1
        return n
    while i < n and s[i] not in " \t\n":
        i += 1
    return i


# =============================================================================
# Structure + read_cif
# =============================================================================

# Element symbols, for salvaging a species out of an atom_site_label.
_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_ELEMENTS_BY_LEN = sorted(_ELEMENTS, key=len, reverse=True)

# Accepts 0.5, .5, -.027, 3.481(2), etc.
_NUM = re.compile(
    r"^\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)\s*(?:\(\d+\))?\s*$"
)


def _num(v, what="value"):
    """Strip an esd in parentheses and parse. '3.481(2)' / '-.027(2)' -> float."""
    if v is None:
        return None
    s = str(v).strip().strip("'\"")
    if s in ("?", ".", ""):
        return None
    m = _NUM.match(s)
    if not m:
        raise SymmetryError("cannot parse %s %r as a number" % (what, v))
    return float(m.group(1))


def _first(block, *names):
    if isinstance(block, CifBlock):
        return block.get_str(*names)
    for n in names:
        for key in (n, n.lower()):
            try:
                if key in block:
                    v = block[key]
                    if isinstance(v, (list, tuple)):
                        v = v[0] if v else None
                    if v is None:
                        continue
                    s = str(v).strip().strip("'\"")
                    if s in ("?", ".", ""):
                        continue
                    return s
            except Exception:
                continue
    return None


def _loop(block, *names):
    if isinstance(block, CifBlock):
        return block.get_loop(*names)
    for n in names:
        for key in (n, n.lower()):
            try:
                if key in block:
                    v = block[key]
                    return list(v) if isinstance(v, (list, tuple)) else [v]
            except Exception:
                continue
    return None


def _species_from_label(label):
    s = re.sub(r"[^A-Za-z]", "", str(label))
    for e in _ELEMENTS_BY_LEN:
        if s[: len(e)].capitalize() == e and (len(s) == len(e) or not s[len(e)].islower()):
            return e
    for e in _ELEMENTS_BY_LEN:
        if s.capitalize().startswith(e):
            return e
    raise SymmetryError("cannot infer an element from atom label %r" % label)


def _sites_snapshot(species, coords, occupancies, uiso):
    """JSON-friendly asymmetric-unit snapshot."""
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    occ = np.asarray(occupancies, dtype=float).reshape(-1)
    u = np.asarray(uiso, dtype=float).reshape(-1)
    return [
        {
            "index": i,
            "symbol": str(species[i]),
            "fract": [float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])],
            "occupancy": float(occ[i]),
            "uiso": float(u[i]),
        }
        for i in range(len(species))
    ]


def _cell_snapshot(cell):
    a, b, c, alpha, beta, gamma = (float(x) for x in cell)
    return {
        "a_angstrom": a,
        "b_angstrom": b,
        "c_angstrom": c,
        "alpha_deg": alpha,
        "beta_deg": beta,
        "gamma_deg": gamma,
    }


def _declared_symmetry(blk):
    """Symmetry tags as written in the CIF (may disagree with resolved SG)."""
    return {
        "it_number": _first(blk, "_space_group_IT_number", "_symmetry_Int_Tables_number"),
        "hall": _first(blk, "_space_group_name_Hall", "_symmetry_space_group_name_Hall"),
        "hm": _first(
            blk,
            "_space_group_name_H-M_alt",
            "_symmetry_space_group_name_H-M",
            "_space_group_name_H-M_ref",
            "_space_group_name_H-M_full",
        ),
    }


class Structure:
    """Everything ebsdsim needs, guaranteed to be in the IT standard setting.

    ``cell`` / ``coords`` / … are the **used** IT-standard crystal.
    ``cif_input`` is the crystal **as read from the CIF** (before setting
    transform, R→hex, or sites-tree fold), for transparent master-pattern
    metadata.
    """

    def __init__(self, number, cell, species, coords, occupancies, uiso, setting,
                 provenance, cif_input=None):
        self.number = number
        self.cell = cell                    # (a, b, c, alpha, beta, gamma)
        self.species = species              # list[str]
        self.coords = coords                # (N, 3) fractional, asymmetric unit
        self.occupancies = occupancies
        self.uiso = uiso
        self.setting = setting              # Setting object: the applied (P, p)
        self.provenance = provenance        # str: where the symmetry came from
        self.cif_input = cif_input          # dict | None: as-read CIF snapshot

    def __repr__(self):
        return "<Structure SG%d %s n_atoms=%d>" % (self.number, self.species, len(self.coords))

    def log(self):
        """One line per load.  Put this in your run log -- see module docstring."""
        what = "standard setting" if self.setting.is_standard else self.setting.describe()
        if self.setting.note:
            what += " [%s]" % self.setting.note
        return (
            "SG %d (%s) | symmetry from: %s | %s | cell = "
            "%.5f %.5f %.5f  %.4f %.4f %.4f | %d sites"
            % (self.number, crystal_system(self.number), self.provenance, what,
               self.cell[0], self.cell[1], self.cell[2],
               self.cell[3], self.cell[4], self.cell[5], len(self.coords))
        )

    def used_cell_dict(self):
        """IT-standard cell + sites actually used for simulation."""
        out = _cell_snapshot(self.cell)
        out.update({
            "space_group": int(self.number),
            "setting": "IT standard",
            "origin_choice": 2 if int(self.number) in _TWORIG else None,
            "transformed": not self.setting.is_standard,
            "setting_note": self.setting.note or None,
            "setting_describe": (
                None if self.setting.is_standard else self.setting.describe()
            ),
            "rhombohedral_input": bool(getattr(self.setting, "rhombohedral_input", False)),
            "P": self.setting.P.astype(int).tolist(),
            "p": [int(x) for x in self.setting.p],
            "n_sites": len(self.species),
            "sites": _sites_snapshot(self.species, self.coords, self.occupancies, self.uiso),
        })
        return out

    def metadata(self):
        """Payload for master-pattern ``meta_json``: CIF-as-read + cell-as-used."""
        return {
            "symmetry_provenance": self.provenance,
            "cif_input": self.cif_input,
            "cell": self.used_cell_dict(),
        }


# IT numbers with two origin choices; standard output is origin 2.
_TWORIG = frozenset({
    48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130, 133, 134, 137, 138,
    141, 142, 201, 203, 222, 224, 227, 228,
})


def symmetry_from_block(block):
    """
    Return (ops, it_number_or_None, provenance, useful).

    `useful` is True only for a real symop loop or Hall symbol (authoritative
    operators).  A non-trivial H-M symbol or IT number is a *trusted group
    identity*: the crystal has that space group, possibly in a non-standard
    setting.  Those come back with useful=False so the reader resolves the
    setting from cell+sites among tabulated settings of that group.
    """
    it = _first(block, "_space_group_IT_number", "_symmetry_Int_Tables_number")
    it = int(float(it)) if it else None

    xyz = _loop(block,
                "_space_group_symop_operation_xyz",
                "_symmetry_equiv_pos_as_xyz",
                "_space_group_symop.operation_xyz")
    if xyz:
        ops = [op_from_xyz(s) for s in xyz if str(s).strip() not in ("", "?", ".")]
        if ops:
            useful = not ops_are_trivial(ops)
            return ops, it, "_space_group_symop_operation_xyz (%d listed)" % len(ops), useful

    hall = _first(block, "_space_group_name_Hall", "_symmetry_space_group_name_Hall")
    if hall:
        ops = hall_ops(hall)
        return ops, it, "Hall symbol %r" % hall, not ops_are_trivial(ops)

    hm = _first(block,
                "_space_group_name_H-M_alt",
                "_symmetry_space_group_name_H-M",
                "_space_group_name_H-M_ref",
                "_space_group_name_H-M_full")
    if hm:
        hm_stripped = re.sub(r"\s+", " ", hm.strip().strip("'\""))
        trivial_hm = hm_stripped.replace(" ", "").lower() in ("p1", "c1")
        if trivial_hm:
            n = it or 1
            return hall_ops(STD_HALL[n]), n, "H-M symbol %r (trivial)" % hm, False
        # IT number is a trusted group identity.  Prefer it over an expensive
        # H-M setting search (non-standard monoclinic H-M can take tens of seconds).
        if it:
            return None, it, "H-M %r + IT %d" % (hm, it), False
        n, ops, note = hm_to_ops(hm, it_number=it)
        extra = (" (%s)" % note) if note else ""
        return ops, n, "H-M symbol %r -> SG%d%s" % (hm, n, extra), False

    if it:
        return None, it, "IT number %d" % it, False

    return None, None, "no declared symmetry", False


def read_cif(path, block=None, eps=DEFAULT_EPS):
    """
    Read a CIF and return a Structure in the IT standard setting.

    Raises SymmetryError rather than guessing whenever the file is internally
    inconsistent.  A wrong master pattern is worse than a failed load.
    """
    blocks = read_cif_file(str(path))
    names = list(blocks.keys())
    if block is None:
        block = names[0] if len(names) == 1 else next(
            (b for b in names if _loop(blocks[b], "_atom_site_fract_x")), names[0])
    elif block not in blocks:
        # allow data_ prefix omission / case
        hit = next((n for n in names if n == block or n.lower() == str(block).lower()), None)
        if hit is None:
            raise SymmetryError("CIF has no data block %r (have %s)" % (block, names))
        block = hit
    blk = blocks[block]
    return structure_from_block(blk, source="%s [%s]" % (path, block), eps=eps)


def _parse_cell_atoms(blk):
    cell = tuple(
        _num(_first(blk, "_cell_length_" + k), "_cell_length_" + k) for k in "abc"
    ) + tuple(
        _num(_first(blk, "_cell_angle_" + k), "_cell_angle_" + k)
        for k in ("alpha", "beta", "gamma")
    )
    if any(v is None for v in cell):
        raise SymmetryError("CIF is missing one or more cell parameters")

    labels = _loop(blk, "_atom_site_label")
    fx = _loop(blk, "_atom_site_fract_x")
    fy = _loop(blk, "_atom_site_fract_y")
    fz = _loop(blk, "_atom_site_fract_z")
    if not (labels and fx and fy and fz):
        raise SymmetryError("CIF is missing the _atom_site fractional coordinate loop")
    types = _loop(blk, "_atom_site_type_symbol")
    occs = _loop(blk, "_atom_site_occupancy")
    uiso = _loop(blk, "_atom_site_U_iso_or_equiv")
    biso = _loop(blk, "_atom_site_B_iso_or_equiv")

    n = len(labels)
    species = []
    for i in range(n):
        raw = (types[i] if types and i < len(types) else labels[i])
        raw = re.sub(r"[0-9+\-.]+$", "", str(raw).strip().strip("'\""))
        species.append(_species_from_label(raw or labels[i]))

    coords = np.array([[_num(fx[i], "fract_x"), _num(fy[i], "fract_y"), _num(fz[i], "fract_z")]
                       for i in range(n)], dtype=float)
    occ = np.array([(_num(occs[i], "occupancy") if occs and i < len(occs) else None) or 1.0
                    for i in range(n)], dtype=float)
    if uiso:
        u = np.array([(_num(uiso[i], "U_iso") or 0.0) for i in range(n)], dtype=float)
    elif biso:
        u = np.array([(_num(biso[i], "B_iso") or 0.0) for i in range(n)], dtype=float) / (8 * np.pi ** 2)
    else:
        u = np.zeros(n)
    return cell, species, coords, occ, u


def _std_ops(k):
    return hall_ops(STD_HALL[k])


def _sites_support_ops(prior_ops, species, coords, eps):
    """True if prior_ops leave the site set invariant (expanding an asym unit first)."""
    prior_ops = close_group(prior_ops)
    sp_e, xyz_e = expand_orbits(prior_ops, species, coords, eps=eps)
    return sites_invariant(prior_ops, sp_e, xyz_e, eps=eps)


def pack_key(ops):
    return fingerprint(close_packed(ops))


def _ops_candidates_for_group(n, preferred=None):
    """
    Operator sets for trusted space group n.

    Keep this list small: full metric-legal P sets are huge (120 monoclinic,
    thousands triclinic) and each close/fingerprint is expensive.  Unique-axis
    / cell-choice changes that actually appear in CIFs are a handful of
    conjugations of the standard ops.
    """
    from .setting import transform_ops

    zero = np.zeros(3, dtype=np.int64)
    std = hall_ops(STD_HALL[n])
    out = []
    seen = set()

    def _add(ops):
        k = pack_key(ops)
        if k not in seen:
            seen.add(k)
            out.append(ops)

    if preferred is not None:
        _add(preferred)
    _add(std)
    if n in ORIGIN1_HALL:
        _add(hall_ops(ORIGIN1_HALL[n]))
    if n in RHOMB_HALL:
        _add(hall_ops(RHOMB_HALL[n]))

    # Compact change-of-basis set covering IT monoclinic unique-axis / cell
    # choices and orthorhombic axis permutations (not the full metric filter).
    sysname = crystal_system(n)
    if sysname == "monoclinic":
        # unique b (std), unique c, unique a; plus a common cell-choice shear
        qs = [
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.int64),   # a->c
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int64),   # cab-like
            np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.int64),
            np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.int64),
            np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=np.int64),   # cell choice
            np.array([[1, 0, -1], [0, 1, 0], [0, 0, 1]], dtype=np.int64),
        ]
    elif sysname == "orthorhombic":
        qs = [
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int64),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int64),
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.int64),
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.int64),
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.int64),
        ]
    else:
        qs = []

    for Q in qs:
        if abs(round(float(np.linalg.det(Q)))) != 1:
            continue
        _add(transform_ops(std, Q, zero))
    return out


def _standard_metric_penalty(n, cell):
    """0 = cell matches the IT conventional metric for group n's system."""
    a, b, c, al, be, ga = (float(x) for x in cell)
    sysname = crystal_system(n)
    ang = lambda x, t: abs(x - t)
    eq = lambda x, y: abs(x - y)
    if sysname == "monoclinic":
        return ang(al, 90.0) + ang(ga, 90.0) + (0.0 if ang(be, 90.0) > 0.5 else 10.0)
    if sysname == "orthorhombic":
        return ang(al, 90.0) + ang(be, 90.0) + ang(ga, 90.0)
    if sysname == "tetragonal":
        return eq(a, b) + ang(al, 90.0) + ang(be, 90.0) + ang(ga, 90.0)
    if sysname in ("trigonal", "hexagonal"):
        return eq(a, b) + ang(al, 90.0) + ang(be, 90.0) + ang(ga, 120.0)
    if sysname == "cubic":
        return eq(a, b) + eq(b, c) + ang(al, 90.0) + ang(be, 90.0) + ang(ga, 90.0)
    return 0.0


def _resolve_group_setting(n, cell, species, coords, eps, preferred_ops=None):
    """
    Trusted space-group number n: find which tabulated setting of that group
    fits the deposited cell+sites.  Does not second-guess n.
    """
    best = None  # (penalty, setting)
    for ops in _ops_candidates_for_group(n, preferred=preferred_ops):
        if not _sites_support_ops(ops, species, coords, eps):
            continue
        try:
            st = find_setting(ops, n, _std_ops)
        except SymmetryError:
            continue
        cell2 = transform_cell(cell, st.P)
        if st.rhombohedral_input:
            from .setting import P_HEX_FROM_RHOMB
            cell2 = transform_cell(cell2, P_HEX_FROM_RHOMB)
        pen = _standard_metric_penalty(n, cell2)
        if best is None or pen < best[0]:
            best = (pen, st)
            if pen == 0.0:
                break
    if best is None:
        raise SymmetryError(
            "could not match cell/sites to any tabulated setting of space group %d "
            "(the IT/H-M group identity is trusted; the setting could not be resolved)"
            % n
        )
    return best[1]


def structure_from_block(blk, source="", eps=DEFAULT_EPS):
    ops, it, provenance, useful = symmetry_from_block(blk)
    cell, species, coords, occ, u = _parse_cell_atoms(blk)

    # Freeze as-read CIF state before any fold / R→hex / (P,p) transform.
    declared = _declared_symmetry(blk)
    cif_input = {
        "source": source or None,
        "declared_symmetry": {
            "it_number": (
                int(float(declared["it_number"])) if declared["it_number"] else None
            ),
            "hall": declared["hall"],
            "hm": declared["hm"],
        },
        "cell": _cell_snapshot(cell),
        "n_sites": len(species),
        "sites": _sites_snapshot(species, coords, occ, u),
    }

    setting = None
    note_bits = [provenance]
    fold_from_sites = False
    sites_ops = None

    if useful and ops is not None:
        # Listed symops / Hall: authoritative operators.
        setting = find_setting(ops, it, _std_ops)

    elif it is not None and not (ops is not None and ops_are_trivial(ops)):
        # Trusted H-M / IT group identity: resolve setting from cell+sites.
        # (Trivial P1 claims fall through to pure sites recovery below.)
        preferred = ops if (ops is not None and not ops_are_trivial(ops)) else None
        setting = _resolve_group_setting(
            it, cell, species, coords, eps, preferred_ops=preferred)
        note_bits.append("setting resolved from cell/sites")

    else:
        # No trusted group (nothing, or trivial P1): recover from sites alone.
        try:
            sites_ops = recover_ops_from_sites(cell, species, coords, eps=eps)
            setting = find_setting(sites_ops, None, _std_ops)
            note_bits = [
                "sites+Hall SG%d (tree descent)" % setting.number,
                "ops path trivial/absent",
            ]
            fold_from_sites = True
        except SymmetryError:
            setting = None

    if setting is None:
        raise SymmetryError(
            "cannot determine symmetry: no useful operators and site tree descent failed"
        )

    if fold_from_sites and sites_ops is not None:
        species, coords, idx = fold_asymmetric(sites_ops, species, coords, eps=eps)
        occ = occ[idx]
        u = u[idx]

    number = setting.number
    prov = " | ".join(b for b in note_bits if b)

    if setting.rhombohedral_input:
        from .setting import P_HEX_FROM_RHOMB
        Pr = P_HEX_FROM_RHOMB.astype(float)
        cell = transform_cell(cell, Pr)
        coords = (coords @ np.linalg.inv(Pr).T) % 1.0

    cell2 = transform_cell(cell, setting.P)
    coords2 = transform_coords(coords, setting.P, setting.p)

    return Structure(
        number, cell2, species, coords2, occ, u, setting,
        prov + ((" from %s" % source) if source else ""),
        cif_input=cif_input,
    )
