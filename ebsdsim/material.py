"""Build internal simulation cells from user-facing material specifications."""

from __future__ import annotations

import re

from ebsdsim.cif import ATOMIC_NUMBERS, CIFAtomSite, CIFCrystal
from ebsdsim.structure import build_cell_from_cif

_DEFAULT_B_ISO_ANGSTROM_SQ = 0.5

# Common Hermann–Mauguin symbols → IT number (normalized: no spaces, lower case).
_HM_TO_SG: dict[str, int] = {
    "fm-3m": 225,
    "fm3m": 225,
    "fd-3m": 227,
    "fd3m": 227,
    "im-3m": 229,
    "im3m": 229,
    "pm-3m": 221,
    "pm3m": 221,
    "p63/mmc": 194,
    "p6_3/mmc": 194,
    "p63mmc": 194,
    "p4/nmm": 129,
    "i41/amd": 141,
    "fd-3": 203,
    "f-43m": 216,
}


def resolve_space_group(space_group: int | str) -> tuple[int | None, str | None]:
    """Return (sg_number, hm_symbol) from an integer or Hermann–Mauguin string."""
    if isinstance(space_group, int):
        if 1 <= space_group <= 230:
            return space_group, None
        raise ValueError(f"space_group must be in [1, 230], got {space_group}")
    key = re.sub(r"\s+", "", str(space_group).strip().lower())
    key = key.replace("_", "")
    if key.isdigit():
        sg = int(key)
        if 1 <= sg <= 230:
            return sg, str(space_group)
        raise ValueError(f"space_group must be in [1, 230], got {sg}")
    if key in _HM_TO_SG:
        return _HM_TO_SG[key], str(space_group)
    raise ValueError(
        f"Unknown space_group symbol {space_group!r}; pass an IT number (1–230) "
        f"or add the symbol to the lookup table."
    )


def build_cell_from_material(
    *,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    space_group: int | str,
    atoms: list,
) -> "Cell":
    """Build an internal :class:`ebsdsim.types.Cell` from lattice + site specs (Å, degrees)."""
    from ebsdsim.types import Cell  # noqa: PLC0415 — avoid circular import at module load

    sg_num, hm = resolve_space_group(space_group)
    sites: list[CIFAtomSite] = []
    for atom in atoms:
        symbol = atom.element if hasattr(atom, "element") else atom.symbol
        sym = str(symbol).strip().capitalize()
        z = ATOMIC_NUMBERS.get(sym)
        if z is None:
            raise ValueError(f"Unknown element symbol: {symbol!r}")
        b_iso = getattr(atom, "b_iso", None)
        if b_iso is None:
            b_iso = _DEFAULT_B_ISO_ANGSTROM_SQ
        sites.append(
            CIFAtomSite(
                label=sym,
                symbol=sym,
                atomic_number=z,
                fract=(float(atom.x), float(atom.y), float(atom.z)),
                occupancy=float(atom.occupancy),
                b_iso=float(b_iso),
            )
        )
    crystal = CIFCrystal(
        a=float(a),
        b=float(b),
        c=float(c),
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        space_group=sg_num,
        hm_symbol=hm,
        atom_sites=sites,
        sym_ops=[],
    )
    return build_cell_from_cif(crystal)
