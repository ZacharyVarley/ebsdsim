"""CIF crystal parser backed by PyCifRW."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from io import StringIO

from CifFile import ReadCif

DEFAULT_B_ISO_ANGSTROM_SQ = 0.5

ELEMENT_SYMBOLS: tuple[str, ...] = (
    "",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac",
    "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
)

ATOMIC_NUMBERS: dict[str, int] = {sym: z for z, sym in enumerate(ELEMENT_SYMBOLS) if sym}
ATOMIC_NUMBERS["D"] = 1
ATOMIC_NUMBERS["T"] = 1

_CELL_LENGTH_TAGS = ("_cell_length_a", "_cell_length_b", "_cell_length_c")
_CELL_ANGLE_TAGS = ("_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")
_SG_NUMBER_TAGS = (
    "_space_group_it_number",
    "_symmetry_int_tables_number",
    "_space_group.it_number",
)
_HM_SYMBOL_TAGS = (
    "_space_group_name_h-m_alt",
    "_space_group.name_h-m_alt",
    "_symmetry_space_group_name_h-m",
    "_space_group_name_h-m",
)
_SYMOP_TAGS = (
    "_space_group_symop_operation_xyz",
    "_symmetry_equiv_pos_as_xyz",
)


@dataclass
class CIFCellParameters:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    space_group: int | None = None
    hm_symbol: str | None = None


@dataclass
class CIFAtomSite:
    label: str
    symbol: str
    atomic_number: int
    fract: tuple[float, float, float]
    occupancy: float
    b_iso: float


@dataclass
class CIFCrystal(CIFCellParameters):
    atom_sites: list[CIFAtomSite] = field(default_factory=list)
    sym_ops: list[str] = field(default_factory=lambda: ["x,y,z"])


def _is_missing(raw: str | None) -> bool:
    return raw is None or str(raw).strip() in ("?", ".", "")


def _strip_uncertainty(s: str) -> str:
    return re.sub(r"\(\d+\)$", "", s.strip())


def _as_float(raw: str | float | int | None) -> float:
    if raw is None:
        raise ValueError("CIF value is missing")
    if isinstance(raw, (int, float)):
        return float(raw)
    cleaned = _strip_uncertainty(str(raw))
    if _is_missing(cleaned):
        raise ValueError("CIF value is missing")
    return float(cleaned)


def _as_float_or(raw: str | float | int | None, fallback: float) -> float:
    if raw is None or _is_missing(str(raw)):
        return fallback
    try:
        return _as_float(raw)
    except ValueError:
        return fallback


def _keys_lower(block) -> dict[str, str]:
    return {k.lower(): k for k in block.keys()}


def _first_tag(block, *names: str) -> str | None:
    lower_map = _keys_lower(block)
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            return block[key]
    return None


def _loop_rows(block, *tags: str) -> list[dict[str, str]] | None:
    if not tags:
        return None
    lower_map = _keys_lower(block)
    first_key = lower_map.get(tags[0].lower())
    if first_key is None:
        return None
    resolved = [lower_map.get(tag.lower()) for tag in tags]
    n_rows = len(block[first_key])
    rows: list[dict[str, str]] = []
    for i in range(n_rows):
        rows.append(
            {tags[j]: block[resolved[j]][i] for j in range(len(tags)) if resolved[j] is not None}
        )
    return rows


def _read_block(text: str):
    cif = ReadCif(StringIO(text))
    if not cif.keys():
        raise ValueError("CIF has no data_ block")
    return cif[list(cif.keys())[0]]


def _clean_symbol(raw: str) -> str:
    stripped = re.sub(r"[0-9+\-.\(\)\[\]].*$", "", raw)
    m = re.match(r"^([A-Za-z]{1,2})", stripped)
    if not m:
        raise ValueError(f"Could not infer atom symbol from '{raw}'")
    candidate = m.group(1)[0].upper() + m.group(1)[1:].lower()
    if candidate in ATOMIC_NUMBERS:
        return candidate
    one_letter = candidate[0]
    if one_letter not in ATOMIC_NUMBERS:
        raise ValueError(f"Unsupported atom symbol '{candidate}' (from raw '{raw}')")
    return one_letter


def _cell_from_block(block) -> CIFCellParameters:
    sg_raw = _first_tag(block, *_SG_NUMBER_TAGS)
    hm_raw = _first_tag(block, *_HM_SYMBOL_TAGS)
    return CIFCellParameters(
        a=_as_float(_first_tag(block, _CELL_LENGTH_TAGS[0])),
        b=_as_float(_first_tag(block, _CELL_LENGTH_TAGS[1])),
        c=_as_float(_first_tag(block, _CELL_LENGTH_TAGS[2])),
        alpha=_as_float(_first_tag(block, _CELL_ANGLE_TAGS[0])),
        beta=_as_float(_first_tag(block, _CELL_ANGLE_TAGS[1])),
        gamma=_as_float(_first_tag(block, _CELL_ANGLE_TAGS[2])),
        space_group=None if _is_missing(sg_raw) else int(_as_float(sg_raw)),
        hm_symbol=None if _is_missing(hm_raw) else str(hm_raw).strip().strip("'\""),
    )


def parse_cif_cell_parameters(text: str) -> CIFCellParameters:
    return _cell_from_block(_read_block(text))


def parse_cif_crystal(text: str) -> CIFCrystal:
    block = _read_block(text)
    cell = _cell_from_block(block)

    site_rows = _loop_rows(
        block,
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_occupancy",
        "_atom_site_b_iso_or_equiv",
        "_atom_site_u_iso_or_equiv",
    )
    if site_rows is None:
        raise ValueError("CIF is missing an atom_site loop with fractional coordinates")

    atom_sites: list[CIFAtomSite] = []
    for i, row in enumerate(site_rows):
        label_raw = row.get("_atom_site_label", f"site{i}")
        symbol_raw = row.get("_atom_site_type_symbol", label_raw)
        if _is_missing(symbol_raw):
            symbol_raw = label_raw
        symbol = _clean_symbol(str(symbol_raw))
        b_raw = row.get("_atom_site_b_iso_or_equiv")
        u_raw = row.get("_atom_site_u_iso_or_equiv")
        if not _is_missing(b_raw):
            b_iso = _as_float(b_raw)
        elif not _is_missing(u_raw):
            b_iso = _as_float(u_raw) * 8 * math.pi * math.pi
        else:
            b_iso = DEFAULT_B_ISO_ANGSTROM_SQ
        atom_sites.append(
            CIFAtomSite(
                label=str(label_raw),
                symbol=symbol,
                atomic_number=ATOMIC_NUMBERS[symbol],
                fract=(
                    _as_float(row["_atom_site_fract_x"]),
                    _as_float(row["_atom_site_fract_y"]),
                    _as_float(row["_atom_site_fract_z"]),
                ),
                occupancy=_as_float_or(row.get("_atom_site_occupancy"), 1.0),
                b_iso=b_iso,
            )
        )

    sym_ops = ["x,y,z"]
    lower_map = _keys_lower(block)
    for tag in _SYMOP_TAGS:
        key = lower_map.get(tag.lower())
        if key is not None:
            sym_ops = [str(op).strip().strip("'\"") for op in block[key]]
            break

    return CIFCrystal(
        a=cell.a,
        b=cell.b,
        c=cell.c,
        alpha=cell.alpha,
        beta=cell.beta,
        gamma=cell.gamma,
        space_group=cell.space_group,
        hm_symbol=cell.hm_symbol,
        atom_sites=atom_sites,
        sym_ops=sym_ops,
    )
