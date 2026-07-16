"""CIF crystal types and load helpers (parser is :mod:`ebsdsim.cif_reader`)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ebsdsim.cif_reader import Structure, SymmetryError, read_cif, warm_caches

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

_caches_warmed = False


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


def clean_symbol(raw: str) -> str:
    """Infer an element symbol from a CIF type/label string."""
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


def ensure_cif_caches() -> None:
    """Load cif_reader ``.npz`` tables once per process."""
    global _caches_warmed
    if not _caches_warmed:
        warm_caches()
        _caches_warmed = True


def load_structure(path: str | Path) -> Structure:
    """Read a CIF into the IT standard setting (:func:`ebsdsim.cif_reader.read_cif`)."""
    ensure_cif_caches()
    try:
        return read_cif(str(path))
    except SymmetryError as exc:
        raise ValueError(str(exc)) from exc


def load_structure_from_text(text: str, *, source: str = "") -> Structure:
    """Parse CIF text into an IT-standard :class:`~ebsdsim.cif_reader.Structure`."""
    from ebsdsim.cif_reader import read_cif_text, structure_from_block

    ensure_cif_caches()
    try:
        blocks = read_cif_text(text)
        if not blocks:
            raise ValueError("CIF has no data_ block")
        name = next(iter(blocks))
        blk = blocks[name]
        src = source or f"data_{name}"
        return structure_from_block(blk, source=src)
    except SymmetryError as exc:
        raise ValueError(str(exc)) from exc
