"""Crystal layer: CIF load helpers (IT-standard Structure)."""

from __future__ import annotations

from pathlib import Path

from ebsdsim.crystal.elements import (
    ATOMIC_NUMBERS,
    ELEMENT_SYMBOLS,
    clean_element_symbol,
)
from ebsdsim.crystal.reader import (
    DEFAULT_B_ISO_ANGSTROM_SQ,
    Structure,
    SymmetryError,
    read_cif,
    read_cif_text,
    structure_from_block,
    warm_caches,
)

# Re-exports for historical imports.
clean_symbol = clean_element_symbol

_caches_warmed = False


def ensure_cif_caches() -> None:
    """Load cif_reader ``.npz`` tables once per process."""
    global _caches_warmed
    if not _caches_warmed:
        warm_caches()
        _caches_warmed = True


def load_structure(path: str | Path) -> Structure:
    """Read a CIF into the IT standard setting."""
    ensure_cif_caches()
    try:
        return read_cif(str(path))
    except SymmetryError as exc:
        raise ValueError(str(exc)) from exc


def load_structure_from_text(text: str, *, source: str = "") -> Structure:
    """Parse CIF text into an IT-standard :class:`~ebsdsim.crystal.reader.Structure`."""
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


__all__ = [
    "ATOMIC_NUMBERS",
    "ELEMENT_SYMBOLS",
    "DEFAULT_B_ISO_ANGSTROM_SQ",
    "Structure",
    "clean_symbol",
    "clean_element_symbol",
    "ensure_cif_caches",
    "load_structure",
    "load_structure_from_text",
]
