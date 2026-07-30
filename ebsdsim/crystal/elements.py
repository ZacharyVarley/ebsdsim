"""Crystal layer: atomic weights, element symbols, and shared constants."""

from __future__ import annotations

import re

# Room-temperature isotropic Debye–Waller default (Å²).
DEFAULT_B_ISO = 0.5
DEFAULT_B_ISO_ANGSTROM_SQ = DEFAULT_B_ISO

ATOMIC_WEIGHTS: tuple[float, ...] = (
    1.00794, 4.002602, 6.941, 9.012182, 10.811, 12.0107, 14.0067, 15.9994,
    18.9984032, 20.1797, 22.98976928, 24.305, 26.9815386, 28.0855,
    30.973762, 32.065, 35.453, 39.948, 39.0983, 40.078, 44.955912, 47.867,
    50.9415, 51.9961, 54.938045, 55.845, 58.933195, 58.6934, 63.546, 65.38,
    69.723, 72.64, 74.9216, 78.96, 79.904, 83.798, 85.4678, 87.62,
    88.90585, 91.224, 92.90638, 95.96, 98.9062, 101.07, 102.9055, 106.42,
    107.8682, 112.411, 114.818, 118.71, 121.76, 127.6, 126.90447,
    131.293, 132.9054519, 137.327, 138.90547, 140.116, 140.90765, 144.242,
    145.0, 150.36, 151.964, 157.25, 158.92535, 162.5, 164.93032, 167.259,
    168.93421, 173.054, 174.9668, 178.49, 180.94788, 183.84, 186.207,
    190.23, 192.217, 195.084, 196.966569, 200.59, 204.3833, 207.2,
    208.9804, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.03806,
    231.03588, 238.02891, 237.0, 244.0, 243.0, 247.0, 251.0, 252.0,
)


def atomic_weight(z: int) -> float:
    if z < 1 or z > len(ATOMIC_WEIGHTS):
        raise ValueError(f"Atomic number {z} out of range")
    return ATOMIC_WEIGHTS[z - 1]


# Index 0 unused so Z matches list position; D/T aliases for deuterium/tritium.
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
    "Es",
)

ATOMIC_NUMBERS: dict[str, int] = {sym: z for z, sym in enumerate(ELEMENT_SYMBOLS) if sym}
ATOMIC_NUMBERS["D"] = 1
ATOMIC_NUMBERS["T"] = 1


def element_symbol(z: int) -> str:
    if z < 1 or z >= len(ELEMENT_SYMBOLS) or not ELEMENT_SYMBOLS[z]:
        return f"Z{z}"
    return ELEMENT_SYMBOLS[z]


def atomic_number(symbol: str) -> int:
    """Return atomic number for a chemical symbol (case-insensitive)."""
    key = symbol.strip()
    key = key[0].upper() + key[1:].lower() if key else key
    z = ATOMIC_NUMBERS.get(key)
    if z is None:
        raise ValueError(f"Unknown element symbol: {symbol!r}")
    return z


def clean_element_symbol(raw: str) -> str:
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


def get_crystal_system(sg: int) -> int:
    """Return crystal-system index 1..7 for IT space-group number ``sg``.

    Name form lives in :func:`ebsdsim.crystal.reader.sym.crystal_system`
    (returns ``\"triclinic\"`` … ``\"cubic\"``); keep both — CIF reader needs
    strings, numeric callers want a compact int.
    """
    if sg <= 2:
        return 1
    if sg <= 15:
        return 2
    if sg <= 74:
        return 3
    if sg <= 142:
        return 4
    if sg <= 167:
        return 5
    if sg <= 194:
        return 6
    return 7
