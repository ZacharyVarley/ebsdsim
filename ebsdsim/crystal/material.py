"""Public Material / Atom / Cell (Å) and Material → Structure adapter."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from ebsdsim.crystal.build import build_cell_from_structure
from ebsdsim.crystal.elements import (
    DEFAULT_B_ISO,
    DEFAULT_B_ISO_ANGSTROM_SQ,
    atomic_number,
    clean_element_symbol,
)
from ebsdsim.crystal.reader import Structure
from ebsdsim.crystal.reader.setting import Setting
from ebsdsim.crystal.reader.sym import I3
from ebsdsim.crystal.simcell import SimCell
from ebsdsim.crystal.spacegroup import require_space_group

_UISO_FROM_BISO = 1.0 / (8.0 * math.pi * math.pi)

# Re-export for callers that historically imported from material.
__all__ = [
    "Atom",
    "Cell",
    "Material",
    "DEFAULT_B_ISO",
    "DEFAULT_B_ISO_ANGSTROM_SQ",
    "resolve_space_group",
    "structure_from_material",
    "build_cell_from_material",
]

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


@dataclass(frozen=True)
class Atom:
    """Crystallographic site in fractional coordinates (direct lattice).

    Parameters
    ----------
    element : str
        Chemical symbol (e.g. ``"Ni"``, ``"Ga"``).
    x, y, z : float
        Fractional coordinates in the direct unit cell.
    occupancy : float, optional
        Site occupancy in ``[0, 1]``. Default ``1.0``.
    b_iso : float or None, optional
        Isotropic Debye–Waller factor in Å². When ``None``, a room-temperature
        default is used during structure-factor evaluation.
    """

    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_iso: float | None = None  # Å²; default room-temperature fallback


@dataclass(frozen=True)
class Cell:
    """Unit-cell lattice parameters (Å and degrees).

    Parameters
    ----------
    a, b, c : float
        Lattice lengths in ångströms.
    alpha, beta, gamma : float, optional
        Interaxial angles in degrees. Default ``90.0`` each.
    space_group : int or str, optional
        International Tables space-group number or Hermann–Mauguin symbol.
        Default ``1`` (triclinic ``P1``).
    """

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    space_group: int | str = 1


@dataclass
class Material:
    """Crystal specification for master-pattern simulation.

    Parameters
    ----------
    cell : Cell
        Unit-cell geometry and space group.
    atoms : list of Atom
        Atomic sites in fractional coordinates.
    name : str, optional
        Human-readable label stored in output metadata. Default ``""``.
    """

    cell: Cell
    atoms: list[Atom]
    name: str = ""

    def to_simulation_cell(self) -> SimCell:
        """Build the internal simulation cell used by the GPU pipeline."""
        return build_cell_from_material(
            a=self.cell.a,
            b=self.cell.b,
            c=self.cell.c,
            alpha=self.cell.alpha,
            beta=self.cell.beta,
            gamma=self.cell.gamma,
            space_group=self.cell.space_group,
            atoms=self.atoms,
        )


def resolve_space_group(space_group: int | str) -> tuple[int | None, str | None]:
    """Return (sg_number, hm_symbol) from an integer or Hermann–Mauguin string."""
    if isinstance(space_group, int):
        return require_space_group(space_group), None
    key = re.sub(r"\s+", "", str(space_group).strip().lower())
    key = key.replace("_", "")
    if key.isdigit():
        sg = int(key)
        return require_space_group(sg), str(space_group)
    if key in _HM_TO_SG:
        return _HM_TO_SG[key], str(space_group)
    raise ValueError(
        f"Unknown space_group symbol {space_group!r}; pass an IT number (1–230) "
        f"or add the symbol to the lookup table."
    )


def structure_from_material(
    *,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    space_group: int | str,
    atoms: list,
    name: str = "",
) -> Structure:
    """Adapt a manual lattice + site list into an IT-standard :class:`Structure`."""
    sg_num, hm = resolve_space_group(space_group)
    assert sg_num is not None

    n = len(atoms)
    coords = np.zeros((n, 3), dtype=np.float64)
    occ = np.ones(n, dtype=np.float64)
    uiso = np.empty(n, dtype=np.float64)
    species: list[str] = []
    # Per-site Python loop: Atom is a dataclass list, not a numeric array.
    for i, atom in enumerate(atoms):
        symbol = atom.element if hasattr(atom, "element") else atom.symbol
        sym = clean_element_symbol(str(symbol))
        atomic_number(sym)  # validate
        species.append(sym)
        coords[i, 0] = float(atom.x)
        coords[i, 1] = float(atom.y)
        coords[i, 2] = float(atom.z)
        occ[i] = float(atom.occupancy)
        b_iso = getattr(atom, "b_iso", None)
        if b_iso is None:
            b_iso = DEFAULT_B_ISO
        uiso[i] = float(b_iso) * _UISO_FROM_BISO

    setting = Setting(sg_num, I3.copy(), np.zeros(3, dtype=np.int64), note="manual Material")
    provenance = "manual Material"
    if hm:
        provenance = f"manual Material ({hm})"
    if name:
        provenance = f"{provenance} [{name}]"
    return Structure(
        sg_num,
        (float(a), float(b), float(c), float(alpha), float(beta), float(gamma)),
        species,
        coords,
        occ,
        uiso,
        setting,
        provenance,
        cif_input=None,
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
) -> SimCell:
    """Build an internal :class:`~ebsdsim.crystal.simcell.SimCell` from lattice + sites (Å)."""
    structure = structure_from_material(
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        space_group=space_group,
        atoms=atoms,
    )
    return build_cell_from_structure(structure)
