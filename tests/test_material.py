"""Manual material API tests."""

from __future__ import annotations

import pytest

from ebsdsim.api import Atom, Cell, Material
from ebsdsim.material import resolve_space_group


def test_resolve_space_group_int():
    assert resolve_space_group(225) == (225, None)


def test_resolve_space_group_hm():
    assert resolve_space_group("Fm-3m") == (225, "Fm-3m")


def test_material_builds_cell():
    material = Material(
        cell=Cell(a=3.52, b=3.52, c=3.52, space_group="Fm-3m"),
        atoms=[Atom("Ni", 0.0, 0.0, 0.0)],
    )
    cell = material.to_simulation_cell()
    assert cell.space_group == 225
    assert cell.pg_num == 32
    assert cell.atom_types.size == 1


def test_unknown_space_group_raises():
    with pytest.raises(ValueError, match="Unknown space_group"):
        resolve_space_group("Not-A-Real-SG")
