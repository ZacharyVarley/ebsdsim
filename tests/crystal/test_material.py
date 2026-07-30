"""Manual material API tests."""

from __future__ import annotations

import numpy as np
import pytest
from ebsdsim.api import Atom as ApiAtom
from ebsdsim.api import Cell as ApiCell
from ebsdsim.api import Material as ApiMaterial
from ebsdsim.crystal._generated.sg_point_groups import SG_PG_FOR_SG
from ebsdsim.crystal.material import Atom, Cell, Material, resolve_space_group
from ebsdsim.crystal.spacegroup import pg_from_sg


def test_crystal_public_types_match_api_fields():
    assert Atom.__dataclass_fields__.keys() == ApiAtom.__dataclass_fields__.keys()
    assert Cell.__dataclass_fields__.keys() == ApiCell.__dataclass_fields__.keys()
    assert Material.__dataclass_fields__.keys() == ApiMaterial.__dataclass_fields__.keys()
    assert Atom.__dataclass_fields__["b_iso"].default is None
    assert Atom.__dataclass_fields__["occupancy"].default == 1.0
    assert Cell.__dataclass_fields__["space_group"].default == 1


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


def test_sg_pg_for_sg_table_shape_and_known_entries():
    """SG_PG_FOR_SG is the 230-entry SG→PG map (faithfulness vs blob verified in CI dump)."""
    assert SG_PG_FOR_SG.shape == (230,)
    assert SG_PG_FOR_SG.dtype == np.int64
    assert int(SG_PG_FOR_SG.min()) >= 1
    assert int(SG_PG_FOR_SG.max()) <= 32
    assert pg_from_sg(1) == 1
    assert pg_from_sg(225) == 32
    assert pg_from_sg(227) == 32
    np.testing.assert_array_equal(
        SG_PG_FOR_SG,
        np.asarray([pg_from_sg(sg) for sg in range(1, 231)], dtype=np.int64),
    )
