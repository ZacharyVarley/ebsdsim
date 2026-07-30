"""Oriented pg_symbol resolution for save/load metadata (CPU-only)."""

from __future__ import annotations

import pytest
from ebsdsim.crystal.material import Atom, Cell, Material
from ebsdsim.crystal.pointgroup import resolve_oriented_symbol
from ebsdsim.engine.metadata import cell_metadata


def test_resolve_oriented_symbol_prefers_stored_symbol():
    assert resolve_oriented_symbol(18, pg_symbol="312", space_group=150) == "312"


def test_resolve_oriented_symbol_derives_from_space_group():
    assert resolve_oriented_symbol(18, space_group=149) == "312"
    assert resolve_oriented_symbol(18, space_group=150) == "32"


def test_resolve_oriented_symbol_raises_without_sg_when_ambiguous():
    with pytest.raises(ValueError, match="orientation-ambiguous"):
        resolve_oriented_symbol(18)


def test_cell_metadata_records_oriented_symbol_for_ambiguous_pair():
    for sg, expected in ((149, "312"), (150, "32")):
        mat = Material(
            cell=Cell(a=5.0, b=5.0, c=7.0, alpha=90.0, beta=90.0, gamma=120.0, space_group=sg),
            atoms=[Atom("Si", 0.0, 0.0, 0.0)],
        )
        cell = mat.to_simulation_cell()
        meta = cell_metadata(cell)
        assert meta["space_group"] == sg
        assert meta["pg_symbol"] == expected
