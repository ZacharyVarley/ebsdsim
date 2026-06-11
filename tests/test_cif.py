"""CIF parsing and cell construction tests."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import numpy as np

from ebsdsim.api import Atom, Cell, Material
from ebsdsim.cif import parse_cif_crystal
from ebsdsim.lookup import build_diff_lookup, BuildLookupOptions
from ebsdsim.structure import build_cell_from_cif


def _ni_cif_text() -> str:
    path = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    return path.read_text(encoding="utf-8")


def test_parse_ni_cif():
    crystal = parse_cif_crystal(_ni_cif_text())
    assert crystal.a > 3.0
    assert len(crystal.atom_sites) >= 1
    assert crystal.atom_sites[0].symbol == "Ni"


def test_build_cell_from_ni_cif():
    cell = build_cell_from_cif(parse_cif_crystal(_ni_cif_text()))
    assert cell.space_group == 225
    assert cell.pg_num == 32  # point group m-3m
    assert cell.volume > 0
    assert cell.density > 0


def test_manual_ni_material_matches_cif_lattice():
    material = Material(
        cell=Cell(a=3.52, b=3.52, c=3.52, alpha=90, beta=90, gamma=90, space_group="Fm-3m"),
        atoms=[Atom("Ni", x=0.0, y=0.0, z=0.0, occupancy=1.0)],
        name="Ni",
    )
    sim = material.to_simulation_cell()
    cif_cell = build_cell_from_cif(parse_cif_crystal(_ni_cif_text()))
    assert abs(sim.a - cif_cell.a) / cif_cell.a < 0.02
    assert sim.space_group == 225
    assert sim.pg_num == cif_cell.pg_num


def test_diff_lookup_finite():
    cell = build_cell_from_cif(parse_cif_crystal(_ni_cif_text()))
    lookup = build_diff_lookup(cell, BuildLookupOptions(voltage_kv=20.0, dmin=0.05))
    assert lookup.hkl.size > 0
    assert np.isfinite(lookup.mlambda)
