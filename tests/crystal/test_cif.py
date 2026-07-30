"""CIF parsing and cell construction tests."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import numpy as np
from ebsdsim.api import Atom, Cell, Material
from ebsdsim.crystal.build import build_cell_from_cif_path, build_cell_from_structure
from ebsdsim.crystal.cif import load_structure, load_structure_from_text
from ebsdsim.physics.lookup import BuildLookupOptions, build_diff_lookup


def _ni_cif_path() -> Path:
    return Path(str(importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")))


def _ni_cif_text() -> str:
    return _ni_cif_path().read_text(encoding="utf-8")


def test_parse_ni_cif():
    structure = load_structure(_ni_cif_path())
    assert structure.cell[0] > 3.0
    assert len(structure.species) >= 1
    assert structure.species[0] == "Ni"
    assert structure.number == 225


def test_build_cell_from_ni_cif():
    cell = build_cell_from_cif_path(_ni_cif_path())
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
    cif_cell = build_cell_from_cif_path(_ni_cif_path())
    assert abs(sim.a - cif_cell.a) / cif_cell.a < 0.02
    assert sim.space_group == 225
    assert sim.pg_num == cif_cell.pg_num


def test_diff_lookup_finite():
    cell = build_cell_from_cif_path(_ni_cif_path())
    lookup = build_diff_lookup(cell, BuildLookupOptions(voltage_kv=20.0, dmin=0.05))
    assert lookup.hkl.size > 0
    assert np.isfinite(lookup.mlambda)


def test_cif_missing_b_iso_defaults_to_half_angstrom_sq(capsys):
    """Preset CIFs omit U/B; CIF path must apply the documented 0.5 Å² default."""
    cell = build_cell_from_cif_path(_ni_cif_path())
    assert np.allclose(cell.atom_data[:, 4], 0.005)
    err = capsys.readouterr()
    assert "default B_iso=0.5" in err.out
    assert "0.005 nm^2" in err.out
    assert "no U_iso/B_iso" in err.out
    assert "unphysical" not in err.out


def test_cif_zero_u_iso_defaults_to_half_angstrom_sq(capsys):
    """Explicit U_iso=0 is rejected; same room-temp B default as missing tags."""
    path = Path(__file__).resolve().parents[1] / "data" / "cif" / "sg_003_1536903.cif"
    structure = load_structure(path)
    assert np.all(np.asarray(structure.uiso) > 0.0)
    # As-read snapshot keeps the authored zeros; working Structure is sanitized.
    assert all(site["uiso"] == 0.0 for site in structure.cif_input["sites"])
    cell = build_cell_from_structure(structure)
    assert np.allclose(cell.atom_data[:, 4], 0.005)
    err = capsys.readouterr()
    assert "non-positive (0)" in err.out
    assert "unphysical at finite temperature" in err.out
    assert "default B_iso=0.5" in err.out
    assert "0.005 nm^2" in err.out


def test_cif_explicit_u_iso_is_preserved():
    path = Path(__file__).resolve().parents[1] / "data" / "cif" / "sg_142_7220794.cif"
    cell = build_cell_from_cif_path(path)
    # Fixture sites carry positive Uiso; converted B must not be the missing-tag default.
    assert not np.allclose(cell.atom_data[:, 4], 0.005)
    assert np.all(cell.atom_data[:, 4] > 0.0)
    # Uiso=0.005 Å² → B = 8π²U ≈ 0.3948 Å² → 0.003948 nm² (one site in this CIF).
    assert any(abs(float(b) - 0.00394784) < 1e-6 for b in cell.atom_data[:, 4])


def _fe_gamma_cif_path() -> Path:
    return Path(str(importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Fe_gamma.cif")))


def test_fe_gamma_cif_hm_tag_case_and_symbol():
    """Regression: capitalized H-M tag and spaced Hermann–Mauguin symbol (issue #3)."""
    structure = load_structure(_fe_gamma_cif_path())
    assert structure.number == 225
    cell = build_cell_from_structure(structure)
    assert cell.space_group == 225
    assert cell.pg_num == 32
    # Also works from raw text (mixed-case tags).
    structure_text = load_structure_from_text(_fe_gamma_cif_path().read_text(encoding="utf-8"))
    assert structure_text.number == 225
