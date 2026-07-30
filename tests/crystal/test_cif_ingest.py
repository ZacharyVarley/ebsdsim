"""CIF ingest smoke tests: setting transform, Hall ops, metadata stamps."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from ebsdsim.crystal.build import build_cell_from_structure
from ebsdsim.crystal.cif import load_structure
from ebsdsim.crystal.material import Atom, Cell, Material
from ebsdsim.crystal.spacegroup import expand_orbit_with_ops, ops_from_hall, pg_from_sg
from ebsdsim.io.save import cell_metadata

_FIXTURES = Path(__file__).resolve().parents[1] / "data" / "cif"

_ALL_CIFS = sorted(_FIXTURES.glob("sg_*.cif"))
assert _ALL_CIFS, f"no CIF fixtures under {_FIXTURES}"

_SG_RE = re.compile(r"^sg_(\d+)_")

# Explicit lattice-setting transforms (non-rhombohedral).
_TRANSFORMED = {
    "sg_003_1536903.cif",
    "sg_004_1004038.cif",
    "sg_004_2012414.cif",
    "sg_005_1530539.cif",
    "sg_005_4336696.cif",
}

# Already in IT standard (and not a R→hex conversion).
_STANDARD = {
    "sg_001_4030618.cif",
    "sg_002_1527508.cif",
    "sg_003_1008916.cif",
    "sg_004_9011677.cif",
    "sg_005_2021616.cif",
    "sg_015_9011089.cif",
    "sg_016_1528186.cif",
    "sg_074_9011656.cif",
    "sg_075_1010348.cif",
    "sg_142_7220794.cif",
    "sg_143_1530161.cif",
    "sg_168_2232341.cif",
    "sg_194_9012036.cif",
    "sg_195_1010043.cif",
}


def _expected_sg(path: Path) -> int:
    m = _SG_RE.match(path.name)
    assert m is not None, path.name
    return int(m.group(1))


@pytest.mark.parametrize("path", _ALL_CIFS, ids=lambda p: p.stem)
def test_all_fixture_cifs_load_and_build(path: Path):
    """Every shipped fixture loads to the filename SG and builds a usable Cell."""
    expected = _expected_sg(path)
    s = load_structure(path)
    assert s.number == expected
    meta = s.metadata()
    assert meta["cell"]["setting"] == "IT standard"
    assert meta["cell"]["space_group"] == expected
    assert meta["cif_input"] is not None
    assert "symmetry_provenance" in meta

    cell = build_cell_from_structure(s)
    assert cell.space_group == expected
    assert cell.pg_num == pg_from_sg(expected)
    assert cell.volume > 0
    assert len(cell.positions) == len(s.species)
    assert all(len(orbit) >= 1 for orbit in cell.positions)
    assert int(cell.multiplicities.sum()) >= len(s.species)


@pytest.mark.parametrize("name", sorted(_TRANSFORMED))
def test_transformed_fixtures(name: str):
    s = load_structure(_FIXTURES / name)
    meta = s.metadata()
    assert meta["cell"]["transformed"] is True
    assert meta["cell"]["rhombohedral_input"] is False
    cif_cell = meta["cif_input"]["cell"]
    used = meta["cell"]
    assert any(
        abs(cif_cell[k] - used[k]) > 1e-4
        for k in ("a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg", "beta_deg", "gamma_deg")
    )


@pytest.mark.parametrize("name", sorted(_STANDARD))
def test_standard_fixtures_match_input_lattice(name: str):
    s = load_structure(_FIXTURES / name)
    meta = s.metadata()
    assert meta["cell"]["transformed"] is False
    assert meta["cell"]["rhombohedral_input"] is False
    cif_cell = meta["cif_input"]["cell"]
    used = meta["cell"]
    for key in ("a_angstrom", "b_angstrom", "c_angstrom", "alpha_deg", "beta_deg", "gamma_deg"):
        assert cif_cell[key] == pytest.approx(used[key], rel=1e-6, abs=1e-6)


def test_nonstandard_monoclinic_sg005_axis_swap():
    """sg_005: unique-axis swap — CIF a/c become used c/a."""
    s = load_structure(_FIXTURES / "sg_005_1530539.cif")
    meta = s.metadata()
    assert s.number == 5
    assert meta["cell"]["transformed"] is True
    cif_cell = meta["cif_input"]["cell"]
    used = meta["cell"]
    assert cif_cell["a_angstrom"] == pytest.approx(5.617, abs=1e-3)
    assert cif_cell["c_angstrom"] == pytest.approx(10.011, abs=1e-3)
    assert used["a_angstrom"] == pytest.approx(10.011, abs=1e-3)
    assert used["c_angstrom"] == pytest.approx(5.617, abs=1e-3)

    cell = build_cell_from_structure(s)
    assert abs(cell.a * 10 - used["a_angstrom"]) < 1e-4


def test_rhombohedral_input_converted_to_hex_sg167():
    s = load_structure(_FIXTURES / "sg_167_1010914.cif")
    meta = s.metadata()
    assert s.number == 167
    assert meta["cell"]["rhombohedral_input"] is True
    assert meta["cell"]["transformed"] is False
    cif_cell = meta["cif_input"]["cell"]
    used = meta["cell"]
    assert cif_cell["alpha_deg"] == pytest.approx(55.28, abs=1e-2)
    assert used["gamma_deg"] == pytest.approx(120.0, abs=1e-4)
    assert used["alpha_deg"] == pytest.approx(90.0, abs=1e-4)
    assert used["c_angstrom"] > used["a_angstrom"]


def test_dual_origin_sg142_origin_choice_2():
    s = load_structure(_FIXTURES / "sg_142_7220794.cif")
    meta = s.metadata()
    assert s.number == 142
    assert meta["cell"]["origin_choice"] == 2
    cell = build_cell_from_structure(s)
    assert sum(len(p) for p in cell.positions) == 32


def test_diamond_227_hall_orbit_multiplicity_8():
    """Fd-3m (227) diamond site (1/8,1/8,1/8) expands to 8 sites under Hall ops."""
    pos = (0.125, 0.125, 0.125)
    hall_orbit = expand_orbit_with_ops(ops_from_hall(227), pos)
    assert len(hall_orbit) == 8
    assert ops_from_hall(227).shape == (192, 12)


def test_manual_material_matches_hall_for_dual_origin_227_and_142():
    """Manual Material uses Hall orbits (same as CIF) for dual-origin groups."""
    cases = (
        (227, (0.125, 0.125, 0.125), 8),
        (142, (0.0, 0.0, 0.0), 16),
    )
    for sg, pos, expected_n in cases:
        hall_n = len(expand_orbit_with_ops(ops_from_hall(sg), pos))
        assert hall_n == expected_n
        cell = Material(
            cell=Cell(a=5.0, b=5.0, c=5.0, space_group=sg),
            atoms=[Atom("Si", pos[0], pos[1], pos[2])],
        ).to_simulation_cell()
        assert int(cell.multiplicities[0]) == expected_n
        assert len(cell.positions[0]) == expected_n


def test_structure_metadata_payload_shape():
    s = load_structure(_FIXTURES / "sg_005_1530539.cif")
    meta = s.metadata()
    assert "symmetry_provenance" in meta
    assert "cif_input" in meta
    assert "cell" in meta
    cif_in = meta["cif_input"]
    assert "source" in cif_in
    assert "declared_symmetry" in cif_in
    assert "sites" in cif_in
    used = meta["cell"]
    for key in ("setting", "origin_choice", "transformed", "P", "p", "rhombohedral_input"):
        assert key in used
    assert used["origin_choice"] is None  # SG 5 is not dual-origin


def test_cif_input_and_cell_stamps_roundtrip_in_meta_json(tmp_path):
    """Persist Structure metadata the same way master patterns do, then reload."""
    s = load_structure(_FIXTURES / "sg_005_1530539.cif")
    structure_meta = s.metadata()
    sim_cell = build_cell_from_structure(s)
    sim_cell_meta = cell_metadata(sim_cell)
    cell_meta = dict(sim_cell_meta)
    used = structure_meta["cell"]
    for key in (
        "setting",
        "origin_choice",
        "transformed",
        "setting_describe",
        "setting_note",
        "P",
        "p",
        "rhombohedral_input",
    ):
        cell_meta[key] = used[key]

    payload = {
        "symmetry_provenance": structure_meta["symmetry_provenance"],
        "cif_input": structure_meta["cif_input"],
        "cell": cell_meta,
    }
    meta_bytes = np.frombuffer(
        json.dumps(payload, indent=2, sort_keys=False).encode("utf-8"),
        dtype=np.uint8,
    )
    out = tmp_path / "meta.npz"
    np.savez_compressed(out, meta_json=meta_bytes)
    loaded = np.load(out, allow_pickle=False)
    restored = json.loads(
        bytes(np.asarray(loaded["meta_json"], dtype=np.uint8).tobytes()).decode("utf-8")
    )
    assert restored["cif_input"]["cell"]["a_angstrom"] == pytest.approx(5.617, abs=1e-3)
    assert restored["cell"]["transformed"] is True
    assert restored["cell"]["a_angstrom"] == pytest.approx(10.011, abs=1e-3)
    assert restored["cell"]["setting"] == "IT standard"
    assert "n_sites" in restored["cell"]
    assert "sites" in restored["cell"]
