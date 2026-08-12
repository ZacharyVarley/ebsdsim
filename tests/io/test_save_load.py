"""Save / load roundtrip tests for the master-pattern .npz format."""

from __future__ import annotations

import importlib.resources

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.gpu.device import require_gpu
from ebsdsim.io.load import load_master_pattern, save_png_gray, to_uint8


def _gpu_available() -> bool:
    try:
        require_gpu()
        return True
    except RuntimeError:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gpu_available(), reason="WebGPU adapter unavailable"),
]


def _gan_cif():
    return importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif")


def _gan_pattern(*, solver: str, halfw: int = 8):
    kwargs: dict = dict(
        voltage_kv=20.0,
        halfw=halfw,
        dmin=0.05,
        energy_binwidth_keV=4.0,
        solver=solver,
        marginal_coverage=1.0,
        mc_backend="surrogate",
    )
    if solver == "lu_smith":
        kwargs.update(rank=6, chunk_size=64)
    elif solver == "smith":
        kwargs.update(rank=16)
    elif solver == "galerkin":
        kwargs.update(rank=6)
    return es.master_pattern_from_cif(_gan_cif(), **kwargs)


@pytest.mark.slow
def test_metadata_records_parameters_and_cell():
    mp = _gan_pattern(solver="lu_smith")
    meta = mp.metadata
    # Solver parameters that change the pattern must be present.
    for key in (
        "rank",
        "bethe_c_strong",
        "bethe_c_weak",
        "bethe_c_cutoff",
        "dbdiff_sg_cutoff",
        "dmin",
        "voltage_kv",
        "energy_binwidth_keV",
    ):
        assert key in meta, f"missing metadata key {key!r}"
    # GaN is non-centrosymmetric -> southern hemisphere needed, two sites.
    assert meta["needs_southern_hemisphere"] is True
    assert meta["is_centrosymmetric"] is False
    cell = meta["cell"]
    assert cell["n_sites"] == 2
    symbols = sorted(site["symbol"] for site in cell["sites"])
    assert symbols == ["Ga", "N"]
    for site in cell["sites"]:
        assert "b_iso_angstrom_sq" in site
        assert "b_iso_nm_sq" in site
        assert site["b_iso_angstrom_sq"] == pytest.approx(site["b_iso_nm_sq"] * 100.0)
    # CIF ingest stamps: as-read snapshot + IT-standard setting fields.
    assert "cif_input" in meta
    assert meta["cif_input"] is not None
    assert "symmetry_provenance" in meta
    assert cell.get("setting") == "IT standard"
    assert "transformed" in cell
    assert "P" in cell and "p" in cell


@pytest.mark.slow
def test_save_load_roundtrip_lu_smith_per_bin(tmp_path):
    """lu_smith stores real per-bin FS patterns; round-trip must preserve them."""
    mp = _gan_pattern(solver="lu_smith")
    assert len(mp.bin_patterns) >= 1
    assert len(mp.bin_patterns) == len(mp.bin_voltages_kv)
    out = mp.save(tmp_path / "gan_lu.npz")
    assert out.exists()

    loaded = load_master_pattern(out)
    assert loaded.meta["format"] == "ebsdsim-master-pattern"
    assert loaded.meta.get("n_bin_patterns") == len(mp.bin_patterns)
    assert loaded.n_sites == 2
    assert loaded.site_weights is not None
    assert loaded.site_weights.shape == (2,)
    assert np.isclose(loaded.site_weights.sum(), 1.0)
    assert loaded.n_bins == len(mp.bin_patterns)
    assert loaded.n_bins == int(loaded.bin_voltages_kv.size)
    assert loaded.needs_southern_hemisphere is True
    assert loaded.data.shape[0] >= 1

    nh, sh = loaded.reconstruct_integrated()
    assert nh.shape == (mp.metadata["grid_size"], mp.metadata["grid_size"])
    assert np.max(np.abs(nh - mp.pattern)) < 1e-4
    assert not np.allclose(nh, sh)


@pytest.mark.slow
def test_save_load_roundtrip_smith_per_bin(tmp_path):
    """smith stores real per-bin FS patterns; round-trip must preserve them."""
    mp = _gan_pattern(solver="smith")
    assert len(mp.bin_patterns) >= 1
    assert len(mp.bin_patterns) == len(mp.bin_voltages_kv)
    assert len(mp.bin_patterns) == len(mp.bin_weights)
    out = mp.save(tmp_path / "gan_smith.npz")
    assert out.exists()

    loaded = load_master_pattern(out)
    assert loaded.meta["format"] == "ebsdsim-master-pattern"
    assert loaded.meta.get("n_bin_patterns") == len(mp.bin_patterns)
    assert loaded.n_sites == 2
    assert loaded.n_bins == len(mp.bin_patterns)
    assert loaded.n_bins == int(loaded.bin_voltages_kv.size)
    assert loaded.needs_southern_hemisphere is True
    assert loaded.data.shape[0] >= 1 + loaded.n_bins

    nh, sh = loaded.reconstruct_integrated()
    assert nh.shape == (mp.metadata["grid_size"], mp.metadata["grid_size"])
    assert np.max(np.abs(nh - mp.pattern)) < 1e-4
    assert not np.allclose(nh, sh)

    nh0, _ = loaded.reconstruct_bin(0)
    assert nh0.shape == (loaded.side, loaded.side)
    assert np.all(np.isfinite(nh0))
    assert np.any(nh0 > 0)

    with pytest.raises(IndexError):
        loaded.reconstruct_bin(loaded.n_bins)


@pytest.mark.slow
def test_bin_reconstruction_and_png(tmp_path):
    mp = _gan_pattern(solver="lu_smith")
    out = mp.save(tmp_path / "gan.npz")
    loaded = load_master_pattern(out)
    assert loaded.n_bins >= 1
    nh0, _ = loaded.reconstruct_bin(0)
    assert nh0.shape == (loaded.side, loaded.side)
    assert np.all(np.isfinite(nh0))
    png = save_png_gray(to_uint8(nh0), tmp_path / "bin0.png")
    assert png.exists()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    with pytest.raises(IndexError):
        loaded.reconstruct_bin(loaded.n_bins)


@pytest.mark.slow
def test_lambert_data_display_scaling(tmp_path):
    mp = _gan_pattern(solver="lu_smith")
    out = mp.save(tmp_path / "gan.npz")
    loaded = load_master_pattern(out)
    nh_raw = loaded.data[0, 0, 0]
    disp, _ = loaded.lambert_data(normalize="minmax")
    nh_scaled = disp[0, 0, 0]
    assert nh_raw.max() > 1.0 or nh_scaled.max() <= 1.0 + 1e-6
    assert not np.allclose(nh_raw, nh_scaled)
    assert np.allclose(loaded.data, load_master_pattern(out).data)


@pytest.mark.slow
def test_master_pattern_lambert_data_method():
    mp = _gan_pattern(solver="lu_smith", halfw=8)
    assert np.allclose(mp.data, mp.lambert_data()[0])
    disp, _ = mp.lambert_data(normalize="minmax")
    assert not np.allclose(mp.data, disp)
    assert not np.allclose(mp.pattern, disp[0, 0, 0])
