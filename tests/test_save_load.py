"""Save / load roundtrip tests for the master-pattern .npz format."""

from __future__ import annotations

import importlib.resources

import numpy as np
import pytest

import ebsdsim as es
from ebsdsim.gpu.device import require_gpu
from ebsdsim.mploader import load_master_pattern, save_png_gray, to_uint8


def _gpu_available() -> bool:
    try:
        require_gpu()
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="WebGPU adapter unavailable")


def _gan_pattern(halfw: int = 17):
    gan = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif")
    return es.master_pattern_from_cif(
        gan,
        voltage_kv=20.0,
        halfw=halfw,
        dmin=0.08,
        energy_binwidth_keV=4.0,
        rank=6,
        chunk_size=64,
        marginal_coverage=1.0,
        mc_backend="surrogate",
    )


@pytest.mark.slow
def test_metadata_records_parameters_and_cell():
    mp = _gan_pattern()
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


@pytest.mark.slow
def test_save_load_roundtrip_matches_gpu(tmp_path):
    mp = _gan_pattern()
    out = mp.save(tmp_path / "gan.npz")
    assert out.exists()

    loaded = load_master_pattern(out)
    assert loaded.meta["format"] == "ebsdsim-master-pattern"
    assert loaded.n_sites == 2
    assert loaded.n_bins == len(mp.bin_patterns)
    assert loaded.needs_southern_hemisphere is True

    nh, sh = loaded.reconstruct_integrated()
    assert nh.shape == (mp.metadata["grid_size"], mp.metadata["grid_size"])
    # Loader reconstruction must match the GPU rasterized NH to f32 precision.
    assert np.max(np.abs(nh - mp.pattern)) < 1e-4
    # Non-centrosymmetric: south differs from north.
    assert not np.allclose(nh, sh)


@pytest.mark.slow
def test_bin_reconstruction_and_png(tmp_path):
    mp = _gan_pattern()
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
