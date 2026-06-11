"""End-to-end API smoke tests (GPU required)."""

from __future__ import annotations

import importlib.resources

import numpy as np
import pytest

import ebsdsim as es
from ebsdsim.gpu.device import require_gpu


def _gpu_available() -> bool:
    try:
        require_gpu()
        return True
    except RuntimeError:
        return False


def test_import():
    from importlib.metadata import version

    assert es.__version__ == version("ebsdsim")
    assert callable(es.master_pattern)
    assert callable(es.master_pattern_from_cif)


def test_preset_cif_and_halfw():
    from ebsdsim.api import _resolve_cif_path, _validate_halfw

    assert _resolve_cif_path("Ni.cif").name == "Ni.cif"
    assert _validate_halfw(250) == 250


@pytest.mark.skipif(not _gpu_available(), reason="WebGPU adapter unavailable")
@pytest.mark.slow
def test_master_pattern_from_cif_tiny():
    ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    result = es.master_pattern_from_cif(
        ni,
        voltage_kv=20.0,
        halfw=8,
        dmin=0.08,
        energy_binwidth_keV=2.0,
        n_trajectories=4096,
        rank=4,
        chunk_size=8,
        marginal_coverage=1.0,
        mc_backend="gpu",
    )
    assert result.pattern.shape == (17, 17)
    assert np.all(np.isfinite(result.pattern))
    assert np.any(result.integrated > 0)
    assert result.metadata["mc_backend"] == "gpu_fly_first"


@pytest.mark.skipif(not _gpu_available(), reason="WebGPU adapter unavailable")
@pytest.mark.slow
def test_master_pattern_manual_ni():
    material = es.Material(
        cell=es.Cell(a=3.52, b=3.52, c=3.52, space_group="Fm-3m"),
        atoms=[es.Atom("Ni", x=0.0, y=0.0, z=0.0)],
        name="Ni",
    )
    result = es.master_pattern(
        material,
        voltage_kv=20.0,
        halfw=8,
        dmin=0.08,
        energy_binwidth_keV=2.0,
        n_trajectories=4096,
        rank=4,
        chunk_size=8,
        marginal_coverage=1.0,
        mc_backend="gpu",
    )
    assert result.pattern.ndim == 2
    assert np.all(np.isfinite(result.pattern))
    assert np.any(result.integrated > 0)
    assert result.n_sites >= 1
