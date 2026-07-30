"""End-to-end API smoke tests (GPU required)."""

from __future__ import annotations

import importlib.resources

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.gpu.device import require_gpu


def _gpu_available() -> bool:
    try:
        require_gpu()
        return True
    except RuntimeError:
        return False


def test_import():
    from ebsdsim._version import __version__ as declared_version

    assert es.__version__ == declared_version
    assert callable(es.master_pattern)
    assert callable(es.master_pattern_from_cif)


def test_preset_cif_and_halfw():
    from ebsdsim.api import _resolve_cif_path
    from ebsdsim.engine.params import SimParams
    from ebsdsim.engine.progress import validate_verbosity

    assert _resolve_cif_path("Ni.cif").name == "Ni.cif"
    assert SimParams(halfw=250).halfw == 250
    with pytest.raises(ValueError, match="halfw"):
        SimParams(halfw=0)
    assert validate_verbosity(0) == 0


@pytest.mark.skipif(not _gpu_available(), reason="WebGPU adapter unavailable")
@pytest.mark.slow
def test_master_pattern_exact_slow_cpu_tiny():
    ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    result = es.master_pattern_from_cif(
        ni,
        voltage_kv=20.0,
        halfw=8,
        dmin=0.05,
        energy_binwidth_keV=20.0,
        rank=4,
        chunk_size=8,
        marginal_coverage=1.0,
        exact_slow_cpu=True,
        mc_backend="gpu",
        n_trajectories=4096,
    )
    assert result.pattern.shape == (17, 17)
    assert np.all(np.isfinite(result.pattern))
    assert result.metadata["exact_slow_cpu"] is True


@pytest.mark.skipif(not _gpu_available(), reason="WebGPU adapter unavailable")
@pytest.mark.slow
def test_master_pattern_from_cif_tiny():
    ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    result = es.master_pattern_from_cif(
        ni,
        voltage_kv=20.0,
        halfw=8,
        dmin=0.05,
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
        dmin=0.05,
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
