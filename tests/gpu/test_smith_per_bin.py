"""Smith per-bin intensity stack (host readback) correctness."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.engine.results import stack_bins
from ebsdsim.gpu.device import require_gpu
from ebsdsim.io.load import load_master_pattern


def _gpu_smith_available() -> bool:
    try:
        require_gpu(required_features=("shader-f16",))
        return True
    except RuntimeError:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _gpu_smith_available(),
        reason="WebGPU adapter with shader-f16 unavailable",
    ),
]


def _run_smith(cif: Path, *, halfw: int = 12, energy_binwidth_keV: float = 5.0) -> es.MasterPattern:
    return es.master_pattern_from_cif(
        cif,
        voltage_kv=20.0,
        halfw=halfw,
        dmin=0.05,
        energy_binwidth_keV=energy_binwidth_keV,
        marginal_coverage=1.0,
        mc_backend="surrogate",
        solver="smith",
        rank=16,
        verbosity=0,
    )


@pytest.mark.parametrize(
    "cif,label",
    [
        (Path(str(importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif"))), "Ni"),
        (Path(str(importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif"))), "GaN"),
    ],
    ids=["Ni", "GaN"],
)
def test_smith_per_bin_stack_matches_integrated(cif: Path, label: str) -> None:
    """Per-bin stack is finite/non-negative and weight-sums to integrated."""
    assert cif.is_file(), f"missing CIF: {cif}"
    mp = _run_smith(cif, halfw=12, energy_binwidth_keV=5.0)
    n_bins = len(mp.bin_voltages_kv)
    assert n_bins >= 1
    assert len(mp.bin_patterns) == n_bins
    assert len(mp.bin_weights) == n_bins

    n_k, n_sites = int(mp.n_k), int(mp.n_sites)
    stack = stack_bins(list(mp.bin_patterns), n_k, n_sites)
    assert stack.shape == (n_bins, n_k, n_sites)
    assert np.all(np.isfinite(stack))
    assert np.all(stack >= 0.0)
    assert np.any(stack > 0.0)

    weights = np.asarray(mp.bin_weights, dtype=np.float64).reshape(n_bins, 1, 1)
    host_sum = (stack.astype(np.float64) * weights).sum(axis=0).astype(np.float32)
    integrated = np.asarray(mp.integrated, dtype=np.float32).reshape(n_k, n_sites)
    # Device integrated (method a) vs host weight-sum of LU-style unweighted bins.
    abs_err = float(np.max(np.abs(host_sum - integrated)))
    rel_err = float(
        np.max(np.abs(host_sum - integrated) / np.maximum(np.abs(integrated), 1e-30))
    )
    assert abs_err < 1e-3 or rel_err < 1e-5, (
        f"{label}: host sum vs integrated max_abs={abs_err:.3e} max_rel={rel_err:.3e}"
    )


def test_smith_per_bin_save_load_reconstruct(tmp_path) -> None:
    """Saved smith run reloads with real bins; reconstruct_bin matches raster."""
    cif = Path(str(importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif")))
    mp = _run_smith(cif, halfw=12, energy_binwidth_keV=5.0)
    assert len(mp.bin_patterns) == len(mp.bin_voltages_kv) >= 1

    out = mp.save(tmp_path / "gan_smith_bins.npz")
    loaded = load_master_pattern(out)
    assert loaded.n_bins == len(mp.bin_voltages_kv)
    assert loaded.n_bins == int(loaded.bin_fs.shape[0])
    assert loaded.bin_fs.shape[1:] == (mp.n_k, mp.n_sites)

    nh0, _ = loaded.reconstruct_bin(0)
    assert nh0.shape == (loaded.side, loaded.side)
    assert np.all(np.isfinite(nh0))
    assert np.any(nh0 > 0)
    # Energy axis: 0=integrated, 1+i=bin i.
    s_int = int(loaded.axes["site_integrated_index"])
    assert np.allclose(nh0, loaded.data[1, s_int, 0], rtol=1e-5, atol=1e-5)
