"""Monte Carlo GPU kernel tests."""

from __future__ import annotations

import pytest

from ebsdsim.cif import parse_cif_crystal
from ebsdsim.gpu import run_monte_carlo_gpu
from ebsdsim.gpu.device import require_gpu
from ebsdsim.structure import build_cell_from_cif
import importlib.resources


def _gpu_available() -> bool:
    try:
        require_gpu()
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="WebGPU adapter unavailable")


def _ni_cell():
    text = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif").read_text()
    return build_cell_from_cif(parse_cif_crystal(text))


def test_mc_returns_normalized_weights():
    cell = _ni_cell()
    mc = run_monte_carlo_gpu(
        cell,
        voltage_kv=20.0,
        energy_binwidth_kev=1.0,
        n_trajectories=8192,
        sigma_deg=70.0,
    )
    assert mc.voltages_kv.size > 0
    assert abs(float(mc.energy_weights.sum()) - 1.0) < 0.05
    assert mc.amplitudes.size == mc.voltages_kv.size
    assert mc.betas.size == mc.voltages_kv.size


def test_mc_auto_stop_converges_within_bounds():
    cell = _ni_cell()
    mc = run_monte_carlo_gpu(
        cell,
        voltage_kv=20.0,
        energy_binwidth_kev=1.0,
        sigma_deg=70.0,
        auto_stop=True,
        relative_tol=0.05,
        min_trajectories=262_144,
        max_trajectories=4_194_304,
        batch_size=262_144,
    )
    # Auto-stop must respect the trajectory bounds and report diagnostics.
    assert mc.n_trajectories >= 262_144
    assert mc.n_trajectories <= 4_194_304
    assert mc.n_convergence_checks >= 1
    if mc.converged:
        assert mc.last_relative_change <= 0.05

