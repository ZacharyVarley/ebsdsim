"""Surrogate MC model tests."""

from __future__ import annotations

import importlib.resources

import numpy as np
from ebsdsim.crystal.build import build_cell_from_cif_path
from ebsdsim.energy.binning import dynamical_voltages_kv
from ebsdsim.energy.surrogate import (
    SurrogateDirectExp,
    get_surrogate_model,
    infer_direct_exp_from_cell_rebinned,
)
from ebsdsim.engine.integrate import surrogate_to_multi_voltage_mc


def _ni_cell():
    path = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    return build_cell_from_cif_path(str(path))


def test_surrogate_load_and_infer_ni():
    cell = _ni_cell()
    out = infer_direct_exp_from_cell_rebinned(
        model=get_surrogate_model(),
        cell=cell,
        sigma_deg=70.0,
        beam_kv=20.0,
        energy_binwidth_keV=1.0,
        n_energy_bins=20,
    )
    assert out.amplitudes.shape == (20,)
    assert np.all(np.isfinite(out.amplitudes))
    assert np.all(np.isfinite(out.betas))
    assert np.all(out.amplitudes >= 0.0)
    assert np.all(out.betas >= 0.0)
    assert np.isclose(out.energy_weights.sum(), 1.0, atol=1e-8)


def test_dynamical_voltages_kv():
    v = dynamical_voltages_kv(20.0, 3, 0.5)
    assert list(v) == [20.0, 19.5, 19.0]


def test_surrogate_to_multi_voltage_mc_voltage_axis():
    cell = _ni_cell()
    direct = infer_direct_exp_from_cell_rebinned(
        cell=cell,
        sigma_deg=70.0,
        beam_kv=20.0,
        energy_binwidth_keV=1.0,
        n_energy_bins=20,
    )
    mc = surrogate_to_multi_voltage_mc(direct, 20.0)
    assert mc.voltages_kv.size == 20
    assert mc.voltages_kv[0] == 20.0
    assert mc.voltages_kv[1] == 19.0
    assert mc.voltages_kv[0] > mc.voltages_kv[-1]
    assert np.isclose(mc.energy_weights.sum(), 1.0)


def test_surrogate_to_multi_voltage_mc_voltages_from_bin_width():
    direct = SurrogateDirectExp(
        amplitudes=np.ones(3),
        betas=np.ones(3),
        energy_mass=np.ones(3),
        energy_weights=np.ones(3) / 3.0,
        energy_centers_keV=np.array([0.25, 0.75, 1.25]),
    )
    mc = surrogate_to_multi_voltage_mc(direct, 20.0)
    assert list(mc.voltages_kv) == [20.0, 19.5, 19.0]
