"""Surrogate MC model tests."""

from __future__ import annotations

import importlib.resources

import numpy as np

from ebsdsim.cif import parse_cif_crystal
from ebsdsim.integrate import surrogate_to_multi_voltage_mc
from ebsdsim.structure import build_cell_from_cif
from ebsdsim.surrogate import get_surrogate_model, infer_direct_exp_from_cell_rebinned


def _ni_cell():
    text = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif").read_text()
    return build_cell_from_cif(parse_cif_crystal(text))


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
    assert mc.voltages_kv[0] > mc.voltages_kv[-1]
    assert np.isclose(mc.energy_weights.sum(), 1.0)
