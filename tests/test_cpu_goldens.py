"""CPU golden and integration tests."""

from __future__ import annotations

import importlib.resources
import json

import numpy as np
from numpy.typing import NDArray

from ebsdsim.golden import compare_float32_golden
from ebsdsim.integrate import compute_mu_eff, trim_multi_voltage_mc_by_coverage
from ebsdsim.kgrid import build_pg_k_grid
from ebsdsim.rasterize import RasterizeOptions, rasterize_pattern
from ebsdsim.types import MultiVoltageMC


def _transpose_flat_square(flat: list[float] | NDArray[np.floating]) -> NDArray[np.float32]:
    side = int(round(np.sqrt(len(flat))))
    arr = np.asarray(flat, dtype=np.float32).reshape(side, side)
    return arr.T.reshape(-1)


def _load_golden(name: str) -> dict:
    path = importlib.resources.files("ebsdsim").joinpath("data/goldens", name)
    return json.loads(path.read_text(encoding="utf-8"))


def test_kgrid_rasterize_golden_case0():
    data = _load_golden("kgrid_rasterize_golden.json")
    opts = RasterizeOptions(robust_norm=False, interp_mode="bilinear")
    for case in data["cases"]:
        grid_info = case["grid"]
        pg_grid = build_pg_k_grid(int(grid_info["pg_num"]), int(grid_info["hw"]))
        assert pg_grid.khat.size // 3 == int(grid_info["n"])
        raster = rasterize_pattern(case["pattern"], pg_grid, n_sites=1, opts=opts)
        expected_nh = _transpose_flat_square(case["nh"])
        cmp = compare_float32_golden(raster.nh, expected_nh)
        assert cmp.passed, (
            f"pg={grid_info['pg_num']} max_abs={cmp.max_abs_error}, max_rel={cmp.max_rel_error}"
        )


def test_compute_mu_eff_bloch():
    mu = compute_mu_eff(0.05, 0.01, 0.001, "bloch")
    assert np.isfinite(mu)


def test_trim_mc_coverage():
    mc = MultiVoltageMC(
        binsize_energy_keV=1.0,
        voltages_kv=np.array([20.0, 19.0, 18.0]),
        energy_weights=np.array([0.5, 0.3, 0.2]),
        amplitudes=np.ones(3),
        betas=np.ones(3) * 0.1,
    )
    trimmed = trim_multi_voltage_mc_by_coverage(mc, 0.9)
    assert trimmed.voltages_kv.size <= mc.voltages_kv.size
