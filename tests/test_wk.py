"""Weickenmeier–Kohl helpers."""

from __future__ import annotations

import warnings

import numpy as np

from ebsdsim.wk import expi, expi_vec


def test_expi_vec_matches_scalar_near_root():
    xs = np.linspace(1e-8, 6.0, 200)
    vec = expi_vec(xs)
    for i, x in enumerate(xs):
        assert np.isclose(vec[i], expi(float(x)), rtol=0, atol=1e-10), x


def test_expi_vec_no_log1p_warning_for_tiny_args():
    """Regression: np.where evaluated log1p(t/root) for xs≈0 where t/root≈-1."""
    xs = np.array([0.0, 1e-30, 1e-12, 1e-6, 0.37, 1.0, 5.0], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = expi_vec(xs)
    assert np.isneginf(out[0])
    assert np.all(np.isfinite(out[1:]))
