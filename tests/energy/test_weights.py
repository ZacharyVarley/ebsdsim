"""Site-weight marginal helpers."""

from __future__ import annotations

import numpy as np
from ebsdsim.energy.weights import reduce_over_sites, site_weights_from_meta_cell


def test_reduce_over_sites_weighted():
    fs = np.array([[1.0, 3.0], [2.0, 6.0]], dtype=np.float32)
    w = np.array([0.25, 0.75], dtype=np.float32)
    out = reduce_over_sites(fs, w)
    assert out[0] == np.float32(1.0 * 0.25 + 3.0 * 0.75)
    assert out[1] == np.float32(2.0 * 0.25 + 6.0 * 0.75)


def test_site_weights_from_meta_cell():
    meta = {
        "sites": [
            {"occupancy": 1.0, "multiplicity": 2},
            {"occupancy": 1.0, "multiplicity": 2},
        ]
    }
    w = site_weights_from_meta_cell(meta)
    assert w is not None
    assert np.allclose(w, [0.5, 0.5])
