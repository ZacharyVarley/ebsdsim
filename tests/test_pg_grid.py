"""Point-group k-grid and fundamental-sector tests."""

from __future__ import annotations

import numpy as np

from ebsdsim.kgrid import build_pg_k_grid
from ebsdsim.pg_ops import fs_normals, pg_num_to_symbol, point_group_operators


def test_p1_fs_normals_empty():
    symbol = pg_num_to_symbol(1)
    normals = fs_normals(symbol)
    assert normals.size == 0
    ops = point_group_operators(symbol)
    assert np.allclose(ops.reshape(3, 3), np.eye(3))


def test_build_pg_k_grid_p1_includes_both_hemispheres():
    hw = 40
    grid = build_pg_k_grid(1, hw)
    side = 2 * hw + 1
    assert grid.pg_num == 1
    assert not grid.is_centro
    kij = grid.kij.reshape(-1, 3)
    nh_pixels = int(np.count_nonzero(kij[:, 2] > 0))
    sh_pixels = int(np.count_nonzero(kij[:, 2] < 0))
    assert nh_pixels == side * side
    assert sh_pixels == side * side
    assert grid.khat.size == (nh_pixels + sh_pixels) * 3
