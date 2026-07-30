"""Point-group k-grid and fundamental-sector tests."""

from __future__ import annotations

import numpy as np
import pytest
from ebsdsim.crystal.pointgroup import (
    folding_symbol,
    fs_normals,
    point_group_operators,
)
from ebsdsim.lambert.kgrid import build_pg_k_grid


def test_p1_fs_normals_empty():
    symbol = folding_symbol(1, 1)
    normals = fs_normals(symbol)
    assert normals.size == 0
    ops = point_group_operators(symbol)
    assert np.allclose(ops.reshape(3, 3), np.eye(3))


def test_build_pg_k_grid_p1_includes_both_hemispheres():
    hw = 40
    grid = build_pg_k_grid(1, hw, space_group=1)
    side = 2 * hw + 1
    assert grid.pg_num == 1
    assert not grid.is_centro
    kij = grid.kij.reshape(-1, 3)
    nh_pixels = int(np.count_nonzero(kij[:, 2] > 0))
    sh_pixels = int(np.count_nonzero(kij[:, 2] < 0))
    assert nh_pixels == side * side
    assert sh_pixels == side * side
    assert grid.khat.size == (nh_pixels + sh_pixels) * 3


def test_build_pg_k_grid_requires_space_group():
    with pytest.raises(TypeError):
        build_pg_k_grid(18, 8)  # type: ignore[call-arg]


def test_orientation_ambiguous_pair_selects_distinct_sectors():
    g149 = build_pg_k_grid(18, 8, space_group=149)
    g150 = build_pg_k_grid(18, 8, space_group=150)
    assert g149.symbol == "312"
    assert g150.symbol == "32"
    assert folding_symbol(18, 149) == "312"
    assert folding_symbol(18, 150) == "32"
    assert g149.khat.size != g150.khat.size
