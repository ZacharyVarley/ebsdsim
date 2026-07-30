"""Lambert layer: PG k-grids and reciprocal-frame transforms (nm⁻¹)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from ebsdsim.crystal.pointgroup import (
    CENTROSYMMETRIC_PG,
    folding_symbol,
    fs_normals,
)
from ebsdsim.lambert.projection import square_to_hemisphere_array


@dataclass
class KGrid:
    hw: int
    output_size: int
    khat: NDArray[np.float32]
    kij: NDArray[np.int32]


@dataclass
class PgKGrid:
    pg_num: int
    hw: int
    side: int
    is_centro: bool
    khat: NDArray[np.float32]
    kij: NDArray[np.int32]
    symbol: str = ""


@dataclass
class KChunk:
    kvecs: NDArray[np.float32]
    output_indices: NDArray[np.uint32]


def _square_to_hemisphere_array(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    southern: bool = False,
) -> NDArray[np.float32]:
    return square_to_hemisphere_array(x, y, southern).astype(np.float32)


def _lambert_pixel_coords(hw: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    coords = np.arange(-hw, hw + 1, dtype=np.int32)
    ii, jj = np.meshgrid(coords, coords, indexing="ij")
    return ii.ravel(), jj.ravel()


def build_lambert_k_grid(
    hw: int,
    *,
    southern: bool = False,
    fundamental_only: bool = False,
) -> KGrid:
    side = 2 * hw + 1
    ii, jj = _lambert_pixel_coords(hw)
    x = ii.astype(np.float64) / hw
    y = jj.astype(np.float64) / hw
    mask = np.ones(ii.size, dtype=bool)
    if fundamental_only:
        mask = (x >= -1e-12) & (y >= -1e-12) & (y <= x + 1e-12)
    dirs = _square_to_hemisphere_array(x[mask], y[mask], southern).reshape(-1)
    pixels = ((jj[mask] + hw) * side + (ii[mask] + hw)).astype(np.int32)
    return KGrid(
        hw=hw,
        output_size=side * side,
        khat=dirs.astype(np.float32, copy=False),
        kij=pixels,
    )


def scale_k_grid_by_wavelength(grid: KGrid, scale: float) -> NDArray[np.float32]:
    return (grid.khat * scale).astype(np.float32)


def transform_k_grid_to_reciprocal(
    grid: KGrid,
    direct_structure_matrix: NDArray[np.floating],
    mlambda: float,
) -> NDArray[np.float32]:
    """k = khat @ dsm / mlambda (row-vector convention, NOT dsm.T)."""
    khat = grid.khat.reshape(-1, 3).astype(np.float32, copy=False)
    matrix = np.asarray(direct_structure_matrix, dtype=np.float32).reshape(3, 3)
    return ((khat @ matrix) / np.float32(mlambda)).astype(np.float32).reshape(-1)


def chunk_k_vectors(
    kvecs: NDArray[np.float32],
    output_indices: NDArray[np.uint32],
    chunk_size: int,
) -> Iterator[KChunk]:
    rows = kvecs.size // 3
    for start in range(0, rows, chunk_size):
        end = min(rows, start + chunk_size)
        yield KChunk(
            kvecs=kvecs[start * 3 : end * 3].copy(),
            output_indices=output_indices[start:end].copy(),
        )


def _lambert_pixel_directions(hw: int, southern: bool) -> NDArray[np.float32]:
    ii, jj = _lambert_pixel_coords(hw)
    x = ii.astype(np.float64) / hw
    y = jj.astype(np.float64) / hw
    return _square_to_hemisphere_array(x, y, southern).reshape(-1)


def build_pg_k_grid(pg_num: int, hw: int, space_group: int) -> PgKGrid:
    """Build the fundamental-sector Lambert k-grid for a crystal.

    ``space_group`` is required so orientation-ambiguous point groups
    (e.g. 321 vs 312) select the correct operators and FS normals.
    """
    symbol = folding_symbol(pg_num, space_group)
    normals = fs_normals(symbol)
    is_centro = pg_num in CENTROSYMMETRIC_PG
    eps = 2.0 / hw
    side = 2 * hw + 1
    ii, jj = _lambert_pixel_coords(hw)
    dirs_nh = _lambert_pixel_directions(hw, False)
    dirs_nh_2d = dirs_nh.reshape(-1, 3)
    normal_2d = normals.reshape(-1, 3)
    mask_nh = np.all(dirs_nh_2d @ normal_2d.T >= -eps, axis=1)
    khat_parts = [dirs_nh_2d[mask_nh]]
    kij_parts = [
        np.column_stack(
            (
                ii[mask_nh],
                jj[mask_nh],
                np.ones(np.count_nonzero(mask_nh), dtype=np.int32),
            )
        )
    ]
    if not is_centro:
        dirs_sh = _lambert_pixel_directions(hw, True)
        dirs_sh_2d = dirs_sh.reshape(-1, 3)
        mask_sh = np.all(dirs_sh_2d @ normal_2d.T >= -eps, axis=1)
        khat_parts.append(dirs_sh_2d[mask_sh])
        kij_parts.append(
            np.column_stack(
                (
                    ii[mask_sh],
                    jj[mask_sh],
                    -np.ones(np.count_nonzero(mask_sh), dtype=np.int32),
                )
            )
        )
    khat = np.concatenate(khat_parts, axis=0).astype(np.float32).reshape(-1)
    kij = np.concatenate(kij_parts, axis=0).astype(np.int32).reshape(-1)
    return PgKGrid(
        pg_num=pg_num,
        hw=hw,
        side=side,
        is_centro=is_centro,
        khat=khat,
        kij=kij,
        symbol=symbol,
    )


def transform_pg_k_grid_to_reciprocal(
    grid: PgKGrid,
    direct_structure_matrix: NDArray[np.floating],
    mlambda: float,
) -> NDArray[np.float32]:
    """k = khat @ dsm / mlambda (row-vector convention, NOT dsm.T)."""
    khat = grid.khat.reshape(-1, 3).astype(np.float32, copy=False)
    matrix = np.asarray(direct_structure_matrix, dtype=np.float32).reshape(3, 3)
    return ((khat @ matrix) / np.float32(mlambda)).astype(np.float32).reshape(-1)


def pg_k_grid_output_indices(grid: PgKGrid) -> NDArray[np.uint32]:
    n = grid.kij.size // 3
    side = grid.side
    out = np.zeros(n, dtype=np.uint32)
    sheet_size = side * side
    hw = grid.hw
    for r in range(n):
        x = int(grid.kij[r * 3]) + hw
        y = int(grid.kij[r * 3 + 1]) + hw
        sign = int(grid.kij[r * 3 + 2])
        sheet = 0 if sign > 0 else 1
        out[r] = sheet * sheet_size + y * side + x
    return out
