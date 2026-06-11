"""Lambert k-grids and reciprocal-frame transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from ebsdsim.pg_ops import CENTROSYMMETRIC_PG, fs_normals, in_fundamental_sector_vec, pg_num_to_symbol


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


@dataclass
class KChunk:
    kvecs: NDArray[np.float32]
    output_indices: NDArray[np.uint32]


def square_to_hemisphere(x: float, y: float, southern: bool = False) -> tuple[float, float, float]:
    scale = np.sqrt(np.pi / 2)
    x_abs = abs(x) * scale
    y_abs = abs(y) * scale
    swap = x_abs >= y_abs
    x_new = x_abs if swap else y_abs
    y_new = y_abs if swap else x_abs
    if x_new == 0:
        return 0.0, 0.0, -1.0 if southern else 1.0
    r = (2 * x_new / np.pi) * np.sqrt(max(np.pi - x_new * x_new, 0.0))
    x_hs = r * np.cos(np.pi * y_new / (4 * x_new))
    y_hs = r * np.sin(np.pi * y_new / (4 * x_new))
    z = -(1 - (2 * x_new * x_new / np.pi)) if southern else (1 - (2 * x_new * x_new / np.pi))
    hx = (x_hs if swap else y_hs) * np.sign(x or 1)
    hy = (y_hs if swap else x_hs) * np.sign(y or 1)
    norm = float(np.sqrt(hx * hx + hy * hy + z * z)) or 1.0
    return hx / norm, hy / norm, z / norm


def _square_to_hemisphere_array(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    southern: bool = False,
) -> NDArray[np.float32]:
    scale = np.sqrt(np.pi / 2)
    x_abs = np.abs(x) * scale
    y_abs = np.abs(y) * scale
    swap = x_abs >= y_abs
    x_new = np.where(swap, x_abs, y_abs)
    y_new = np.where(swap, y_abs, x_abs)

    r = np.zeros_like(x_new)
    nonzero = x_new != 0
    r[nonzero] = (
        (2 * x_new[nonzero] / np.pi)
        * np.sqrt(np.maximum(np.pi - x_new[nonzero] * x_new[nonzero], 0.0))
    )
    angle = np.zeros_like(x_new)
    angle[nonzero] = np.pi * y_new[nonzero] / (4 * x_new[nonzero])
    x_hs = r * np.cos(angle)
    y_hs = r * np.sin(angle)
    z = 1 - (2 * x_new * x_new / np.pi)
    if southern:
        z = -z
    hx = np.where(swap, x_hs, y_hs) * np.where(x == 0, 1.0, np.sign(x))
    hy = np.where(swap, y_hs, x_hs) * np.where(y == 0, 1.0, np.sign(y))
    norm = np.sqrt(hx * hx + hy * hy + z * z)
    norm = np.where(norm == 0, 1.0, norm)
    return np.stack((hx / norm, hy / norm, z / norm), axis=1).astype(np.float32)


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


def build_pg_k_grid(pg_num: int, hw: int) -> PgKGrid:
    symbol = pg_num_to_symbol(pg_num)
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
