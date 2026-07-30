"""Lambert layer: CPU raster of FS intensities onto Lambert squares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ebsdsim.crystal.pointgroup import fs_normals, point_group_operators
from ebsdsim.lambert.display import NormalizeMode, scale_fs_channel
from ebsdsim.lambert.kgrid import PgKGrid
from ebsdsim.lambert.projection import (
    hemisphere_to_square_array,
    orbit_fs_representative_array,
    sample_sheet,
    square_to_hemisphere_array,
)


@dataclass
class RasterizeOptions:
    normalize: NormalizeMode | None = None
    robust_p_low: float = 0.01
    robust_p_high: float = 0.99
    interp_mode: Literal["nearest", "bilinear"] = "bilinear"
    fs_only: bool = False


@dataclass
class RasterizedPattern:
    nh: NDArray[np.float32]
    sh: NDArray[np.float32]
    side: int


@dataclass
class PixelSourceMap:
    side: int
    src_x: NDArray[np.float32]
    src_y: NDArray[np.float32]
    from_sh: NDArray[np.uint8]


@dataclass
class RasterizePixelMaps:
    side: int
    nh: PixelSourceMap
    sh: PixelSourceMap | None


def _square_to_hemisphere_array(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    southern: bool,
) -> NDArray[np.float64]:
    return square_to_hemisphere_array(x, y, southern)


def _hemisphere_to_square_array(k: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    return hemisphere_to_square_array(k)


def _orbit_fs_representative_array(
    dirs: NDArray[np.float64],
    ops: NDArray[np.float64],
    normals: NDArray[np.float64],
    eps: float,
) -> NDArray[np.float64]:
    return orbit_fs_representative_array(dirs, ops, normals, eps)


def _build_pixel_source_map(hw: int, southern: bool, symbol: str) -> PixelSourceMap:
    side = 2 * hw + 1
    ops = point_group_operators(symbol)
    normals = fs_normals(symbol)
    eps = 1.0 / hw
    coords = np.linspace(-1.0, 1.0, side, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)
    x = xx.ravel()
    y = yy.ravel()
    src_x = np.empty(side * side, dtype=np.float32)
    src_y = np.empty(side * side, dtype=np.float32)
    from_sh = np.empty(side * side, dtype=np.uint8)
    chunk_size = 16384
    for start in range(0, x.size, chunk_size):
        end = min(start + chunk_size, x.size)
        dirs = _square_to_hemisphere_array(x[start:end], y[start:end], southern)
        reps = _orbit_fs_representative_array(dirs, ops, normals, eps)
        sx, sy = _hemisphere_to_square_array(reps)
        src_x[start:end] = sx
        src_y[start:end] = sy
        from_sh[start:end] = (reps[:, 2] < 0).astype(np.uint8)
    return PixelSourceMap(side=side, src_x=src_x, src_y=src_y, from_sh=from_sh)


def _grid_symbol(grid: PgKGrid) -> str:
    if not grid.symbol:
        raise ValueError(
            "PgKGrid.symbol is empty; build_pg_k_grid always sets the oriented "
            "folding symbol when space_group is provided"
        )
    return grid.symbol


def build_rasterize_pixel_maps(grid: PgKGrid) -> RasterizePixelMaps:
    symbol = _grid_symbol(grid)
    return RasterizePixelMaps(
        side=grid.side,
        nh=_build_pixel_source_map(grid.hw, False, symbol),
        sh=None if grid.is_centro else _build_pixel_source_map(grid.hw, True, symbol),
    )


def _sample_sheet(
    sheet: NDArray[np.float32],
    side: int,
    sx: float,
    sy: float,
    mode: Literal["nearest", "bilinear"],
) -> float:
    fj = ((sx + 1) * 0.5) * (side - 1)
    fi = ((sy + 1) * 0.5) * (side - 1)

    def clamp(x: float) -> float:
        return max(0.0, min(side - 1, float(x)))

    if mode == "nearest":
        ii = int(round(clamp(fi)))
        jj = int(round(clamp(fj)))
        return float(sheet[ii * side + jj])
    i0 = int(np.floor(clamp(fi)))
    j0 = int(np.floor(clamp(fj)))
    i1 = min(side - 1, i0 + 1)
    j1 = min(side - 1, j0 + 1)
    ti = clamp(fi) - i0
    tj = clamp(fj) - j0
    v00 = sheet[i0 * side + j0]
    v01 = sheet[i0 * side + j1]
    v10 = sheet[i1 * side + j0]
    v11 = sheet[i1 * side + j1]
    return float((v00 * (1 - ti) + v10 * ti) * (1 - tj) + (v01 * (1 - ti) + v11 * ti) * tj)


def _sample_sheet_array(
    sheet: NDArray[np.float32],
    side: int,
    sx: NDArray[np.float32],
    sy: NDArray[np.float32],
    mode: Literal["nearest", "bilinear"],
) -> NDArray[np.float32]:
    return sample_sheet(sheet, side, sx, sy, mode)


def rasterize_pattern(
    pattern: NDArray[np.floating] | list[float],
    grid: PgKGrid,
    n_sites: int = 1,
    opts: RasterizeOptions | None = None,
) -> RasterizedPattern:
    options = opts or RasterizeOptions()
    n = grid.kij.size // 3
    arr = np.asarray(pattern, dtype=np.float32)
    if n_sites > 1:
        per_dir = np.zeros(n, dtype=np.float32)
        for r in range(n):
            per_dir[r] = float(np.mean(arr[r * n_sites : (r + 1) * n_sites]))
    else:
        per_dir = arr if arr.ndim == 1 else arr.astype(np.float32)
    vals_f = scale_fs_channel(
        per_dir,
        options.normalize,
        robust_p_low=options.robust_p_low,
        robust_p_high=options.robust_p_high,
    )

    side = grid.side
    hw = grid.hw
    sheet_nh = np.zeros(side * side, dtype=np.float32)
    sheet_sh = np.zeros(side * side, dtype=np.float32)
    for r in range(n):
        x = int(grid.kij[r * 3]) + hw
        y = int(grid.kij[r * 3 + 1]) + hw
        sign = int(grid.kij[r * 3 + 2])
        idx = y * side + x
        if sign > 0:
            sheet_nh[idx] = vals_f[r]
        else:
            sheet_sh[idx] = vals_f[r]

    if options.fs_only:
        if grid.is_centro:
            return RasterizedPattern(nh=sheet_nh, sh=sheet_nh.copy(), side=side)
        return RasterizedPattern(nh=sheet_nh, sh=sheet_sh, side=side)

    def _maybe_clip(arr: NDArray[np.float32]) -> NDArray[np.float32]:
        if options.normalize is not None:
            return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    symbol = _grid_symbol(grid)
    nh_map = _build_pixel_source_map(hw, False, symbol)
    nh_from_nh = _sample_sheet_array(sheet_nh, side, nh_map.src_x, nh_map.src_y, options.interp_mode)
    if grid.is_centro:
        nh = _maybe_clip(nh_from_nh)
        return RasterizedPattern(nh=nh, sh=nh.copy(), side=side)
    nh_from_sh = _sample_sheet_array(sheet_sh, side, nh_map.src_x, nh_map.src_y, options.interp_mode)
    nh = _maybe_clip(np.where(nh_map.from_sh != 0, nh_from_sh, nh_from_nh).astype(np.float32))
    sh_map = _build_pixel_source_map(hw, True, symbol)
    sh_from_nh = _sample_sheet_array(sheet_nh, side, sh_map.src_x, sh_map.src_y, options.interp_mode)
    sh_from_sh = _sample_sheet_array(sheet_sh, side, sh_map.src_x, sh_map.src_y, options.interp_mode)
    sh = _maybe_clip(np.where(sh_map.from_sh != 0, sh_from_sh, sh_from_nh).astype(np.float32))
    return RasterizedPattern(nh=nh, sh=sh, side=side)


def float01_to_uint8(img: NDArray[np.float32]) -> NDArray[np.uint8]:
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
