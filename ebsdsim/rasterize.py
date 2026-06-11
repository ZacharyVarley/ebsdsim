"""Rasterize FS-only patterns onto full Lambert squares."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ebsdsim.kgrid import PgKGrid, square_to_hemisphere
from ebsdsim.normalize import NormalizeMode, scale_fs_channel
from ebsdsim.pg_ops import fs_normals, orbit_fs_representative, pg_num_to_symbol, point_group_operators


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


def _square_to_hemisphere_in_place(x: float, y: float, southern: bool) -> tuple[float, float, float]:
    return square_to_hemisphere(x, y, southern)


def _square_to_hemisphere_array(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    southern: bool,
) -> NDArray[np.float64]:
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
    return np.stack((hx / norm, hy / norm, z / norm), axis=1)


def _hemisphere_to_square(kx: float, ky: float, kz: float) -> tuple[float, float]:
    xn = np.sqrt(max(np.pi * (1 - kz) / 2, 0.0))
    ax = abs(kx)
    ay = abs(ky)
    swap = ax >= ay
    safe_ax = max(ax, 1e-15)
    safe_ay = max(ay, 1e-15)
    b_swap = (4 * xn / np.pi) * np.arctan2(ay, safe_ax)
    a_nswap = (4 * xn / np.pi) * np.arctan2(ax, safe_ay)
    u_abs = xn if swap else a_nswap
    v_abs = b_swap if swap else xn
    scale = np.sqrt(np.pi / 2)
    u = (u_abs / scale) * np.sign(kx or 1)
    v = (v_abs / scale) * np.sign(ky or 1)
    if kz >= 1 - 1e-12:
        u = 0.0
        v = 0.0
    return u, v


def _hemisphere_to_square_array(k: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    kx = k[:, 0]
    ky = k[:, 1]
    kz = np.abs(k[:, 2])
    xn = np.sqrt(np.maximum(np.pi * (1 - kz) / 2, 0.0))
    ax = np.abs(kx)
    ay = np.abs(ky)
    swap = ax >= ay
    safe_ax = np.maximum(ax, 1e-15)
    safe_ay = np.maximum(ay, 1e-15)
    b_swap = (4 * xn / np.pi) * np.arctan2(ay, safe_ax)
    a_nswap = (4 * xn / np.pi) * np.arctan2(ax, safe_ay)
    u_abs = np.where(swap, xn, a_nswap)
    v_abs = np.where(swap, b_swap, xn)
    scale = np.sqrt(np.pi / 2)
    u = (u_abs / scale) * np.where(kx == 0, 1.0, np.sign(kx))
    v = (v_abs / scale) * np.where(ky == 0, 1.0, np.sign(ky))
    at_pole = kz >= 1 - 1e-12
    u = np.where(at_pole, 0.0, u)
    v = np.where(at_pole, 0.0, v)
    return u.astype(np.float32), v.astype(np.float32)


def _orbit_fs_representative_array(
    dirs: NDArray[np.float64],
    ops: NDArray[np.float64],
    normals: NDArray[np.float64],
    eps: float,
) -> NDArray[np.float64]:
    op_mats = ops.reshape(-1, 3, 3)
    transformed = np.einsum("gij,nj->ngi", op_mats, dirs, optimize=True)
    if normals.size == 0:
        return transformed[:, 0, :]

    normal_mats = normals.reshape(-1, 3)
    margins = np.einsum("ngi,mi->ngm", transformed, normal_mats, optimize=True)
    min_margins = margins.min(axis=2)
    in_fs = min_margins >= -eps
    has_in_fs = in_fs.any(axis=1)
    fs_scores = np.where(in_fs, min_margins, -np.inf)
    best_in_fs = np.argmax(fs_scores, axis=1)
    best_any = np.argmax(min_margins, axis=1)
    best = np.where(has_in_fs, best_in_fs, best_any)
    return transformed[np.arange(dirs.shape[0]), best, :]


def _build_pixel_source_map(hw: int, southern: bool, pg_num: int) -> PixelSourceMap:
    side = 2 * hw + 1
    ops = point_group_operators(pg_num_to_symbol(pg_num))
    normals = fs_normals(pg_num_to_symbol(pg_num))
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


def build_rasterize_pixel_maps(grid: PgKGrid) -> RasterizePixelMaps:
    return RasterizePixelMaps(
        side=grid.side,
        nh=_build_pixel_source_map(grid.hw, False, grid.pg_num),
        sh=None if grid.is_centro else _build_pixel_source_map(grid.hw, True, grid.pg_num),
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
    fj = ((sx.astype(np.float32, copy=False) + 1.0) * 0.5) * (side - 1)
    fi = ((sy.astype(np.float32, copy=False) + 1.0) * 0.5) * (side - 1)
    fj = np.clip(fj, 0.0, side - 1)
    fi = np.clip(fi, 0.0, side - 1)
    if mode == "nearest":
        ii = np.rint(fi).astype(np.intp)
        jj = np.rint(fj).astype(np.intp)
        return sheet[ii * side + jj]

    i0 = np.floor(fi).astype(np.intp)
    j0 = np.floor(fj).astype(np.intp)
    i1 = np.minimum(side - 1, i0 + 1)
    j1 = np.minimum(side - 1, j0 + 1)
    ti = fi - i0.astype(np.float32)
    tj = fj - j0.astype(np.float32)
    v00 = sheet[i0 * side + j0]
    v01 = sheet[i0 * side + j1]
    v10 = sheet[i1 * side + j0]
    v11 = sheet[i1 * side + j1]
    out = (v00 * (1 - ti) + v10 * ti) * (1 - tj) + (v01 * (1 - ti) + v11 * ti) * tj
    return out.astype(np.float32, copy=False)


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

    nh_map = _build_pixel_source_map(hw, False, grid.pg_num)
    nh_from_nh = _sample_sheet_array(sheet_nh, side, nh_map.src_x, nh_map.src_y, options.interp_mode)
    if grid.is_centro:
        nh = _maybe_clip(nh_from_nh)
        return RasterizedPattern(nh=nh, sh=nh.copy(), side=side)
    nh_from_sh = _sample_sheet_array(sheet_sh, side, nh_map.src_x, nh_map.src_y, options.interp_mode)
    nh = _maybe_clip(np.where(nh_map.from_sh != 0, nh_from_sh, nh_from_nh).astype(np.float32))
    sh_map = _build_pixel_source_map(hw, True, grid.pg_num)
    sh_from_nh = _sample_sheet_array(sheet_nh, side, sh_map.src_x, sh_map.src_y, options.interp_mode)
    sh_from_sh = _sample_sheet_array(sheet_sh, side, sh_map.src_x, sh_map.src_y, options.interp_mode)
    sh = _maybe_clip(np.where(sh_map.from_sh != 0, sh_from_sh, sh_from_nh).astype(np.float32))
    return RasterizedPattern(nh=nh, sh=sh, side=side)


def float01_to_uint8(img: NDArray[np.float32]) -> NDArray[np.uint8]:
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
