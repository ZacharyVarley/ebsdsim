"""Lambert layer: square ↔ hemisphere projection (unit sphere).

Used by k-grid construction, CPU rasterize, and master-pattern load/expand."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


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


def square_to_hemisphere_array(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    southern: bool = False,
) -> NDArray[np.float64]:
    """Map equal-area square coords in ``[-1, 1]`` to unit hemisphere vectors."""
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


def hemisphere_to_square(kx: float, ky: float, kz: float) -> tuple[float, float]:
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


def hemisphere_to_square_array(k: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Inverse of :func:`square_to_hemisphere_array` (uses ``|kz|``)."""
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


def orbit_fs_representative_array(
    dirs: NDArray[np.float64],
    ops: NDArray[np.float64],
    normals: NDArray[np.float64],
    eps: float,
) -> NDArray[np.float64]:
    """Map each direction to its fundamental-sector representative (vectorized)."""
    op_mats = np.asarray(ops, dtype=np.float64).reshape(-1, 3, 3)
    transformed = np.einsum("gij,nj->ngi", op_mats, dirs, optimize=True)
    if normals.size == 0:
        return transformed[:, 0, :]
    normal_mats = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    margins = np.einsum("ngi,mi->ngm", transformed, normal_mats, optimize=True)
    min_margins = margins.min(axis=2)
    in_fs = min_margins >= -eps
    has_in_fs = in_fs.any(axis=1)
    fs_scores = np.where(in_fs, min_margins, -np.inf)
    best_in_fs = np.argmax(fs_scores, axis=1)
    best_any = np.argmax(min_margins, axis=1)
    best = np.where(has_in_fs, best_in_fs, best_any)
    return transformed[np.arange(dirs.shape[0]), best, :]


def sample_sheet(
    sheet: NDArray[np.float32],
    side: int,
    sx: NDArray[np.float32],
    sy: NDArray[np.float32],
    mode: Literal["nearest", "bilinear"],
) -> NDArray[np.float32]:
    fj = ((sx.astype(np.float32) + 1.0) * 0.5) * (side - 1)
    fi = ((sy.astype(np.float32) + 1.0) * 0.5) * (side - 1)
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
    return out.astype(np.float32)


def sample_sheets(
    sheet: NDArray[np.float32],
    side: int,
    sx: NDArray[np.float32],
    sy: NDArray[np.float32],
    mode: Literal["nearest", "bilinear"],
) -> NDArray[np.float32]:
    """Sample a stack of sheets ``(side*side, C)`` at ``(sx, sy)`` → ``(n_out, C)``.

    Vectorized over the channel axis.
    """
    fj = ((sx.astype(np.float32) + 1.0) * 0.5) * (side - 1)
    fi = ((sy.astype(np.float32) + 1.0) * 0.5) * (side - 1)
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
    ti = (fi - i0.astype(np.float32))[:, None]
    tj = (fj - j0.astype(np.float32))[:, None]
    v00 = sheet[i0 * side + j0]
    v01 = sheet[i0 * side + j1]
    v10 = sheet[i1 * side + j0]
    v11 = sheet[i1 * side + j1]
    out = (v00 * (1 - ti) + v10 * ti) * (1 - tj) + (v01 * (1 - ti) + v11 * ti) * tj
    return out.astype(np.float32)
