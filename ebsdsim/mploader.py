"""Standalone, NumPy-only loader for ebsdsim master-pattern ``.npz`` files.

This module is intentionally self-contained: it depends on **NumPy only** and
imports nothing from the rest of ``ebsdsim``. Drop it into any project that
needs to read the ``.npz`` files written by :func:`ebsdsim.save_master_pattern`
and expand the symmetry-reduced *fundamental sector* back into full Lambert
hemispheres.

Everything required for the expansion — the point-group operators and the
fundamental-sector normals — is embedded in the ``.npz`` itself, so the loader
needs no crystallographic tables of its own.

Quick start
-----------
>>> from ebsdsim.mploader import load_master_pattern, to_uint8, save_png_gray
>>> mp = load_master_pattern("GaN-master-pattern.npz")
>>> mp.data.shape          # (energy, site, hemisphere, H, W), built on load
>>> nh = mp.data[0, 0, 0]  # energy-integrated, site-integrated, north hemisphere
>>> save_png_gray(to_uint8(nh), "GaN_integrated_nh.png")
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "LoadedMasterPattern",
    "load_master_pattern",
    "build_master_pattern_data",
    "to_uint8",
    "save_png_gray",
]


# --------------------------------------------------------------------------- #
# Vectorized Lambert <-> hemisphere geometry (NumPy only).
# --------------------------------------------------------------------------- #
def _square_to_hemisphere(
    x: NDArray[np.float64], y: NDArray[np.float64], southern: bool
) -> NDArray[np.float64]:
    """Map equal-area square coordinates in [-1, 1] to unit hemisphere vectors."""
    scale = np.sqrt(np.pi / 2)
    x_abs = np.abs(x) * scale
    y_abs = np.abs(y) * scale
    swap = x_abs >= y_abs
    x_new = np.where(swap, x_abs, y_abs)
    y_new = np.where(swap, y_abs, x_abs)

    r = np.zeros_like(x_new)
    nonzero = x_new != 0
    r[nonzero] = (2 * x_new[nonzero] / np.pi) * np.sqrt(
        np.maximum(np.pi - x_new[nonzero] * x_new[nonzero], 0.0)
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


def _hemisphere_to_square(k: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Inverse of :func:`_square_to_hemisphere` (uses ``|kz|``)."""
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


def _orbit_fs_representative(
    dirs: NDArray[np.float64],
    ops: NDArray[np.float64],
    normals: NDArray[np.float64],
    eps: float,
) -> NDArray[np.float64]:
    """Map each direction to its fundamental-sector representative (vectorized)."""
    transformed = np.einsum("gij,nj->ngi", ops, dirs, optimize=True)
    if normals.size == 0:
        return transformed[:, 0, :]
    margins = np.einsum("ngi,mi->ngm", transformed, normals, optimize=True)
    min_margins = margins.min(axis=2)
    in_fs = min_margins >= -eps
    has_in_fs = in_fs.any(axis=1)
    fs_scores = np.where(in_fs, min_margins, -np.inf)
    best_in_fs = np.argmax(fs_scores, axis=1)
    best_any = np.argmax(min_margins, axis=1)
    best = np.where(has_in_fs, best_in_fs, best_any)
    return transformed[np.arange(dirs.shape[0]), best, :]


def _sample_sheet(
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


def _sample_sheets(
    sheet: NDArray[np.float32],
    side: int,
    sx: NDArray[np.float32],
    sy: NDArray[np.float32],
    mode: Literal["nearest", "bilinear"],
) -> NDArray[np.float32]:
    """Sample a stack of sheets ``(side*side, C)`` at ``(sx, sy)`` -> ``(n_out, C)``.

    Vectorized over the channel axis so every energy/site channel is expanded
    with a single pass over the (shared) pixel-source map.
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


def _robust_normalize(vals: NDArray[np.float32], robust: bool) -> NDArray[np.float32]:
    """Normalize to [0, 1], optionally clipping IQR outliers (matches the viewer)."""
    if vals.size == 0:
        return vals.astype(np.float32, copy=True)
    out = np.zeros_like(vals, dtype=np.float32)
    if robust:
        sorted_vals = np.sort(vals)

        def q(p: float) -> float:
            idx = min(sorted_vals.size - 1, max(0, int(np.floor(p * (sorted_vals.size - 1)))))
            return float(sorted_vals[idx])

        q1, q3 = q(0.25), q(0.75)
        iqr = q3 - q1
        if iqr > 0:
            good = sorted_vals[(sorted_vals >= q1 - 3 * iqr) & (sorted_vals <= q3 + 3 * iqr)]
            if good.size > 0:
                lo = float(good[int(np.floor(0.01 * (good.size - 1)))])
                hi = float(good[int(np.floor(0.99 * (good.size - 1)))])
            else:
                lo, hi = float(sorted_vals[0]), float(sorted_vals[-1])
        else:
            lo, hi = float(sorted_vals[0]), float(sorted_vals[-1])
    else:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    span = hi - lo
    if span <= 1e-15:
        return out
    out[:] = np.clip((vals - lo) / span, 0.0, 1.0)
    return out


def _build_pixel_source_map(
    hw: int,
    southern: bool,
    ops: NDArray[np.float64],
    normals: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint8]]:
    """For every output pixel, find the source (sx, sy) in the FS sheet."""
    side = 2 * hw + 1
    eps = 1.0 / hw
    coords = np.linspace(-1.0, 1.0, side, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)
    x = xx.ravel()
    y = yy.ravel()
    dirs = _square_to_hemisphere(x, y, southern)
    reps = _orbit_fs_representative(dirs, ops, normals, eps)
    sx, sy = _hemisphere_to_square(reps)
    from_sh = (reps[:, 2] < 0).astype(np.uint8)
    return sx, sy, from_sh


def build_master_pattern_data(
    *,
    integrated_fs: NDArray[np.floating],
    bin_fs: NDArray[np.floating],
    kij: NDArray[np.integer],
    pg_operators: NDArray[np.floating],
    fs_normals: NDArray[np.floating],
    hw: int,
    side: int,
    needs_southern_hemisphere: bool,
    normalize: bool = True,
    robust: bool = True,
    interp: Literal["nearest", "bilinear"] = "bilinear",
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Expand the fundamental sector into a dense ``(E, S, H, side, side)`` tensor.

    Axes
    ----
    ``E`` (energy): ``1 + n_bins`` when ``n_bins > 1`` (index 0 is the
        energy-integrated pattern, index ``1 + b`` is bin ``b``); otherwise
        ``1`` (the single bin, which is also the integrated pattern).
    ``S`` (site): ``1 + n_sites`` when ``n_sites > 1`` (index 0 is the
        site-integrated pattern, index ``1 + s`` is site ``s``); otherwise
        ``1`` (the single site).
    ``H`` (hemisphere): ``2`` when ``needs_southern_hemisphere`` else ``1``
        (index 0 north, index 1 south).

    The point-group pixel-source map is built once per hemisphere and shared
    across every energy/site channel, so the fundamental sector is expanded
    without redundant per-channel geometry work.

    Returns ``(data, axes)`` where ``axes`` documents the index layout.
    """
    integrated_fs = np.asarray(integrated_fs, dtype=np.float32)
    bin_fs = np.asarray(bin_fs, dtype=np.float32)
    kij = np.asarray(kij)
    n_k, n_sites = int(integrated_fs.shape[0]), int(integrated_fs.shape[1])
    n_bins = int(bin_fs.shape[0])

    # Energy axis: integrated first (only as a distinct slice when >1 bin).
    if n_bins > 1:
        energy_sources = [integrated_fs] + [bin_fs[b] for b in range(n_bins)]
        energy_integrated_index = 0
        bin_to_energy_index = [1 + b for b in range(n_bins)]
    else:
        energy_sources = [integrated_fs]
        energy_integrated_index = 0
        bin_to_energy_index = [0] * n_bins  # the single bin shares index 0
    e_dim = len(energy_sources)

    # Site axis: site-integrated (mean) first (only distinct when >1 site).
    multi_site = n_sites > 1
    s_dim = 1 + n_sites if multi_site else 1
    site_integrated_index = 0
    site_to_index = [1 + s for s in range(n_sites)] if multi_site else [0]

    n_channels = e_dim * s_dim
    channels = np.empty((n_k, n_channels), dtype=np.float32)
    c = 0
    for esrc in energy_sources:
        mean_dir = esrc.mean(axis=1, dtype=np.float32) if multi_site else esrc[:, 0]
        if multi_site:
            channels[:, c] = mean_dir
            c += 1
            for s in range(n_sites):
                channels[:, c] = esrc[:, s]
                c += 1
        else:
            channels[:, c] = mean_dir
            c += 1

    if normalize:
        for ci in range(n_channels):
            channels[:, ci] = _robust_normalize(channels[:, ci], robust)

    plane = side * side
    ix = kij[:, 0].astype(np.intp) + hw
    iy = kij[:, 1].astype(np.intp) + hw
    sign = kij[:, 2]
    dst = iy * side + ix
    north = sign > 0
    sheet_nh = np.zeros((plane, n_channels), dtype=np.float32)
    sheet_nh[dst[north]] = channels[north]

    h_dim = 2 if needs_southern_hemisphere else 1
    ops = np.asarray(pg_operators, dtype=np.float64)
    normals = np.asarray(fs_normals, dtype=np.float64)

    def _to_chw(out_pc: NDArray[np.float32]) -> NDArray[np.float32]:
        return out_pc.T.reshape(n_channels, side, side)

    sx_n, sy_n, from_sh_n = _build_pixel_source_map(hw, False, ops, normals)
    nh_from_nh = _sample_sheets(sheet_nh, side, sx_n, sy_n, interp)

    data = np.empty((e_dim, s_dim, h_dim, side, side), dtype=np.float32)

    if h_dim == 1:
        nh = np.clip(nh_from_nh, 0.0, 1.0)
        data[:, :, 0] = _to_chw(nh).reshape(e_dim, s_dim, side, side)
    else:
        sheet_sh = np.zeros((plane, n_channels), dtype=np.float32)
        sheet_sh[dst[~north]] = channels[~north]
        from_sh_n_col = (from_sh_n != 0)[:, None]
        nh_from_sh = _sample_sheets(sheet_sh, side, sx_n, sy_n, interp)
        nh = np.clip(np.where(from_sh_n_col, nh_from_sh, nh_from_nh), 0.0, 1.0)
        data[:, :, 0] = _to_chw(nh).reshape(e_dim, s_dim, side, side)

        sx_s, sy_s, from_sh_s = _build_pixel_source_map(hw, True, ops, normals)
        from_sh_s_col = (from_sh_s != 0)[:, None]
        sh_from_nh = _sample_sheets(sheet_nh, side, sx_s, sy_s, interp)
        sh_from_sh = _sample_sheets(sheet_sh, side, sx_s, sy_s, interp)
        sh = np.clip(np.where(from_sh_s_col, sh_from_sh, sh_from_nh), 0.0, 1.0)
        data[:, :, 1] = _to_chw(sh).reshape(e_dim, s_dim, side, side)

    axes = {
        "dims": ["energy", "site", "hemisphere", "height", "width"],
        "energy_dim": e_dim,
        "site_dim": s_dim,
        "hemisphere_dim": h_dim,
        "energy_integrated_index": energy_integrated_index,
        "bin_to_energy_index": bin_to_energy_index,
        "site_integrated_index": site_integrated_index,
        "site_to_index": site_to_index,
        "hemispheres": ["north"] if h_dim == 1 else ["north", "south"],
    }
    return data, axes


# --------------------------------------------------------------------------- #
# Loaded master pattern container.
# --------------------------------------------------------------------------- #
@dataclass
class LoadedMasterPattern:
    """A master pattern loaded from a ``.npz`` file, with hemisphere expansion."""

    meta: dict[str, Any]
    integrated_fs: NDArray[np.float32]  # (n_k, n_sites)
    bin_fs: NDArray[np.float32]  # (n_bins, n_k, n_sites)
    kij: NDArray[np.int32]  # (n_k, 3)
    khat: NDArray[np.float32]  # (n_k, 3)
    pg_operators: NDArray[np.float64]  # (n_ops, 3, 3)
    fs_normals: NDArray[np.float64]  # (n_normals, 3)
    bin_voltages_kv: NDArray[np.float32]
    bin_weights: NDArray[np.float32]
    data: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0,), np.float32))
    axes: dict[str, Any] = field(default_factory=dict)
    _maps: dict[bool, tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint8]]] = field(
        default_factory=dict, repr=False
    )

    # -- convenience accessors -------------------------------------------- #
    @property
    def halfw(self) -> int:
        return int(self.meta.get("halfw", (self.side - 1) // 2))

    @property
    def side(self) -> int:
        return int(self.meta.get("grid_size", int(np.max(np.abs(self.kij[:, :2])) * 2 + 1)))

    @property
    def n_k(self) -> int:
        return int(self.integrated_fs.shape[0])

    @property
    def n_sites(self) -> int:
        return int(self.integrated_fs.shape[1])

    @property
    def n_bins(self) -> int:
        return int(self.bin_fs.shape[0])

    @property
    def is_centrosymmetric(self) -> bool:
        return bool(self.meta.get("is_centrosymmetric", not self.needs_southern_hemisphere))

    @property
    def needs_southern_hemisphere(self) -> bool:
        if "needs_southern_hemisphere" in self.meta:
            return bool(self.meta["needs_southern_hemisphere"])
        return bool(np.any(self.khat[:, 2] < -1e-9))

    # -- expansion -------------------------------------------------------- #
    def reduce_over_sites(self, values_fs: NDArray[np.floating]) -> NDArray[np.float32]:
        """Collapse a ``(n_k, n_sites)`` array to one value per direction (site mean)."""
        arr = np.asarray(values_fs, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr.mean(axis=1, dtype=np.float32)

    def _pixel_map(self, southern: bool):
        if southern not in self._maps:
            self._maps[southern] = _build_pixel_source_map(
                self.halfw, southern, self.pg_operators, self.fs_normals
            )
        return self._maps[southern]

    def reconstruct(
        self,
        values_fs: NDArray[np.floating] | None = None,
        *,
        normalize: bool = True,
        robust: bool = True,
        interp: Literal["nearest", "bilinear"] = "bilinear",
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Expand fundamental-sector values into full ``(side, side)`` hemispheres.

        Returns ``(nh, sh)``. For centrosymmetric groups ``sh`` is a copy of
        ``nh``. If ``values_fs`` is ``None`` the integrated pattern is used.
        """
        if values_fs is None:
            values_fs = self.integrated_fs
        per_dir = self.reduce_over_sites(values_fs)
        vals = _robust_normalize(per_dir.astype(np.float32), robust) if normalize else per_dir.astype(np.float32)

        side = self.side
        hw = self.halfw
        plane = side * side
        sheet_nh = np.zeros(plane, dtype=np.float32)
        sheet_sh = np.zeros(plane, dtype=np.float32)
        ix = self.kij[:, 0].astype(np.intp) + hw
        iy = self.kij[:, 1].astype(np.intp) + hw
        sign = self.kij[:, 2]
        dst = iy * side + ix
        north = sign > 0
        sheet_nh[dst[north]] = vals[north]
        sheet_sh[dst[~north]] = vals[~north]

        sx_n, sy_n, from_sh_n = self._pixel_map(False)
        nh_from_nh = _sample_sheet(sheet_nh, side, sx_n, sy_n, interp)
        if self.is_centrosymmetric:
            nh = np.clip(nh_from_nh, 0.0, 1.0).astype(np.float32).reshape(side, side)
            return nh, nh.copy()

        nh_from_sh = _sample_sheet(sheet_sh, side, sx_n, sy_n, interp)
        nh = np.clip(np.where(from_sh_n != 0, nh_from_sh, nh_from_nh), 0.0, 1.0).astype(np.float32)

        sx_s, sy_s, from_sh_s = self._pixel_map(True)
        sh_from_nh = _sample_sheet(sheet_nh, side, sx_s, sy_s, interp)
        sh_from_sh = _sample_sheet(sheet_sh, side, sx_s, sy_s, interp)
        sh = np.clip(np.where(from_sh_s != 0, sh_from_sh, sh_from_nh), 0.0, 1.0).astype(np.float32)
        return nh.reshape(side, side), sh.reshape(side, side)

    def reconstruct_integrated(
        self, **kwargs: Any
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Expand the energy-integrated master pattern."""
        return self.reconstruct(self.integrated_fs, **kwargs)

    def reconstruct_bin(
        self, bin_index: int, **kwargs: Any
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Expand a single per-energy-bin intermediate."""
        if bin_index < 0 or bin_index >= self.n_bins:
            raise IndexError(f"bin_index {bin_index} out of range [0, {self.n_bins})")
        return self.reconstruct(self.bin_fs[bin_index], **kwargs)


def _deconsolidate_fundamental_sector(
    fs: NDArray[np.float32], n_bins: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Unpack a ``(E, S, n_k)`` array into ``integrated_fs`` and ``bin_fs``.

    Inverse of the consolidation done by :func:`ebsdsim.save.save_master_pattern`:
    energy index 0 is the energy-integrated slice and ``1 + b`` is bin ``b``
    (when ``E > 1``); site index 0 is the site mean and ``1 + s`` is site ``s``
    (when ``S > 1``).
    """
    fs = np.asarray(fs, dtype=np.float32)
    e_dim, s_dim, n_k = int(fs.shape[0]), int(fs.shape[1]), int(fs.shape[2])
    multi_bin = e_dim > 1
    multi_site = s_dim > 1
    n_sites = (s_dim - 1) if multi_site else 1
    site_idx = [(1 + s) if multi_site else 0 for s in range(n_sites)]

    integrated_fs = np.empty((n_k, n_sites), dtype=np.float32)
    for s in range(n_sites):
        integrated_fs[:, s] = fs[0, site_idx[s]]

    bin_fs = np.empty((n_bins, n_k, n_sites), dtype=np.float32)
    for b in range(n_bins):
        e_idx = (1 + b) if multi_bin else 0
        for s in range(n_sites):
            bin_fs[b, :, s] = fs[e_idx, site_idx[s]]
    return integrated_fs, bin_fs


def load_master_pattern(path: str | Path) -> LoadedMasterPattern:
    """Load an ebsdsim master-pattern ``.npz`` written by ``save_master_pattern``."""
    with np.load(Path(path), allow_pickle=False) as data:
        meta_bytes = bytes(np.asarray(data["meta_json"], dtype=np.uint8).tobytes())
        meta = json.loads(meta_bytes.decode("utf-8")) if meta_bytes else {}
        bin_voltages_kv = np.asarray(data["bin_voltages_kv"], dtype=np.float32)
        bin_weights = np.asarray(data["bin_weights"], dtype=np.float32)
        if "fundamental_sector" in data:
            integrated_fs, bin_fs = _deconsolidate_fundamental_sector(
                np.asarray(data["fundamental_sector"], dtype=np.float32),
                int(bin_voltages_kv.size),
            )
        else:  # legacy format (format_version 1): two separate arrays
            integrated_fs = np.asarray(data["integrated_fundamental_sector"], dtype=np.float32)
            bin_fs = np.asarray(data["bin_fundamental_sector"], dtype=np.float32)
        loaded = LoadedMasterPattern(
            meta=meta,
            integrated_fs=integrated_fs,
            bin_fs=bin_fs,
            kij=np.asarray(data["fundamental_kij"], dtype=np.int32),
            khat=np.asarray(data["fundamental_khat"], dtype=np.float32),
            pg_operators=np.asarray(data["pg_operators"], dtype=np.float64),
            fs_normals=np.asarray(data["fs_normals"], dtype=np.float64),
            bin_voltages_kv=bin_voltages_kv,
            bin_weights=bin_weights,
        )
    loaded.data, loaded.axes = build_master_pattern_data(
        integrated_fs=loaded.integrated_fs,
        bin_fs=loaded.bin_fs,
        kij=loaded.kij,
        pg_operators=loaded.pg_operators,
        fs_normals=loaded.fs_normals,
        hw=loaded.halfw,
        side=loaded.side,
        needs_southern_hemisphere=loaded.needs_southern_hemisphere,
    )
    return loaded


# --------------------------------------------------------------------------- #
# Tiny dependency-free image helpers.
# --------------------------------------------------------------------------- #
def to_uint8(img01: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Convert a float image in [0, 1] to ``uint8`` [0, 255]."""
    return np.clip(np.round(np.asarray(img01, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def save_png_gray(img_uint8: NDArray[np.uint8], path: str | Path) -> Path:
    """Write a 2-D ``uint8`` array as an 8-bit grayscale PNG (stdlib only)."""
    arr = np.ascontiguousarray(np.asarray(img_uint8, dtype=np.uint8))
    if arr.ndim != 2:
        raise ValueError("save_png_gray expects a 2-D grayscale array")
    height, width = arr.shape

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    # Each scanline is prefixed with a filter-type byte (0 = none).
    raw = bytearray()
    for row in arr:
        raw.append(0)
        raw.extend(row.tobytes())
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _chunk(b"IEND", b"")
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png)
    return out_path
