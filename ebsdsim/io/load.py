"""I/O layer: load master-pattern .npz (raw FS + Lambert expand).

Lambert geometry and display scaling live in :mod:`ebsdsim.lambert`; this module
owns NPZ I/O and the expand-to-Lambert orchestration.

Quick start
-----------
>>> from ebsdsim.io.load import load_master_pattern, to_uint8, save_png_gray
>>> mp = load_master_pattern("GaN-master-pattern.npz")  # raw Lambert data in mp.data
>>> disp, _ = mp.lambert_data(normalize="robust")  # display scaling on demand
>>> nh = disp[0, 0, 0]  # energy-integrated, site-integrated, north hemisphere
>>> save_png_gray(to_uint8(nh), "GaN_integrated_nh.png")"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ebsdsim.lambert.display import NormalizeMode, scale_fs_channel
from ebsdsim.lambert.projection import (
    hemisphere_to_square_array,
    orbit_fs_representative_array,
    sample_sheet,
    sample_sheets,
    square_to_hemisphere_array,
)

__all__ = [
    "LoadedMasterPattern",
    "NormalizeMode",
    "load_master_pattern",
    "build_master_pattern_data",
    "to_uint8",
    "save_png_gray",
]


def _square_to_hemisphere(
    x: NDArray[np.float64], y: NDArray[np.float64], southern: bool
) -> NDArray[np.float64]:
    return square_to_hemisphere_array(x, y, southern)


def _hemisphere_to_square(k: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    return hemisphere_to_square_array(k)


def _orbit_fs_representative(
    dirs: NDArray[np.float64],
    ops: NDArray[np.float64],
    normals: NDArray[np.float64],
    eps: float,
) -> NDArray[np.float64]:
    return orbit_fs_representative_array(dirs, ops, normals, eps)


def _sample_sheet(
    sheet: NDArray[np.float32],
    side: int,
    sx: NDArray[np.float32],
    sy: NDArray[np.float32],
    mode: Literal["nearest", "bilinear"],
) -> NDArray[np.float32]:
    return sample_sheet(sheet, side, sx, sy, mode)


def _sample_sheets(
    sheet: NDArray[np.float32],
    side: int,
    sx: NDArray[np.float32],
    sy: NDArray[np.float32],
    mode: Literal["nearest", "bilinear"],
) -> NDArray[np.float32]:
    return sample_sheets(sheet, side, sx, sy, mode)


def _reduce_over_sites(
    values_fs: NDArray[np.floating],
    site_weights: NDArray[np.floating] | None,
) -> NDArray[np.float32]:
    arr = np.asarray(values_fs, dtype=np.float32)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] == 1:
        return arr[:, 0].astype(np.float32, copy=False)
    if site_weights is None:
        return arr.mean(axis=1, dtype=np.float32)
    w = np.asarray(site_weights, dtype=np.float64).reshape(-1)
    if w.size != arr.shape[1]:
        return arr.mean(axis=1, dtype=np.float32)
    w = w / w.sum()
    return (arr.astype(np.float64) * w).sum(axis=1).astype(np.float32)


def _site_weights_from_meta_cell(meta_cell: dict) -> NDArray[np.float32] | None:
    sites = meta_cell.get("sites")
    if not sites:
        return None
    w = np.array(
        [
            float(s.get("occupancy", 1.0)) * float(s.get("multiplicity") or 1)
            for s in sites
        ],
        dtype=np.float64,
    )
    total = float(w.sum())
    if total <= 0:
        return None
    return (w / total).astype(np.float32)


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
    site_weights: NDArray[np.floating] | None = None,
    normalize: NormalizeMode | None = None,
    robust_p_low: float = 0.01,
    robust_p_high: float = 0.99,
    interp: Literal["nearest", "bilinear"] = "bilinear",
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Expand the fundamental sector into a dense ``(E, S, H, side, side)`` tensor.

    Axes
    ----
    ``E`` (energy): ``1 + n_bins`` when ``n_bins > 1`` (index 0 is the
        energy-integrated pattern, index ``1 + b`` is bin ``b``); otherwise
        ``1`` (integrated-only when ``n_bins == 0``, or a single stored bin
        that shares index 0 when ``n_bins == 1``).
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
    # n_bins == 0 (integrated-only): single energy channel, empty map.
    if n_bins > 1:
        energy_sources = [integrated_fs] + [bin_fs[b] for b in range(n_bins)]
        energy_integrated_index = 0
        bin_to_energy_index = [1 + b for b in range(n_bins)]
    else:
        energy_sources = [integrated_fs]
        energy_integrated_index = 0
        bin_to_energy_index = [0] * n_bins  # [] when n_bins==0; [0] when n_bins==1
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
        mean_dir = _reduce_over_sites(esrc, site_weights) if multi_site else esrc[:, 0]
        if multi_site:
            channels[:, c] = mean_dir
            c += 1
            for s in range(n_sites):
                channels[:, c] = esrc[:, s]
                c += 1
        else:
            channels[:, c] = mean_dir
            c += 1

    if normalize is not None:
        for ci in range(n_channels):
            channels[:, ci] = scale_fs_channel(
                channels[:, ci],
                normalize,
                robust_p_low=robust_p_low,
                robust_p_high=robust_p_high,
            )

    plane = side * side
    ix = kij[:, 0].astype(np.intp) + hw
    iy = kij[:, 1].astype(np.intp) + hw
    sign = kij[:, 2]
    dst = iy * side + ix
    north = sign > 0
    sheet_nh = np.zeros((plane, n_channels), dtype=np.float32)
    sheet_nh[dst[north]] = channels[north]

    h_dim = 2 if needs_southern_hemisphere else 1
    ops = np.asarray(pg_operators, dtype=np.float64).reshape(-1, 3, 3)
    normals = np.asarray(fs_normals, dtype=np.float64).reshape(-1, 3)

    def _to_chw(out_pc: NDArray[np.float32]) -> NDArray[np.float32]:
        return out_pc.T.reshape(n_channels, side, side)

    sx_n, sy_n, from_sh_n = _build_pixel_source_map(hw, False, ops, normals)
    nh_from_nh = _sample_sheets(sheet_nh, side, sx_n, sy_n, interp)

    data = np.empty((e_dim, s_dim, h_dim, side, side), dtype=np.float32)

    def _maybe_clip(arr: NDArray[np.float32]) -> NDArray[np.float32]:
        if normalize is not None:
            return np.clip(arr, 0.0, 1.0)
        return arr

    if h_dim == 1:
        nh = _maybe_clip(nh_from_nh)
        data[:, :, 0] = _to_chw(nh).reshape(e_dim, s_dim, side, side)
    else:
        sheet_sh = np.zeros((plane, n_channels), dtype=np.float32)
        sheet_sh[dst[~north]] = channels[~north]
        from_sh_n_col = (from_sh_n != 0)[:, None]
        nh_from_sh = _sample_sheets(sheet_sh, side, sx_n, sy_n, interp)
        nh = _maybe_clip(np.where(from_sh_n_col, nh_from_sh, nh_from_nh).astype(np.float32))
        data[:, :, 0] = _to_chw(nh).reshape(e_dim, s_dim, side, side)

        sx_s, sy_s, from_sh_s = _build_pixel_source_map(hw, True, ops, normals)
        from_sh_s_col = (from_sh_s != 0)[:, None]
        sh_from_nh = _sample_sheets(sheet_nh, side, sx_s, sy_s, interp)
        sh_from_sh = _sample_sheets(sheet_sh, side, sx_s, sy_s, interp)
        sh = _maybe_clip(np.where(from_sh_s_col, sh_from_sh, sh_from_nh).astype(np.float32))
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
    """Master pattern loaded from a ``.npz`` file.

    :attr:`meta` holds simulation metadata. :attr:`integrated_fs` and
    :attr:`bin_fs` are fundamental-sector intensities; :attr:`data` is the
    expanded Lambert tensor of raw values. See :attr:`axes` for index maps.

    ``n_bins == 0`` (empty ``bin_fs``) means energy-integrated only: voltages
    and weights describe the energy model but no per-bin intensity slices were
    stored. Current solvers (``lu_smith`` and Smith iterative) store one pattern
    per voltage bin; ``bin_voltages_kv.size`` must not be treated as ``n_bins``
    for older integrated-only files.
    """

    meta: dict[str, Any]
    integrated_fs: NDArray[np.float32]  # (n_k, n_sites)
    bin_fs: NDArray[np.float32]  # (n_bins, n_k, n_sites); n_bins==0 if integrated-only
    kij: NDArray[np.int32]  # (n_k, 3)
    khat: NDArray[np.float32]  # (n_k, 3)
    pg_operators: NDArray[np.float64]  # (n_ops, 3, 3)
    fs_normals: NDArray[np.float64]  # (n_normals, 3)
    bin_voltages_kv: NDArray[np.float32]
    bin_weights: NDArray[np.float32]
    site_weights: NDArray[np.float32] | None = None
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
        """Number of stored per-bin intensity slices (0 = integrated-only)."""
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
        """Collapse a ``(n_k, n_sites)`` array to one value per direction."""
        return _reduce_over_sites(values_fs, self.site_weights)

    def lambert_data(
        self,
        *,
        normalize: NormalizeMode | None = None,
        robust_p_low: float = 0.01,
        robust_p_high: float = 0.99,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Expand raw FS intensities to Lambert ``(E, S, H, side, side)``.

        ``normalize`` is ``None`` (raw), ``"minmax"``, or ``"robust"``. Display
        scaling is applied on demand only; :attr:`data` always stays raw.
        """
        if normalize is None:
            return self.data, self.axes
        return build_master_pattern_data(
            integrated_fs=self.integrated_fs,
            bin_fs=self.bin_fs,
            kij=self.kij,
            pg_operators=self.pg_operators,
            fs_normals=self.fs_normals,
            hw=self.halfw,
            side=self.side,
            needs_southern_hemisphere=self.needs_southern_hemisphere,
            site_weights=self.site_weights,
            normalize=normalize,
            robust_p_low=robust_p_low,
            robust_p_high=robust_p_high,
        )

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
        normalize: NormalizeMode | None = None,
        robust_p_low: float = 0.01,
        robust_p_high: float = 0.99,
        interp: Literal["nearest", "bilinear"] = "bilinear",
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Expand fundamental-sector values into full ``(side, side)`` hemispheres.

        Returns ``(nh, sh)``. For centrosymmetric groups ``sh`` is a copy of
        ``nh``. If ``values_fs`` is ``None`` the integrated pattern is used.
        """
        if values_fs is None:
            values_fs = self.integrated_fs
        per_dir = self.reduce_over_sites(values_fs)
        vals = scale_fs_channel(
            per_dir,
            normalize,
            robust_p_low=robust_p_low,
            robust_p_high=robust_p_high,
        )

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
        def _maybe_clip(arr: NDArray[np.float32]) -> NDArray[np.float32]:
            if normalize is not None:
                return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
            return arr.astype(np.float32, copy=False)

        if self.is_centrosymmetric:
            nh = _maybe_clip(nh_from_nh).reshape(side, side)
            return nh, nh.copy()

        nh_from_sh = _sample_sheet(sheet_sh, side, sx_n, sy_n, interp)
        nh = _maybe_clip(np.where(from_sh_n != 0, nh_from_sh, nh_from_nh).astype(np.float32))

        sx_s, sy_s, from_sh_s = self._pixel_map(True)
        sh_from_nh = _sample_sheet(sheet_nh, side, sx_s, sy_s, interp)
        sh_from_sh = _sample_sheet(sheet_sh, side, sx_s, sy_s, interp)
        sh = _maybe_clip(np.where(from_sh_s != 0, sh_from_sh, sh_from_nh).astype(np.float32))
        return nh.reshape(side, side), sh.reshape(side, side)

    def reconstruct_integrated(
        self, **kwargs: Any
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Expand the energy-integrated master pattern."""
        return self.reconstruct(self.integrated_fs, **kwargs)

    def reconstruct_bin(
        self, bin_index: int, **kwargs: Any
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Expand a single per-energy-bin intermediate.

        Raises ``IndexError`` when no per-bin slices were stored
        (``n_bins == 0``, integrated-only).
        """
        if self.n_bins == 0:
            raise IndexError(
                "no per-bin intensity slices stored (integrated-only master pattern); "
                "use reconstruct_integrated() instead"
            )
        if bin_index < 0 or bin_index >= self.n_bins:
            raise IndexError(f"bin_index {bin_index} out of range [0, {self.n_bins})")
        return self.reconstruct(self.bin_fs[bin_index], **kwargs)


def _n_stored_bin_patterns(
    fs: NDArray[np.float32],
    meta: dict[str, Any],
    n_voltages: int,
) -> int:
    """Count stored per-bin intensity slices (not energy-model voltage count).

    Prefer explicit ``meta['n_bin_patterns']`` (format_version ≥ 3), then
    ``fundamental_sector_axes['bin_to_energy_index']`` (format_version ≥ 2).
    Fallback for older files: distinct energy axis when ``E > 1``; when
    ``E == 1`` and more than one voltage was recorded, treat as integrated-only
    (legacy files that stored voltages without per-bin intensity slices).
    """
    if "n_bin_patterns" in meta:
        return int(meta["n_bin_patterns"])
    axes = meta.get("fundamental_sector_axes") or {}
    if "bin_to_energy_index" in axes:
        return len(axes["bin_to_energy_index"])
    e_dim = int(np.asarray(fs).shape[0])
    if e_dim > 1:
        return e_dim - 1
    # E == 1: either one real bin (lu_smith) or integrated-only with N voltages.
    if n_voltages > 1:
        return 0
    return 1 if n_voltages == 1 else 0


def _deconsolidate_fundamental_sector(
    fs: NDArray[np.float32], n_bins: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Unpack a ``(E, S, n_k)`` array into ``integrated_fs`` and ``bin_fs``.

    Inverse of the consolidation done by :func:`ebsdsim.save.save_master_pattern`:
    energy index 0 is the energy-integrated slice and ``1 + b`` is bin ``b``
    (when ``E > 1``); site index 0 is the site mean and ``1 + s`` is site ``s``
    (when ``S > 1``). ``n_bins == 0`` yields an empty ``bin_fs``.
    """
    fs = np.asarray(fs, dtype=np.float32)
    e_dim, s_dim, n_k = int(fs.shape[0]), int(fs.shape[1]), int(fs.shape[2])
    multi_bin = e_dim > 1
    multi_site = s_dim > 1
    n_sites = (s_dim - 1) if multi_site else 1
    site_idx = np.arange(1, 1 + n_sites, dtype=np.intp) if multi_site else np.zeros(1, dtype=np.intp)

    # fs[0, site_idx, :] -> (n_sites, n_k) then transpose to (n_k, n_sites)
    integrated_fs = np.ascontiguousarray(fs[0, site_idx, :].T)

    if n_bins == 0:
        bin_fs = np.zeros((0, n_k, n_sites), dtype=np.float32)
    elif multi_bin:
        e_idx = np.arange(1, 1 + n_bins, dtype=np.intp)
        # fs[e_idx][:, site_idx, :] -> (n_bins, n_sites, n_k) -> (n_bins, n_k, n_sites)
        bin_fs = np.ascontiguousarray(fs[np.ix_(e_idx, site_idx, np.arange(n_k))].transpose(0, 2, 1))
    else:
        # Single energy slice shared by each stored bin (n_bins == 1 typically).
        shared = fs[0, site_idx, :].T  # (n_k, n_sites)
        bin_fs = np.broadcast_to(shared[None, ...], (n_bins, n_k, n_sites)).copy()
    return integrated_fs, bin_fs


def load_master_pattern(path: str | Path) -> LoadedMasterPattern:
    """Load an ebsdsim master-pattern ``.npz``.

    Parameters
    ----------
    path : str or Path
        File written by :func:`ebsdsim.save_master_pattern`.

    Returns
    -------
    LoadedMasterPattern
        Raw Lambert data in :attr:`LoadedMasterPattern.data`; call
        :meth:`LoadedMasterPattern.lambert_data` for display scaling.
        ``n_bins`` is the number of stored intensity slices (0 when the file
        is integrated-only); do not equate it with ``bin_voltages_kv.size``.

    Notes
    -----
    Folding geometry is taken from the stored ``pg_operators`` / ``fs_normals``
    arrays (and the oriented ``meta['pg_symbol']`` when present). The loader
    never recomputes operators from a bare ``pg_num``, which would be
    orientation-ambiguous for some crystal families.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        meta_bytes = bytes(np.asarray(data["meta_json"], dtype=np.uint8).tobytes())
        meta = json.loads(meta_bytes.decode("utf-8")) if meta_bytes else {}
        bin_voltages_kv = np.asarray(data["bin_voltages_kv"], dtype=np.float32)
        bin_weights = np.asarray(data["bin_weights"], dtype=np.float32)
        if "site_weights" in data:
            site_weights = np.asarray(data["site_weights"], dtype=np.float32).reshape(-1)
        else:
            site_weights = _site_weights_from_meta_cell(meta.get("cell", {}))
        if "fundamental_sector" in data:
            fs = np.asarray(data["fundamental_sector"], dtype=np.float32)
            n_bin_patterns = _n_stored_bin_patterns(fs, meta, int(bin_voltages_kv.size))
            integrated_fs, bin_fs = _deconsolidate_fundamental_sector(fs, n_bin_patterns)
        else:  # legacy format (format_version 1): two separate arrays
            integrated_fs = np.asarray(data["integrated_fundamental_sector"], dtype=np.float32)
            bin_fs = np.asarray(data["bin_fundamental_sector"], dtype=np.float32)
        # Prefer on-disk oriented operators; never rebuild from bare pg_num.
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
            site_weights=site_weights,
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
        site_weights=loaded.site_weights,
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
