"""I/O layer: compressed .npz export of master patterns.

The on-disk format stores the **fundamental-sector** intensities only (the
minimal, symmetry-reduced representation) and is always compressed. Per-energy
bin intensity slices are written when :attr:`MasterPattern.bin_patterns` is
non-empty (``len(bin_patterns) == len(bin_voltages_kv)`` for both ``lu_smith``
and Smith iterative). Smith iterative runs also keep a separate device-accumulated integrated
total; ``bin_voltages_kv`` / ``bin_weights`` remain the energy-model metadata.
The embedded point-group operators and fundamental-sector normals make the
file self-describing: it can be expanded back into full Lambert hemispheres
with NumPy alone (see :mod:`ebsdsim.io.load`).

Arrays written
--------------
``fundamental_sector`` : float32 ``(n_energy, n_site, n_k)``
    The symmetry-reduced intensities for every direction, packed along an
    energy axis and a site axis (raw, un-normalized). The energy axis is
    ``1 + n_bins`` long when ``n_bins > 1`` (index 0 is the energy-integrated
    weighted sum, index ``1 + b`` is bin ``b``) and length 1 otherwise
    (integrated-only when ``bin_patterns`` is empty, or a single stored bin).
    The site axis is ``1 + n_sites`` long when ``n_sites > 1`` (index 0 is the
    site mean, index ``1 + s`` is site ``s``) and length 1 otherwise. The
    north/south hemisphere of each of the ``n_k`` directions is carried by the
    sign column of ``fundamental_kij`` (the per-group N/S split is irregular,
    so hemisphere is not a separate array axis).

``fundamental_kij`` : int32 ``(n_k, 3)``
    Lambert pixel indices ``(i, j, sign)`` for each fundamental-sector
    direction (``sign > 0`` north, ``sign < 0`` south).

``fundamental_khat`` : float32 ``(n_k, 3)``
    Unit propagation directions for each fundamental-sector pixel.

``pg_operators`` : float64 ``(n_ops, 3, 3)``
    Proper/improper point-group rotation matrices.

``fs_normals`` : float64 ``(n_normals, 3)``
    Inward normals bounding the fundamental sector.

``bin_voltages_kv`` / ``bin_weights`` : float32 ``(n_energy_bins,)``
    Dynamical voltage (kV) and energy weight of each *energy-model* bin.
    When per-bin patterns are stored their count matches this length; an empty
    ``bin_patterns`` list is the integrated-only edge case (no intensity slices).

``site_weights`` : float32 ``(n_sites,)``
    Normalized occupancy × multiplicity weights used for the site-integrated
    marginal (index 0 on the site axis). Per-site slices are raw intensities.

``meta_json`` : uint8 ``(n_bytes,)``
    UTF-8 JSON metadata blob (decode with ``bytes(arr).decode("utf-8")``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ebsdsim.crystal.pointgroup import fs_normals, point_group_operators, resolve_oriented_symbol
from ebsdsim.energy.weights import reduce_over_sites, site_weights_from_meta_cell
from ebsdsim.engine.metadata import cell_metadata
from ebsdsim.engine.results import MasterPattern, stack_bins, validate_bin_contract

__all__ = [
    "cell_metadata",
    "save_master_pattern",
    "stack_bins",
]


# On-disk semantic version: v3 records ``n_bin_patterns`` explicitly so loaders
# never invent per-bin intensity slices from energy-model voltages alone.
_FORMAT_VERSION = 3


def _consolidate_fundamental_sector(
    integrated_fs: np.ndarray,  # (n_k, n_sites)
    bin_fs: np.ndarray,  # (n_bins, n_k, n_sites)
    site_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pack integrated + per-bin and mean + per-site into one ``(E, S, n_k)`` array.

    Mirrors the in-memory ``MasterPattern.data`` axis convention (minus the
    hemisphere and Lambert image axes): energy index 0 is the energy-integrated
    weighted sum, ``1 + b`` is bin ``b`` (a distinct slice only when
    ``n_bins > 1``); site index 0 is the site mean, ``1 + s`` is site ``s`` (a
    distinct slice only when ``n_sites > 1``). Values are raw (un-normalized)
    so the loader can reproduce the display normalization.

    When ``n_bins == 0`` (integrated-only; no stored per-bin slices), only the
    energy-integrated slice is written and ``bin_to_energy_index`` is empty.
    """
    integrated_fs = np.asarray(integrated_fs, dtype=np.float32)
    bin_fs = np.asarray(bin_fs, dtype=np.float32)
    n_k, n_sites = int(integrated_fs.shape[0]), int(integrated_fs.shape[1])
    n_bins = int(bin_fs.shape[0])
    multi_bin = n_bins > 1
    multi_site = n_sites > 1
    e_dim = 1 + n_bins if multi_bin else 1
    s_dim = 1 + n_sites if multi_site else 1
    out = np.empty((e_dim, s_dim, n_k), dtype=np.float32)

    def _fill(e_idx: int, src: np.ndarray) -> None:  # src: (n_k, n_sites)
        if multi_site:
            out[e_idx, 0] = reduce_over_sites(src, site_weights)
            # Vectorized: src[:, s] -> out[e_idx, 1 + s, :]
            out[e_idx, 1 : 1 + n_sites] = src.T
        else:
            out[e_idx, 0] = src[:, 0]

    _fill(0, integrated_fs)
    if multi_bin:
        for b in range(n_bins):
            _fill(1 + b, bin_fs[b])

    axes_meta = {
        "dims": ["energy", "site", "direction"],
        "energy_integrated_index": 0,
        "bin_to_energy_index": [1 + b for b in range(n_bins)] if multi_bin else [0] * n_bins,
        "site_integrated_index": 0,
        "site_to_index": [1 + s for s in range(n_sites)] if multi_site else [0],
    }
    return out, axes_meta


def save_master_pattern(mp: MasterPattern, path: str | Path) -> Path:
    """Write a master pattern (with intermediates) to a compressed ``.npz``.

    The file stores symmetry-reduced fundamental-sector intensities plus
    embedded point-group operators so it can be expanded offline with
    :mod:`ebsdsim.io.load`.

    Per-bin intensity slices are written when ``mp.bin_patterns`` is non-empty
    (same count contract as ``lu_smith`` / Smith iterative:
    ``len(bin_patterns) == len(bin_voltages_kv)``). Empty ``bin_patterns`` with
    non-empty ``bin_voltages_kv`` remains a legal integrated-only edge case:
    voltages/weights are energy-model metadata, not a claim that one pattern
    exists per voltage.

    Parameters
    ----------
    mp : MasterPattern
        Result from :func:`~ebsdsim.api.master_pattern` or
        :func:`~ebsdsim.api.master_pattern_from_cif`.
    path : str or Path
        Output path; ``.npz`` is appended if missing.

    Returns
    -------
    Path
        Resolved output path.
    """
    out_path = Path(path)
    if out_path.suffix.lower() != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mp.pg_num is None or mp.kij is None or mp.khat is None:
        raise ValueError(
            "MasterPattern is missing fundamental-sector grid data; it must be "
            "produced by ebsdsim.master_pattern[_from_cif]() to be saved."
        )

    validate_bin_contract(mp.bin_patterns, mp.bin_voltages_kv, mp.bin_weights)

    n_k = int(mp.n_k)
    n_sites = int(mp.n_sites)
    integrated_fs = np.asarray(mp.integrated, dtype=np.float32).reshape(n_k, n_sites)
    bin_fs = stack_bins(list(mp.bin_patterns), n_k, n_sites)
    site_weights = site_weights_from_meta_cell(mp.metadata.get("cell", {}))
    if site_weights is None:
        site_weights = np.full(n_sites, 1.0 / max(n_sites, 1), dtype=np.float32)
    fundamental_sector, fs_axes = _consolidate_fundamental_sector(
        integrated_fs, bin_fs, site_weights
    )

    kij = np.asarray(mp.kij, dtype=np.int32).reshape(n_k, 3)
    khat = np.asarray(mp.khat, dtype=np.float32).reshape(n_k, 3)
    cell_meta = mp.metadata.get("cell") or {}
    space_group = mp.metadata.get("space_group", cell_meta.get("space_group"))
    symbol = resolve_oriented_symbol(
        int(mp.pg_num),
        pg_symbol=mp.pg_symbol or mp.metadata.get("pg_symbol"),
        space_group=int(space_group) if space_group is not None else None,
    )
    ops = point_group_operators(symbol).reshape(-1, 3, 3).astype(np.float64)
    normals = fs_normals(symbol).reshape(-1, 3).astype(np.float64)

    bin_voltages_kv = np.asarray(mp.bin_voltages_kv, dtype=np.float32).reshape(-1)
    bin_weights = np.asarray(mp.bin_weights, dtype=np.float32).reshape(-1)

    meta = dict(mp.metadata)
    meta.setdefault("format", "ebsdsim-master-pattern")
    meta["format_version"] = _FORMAT_VERSION
    meta["n_bin_patterns"] = int(len(mp.bin_patterns))
    meta["array_layout"] = {
        "fundamental_sector": ["energy", "site", "direction"],
        "fundamental_kij": ["direction", "ij_sign"],
        "fundamental_khat": ["direction", "xyz"],
        "pg_operators": ["op", "row", "col"],
        "fs_normals": ["normal", "xyz"],
    }
    meta["fundamental_sector_axes"] = fs_axes
    meta_bytes = np.frombuffer(
        json.dumps(meta, indent=2, sort_keys=False).encode("utf-8"), dtype=np.uint8
    )

    np.savez_compressed(
        out_path,
        fundamental_sector=fundamental_sector,
        fundamental_kij=kij,
        fundamental_khat=khat,
        pg_operators=ops,
        fs_normals=normals,
        bin_voltages_kv=bin_voltages_kv,
        bin_weights=bin_weights,
        site_weights=np.asarray(site_weights, dtype=np.float32).reshape(-1),
        meta_json=meta_bytes,
    )
    return out_path
