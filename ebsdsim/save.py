"""Compressed ``.npz`` export of ebsdsim master patterns.

The on-disk format stores the **fundamental-sector** intensities only (the
minimal, symmetry-reduced representation), always keeps the per-energy-bin
intermediates, and is always compressed. The embedded point-group operators
and fundamental-sector normals make the file self-describing: it can be
expanded back into full Lambert hemispheres with NumPy alone (see
:mod:`ebsdsim.mploader`), without installing ebsdsim.

Arrays written
--------------
``fundamental_sector`` : float32 ``(n_energy, n_site, n_k)``
    The symmetry-reduced intensities for every direction, packed along an
    energy axis and a site axis (raw, un-normalized). The energy axis is
    ``1 + n_bins`` long when ``n_bins > 1`` (index 0 is the energy-integrated
    weighted sum, index ``1 + b`` is bin ``b``) and length 1 otherwise. The
    site axis is ``1 + n_sites`` long when ``n_sites > 1`` (index 0 is the
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
``bin_voltages_kv`` / ``bin_weights`` : float32 ``(n_bins,)``
    Beam voltage and energy weight of each saved bin.
``meta_json`` : uint8 ``(n_bytes,)``
    UTF-8 JSON metadata blob (decode with ``bytes(arr).decode("utf-8")``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ebsdsim.elements import element_symbol
from ebsdsim.pg_ops import fs_normals, pg_num_to_symbol, point_group_operators
from ebsdsim.types import Cell

if TYPE_CHECKING:  # pragma: no cover
    from ebsdsim.api import MasterPattern

_NM_TO_ANGSTROM = 10.0
_NM_SQ_TO_ANGSTROM_SQ = 100.0


def cell_metadata(cell: Cell) -> dict[str, Any]:
    """Serialize a :class:`~ebsdsim.types.Cell` to a JSON-friendly dict.

    Per-site isotropic Debye–Waller factors are recorded in both Å² and nm²
    because they are commonly estimated and materially affect the pattern.
    """
    atom_types = np.asarray(cell.atom_types).reshape(-1)
    n_sites = int(atom_types.size)
    atom_data = np.asarray(cell.atom_data, dtype=np.float64).reshape(n_sites, 5)
    mult = np.asarray(cell.multiplicities).reshape(-1)
    sites: list[dict[str, Any]] = []
    for i in range(n_sites):
        z = int(atom_types[i])
        b_iso_nm2 = float(atom_data[i, 4])
        sites.append(
            {
                "index": i,
                "atomic_number": z,
                "symbol": element_symbol(z),
                "fract": [float(atom_data[i, 0]), float(atom_data[i, 1]), float(atom_data[i, 2])],
                "occupancy": float(atom_data[i, 3]),
                "b_iso_angstrom_sq": b_iso_nm2 * _NM_SQ_TO_ANGSTROM_SQ,
                "b_iso_nm_sq": b_iso_nm2,
                "multiplicity": int(mult[i]) if i < mult.size else None,
            }
        )
    return {
        "a_angstrom": float(cell.a) * _NM_TO_ANGSTROM,
        "b_angstrom": float(cell.b) * _NM_TO_ANGSTROM,
        "c_angstrom": float(cell.c) * _NM_TO_ANGSTROM,
        "alpha_deg": float(cell.alpha),
        "beta_deg": float(cell.beta),
        "gamma_deg": float(cell.gamma),
        "volume_angstrom3": float(cell.volume) * (_NM_TO_ANGSTROM**3),
        "density_g_cm3": float(cell.density),
        "lattice_centering": cell.lattice_centering,
        "space_group": cell.space_group,
        "pg_num": cell.pg_num,
        "pg_symbol": pg_num_to_symbol(int(cell.pg_num)) if cell.pg_num else None,
        "average_atomic_number": float(cell.average_atomic_number),
        "average_atomic_weight": float(cell.average_atomic_weight),
        "n_sites": n_sites,
        "sites": sites,
    }


def _stack_bins(bin_patterns: list[np.ndarray], n_k: int, n_sites: int) -> np.ndarray:
    if not bin_patterns:
        return np.zeros((0, n_k, n_sites), dtype=np.float32)
    out = np.empty((len(bin_patterns), n_k, n_sites), dtype=np.float32)
    for b, pat in enumerate(bin_patterns):
        out[b] = np.asarray(pat, dtype=np.float32).reshape(n_k, n_sites)
    return out


def _consolidate_fundamental_sector(
    integrated_fs: np.ndarray,  # (n_k, n_sites)
    bin_fs: np.ndarray,  # (n_bins, n_k, n_sites)
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pack integrated + per-bin and mean + per-site into one ``(E, S, n_k)`` array.

    Mirrors the in-memory ``MasterPattern.data`` axis convention (minus the
    hemisphere and Lambert image axes): energy index 0 is the energy-integrated
    weighted sum, ``1 + b`` is bin ``b`` (a distinct slice only when
    ``n_bins > 1``); site index 0 is the site mean, ``1 + s`` is site ``s`` (a
    distinct slice only when ``n_sites > 1``). Values are raw (un-normalized)
    so the loader can reproduce the display normalization.
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
            out[e_idx, 0] = src.mean(axis=1, dtype=np.float32)
            for s in range(n_sites):
                out[e_idx, 1 + s] = src[:, s]
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


def save_master_pattern(mp: "MasterPattern", path: str | Path) -> Path:
    """Write a master pattern (with intermediates) to a compressed ``.npz``.

    Returns the resolved output path. The ``.npz`` suffix is added if missing.
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

    n_k = int(mp.n_k)
    n_sites = int(mp.n_sites)
    integrated_fs = np.asarray(mp.integrated, dtype=np.float32).reshape(n_k, n_sites)
    bin_fs = _stack_bins(list(mp.bin_patterns), n_k, n_sites)
    fundamental_sector, fs_axes = _consolidate_fundamental_sector(integrated_fs, bin_fs)

    kij = np.asarray(mp.kij, dtype=np.int32).reshape(n_k, 3)
    khat = np.asarray(mp.khat, dtype=np.float32).reshape(n_k, 3)
    symbol = pg_num_to_symbol(int(mp.pg_num))
    ops = point_group_operators(symbol).reshape(-1, 3, 3).astype(np.float64)
    normals = fs_normals(symbol).reshape(-1, 3).astype(np.float64)

    bin_voltages_kv = np.asarray(mp.bin_voltages_kv, dtype=np.float32).reshape(-1)
    bin_weights = np.asarray(mp.bin_weights, dtype=np.float32).reshape(-1)

    meta = dict(mp.metadata)
    meta.setdefault("format", "ebsdsim-master-pattern")
    meta["format_version"] = 2
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
        meta_json=meta_bytes,
    )
    return out_path
