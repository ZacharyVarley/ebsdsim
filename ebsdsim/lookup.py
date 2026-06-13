"""Structure-factor lookup tables and coupling scales."""

from __future__ import annotations

import multiprocessing
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ebsdsim.types import Cell, MasterPatternMode
from ebsdsim.wk import scatter_factor_wk_array

Vec3 = tuple[float, float, float]
PI = np.pi

# On macOS/Linux, force "fork" so workers don't re-import the caller's
# __main__ module, which would fail when the library is used from a
# plain script
if sys.platform == "win32":
    _MP_CONTEXT = None
else:
    _MP_CONTEXT = multiprocessing.get_context("fork")


@dataclass
class DiffLookupData:
    hkl: NDArray[np.int32]
    hkl_hash: NDArray[np.int32]
    hkl_diff: NDArray[np.int32]
    diff_table: NDArray[np.float32]
    coupling: NDArray[np.float32]
    reflection_dbdiff: NDArray[np.uint32]
    dbdiff_table: NDArray[np.uint32]
    sgh_tables: NDArray[np.float32]
    table_size: int
    offset: int
    stride_h: int
    stride_k: int
    hmax: int
    kmax: int
    lmax: int
    prefactor: tuple[float, float]
    mlambda: float
    diag_imag: float
    n_dbdiff: int


@dataclass
class BuildLookupOptions:
    voltage_kv: float
    dmin: float
    mode: MasterPatternMode = "bloch"
    absorption: bool = True
    dbdiff_threshold: float = 1e-10
    beam_chunk: int = 512
    on_progress: Callable[[str], None] | None = None


def _metric_dot(h: Vec3, m: NDArray[np.float64]) -> float:
    x = m[0] * h[0] + m[1] * h[1] + m[2] * h[2]
    y = m[3] * h[0] + m[4] * h[1] + m[5] * h[2]
    z = m[6] * h[0] + m[7] * h[1] + m[8] * h[2]
    return h[0] * x + h[1] * y + h[2] * z


def reciprocal_length(h: Vec3, cell: Cell) -> float:
    return float(np.sqrt(max(_metric_dot(h, cell.reciprocal_metric), 0.0)))


def reciprocal_length_squared(h: Vec3, cell: Cell) -> float:
    return max(_metric_dot(h, cell.reciprocal_metric), 0.0)


def reciprocal_length_squared_batch(hkl: NDArray[np.int32], metric: NDArray[np.float64]) -> NDArray[np.float64]:
    h = hkl.reshape(-1, 3).astype(np.float64, copy=False)
    x = metric[0] * h[:, 0] + metric[1] * h[:, 1] + metric[2] * h[:, 2]
    y = metric[3] * h[:, 0] + metric[4] * h[:, 1] + metric[5] * h[:, 2]
    z = metric[6] * h[:, 0] + metric[7] * h[:, 1] + metric[8] * h[:, 2]
    return np.maximum(h[:, 0] * x + h[:, 1] * y + h[:, 2] * z, 0.0)


def reciprocal_length_batch(hkl: NDArray[np.int32], metric: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(reciprocal_length_squared_batch(hkl, metric))


def _first_below_dmin(axis: int, cell: Cell, dmin: float) -> int:
    for n in range(1, 100):
        h: list[int] = [0, 0, 0]
        h[axis] = n
        length = reciprocal_length((h[0], h[1], h[2]), cell)
        if length > 0 and 1 / length < dmin:
            return n - 1
    return 99


def hkl_limits(cell: Cell, dmin: float) -> tuple[int, int, int]:
    return (
        _first_below_dmin(0, cell, dmin),
        _first_below_dmin(1, cell, dmin),
        _first_below_dmin(2, cell, dmin),
    )


def is_reflection_allowed(h: int, k: int, l: int, cell: Cell) -> bool:
    centering = cell.lattice_centering
    if centering == "P":
        return True
    if centering == "F":
        hp = ((h % 2) + 2) % 2
        kp = ((k % 2) + 2) % 2
        lp = ((l % 2) + 2) % 2
        return hp == kp == lp
    if centering == "I":
        return ((h + k + l) & 1) == 0
    if centering == "A":
        return ((k + l) & 1) == 0
    if centering == "B":
        return ((h + l) & 1) == 0
    if centering == "C":
        return ((h + k) & 1) == 0
    if centering == "R":
        return (((-h + k + l) % 3) + 3) % 3 == 0
    return True


def generate_reflections(cell: Cell, dmin: float, difference_table: bool) -> NDArray[np.int32]:
    mh, mk, ml = hkl_limits(cell, dmin)
    hr = 2 * mh if difference_table else mh
    kr = 2 * mk if difference_table else mk
    lr = 2 * ml if difference_table else ml
    rows: list[int] = []
    for h in range(-hr, hr + 1):
        for k in range(-kr, kr + 1):
            for l in range(-lr, lr + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if not is_reflection_allowed(h, k, l, cell):
                    continue
                if not difference_table:
                    length = reciprocal_length((h, k, l), cell)
                    if length <= 0 or 1 / length <= dmin:
                        continue
                rows.extend([h, k, l])
    return np.array(rows, dtype=np.int32)


def _cx_add(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] + b[0], a[1] + b[1]


def _cx_abs(a: tuple[float, float]) -> float:
    return float(np.hypot(a[0], a[1]))


def _prepend_zero(hkl: NDArray[np.int32]) -> NDArray[np.int32]:
    out = np.zeros(hkl.size + 3, dtype=np.int32)
    out[3:] = hkl
    return out


@dataclass
class DiffLookupGeometry:
    """Voltage-invariant diff-lookup tables and per-row phase/site data."""

    hkl: NDArray[np.int32]
    hkl_diff: NDArray[np.int32]
    hkl_hash: NDArray[np.int32]
    table_size: int
    offset: int
    stride_h: int
    stride_k: int
    hmax: int
    kmax: int
    lmax: int
    pref: float
    pre: float
    row_g_sq: NDArray[np.float64]
    row_table_hash: NDArray[np.int32]
    row_phase_re: NDArray[np.float64]
    row_phase_im: NDArray[np.float64]
    site_occ: NDArray[np.float64]
    site_species: NDArray[np.int32]
    species_z: list[int]
    species_thermal_sigma: list[float]


def _stack_site_positions(cell: Cell) -> tuple[NDArray[np.float64], int]:
    n_sites = cell.atom_types.size
    max_pos = max((len(cell.positions[s]) for s in range(n_sites)), default=0)
    pos = np.zeros((n_sites, max(max_pos, 1), 3), dtype=np.float64)
    for site in range(n_sites):
        for p, xyz in enumerate(cell.positions[site]):
            pos[site, p] = xyz
    return pos, max_pos


def prepare_diff_lookup_geometry(cell: Cell, dmin: float) -> DiffLookupGeometry:
    """Precompute reflection geometry and structure-factor phases (once per cell)."""
    hmax, kmax, lmax = hkl_limits(cell, dmin)
    stride_k = 4 * lmax + 1
    stride_h = (4 * kmax + 1) * stride_k
    table_size = (4 * hmax + 1) * stride_h
    offset = 2 * hmax * stride_h + 2 * kmax * stride_k + 2 * lmax
    hkl = _prepend_zero(generate_reflections(cell, dmin, False))
    hkl_diff = _prepend_zero(generate_reflections(cell, dmin, True))
    pref = 0.04787801 / cell.volume
    pre = pref * 0.664840340614319

    n_diff = hkl_diff.size // 3
    n_sites = cell.atom_types.size
    site_occ = cell.atom_data[:, 3]
    site_b_iso = cell.atom_data[:, 4]
    site_thermal_sigma = np.maximum(1e-7, 10 * np.sqrt(np.maximum(site_b_iso, 0) / (8 * PI * PI)))
    site_pos, _ = _stack_site_positions(cell)
    species_index_by_key: dict[str, int] = {}
    site_species = np.zeros(n_sites, dtype=np.int32)
    species_z: list[int] = []
    species_thermal_sigma: list[float] = []

    for site in range(n_sites):
        z = int(cell.atom_types[site])
        thermal_sigma = float(site_thermal_sigma[site])
        species_key = f"{z}:{thermal_sigma:.10g}"
        sp = species_index_by_key.get(species_key)
        if sp is None:
            sp = len(species_z)
            species_index_by_key[species_key] = sp
            species_z.append(z)
            species_thermal_sigma.append(thermal_sigma)
        site_species[site] = sp

    h_diff = hkl_diff.reshape(-1, 3).astype(np.float64, copy=False)
    row_g_sq = reciprocal_length_squared_batch(hkl_diff, cell.reciprocal_metric)
    row_table_hash = (
        (h_diff[:, 0] + 2 * hmax) * stride_h
        + (h_diff[:, 1] + 2 * kmax) * stride_k
        + (h_diff[:, 2] + 2 * lmax)
    ).astype(np.int32)
    ang = -2 * PI * (
        h_diff[:, None, None, 0] * site_pos[None, :, :, 0]
        + h_diff[:, None, None, 1] * site_pos[None, :, :, 1]
        + h_diff[:, None, None, 2] * site_pos[None, :, :, 2]
    )
    row_phase_re = np.cos(ang).sum(axis=2)
    row_phase_im = np.sin(ang).sum(axis=2)

    h_refl = hkl.reshape(-1, 3)
    hkl_hash = (
        h_refl[:, 0] * stride_h + h_refl[:, 1] * stride_k + h_refl[:, 2]
    ).astype(np.int32)

    return DiffLookupGeometry(
        hkl=hkl,
        hkl_diff=hkl_diff,
        hkl_hash=hkl_hash,
        table_size=table_size,
        offset=offset,
        stride_h=stride_h,
        stride_k=stride_k,
        hmax=hmax,
        kmax=kmax,
        lmax=lmax,
        pref=pref,
        pre=pre,
        row_g_sq=row_g_sq,
        row_table_hash=row_table_hash,
        row_phase_re=row_phase_re,
        row_phase_im=row_phase_im,
        site_occ=site_occ,
        site_species=site_species,
        species_z=species_z,
        species_thermal_sigma=species_thermal_sigma,
    )


def _wk_scatter_factors_per_row(
    g_sq: NDArray[np.float64],
    species_z: list[int],
    species_thermal_sigma: list[float],
    voltage_kv: float,
    *,
    absorption: bool,
    wk_opts: dict[str, bool],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate WK scattering factors once per unique |g|^2, then map to every diff row."""
    unique_g_sq, inverse = np.unique(g_sq, return_inverse=True)
    g_unique = 0.1 * 2 * PI * np.sqrt(np.maximum(unique_g_sq, 0.0))
    n_species = len(species_z)
    n_diff = int(g_sq.size)
    sf_re = np.empty((n_species, n_diff), dtype=np.float64)
    sf_im = np.empty((n_species, n_diff), dtype=np.float64)
    for sp in range(n_species):
        re_u, im_u = scatter_factor_wk_array(
            g_unique,
            species_z[sp],
            species_thermal_sigma[sp],
            voltage_kv,
            wk_opts,
        )
        sf_re[sp] = re_u[inverse]
        sf_im[sp] = im_u[inverse] if absorption else 0.0
    return sf_re, sf_im


def build_diff_lookup_from_geometry(
    geometry: DiffLookupGeometry,
    opts: BuildLookupOptions,
) -> DiffLookupData:
    """Build a voltage-dependent diff lookup from cached geometry."""
    if opts.voltage_kv <= 0:
        raise ValueError("voltage_kv must be positive")
    mode = opts.mode
    absorption = opts.absorption
    dbdiff_threshold = opts.dbdiff_threshold
    geom = geometry
    table = np.zeros(geom.table_size * 2, dtype=np.float32)
    dbdiff_table = np.zeros(geom.table_size, dtype=np.uint32)
    n_diff = geom.row_g_sq.size
    n_sites = geom.site_occ.size
    voltage_kv = opts.voltage_kv
    wk_opts = {"include_core": True, "include_phonon": True}
    opts.on_progress and opts.on_progress(
        f"DiffLookup: {geom.hkl_diff.size // 3} difference reflections, {n_sites} sites, {len(geom.species_z)} species"
    )

    sf_re, sf_im = _wk_scatter_factors_per_row(
        geom.row_g_sq,
        geom.species_z,
        geom.species_thermal_sigma,
        voltage_kv,
        absorption=absorption,
        wk_opts=wk_opts,
    )

    site_sf_re = sf_re[geom.site_species].T
    site_sf_im = sf_im[geom.site_species].T
    occ = np.where(geom.site_occ > 0, geom.site_occ, 1.0)
    pr = geom.row_phase_re
    pi = geom.row_phase_im
    # Match the legacy site loop: f += occ*phase*sf with (f_im, gp) using sf_re/sf_im cross-terms.
    f_re = np.sum(occ[None, :] * pr * site_sf_re, axis=1)
    f_im = np.sum(occ[None, :] * pi * site_sf_re, axis=1)
    gp_re = np.sum(occ[None, :] * pr * site_sf_im, axis=1)
    gp_im = np.sum(occ[None, :] * pi * site_sf_im, axis=1)

    vmod0 = geom.pref * float(np.hypot(f_re[0], f_im[0]))
    temp2 = 0.00097847561649 * opts.voltage_kv
    psi_hat = vmod0 + opts.voltage_kv * (1 + temp2) * 1000
    mlambda = 1.22642584554 / np.sqrt(psi_hat)
    preg = 0.664840340614319

    if absorption:
        ucg_re = geom.pre * (f_re - gp_im)
        ucg_im = geom.pre * (f_im + gp_re)
    else:
        ucg_re = geom.pre * f_re
        ucg_im = geom.pre * f_im

    ucg_abs = np.hypot(ucg_re, ucg_im)
    is_dbdiff = (ucg_abs <= dbdiff_threshold) & (np.arange(n_diff) != 0)
    n_dbdiff = int(np.count_nonzero(is_dbdiff))

    if mode == "structure":
        vmod = geom.pref * np.hypot(f_re, f_im)
        umod = preg * vmod
        upmod = preg * geom.pref * np.hypot(gp_re, gp_im) if absorption else np.zeros(n_diff)
        vphase = np.arctan2(f_im, f_re)
        vpphase = np.arctan2(gp_im, gp_re) if absorption else np.zeros(n_diff)
        xg = np.where(umod > 0, 1 / (umod * mlambda), 1e8)
        xgp = np.where(upmod > 0, 1 / (upmod * mlambda), 1e8)
        if absorption:
            diff_re = np.cos(vphase) / xg - np.sin(vpphase) / xgp
            diff_im = np.cos(vpphase) / xgp + np.sin(vphase) / xg
        else:
            diff_re = 1 / xg
            diff_im = np.zeros(n_diff)
        diag_imag = float(1 / (upmod[0] * mlambda) if upmod[0] > 0 else 1e8)
    else:
        diff_re = ucg_re
        diff_im = ucg_im
        diag_imag = float(preg * geom.pref * np.hypot(gp_re[0], gp_im[0])) if absorption else 0.0

    hash_idx = geom.row_table_hash
    table[hash_idx * 2] = diff_re.astype(np.float32)
    table[hash_idx * 2 + 1] = diff_im.astype(np.float32)
    dbdiff_table[hash_idx] = is_dbdiff.astype(np.uint32)
    table[geom.offset * 2 : geom.offset * 2 + 2] = 0.0

    reflection_dbdiff = dbdiff_table[geom.hkl_hash + geom.offset].astype(np.uint32)
    reflection_dbdiff[0] = 0

    coupling = coupling_scale_from_diff_candidates(
        geom.hkl_hash,
        geom.hkl_diff,
        table,
        dbdiff_table,
        geom.offset,
        geom.stride_h,
        geom.stride_k,
        mlambda,
    )

    return DiffLookupData(
        hkl=geom.hkl,
        hkl_hash=geom.hkl_hash,
        hkl_diff=geom.hkl_diff,
        diff_table=table,
        coupling=coupling,
        reflection_dbdiff=reflection_dbdiff,
        dbdiff_table=dbdiff_table,
        sgh_tables=np.zeros(0, dtype=np.float32),
        table_size=geom.table_size,
        offset=geom.offset,
        stride_h=geom.stride_h,
        stride_k=geom.stride_k,
        hmax=geom.hmax,
        kmax=geom.kmax,
        lmax=geom.lmax,
        prefactor=(0.0, PI) if mode == "structure" else (0.0, PI * mlambda),
        mlambda=float(mlambda),
        diag_imag=float(diag_imag),
        n_dbdiff=n_dbdiff,
    )


@dataclass
class LookupCache:
    """Pre-built diff lookups keyed by voltage (kV)."""

    by_voltage_kv: dict[float, DiffLookupData]

    def get(self, voltage_kv: float) -> DiffLookupData:
        key = float(voltage_kv)
        try:
            return self.by_voltage_kv[key]
        except KeyError as exc:
            raise KeyError(f"no diff lookup cached for {key:.6f} kV") from exc


def _build_lookup_entry(args: tuple[DiffLookupGeometry, float, float, MasterPatternMode]) -> tuple[float, DiffLookupData]:
    geometry, voltage_kv, dmin, mode = args
    opts = BuildLookupOptions(
        voltage_kv=voltage_kv,
        dmin=dmin,
        mode=mode,
        absorption=True,
        dbdiff_threshold=1e-10,
    )
    return voltage_kv, build_diff_lookup_from_geometry(geometry, opts)


def build_lookup_cache(
    geometry: DiffLookupGeometry,
    voltages_kv: NDArray[np.floating],
    dmin: float,
    mode: MasterPatternMode,
    *,
    parallel: bool = True,
) -> LookupCache:
    """Precompute every voltage's diff lookup (used by tests and offline tooling)."""
    unique = sorted({float(v) for v in np.asarray(voltages_kv).reshape(-1) if float(v) > 0})
    if not unique:
        return LookupCache({})
    tasks = [(geometry, vkv, dmin, mode) for vkv in unique]
    if parallel and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(len(tasks), 8), mp_context=_MP_CONTEXT) as pool:
            pairs = list(pool.map(_build_lookup_entry, tasks))
    else:
        pairs = [_build_lookup_entry(task) for task in tasks]
    return LookupCache(dict(pairs))


@dataclass
class LookupPrefetcher:
    """Build the next voltage's diff lookup on CPU while GPU runs the current bin."""

    geometry: DiffLookupGeometry
    dmin: float
    mode: MasterPatternMode
    _pool: ProcessPoolExecutor | None = field(default=None, init=False, repr=False)
    _future: Future | None = field(default=None, init=False, repr=False)
    _future_vkv: float | None = field(default=None, init=False, repr=False)

    def _pool_or_create(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=_MP_CONTEXT)
        return self._pool

    def prefetch(self, voltage_kv: float) -> None:
        key = float(voltage_kv)
        if key <= 0:
            return
        if self._future is not None:
            if self._future_vkv == key:
                return
            if not self._future.done():
                return
            self._future.result()
            self._future = None
            self._future_vkv = None
        task = (self.geometry, key, self.dmin, self.mode)
        self._future = self._pool_or_create().submit(_build_lookup_entry, task)
        self._future_vkv = key

    def get(self, voltage_kv: float) -> DiffLookupData:
        key = float(voltage_kv)
        if self._future is not None and self._future_vkv == key:
            _, data = self._future.result()
            self._future = None
            self._future_vkv = None
            return data
        if self._future is not None and self._future.done():
            self._future.result()
            self._future = None
            self._future_vkv = None
        return _build_lookup_entry((self.geometry, key, self.dmin, self.mode))[1]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._future = None
        self._future_vkv = None


def build_diff_lookup(cell: Cell, opts: BuildLookupOptions) -> DiffLookupData:
    geometry = prepare_diff_lookup_geometry(cell, opts.dmin)
    return build_diff_lookup_from_geometry(geometry, opts)


def coupling_scale_streaming(
    hkl_hash: NDArray[np.int32],
    table: NDArray[np.float32],
    dbdiff_table: NDArray[np.uint32],
    offset: int,
    mlambda: float,
    _beam_chunk: int,
) -> NDArray[np.float32]:
    n = hkl_hash.size
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        hi = int(hkl_hash[i])
        best_sq = 0.0
        for j in range(n):
            dh = hi - int(hkl_hash[j]) + offset
            if dbdiff_table[dh]:
                continue
            re = table[dh * 2]
            im = table[dh * 2 + 1]
            sq = re * re + im * im
            if sq > best_sq:
                best_sq = sq
        best = np.sqrt(best_sq)
        out[i] = mlambda * best if mlambda * best > 1e-12 else 1e-12
    return out


def coupling_scale_from_diff_candidates(
    hkl_hash: NDArray[np.int32],
    hkl_diff: NDArray[np.int32],
    table: NDArray[np.float32],
    dbdiff_table: NDArray[np.uint32],
    offset: int,
    stride_h: int,
    stride_k: int,
    mlambda: float,
) -> NDArray[np.float32]:
    reflection_set = set(int(hkl_hash[i]) for i in range(hkl_hash.size))
    candidates: list[tuple[int, float]] = []
    for i in range(0, hkl_diff.size, 3):
        hash_idx = int(hkl_diff[i] * stride_h + hkl_diff[i + 1] * stride_k + hkl_diff[i + 2])
        table_index = hash_idx + offset
        if table_index < 0 or table_index >= dbdiff_table.size or dbdiff_table[table_index]:
            continue
        magnitude = float(np.hypot(table[table_index * 2], table[table_index * 2 + 1]))
        if magnitude > 0:
            candidates.append((hash_idx, magnitude))
    candidates.sort(key=lambda x: x[1], reverse=True)

    out = np.zeros(hkl_hash.size, dtype=np.float32)
    for i in range(hkl_hash.size):
        hi = int(hkl_hash[i])
        best = 0.0
        for hash_idx, magnitude in candidates:
            if magnitude <= best:
                break
            if hi - hash_idx in reflection_set:
                best = magnitude
                break
        out[i] = max(mlambda * best, 1e-12)
    return out
