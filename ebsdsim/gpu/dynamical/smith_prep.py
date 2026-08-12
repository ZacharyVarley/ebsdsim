"""Multi-voltage prep: do shared work once; only swap voltage-dependent tables.

Mirrors production LookupPrefetcher + ResidentTables:
  once:  cell, geometry, SGH, PG grid, kernels, metric, Bl, Sgh transpose
  /bin:  diff_table (+coupling/dbdiff), kvecs, mu, amp*w, Bethe prescan

Never rebuild geometry / SGH / device between bins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu import device as gpu_device_mod
from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.gpu.device import require_gpu
from ebsdsim.gpu.dynamical import EBSDDynamicalKernels
from ebsdsim.gpu.dynamical import smith_repack as zr
from ebsdsim.gpu.pipelines import PipelineCache, load_wgsl
from ebsdsim.gpu.resident import ResidentTables, make_metric_buffer
from ebsdsim.lambert.kgrid import build_pg_k_grid, transform_pg_k_grid_to_reciprocal
from ebsdsim.physics.lookup import LookupPrefetcher, prepare_diff_lookup_geometry
from ebsdsim.physics.prescan import compute_mu_eff
from ebsdsim.physics.site_tables import prepare_site_sgh_tables


@dataclass
class VoltageBin:
    voltage_kv: float
    next_voltage_kv: float | None
    beta: float
    amplitude: float
    energy_weight: float
    bin_index: int


@dataclass
class SmithCrystalPrep:
    """Voltage-independent state for the smith hotpath."""

    label: str
    cell: object
    lookup_geometry: object
    prefetcher: LookupPrefetcher
    sgh_tables_site_major: NDArray[np.float32]
    sgh_tables_delta_major: StorageBuffer
    n_sites: int
    kernels: EBSDDynamicalKernels
    device: object
    queue: object
    metric: StorageBuffer
    pipelines: PipelineCache
    resident: ResidentTables
    bl: NDArray[np.float32]
    pg: object
    dmin: float
    bethe_c_strong: float
    bethe_c_weak: float
    bethe_c_cutoff: float
    dbdiff_sg_cutoff: float
    kvec_cache: dict[float, NDArray[np.float32]] = field(default_factory=dict)
    prep_timings: dict[str, float] = field(default_factory=dict)
    _current_vkv: float | None = None
    _current_lookup: object | None = None

    def close(self) -> None:
        self.prefetcher.close()
        self.resident.destroy()
        self.metric.destroy()
        self.sgh_tables_delta_major.destroy()
        self.pipelines.clear()


def _crystal_bl_from_hkl(hkl: NDArray[np.int32]) -> NDArray[np.float32]:
    """Cheap Bl: HNF of reflection differences (no GPU extract_k_system)."""
    H = np.asarray(hkl, dtype=np.int64).reshape(-1, 3)
    if H.shape[0] < 2:
        return np.eye(3, dtype=np.float32)
    return zr.delta_lattice_basis(H).astype(np.float32)


def _transpose_sgh(
    pipelines: PipelineCache,
    queue,
    device,
    site_major: NDArray[np.float32],
    *,
    table_size: int,
    n_sites: int,
) -> StorageBuffer:
    import struct

    src = StorageBuffer(device, queue, label="sgh:src", data=site_major)
    dst = StorageBuffer(
        device,
        queue,
        label="sgh:delta_major",
        byte_length=site_major.nbytes,
        copy_src=False,
        copy_dst=True,
    )
    code = load_wgsl("dynamical/transpose_sgh_delta_major.wgsl")
    total = table_size * n_sites
    pipelines.dispatch_with_params(
        queue,
        "sgh:transpose",
        code,
        struct.pack("<4I", table_size, n_sites, 0, 0),
        [src, dst],
        ((total + 255) // 256, 1, 1),
        label="sgh:transpose",
        n_storage_bindings=2,
        uniform_size=16,
    )
    src.destroy()
    return dst


def open_smith_prep(
    label: str,
    *,
    halfw: int,
    voltage_kv: float,
    dmin: float,
    bins: list[VoltageBin],
    bethe_c_strong: float = 20.0,
    bethe_c_weak: float = 40.0,
    bethe_c_cutoff: float = 200.0,
    dbdiff_sg_cutoff: float = 1.0,
    cell: object | None = None,
    reuse_kernels: EBSDDynamicalKernels | None = None,
) -> tuple[SmithCrystalPrep, list[VoltageBin]]:
    """Open shared smith prep. ``bins`` is required (caller supplies energy model)."""
    import time

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if cell is None:
        raise ValueError("open_smith_prep requires cell= (SimCell)")
    timings["cell"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if reuse_kernels is not None:
        ctx_kernels = reuse_kernels
        feat_device, feat_queue = ctx_kernels.device, ctx_kernels.queue
    else:
        feat = require_gpu(required_features=("shader-f16",))
        gpu_device_mod._context = feat
        ctx_kernels = EBSDDynamicalKernels(feat.device, feat.queue)
        feat_device, feat_queue = feat.device, feat.queue
    timings["device_kernels"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    geom = prepare_diff_lookup_geometry(cell, dmin)
    timings["lookup_geometry"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sgh = prepare_site_sgh_tables(cell, dmin)
    timings["sgh"] = time.perf_counter() - t0

    if not bins:
        raise RuntimeError("no active voltage bins")

    t0 = time.perf_counter()
    prefetcher = LookupPrefetcher(geom, dmin, "bloch")
    prefetcher.prefetch(bins[0].voltage_kv)
    first_lookup = prefetcher.get(bins[0].voltage_kv)
    if bins[0].next_voltage_kv is not None:
        prefetcher.prefetch(bins[0].next_voltage_kv)
    timings["first_lookup"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    resident = ResidentTables.create(ctx_kernels, first_lookup, sgh.tables)
    metric = make_metric_buffer(ctx_kernels, cell)
    pipelines = PipelineCache(feat_device)
    timings["persistent_metric"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sgh_dm = _transpose_sgh(
        pipelines,
        feat_queue,
        feat_device,
        sgh.tables,
        table_size=int(first_lookup.table_size),
        n_sites=int(sgh.n_sites),
    )
    timings["sgh_transpose"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pg = build_pg_k_grid(cell.pg_num, halfw, cell.space_group)
    bl = _crystal_bl_from_hkl(np.asarray(first_lookup.hkl, dtype=np.int32))
    timings["pg_bl"] = time.perf_counter() - t0

    _ = voltage_kv  # signature stability; bins already carry voltages

    prep = SmithCrystalPrep(
        label=label,
        cell=cell,
        lookup_geometry=geom,
        prefetcher=prefetcher,
        sgh_tables_site_major=sgh.tables,
        sgh_tables_delta_major=sgh_dm,
        n_sites=int(sgh.n_sites),
        kernels=ctx_kernels,
        device=feat_device,
        queue=feat_queue,
        metric=metric,
        pipelines=pipelines,
        resident=resident,
        bl=bl,
        pg=pg,
        dmin=dmin,
        bethe_c_strong=bethe_c_strong,
        bethe_c_weak=bethe_c_weak,
        bethe_c_cutoff=bethe_c_cutoff,
        dbdiff_sg_cutoff=dbdiff_sg_cutoff,
        prep_timings=timings,
        _current_vkv=float(bins[0].voltage_kv),
        _current_lookup=first_lookup,
    )
    return prep, bins


def activate_voltage_bin(prep: SmithCrystalPrep, bin: VoltageBin):
    """Bind lookup/resident/kvecs/mu for one bin; prefetch the next lookup on CPU."""
    import time

    reused = prep._current_vkv == float(bin.voltage_kv) and prep._current_lookup is not None
    t_update = 0.0
    if reused:
        t0 = time.perf_counter()
        lookup = prep._current_lookup
        t_lookup = 0.0
        _ = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        lookup = prep.prefetcher.get(bin.voltage_kv)
        prep._current_vkv = float(bin.voltage_kv)
        prep._current_lookup = lookup
        t_lookup = time.perf_counter() - t0
        t0 = time.perf_counter()
        prep.resident.update(lookup)
        t_update = time.perf_counter() - t0

    if bin.next_voltage_kv is not None:
        prep.prefetcher.prefetch(bin.next_voltage_kv)

    t0 = time.perf_counter()
    key = float(bin.voltage_kv)
    kvecs = prep.kvec_cache.get(key)
    if kvecs is None:
        kvecs = transform_pg_k_grid_to_reciprocal(
            prep.pg, prep.cell.direct_structure_matrix, lookup.mlambda
        ).astype(np.float32, copy=False)
        prep.kvec_cache[key] = kvecs
    t_k = time.perf_counter() - t0

    mu = compute_mu_eff(bin.beta, lookup.mlambda, lookup.diag_imag, "bloch")
    amp = float(bin.amplitude) * float(bin.energy_weight)

    t0 = time.perf_counter()
    n_g = lookup.hkl.size // 3
    n_k = kvecs.size // 3
    prescan = prep.kernels.prescan_bethe_beam_counts_gpu(
        kvecs,
        prep.resident.buffers,
        prep.metric,
        batch_count=n_k,
        n_g=n_g,
        bethe_c_cutoff=prep.bethe_c_cutoff,
        dbdiff_sg_cutoff=prep.dbdiff_sg_cutoff,
        bethe_c_strong=prep.bethe_c_strong,
        bethe_c_weak=prep.bethe_c_weak,
    )
    t_prescan = time.perf_counter() - t0

    return {
        "lookup": lookup,
        "kvecs": kvecs,
        "mu": float(mu),
        "amplitude": amp,
        "n_strong": max(1, int(prescan.n_strong)),
        "n_weak": max(0, int(prescan.n_weak)),
        "strong_needed": prescan.strong_needed,
        "weak_needed": prescan.weak_needed,
        "candidate_needed": prescan.candidate_needed,
        "n_g": n_g,
        "timings": {
            "lookup_get": t_lookup,
            "persistent_update": t_update,
            "kvecs": t_k,
            "prescan": t_prescan,
            "reused_lookup": reused,
        },
    }
