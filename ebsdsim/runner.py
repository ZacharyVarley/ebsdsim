"""Per-voltage dynamical solve orchestration for integrate.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.buffers import StorageBuffer, c64_bytes, f32_bytes, u32_bytes
from ebsdsim.gpu.limits import device_limit
from ebsdsim.gpu.dynamical import (
    EBSDDynamicalKernels,
    FixedRankChunkDescriptor,
    PersistentBuffers,
    RunChunk,
    _to_u32,
)
from ebsdsim.integrate import PerVoltageContext, PerVoltageResult, compute_mu_eff
from ebsdsim.kgrid import PgKGrid, chunk_k_vectors, transform_pg_k_grid_to_reciprocal
from ebsdsim.lookup import (
    BuildLookupOptions,
    DiffLookupData,
    DiffLookupGeometry,
    LookupCache,
    LookupPrefetcher,
    build_diff_lookup,
    build_diff_lookup_from_geometry,
)
from ebsdsim.sgh import SghTableData
from ebsdsim.structure import metric_to_float32
from ebsdsim.types import Cell, MasterPatternMode


def _estimate_workspace_buffer_sizes(
    batch_count: int,
    *,
    n_g: int,
    n_strong: int,
    n_weak: int,
    rank: int,
    n_sites: int,
) -> dict[str, int]:
    mat_a = batch_count * n_strong * n_strong
    mat_sw = batch_count * n_strong * n_weak
    return {
        "sg": f32_bytes(batch_count * n_g),
        "scores": f32_bytes(batch_count * n_g),
        "candidate_mask": u32_bytes(batch_count * n_g),
        "selected_flags": u32_bytes(batch_count * n_g),
        "idx_a": u32_bytes(batch_count * n_strong),
        "idx_w": u32_bytes(max(1, batch_count * n_weak)),
        "v_aa": c64_bytes(mat_a),
        "v_sw": c64_bytes(max(1, mat_sw)),
        "d_a": c64_bytes(batch_count * n_strong),
        "d_w": c64_bytes(max(1, batch_count * n_weak)),
        "weighted_vws": c64_bytes(max(1, batch_count * n_weak * n_strong)),
        "sigma": c64_bytes(mat_a),
        "geff": c64_bytes(mat_a),
        "q_values": f32_bytes(batch_count * 4),
        "q_i_minus_a": c64_bytes(mat_a),
        "q_i_plus_a": c64_bytes(mat_a),
        "e0": c64_bytes(batch_count * n_strong),
        "pivots": u32_bytes(batch_count * n_strong),
        "w_a": c64_bytes(batch_count * n_strong),
        "w_b": c64_bytes(batch_count * n_strong),
        "rhs": c64_bytes(batch_count * n_strong),
        "w_stack": c64_bytes(batch_count * n_strong * rank),
        "dh": u32_bytes(mat_a),
        "intensities": f32_bytes(batch_count * n_sites),
    }


def plan_safe_chunk_size(
    requested_chunk_size: int,
    device_limits: Any,
    *,
    n_g: int,
    n_strong: int,
    n_weak: int,
    rank: int,
    n_sites: int,
) -> int:
    """Reduce chunk size until workspace buffers fit device storage limits."""
    limit_bytes = min(
        int(device_limit(device_limits, "max_buffer_size", 256 * 1024 * 1024)),
        int(device_limit(device_limits, "max_storage_buffer_binding_size", 128 * 1024 * 1024)),
    )

    def fits(batch_count: int) -> bool:
        sizes = _estimate_workspace_buffer_sizes(
            batch_count,
            n_g=n_g,
            n_strong=n_strong,
            n_weak=n_weak,
            rank=rank,
            n_sites=n_sites,
        )
        return max(sizes.values()) <= limit_bytes

    if fits(requested_chunk_size):
        return requested_chunk_size

    low, high = 1, requested_chunk_size
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if fits(mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


@dataclass
class ReusablePersistent:
    """Reuse GPU persistent buffers; only voltage-dependent tables are updated per bin."""

    buffers: PersistentBuffers

    @classmethod
    def create(
        cls,
        kernels: EBSDDynamicalKernels,
        lookup: DiffLookupData,
        sgh_tables: NDArray[np.float32],
    ) -> ReusablePersistent:
        return cls(
            buffers=kernels.create_persistent_buffers(
                hkl=lookup.hkl,
                hkl_hash=lookup.hkl_hash,
                diff_table=lookup.diff_table,
                coupling=lookup.coupling,
                reflection_dbdiff=lookup.reflection_dbdiff,
                sgh_tables=sgh_tables,
            )
        )

    def update(self, lookup: DiffLookupData) -> None:
        self.buffers.diff_table.write(lookup.diff_table)
        self.buffers.coupling.write(lookup.coupling)
        self.buffers.reflection_dbdiff.write(_to_u32(lookup.reflection_dbdiff))

    def destroy(self) -> None:
        _destroy_persistent(self.buffers)


@dataclass
class RunOneVoltageDeps:
    cell: Cell
    pg_grid: PgKGrid
    sgh: SghTableData
    kernels: EBSDDynamicalKernels
    metric: StorageBuffer
    chunk_size: int
    rank: int
    dmin: float
    bethe_c_strong: float = 20.0
    bethe_c_weak: float = 40.0
    bethe_c_cutoff: float = 200.0
    dbdiff_sg_cutoff: float = 1.0
    working_chunk_size: int | None = None
    lookup_geometry: DiffLookupGeometry | None = None
    lookup_cache: LookupCache | None = None
    lookup_prefetcher: LookupPrefetcher | None = None
    reusable_persistent: ReusablePersistent | None = None
    kvec_cache: dict[float, NDArray[np.float32]] = field(default_factory=dict)


def _resolve_diff_lookup(ctx: PerVoltageContext, deps: RunOneVoltageDeps) -> DiffLookupData:
    if deps.lookup_prefetcher is not None:
        lookup = deps.lookup_prefetcher.get(ctx.voltage_kv)
        if ctx.next_voltage_kv is not None:
            deps.lookup_prefetcher.prefetch(ctx.next_voltage_kv)
        return lookup
    if deps.lookup_cache is not None:
        return deps.lookup_cache.get(ctx.voltage_kv)
    opts = BuildLookupOptions(
        voltage_kv=ctx.voltage_kv,
        dmin=deps.dmin,
        mode=ctx.mode,
        absorption=True,
        dbdiff_threshold=1e-10,
    )
    if deps.lookup_geometry is not None:
        return build_diff_lookup_from_geometry(deps.lookup_geometry, opts)
    return build_diff_lookup(deps.cell, opts)


def _resolve_persistent(
    deps: RunOneVoltageDeps,
    lookup: DiffLookupData,
) -> PersistentBuffers:
    if deps.reusable_persistent is None:
        deps.reusable_persistent = ReusablePersistent.create(deps.kernels, lookup, deps.sgh.tables)
    else:
        deps.reusable_persistent.update(lookup)
    return deps.reusable_persistent.buffers


def _resolve_kvecs(
    deps: RunOneVoltageDeps,
    voltage_kv: float,
    mlambda: float,
) -> NDArray[np.float32]:
    key = float(voltage_kv)
    cached = deps.kvec_cache.get(key)
    if cached is not None:
        return cached
    kvecs = transform_pg_k_grid_to_reciprocal(
        deps.pg_grid,
        deps.cell.direct_structure_matrix,
        mlambda,
    )
    deps.kvec_cache[key] = kvecs
    return kvecs


def run_one_voltage(ctx: PerVoltageContext, deps: RunOneVoltageDeps) -> PerVoltageResult | None:
    """Run one voltage-bin GPU dynamical solve and return flattened intensities."""
    w = float(ctx.energy_weight)
    a = float(ctx.amplitude)
    if w < 1e-12 or a < 1e-12 or ctx.voltage_kv <= 0:
        return None

    lookup = _resolve_diff_lookup(ctx, deps)
    n_g = lookup.hkl.size // 3
    n_sites = deps.sgh.n_sites
    n_k = deps.pg_grid.khat.size // 3
    mu_eff = compute_mu_eff(ctx.beta, lookup.mlambda, lookup.diag_imag, ctx.mode)

    kvecs = _resolve_kvecs(deps, ctx.voltage_kv, lookup.mlambda)
    persistent = _resolve_persistent(deps, lookup)
    prescan = deps.kernels.prescan_bethe_beam_counts_gpu(
        kvecs,
        persistent,
        deps.metric,
        batch_count=n_k,
        n_g=n_g,
        bethe_c_cutoff=deps.bethe_c_cutoff,
        dbdiff_sg_cutoff=deps.dbdiff_sg_cutoff,
        bethe_c_strong=deps.bethe_c_strong,
        bethe_c_weak=deps.bethe_c_weak,
    )
    n_strong = max(1, prescan.n_strong)
    n_weak = max(0, prescan.n_weak)

    working_chunk = deps.working_chunk_size if deps.working_chunk_size is not None else deps.chunk_size
    effective_chunk = plan_safe_chunk_size(
        working_chunk,
        deps.kernels.device.limits,
        n_g=n_g,
        n_strong=n_strong,
        n_weak=n_weak,
        rank=deps.rank,
        n_sites=n_sites,
    )
    if effective_chunk < 1:
        raise RuntimeError("GPU workspace does not fit even at chunk size 1")
    if effective_chunk < working_chunk:
        deps.working_chunk_size = effective_chunk
    elif deps.working_chunk_size is None:
        deps.working_chunk_size = working_chunk

    workspace = deps.kernels.create_workspace(
        FixedRankChunkDescriptor(
            batch_count=effective_chunk,
            n_g=n_g,
            n_strong=n_strong,
            n_weak=n_weak,
            rank=deps.rank,
            table_size=lookup.table_size,
            n_sites=n_sites,
        )
    )
    output = StorageBuffer(
        deps.kernels.device,
        deps.kernels.queue,
        label=f"ebsd:bin{ctx.bin_index}",
        byte_length=n_k * n_sites * 4,
        copy_src=True,
        copy_dst=True,
    )

    local_idx = np.arange(n_k, dtype=np.uint32)
    try:
        for chunk in chunk_k_vectors(kvecs, local_idx, effective_chunk):
            deps.kernels.run_fixed_rank_chunks(
                chunks=[RunChunk(kvecs=chunk.kvecs, output_indices=chunk.output_indices)],
                persistent=persistent,
                metric=deps.metric,
                workspace=workspace,
                batch_count=effective_chunk,
                n_g=n_g,
                n_strong=n_strong,
                n_weak=n_weak,
                rank=deps.rank,
                table_size=lookup.table_size,
                offset=lookup.offset,
                prefactor=lookup.prefactor,
                bethe_c_cutoff=deps.bethe_c_cutoff,
                dbdiff_sg_cutoff=deps.dbdiff_sg_cutoff,
                bethe_c_strong=deps.bethe_c_strong,
                bethe_c_weak=deps.bethe_c_weak,
                diag_imag=lookup.diag_imag,
                mlambda=lookup.mlambda,
                mu_shift=mu_eff,
                amplitude=a,
                n_sites=n_sites,
                mode=ctx.mode,
                output=output,
                output_count=n_k,
            )
        pattern = output.read_as(np.float32)
    finally:
        output.destroy()
        _destroy_workspace(workspace)

    if not np.all(np.isfinite(pattern)):
        raise ValueError(f"non-finite intensities at bin {ctx.bin_index} ({ctx.voltage_kv:.2f} kV)")
    return PerVoltageResult(pattern=pattern, n_k=n_k, n_sites=n_sites)


def make_metric_buffer(kernels: EBSDDynamicalKernels, cell: Cell) -> StorageBuffer:
    return kernels.create_metric_buffer(metric_to_float32(cell))


def _destroy_workspace(ws: Any) -> None:
    for name in vars(ws):
        buf = getattr(ws, name)
        if isinstance(buf, StorageBuffer):
            buf.destroy()


def _destroy_persistent(pb: PersistentBuffers) -> None:
    for name in vars(pb):
        buf = getattr(pb, name)
        if isinstance(buf, StorageBuffer):
            buf.destroy()
