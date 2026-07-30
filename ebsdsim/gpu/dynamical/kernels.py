"""GPU dynamical-theory kernels (LU–Smith façade + re-exports).

Stage implementations are free functions in sibling modules; this class holds
device/queue/pipeline-cache/LU state and delegates.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.gpu.dynamical.assemble import assemble_geff_q as _fn_assemble_geff_q
from ebsdsim.gpu.dynamical.assemble import (
    build_sigma_with_bethe_gemm as _fn_build_sigma_with_bethe_gemm,
)
from ebsdsim.gpu.dynamical.assemble import gather_diagonal as _fn_gather_diagonal
from ebsdsim.gpu.dynamical.assemble import gemm_c64 as _fn_gemm_c64
from ebsdsim.gpu.dynamical.assemble import gemv_smith_rhs as _fn_gemv_smith_rhs
from ebsdsim.gpu.dynamical.assemble import hash_diff as _fn_hash_diff
from ebsdsim.gpu.dynamical.assemble import lookup_submatrix as _fn_lookup_submatrix
from ebsdsim.gpu.dynamical.assemble import weight_bethe_vws as _fn_weight_bethe_vws
from ebsdsim.gpu.dynamical.chunks import (
    run_fixed_rank_bethe_chunk as _fn_run_fixed_rank_bethe_chunk,
)
from ebsdsim.gpu.dynamical.chunks import run_fixed_rank_chunks as _fn_run_fixed_rank_chunks
from ebsdsim.gpu.dynamical.intensity import contract_intensity as _fn_contract_intensity
from ebsdsim.gpu.dynamical.intensity import pack_w as _fn_pack_w
from ebsdsim.gpu.dynamical.intensity import writeback_output as _fn_writeback_output
from ebsdsim.gpu.dynamical.score import excitation_score as _fn_excitation_score
from ebsdsim.gpu.dynamical.score import (
    prescan_bethe_beam_counts_gpu as _fn_prescan_bethe_beam_counts_gpu,
)
from ebsdsim.gpu.dynamical.score import top_k as _fn_top_k
from ebsdsim.gpu.dynamical.solve import _solve_c64 as _fn__solve_c64
from ebsdsim.gpu.dynamical.solve import run_exact_lyapunov_cpu as _fn_run_exact_lyapunov_cpu
from ebsdsim.gpu.dynamical.solve import run_fixed_smith_loop as _fn_run_fixed_smith_loop
from ebsdsim.gpu.dynamical.workspace import (
    C64,
    BetheMode,
    FixedRankChunkDescriptor,
    FixedRankWorkspace,
    PersistentBuffers,
    RunChunk,
    _to_u32,
)
from ebsdsim.gpu.dynamical.workspace import _dispatch_with_params as _fn__dispatch_with_params
from ebsdsim.gpu.dynamical.workspace import _storage as _fn__storage
from ebsdsim.gpu.dynamical.workspace import create_k_chunk_buffer as _fn_create_k_chunk_buffer
from ebsdsim.gpu.dynamical.workspace import create_metric_buffer as _fn_create_metric_buffer
from ebsdsim.gpu.dynamical.workspace import (
    create_persistent_buffers as _fn_create_persistent_buffers,
)
from ebsdsim.gpu.dynamical.workspace import create_workspace as _fn_create_workspace
from ebsdsim.gpu.lu import LuKernels
from ebsdsim.gpu.pipelines import PipelineCache
from ebsdsim.physics.prescan import PrescanResult

__all__ = [
    "BetheMode",
    "C64",
    "EBSDDynamicalKernels",
    "FixedRankChunkDescriptor",
    "FixedRankWorkspace",
    "PersistentBuffers",
    "RunChunk",
    "_to_u32",
]


class EBSDDynamicalKernels:
    """GPU kernel orchestration for fixed-rank Bethe dynamical solves."""

    def __init__(self, device: Any, queue: Any, lu_kernels: LuKernels | None = None) -> None:
        self.device = device
        self.queue = queue
        self.pipelines = PipelineCache(device)
        self.lu_kernels = lu_kernels or LuKernels(device, queue, self.pipelines)
        self._owns_lu = lu_kernels is None

    def destroy(self) -> None:
        self.pipelines.clear()
        if self._owns_lu:
            self.lu_kernels.destroy()

    def create_workspace(self, desc: FixedRankChunkDescriptor) -> FixedRankWorkspace:
        return _fn_create_workspace(self, desc)

    def create_persistent_buffers(self, *, hkl: NDArray[np.int32], hkl_hash: NDArray[np.int32], diff_table: NDArray[np.float32], coupling: NDArray[np.float32], reflection_dbdiff: NDArray[np.integer], sgh_tables: NDArray[np.float32], hkl_packed_vec4: bool=False) -> PersistentBuffers:
        return _fn_create_persistent_buffers(self, hkl=hkl, hkl_hash=hkl_hash, diff_table=diff_table, coupling=coupling, reflection_dbdiff=reflection_dbdiff, sgh_tables=sgh_tables, hkl_packed_vec4=hkl_packed_vec4)

    def create_metric_buffer(self, metric: NDArray[np.float32]) -> StorageBuffer:
        return _fn_create_metric_buffer(self, metric)

    def create_k_chunk_buffer(self, kvecs: NDArray[np.float32]) -> StorageBuffer:
        return _fn_create_k_chunk_buffer(self, kvecs)

    def _storage(self, label: str, byte_length: int, copy_src: bool=False) -> StorageBuffer:
        return _fn__storage(self, label, byte_length, copy_src)

    def _dispatch_with_params(self, key: str, wgsl_name: str, params_data: bytes, resources: list[StorageBuffer], resource_sizes: list[int | None], workgroups: tuple[int, int, int], label: str, *, uniform_size: int=16, storage_read_only: bool | list[bool] | None=None) -> None:
        return _fn__dispatch_with_params(self, key, wgsl_name, params_data, resources, resource_sizes, workgroups, label, uniform_size=uniform_size, storage_read_only=storage_read_only)

    def excitation_score(self, kvecs: StorageBuffer, persistent: PersistentBuffers, metric: StorageBuffer, ws: FixedRankWorkspace, *, batch_count: int, n_g: int, bethe_c_cutoff: float, dbdiff_sg_cutoff: float, bethe_c_strong: float, bethe_c_weak: float) -> None:
        return _fn_excitation_score(self, kvecs, persistent, metric, ws, batch_count=batch_count, n_g=n_g, bethe_c_cutoff=bethe_c_cutoff, dbdiff_sg_cutoff=dbdiff_sg_cutoff, bethe_c_strong=bethe_c_strong, bethe_c_weak=bethe_c_weak)

    def top_k(self, ws: FixedRankWorkspace, *, batch_count: int, n_g: int, n_strong: int, n_weak: int) -> None:
        return _fn_top_k(self, ws, batch_count=batch_count, n_g=n_g, n_strong=n_strong, n_weak=n_weak)

    def prescan_bethe_beam_counts_gpu(self, kvecs: NDArray[np.float32], persistent: PersistentBuffers, metric: StorageBuffer | NDArray[np.float32], *, batch_count: int, n_g: int, bethe_c_cutoff: float=200.0, dbdiff_sg_cutoff: float=1.0, bethe_c_strong: float=20.0, bethe_c_weak: float=40.0) -> PrescanResult:
        return _fn_prescan_bethe_beam_counts_gpu(self, kvecs, persistent, metric, batch_count=batch_count, n_g=n_g, bethe_c_cutoff=bethe_c_cutoff, dbdiff_sg_cutoff=dbdiff_sg_cutoff, bethe_c_strong=bethe_c_strong, bethe_c_weak=bethe_c_weak)

    def gemm_c64(self, a: StorageBuffer, b: StorageBuffer, c: StorageBuffer, *, batch_count: int, m: int, n: int, k: int, a_stride: int | None=None, b_stride: int | None=None, c_stride: int | None=None, lda: int | None=None, ldb: int | None=None, ldc: int | None=None, alpha: C64=(1.0, 0.0), beta: C64=(0.0, 0.0)) -> None:
        return _fn_gemm_c64(self, a, b, c, batch_count=batch_count, m=m, n=n, k=k, a_stride=a_stride, b_stride=b_stride, c_stride=c_stride, lda=lda, ldb=ldb, ldc=ldc, alpha=alpha, beta=beta)

    def gemv_smith_rhs(self, geff: StorageBuffer, w: StorageBuffer, out: StorageBuffer, *, batch_count: int, n: int, q_offset: int=2) -> None:
        return _fn_gemv_smith_rhs(self, geff, w, out, batch_count=batch_count, n=n, q_offset=q_offset)

    def lookup_submatrix(self, idx_rows: StorageBuffer, idx_cols: StorageBuffer, hkl_hash: StorageBuffer, table: StorageBuffer, out: StorageBuffer, *, batch_count: int, n_rows: int, n_cols: int, table_size: int, offset: int, prefactor: C64, zero_diagonal: bool=False) -> None:
        return _fn_lookup_submatrix(self, idx_rows, idx_cols, hkl_hash, table, out, batch_count=batch_count, n_rows=n_rows, n_cols=n_cols, table_size=table_size, offset=offset, prefactor=prefactor, zero_diagonal=zero_diagonal)

    def hash_diff(self, ws: FixedRankWorkspace, hkl_hash: StorageBuffer, *, batch_count: int, n: int, table_size: int, offset: int) -> None:
        return _fn_hash_diff(self, ws, hkl_hash, batch_count=batch_count, n=n, table_size=table_size, offset=offset)

    def gather_diagonal(self, sg: StorageBuffer, idx: StorageBuffer, out: StorageBuffer, *, batch_count: int, n_g: int, n: int, mode: BetheMode, diag_imag: float, mlambda: float) -> None:
        return _fn_gather_diagonal(self, sg, idx, out, batch_count=batch_count, n_g=n_g, n=n, mode=mode, diag_imag=diag_imag, mlambda=mlambda)

    def weight_bethe_vws(self, ws: FixedRankWorkspace, *, batch_count: int, n_a: int, n_w: int, eps: float=1e-07) -> None:
        return _fn_weight_bethe_vws(self, ws, batch_count=batch_count, n_a=n_a, n_w=n_w, eps=eps)

    def build_sigma_with_bethe_gemm(self, ws: FixedRankWorkspace, *, batch_count: int, n_a: int, n_w: int, eps: float=1e-07) -> None:
        return _fn_build_sigma_with_bethe_gemm(self, ws, batch_count=batch_count, n_a=n_a, n_w=n_w, eps=eps)

    def assemble_geff_q(self, ws: FixedRankWorkspace, *, batch_count: int, n: int, use_sigma: bool=True, mu_shift: float=0.0, eps: float=1e-12) -> None:
        return _fn_assemble_geff_q(self, ws, batch_count=batch_count, n=n, use_sigma=use_sigma, mu_shift=mu_shift, eps=eps)

    def _solve_c64(self, lu: StorageBuffer, pivots: StorageBuffer, rhs: StorageBuffer, out: StorageBuffer, *, batch_count: int, n: int) -> None:
        return _fn__solve_c64(self, lu, pivots, rhs, out, batch_count=batch_count, n=n)

    def run_exact_lyapunov_cpu(self, ws: FixedRankWorkspace, *, batch_count: int, n: int, use_sigma: bool, mu_shift: float=0.0) -> None:
        return _fn_run_exact_lyapunov_cpu(self, ws, batch_count=batch_count, n=n, use_sigma=use_sigma, mu_shift=mu_shift)

    def run_fixed_smith_loop(self, ws: FixedRankWorkspace, *, batch_count: int, n: int, rank: int) -> None:
        return _fn_run_fixed_smith_loop(self, ws, batch_count=batch_count, n=n, rank=rank)

    def pack_w(self, w: StorageBuffer, ws: FixedRankWorkspace, *, batch_count: int, n: int, rank: int, iter: int, q_offset: int=0, apply_scale: bool=False) -> None:
        return _fn_pack_w(self, w, ws, batch_count=batch_count, n=n, rank=rank, iter=iter, q_offset=q_offset, apply_scale=apply_scale)

    def contract_intensity(self, ws: FixedRankWorkspace, sgh_tables: StorageBuffer, *, batch_count: int, n: int, rank: int, n_sites: int, table_size: int, amplitude: float) -> None:
        return _fn_contract_intensity(self, ws, sgh_tables, batch_count=batch_count, n=n, rank=rank, n_sites=n_sites, table_size=table_size, amplitude=amplitude)

    def writeback_output(self, chunk_values: StorageBuffer, output_indices: StorageBuffer, output: StorageBuffer, *, batch_count: int, n_sites: int, output_count: int | None=None) -> None:
        return _fn_writeback_output(self, chunk_values, output_indices, output, batch_count=batch_count, n_sites=n_sites, output_count=output_count)

    def run_fixed_rank_bethe_chunk(self, kvecs: StorageBuffer, metric: StorageBuffer, persistent: PersistentBuffers, ws: FixedRankWorkspace, *, batch_count: int, n_g: int, n_strong: int, n_weak: int, rank: int, table_size: int, offset: int, prefactor: C64, bethe_c_cutoff: float, dbdiff_sg_cutoff: float, bethe_c_strong: float, bethe_c_weak: float, mode: BetheMode='bloch', diag_imag: float, mlambda: float, mu_shift: float=0.0, amplitude: float, n_sites: int, exact_slow_cpu: bool=False) -> None:
        return _fn_run_fixed_rank_bethe_chunk(self, kvecs, metric, persistent, ws, batch_count=batch_count, n_g=n_g, n_strong=n_strong, n_weak=n_weak, rank=rank, table_size=table_size, offset=offset, prefactor=prefactor, bethe_c_cutoff=bethe_c_cutoff, dbdiff_sg_cutoff=dbdiff_sg_cutoff, bethe_c_strong=bethe_c_strong, bethe_c_weak=bethe_c_weak, mode=mode, diag_imag=diag_imag, mlambda=mlambda, mu_shift=mu_shift, amplitude=amplitude, n_sites=n_sites, exact_slow_cpu=exact_slow_cpu)

    def run_fixed_rank_chunks(self, *, chunks: Iterable[RunChunk], persistent: PersistentBuffers, metric: StorageBuffer | NDArray[np.float32], workspace: FixedRankWorkspace, batch_count: int, n_g: int, n_strong: int, n_weak: int, rank: int, table_size: int, offset: int, prefactor: C64, bethe_c_cutoff: float, dbdiff_sg_cutoff: float, bethe_c_strong: float, bethe_c_weak: float, diag_imag: float, mlambda: float, mu_shift: float=0.0, amplitude: float, n_sites: int, mode: BetheMode='bloch', output: StorageBuffer | None=None, output_count: int | None=None, on_progress: Callable[[int, int], None] | None=None, exact_slow_cpu: bool=False, max_chunks: int | None=None) -> None:
        return _fn_run_fixed_rank_chunks(self, chunks=chunks, persistent=persistent, metric=metric, workspace=workspace, batch_count=batch_count, n_g=n_g, n_strong=n_strong, n_weak=n_weak, rank=rank, table_size=table_size, offset=offset, prefactor=prefactor, bethe_c_cutoff=bethe_c_cutoff, dbdiff_sg_cutoff=dbdiff_sg_cutoff, bethe_c_strong=bethe_c_strong, bethe_c_weak=bethe_c_weak, diag_imag=diag_imag, mlambda=mlambda, mu_shift=mu_shift, amplitude=amplitude, n_sites=n_sites, mode=mode, output=output, output_count=output_count, on_progress=on_progress, exact_slow_cpu=exact_slow_cpu, max_chunks=max_chunks)
