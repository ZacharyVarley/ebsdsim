"""Fixed-rank chunk orchestration (GPU dynamical layer).

Free functions take the kernels façade (device/queue/pipelines/LU) as their first argument.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.gpu.device import sync_device
from ebsdsim.gpu.dynamical.workspace import (
    C64,
    BetheMode,
    FixedRankWorkspace,
    PersistentBuffers,
    RunChunk,
)


def run_fixed_rank_bethe_chunk(
    kernels,
    kvecs: StorageBuffer,
    metric: StorageBuffer,
    persistent: PersistentBuffers,
    ws: FixedRankWorkspace,
    *,
    batch_count: int,
    n_g: int,
    n_strong: int,
    n_weak: int,
    rank: int,
    table_size: int,
    offset: int,
    prefactor: C64,
    bethe_c_cutoff: float,
    dbdiff_sg_cutoff: float,
    bethe_c_strong: float,
    bethe_c_weak: float,
    mode: BetheMode = "bloch",
    diag_imag: float,
    mlambda: float,
    mu_shift: float = 0.0,
    amplitude: float,
    n_sites: int,
    exact_slow_cpu: bool = False,
) -> None:
    score_kw = dict(
        batch_count=batch_count,
        n_g=n_g,
        bethe_c_cutoff=bethe_c_cutoff,
        dbdiff_sg_cutoff=dbdiff_sg_cutoff,
        bethe_c_strong=bethe_c_strong,
        bethe_c_weak=bethe_c_weak,
    )
    kernels.excitation_score(kvecs, persistent, metric, ws, **score_kw)
    kernels.top_k(ws, batch_count=batch_count, n_g=n_g, n_strong=n_strong, n_weak=n_weak)
    kernels.lookup_submatrix(
        ws.idx_a,
        ws.idx_a,
        persistent.hkl_hash,
        persistent.diff_table,
        ws.v_aa,
        batch_count=batch_count,
        n_rows=n_strong,
        n_cols=n_strong,
        table_size=table_size,
        offset=offset,
        prefactor=prefactor,
        zero_diagonal=True,
    )
    kernels.gather_diagonal(
        ws.sg,
        ws.idx_a,
        ws.d_a,
        batch_count=batch_count,
        n_g=n_g,
        n=n_strong,
        mode=mode,
        diag_imag=diag_imag,
        mlambda=mlambda,
    )
    if n_weak > 0:
        kernels.lookup_submatrix(
            ws.idx_a,
            ws.idx_w,
            persistent.hkl_hash,
            persistent.diff_table,
            ws.v_sw,
            batch_count=batch_count,
            n_rows=n_strong,
            n_cols=n_weak,
            table_size=table_size,
            offset=offset,
            prefactor=prefactor,
        )
        kernels.gather_diagonal(
            ws.sg,
            ws.idx_w,
            ws.d_w,
            batch_count=batch_count,
            n_g=n_g,
            n=n_weak,
            mode="bethe",
            diag_imag=diag_imag,
            mlambda=mlambda,
        )
        kernels.build_sigma_with_bethe_gemm(ws, batch_count=batch_count, n_a=n_strong, n_w=n_weak)
    use_sigma = n_weak > 0
    intensity_rank = n_strong if exact_slow_cpu else rank
    if exact_slow_cpu:
        kernels.run_exact_lyapunov_cpu(
            ws,
            batch_count=batch_count,
            n=n_strong,
            use_sigma=use_sigma,
            mu_shift=mu_shift,
        )
    else:
        kernels.assemble_geff_q(
            ws,
            batch_count=batch_count,
            n=n_strong,
            use_sigma=use_sigma,
            mu_shift=mu_shift,
        )
        kernels.run_fixed_smith_loop(ws, batch_count=batch_count, n=n_strong, rank=rank)
    kernels.hash_diff(ws, persistent.hkl_hash, batch_count=batch_count, n=n_strong, table_size=table_size, offset=offset)
    kernels.contract_intensity(
        ws,
        persistent.sgh_tables,
        batch_count=batch_count,
        n=n_strong,
        rank=intensity_rank,
        n_sites=n_sites,
        table_size=table_size,
        amplitude=amplitude,
    )


def run_fixed_rank_chunks(
    kernels,
    *,
    chunks: Iterable[RunChunk],
    persistent: PersistentBuffers,
    metric: StorageBuffer | NDArray[np.float32],
    workspace: FixedRankWorkspace,
    batch_count: int,
    n_g: int,
    n_strong: int,
    n_weak: int,
    rank: int,
    table_size: int,
    offset: int,
    prefactor: C64,
    bethe_c_cutoff: float,
    dbdiff_sg_cutoff: float,
    bethe_c_strong: float,
    bethe_c_weak: float,
    diag_imag: float,
    mlambda: float,
    mu_shift: float = 0.0,
    amplitude: float,
    n_sites: int,
    mode: BetheMode = "bloch",
    output: StorageBuffer | None = None,
    output_count: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    exact_slow_cpu: bool = False,
    max_chunks: int | None = None,
) -> None:
    metric_buf = metric if isinstance(metric, StorageBuffer) else kernels.create_metric_buffer(metric)
    owns_metric = not isinstance(metric, StorageBuffer)
    chunks_done = 0
    rows_done = 0
    try:
        for chunk in chunks:
            if max_chunks is not None and chunks_done >= max_chunks:
                break
            rows = chunk.kvecs.size // 3
            if rows > batch_count:
                raise ValueError(f"chunk has {rows} k-vectors but batch_count is {batch_count}")
            kvecs = kernels.create_k_chunk_buffer(chunk.kvecs)
            output_indices: StorageBuffer | None = None
            try:
                kernels.run_fixed_rank_bethe_chunk(
                    kvecs,
                    metric_buf,
                    persistent,
                    workspace,
                    batch_count=rows,
                    n_g=n_g,
                    n_strong=n_strong,
                    n_weak=n_weak,
                    rank=rank,
                    table_size=table_size,
                    offset=offset,
                    prefactor=prefactor,
                    bethe_c_cutoff=bethe_c_cutoff,
                    dbdiff_sg_cutoff=dbdiff_sg_cutoff,
                    bethe_c_strong=bethe_c_strong,
                    bethe_c_weak=bethe_c_weak,
                    mode=mode,
                    diag_imag=diag_imag,
                    mlambda=mlambda,
                    mu_shift=mu_shift,
                    amplitude=amplitude,
                    n_sites=n_sites,
                    exact_slow_cpu=exact_slow_cpu,
                )
                if output is not None and chunk.output_indices is not None:
                    output_indices = StorageBuffer(
                        kernels.device,
                        kernels.queue,
                        label="ebsd:outputIndices",
                        data=chunk.output_indices,
                    )
                    kernels.writeback_output(
                        workspace.intensities,
                        output_indices,
                        output,
                        batch_count=rows,
                        n_sites=n_sites,
                        output_count=output_count,
                    )
                sync_device(kernels.device)
            finally:
                kvecs.destroy()
                if output_indices is not None:
                    output_indices.destroy()
            chunks_done += 1
            rows_done += rows
            if on_progress is not None:
                on_progress(chunks_done, rows_done)
    finally:
        if owns_metric:
            metric_buf.destroy()

