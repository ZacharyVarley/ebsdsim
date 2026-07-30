"""Score / top-k / GPU Bethe prescan (GPU dynamical layer).

Free functions take the kernels façade (device/queue/pipelines/LU) as their first argument.
"""
from __future__ import annotations

import struct

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.buffers import StorageBuffer, f32_bytes, u32_bytes
from ebsdsim.gpu.device import sync_device
from ebsdsim.gpu.dynamical.workspace import (
    FixedRankWorkspace,
    PersistentBuffers,
)
from ebsdsim.gpu.limits import max_compute_workgroups_per_dimension
from ebsdsim.gpu.pipelines import workgroups_1d
from ebsdsim.physics.prescan import PrescanResult


def excitation_score(
    kernels,
    kvecs: StorageBuffer,
    persistent: PersistentBuffers,
    metric: StorageBuffer,
    ws: FixedRankWorkspace,
    *,
    batch_count: int,
    n_g: int,
    bethe_c_cutoff: float,
    dbdiff_sg_cutoff: float,
    bethe_c_strong: float,
    bethe_c_weak: float,
) -> None:
    params = struct.pack(
        "<IIIIffff",
        batch_count,
        n_g,
        0,
        0,
        bethe_c_cutoff,
        dbdiff_sg_cutoff,
        bethe_c_strong,
        bethe_c_weak,
    )
    kernels._dispatch_with_params(
        "ebsd:excitationScore",
        "dynamical/excitation_score.wgsl",
        params,
        [
            kvecs,
            persistent.hkl,
            metric,
            persistent.coupling,
            persistent.reflection_dbdiff,
            ws.sg,
            ws.scores,
            ws.candidate_mask,
        ],
        [
            batch_count * 16,
            n_g * 16,
            48,
            f32_bytes(n_g),
            u32_bytes(n_g),
            f32_bytes(batch_count * n_g),
            f32_bytes(batch_count * n_g),
            u32_bytes(batch_count * n_g),
        ],
        workgroups_1d(batch_count * n_g, 256),
        "ebsd:excitationScore",
        uniform_size=32,
    )


def top_k(
    kernels,
    ws: FixedRankWorkspace,
    *,
    batch_count: int,
    n_g: int,
    n_strong: int,
    n_weak: int,
) -> None:
    params = struct.pack("<IIII", batch_count, n_g, n_strong, n_weak)
    kernels._dispatch_with_params(
        "ebsd:topK",
        "dynamical/topk_indices.wgsl",
        params,
        [ws.scores, ws.candidate_mask, ws.selected_flags, ws.idx_a, ws.idx_w],
        [
            f32_bytes(batch_count * n_g),
            u32_bytes(batch_count * n_g),
            u32_bytes(batch_count * n_g),
            u32_bytes(batch_count * n_strong),
            u32_bytes(max(1, batch_count * n_weak)),
        ],
        (batch_count, 1, 1),
        "ebsd:topK",
    )


def prescan_bethe_beam_counts_gpu(
    kernels,
    kvecs: NDArray[np.float32],
    persistent: PersistentBuffers,
    metric: StorageBuffer | NDArray[np.float32],
    *,
    batch_count: int,
    n_g: int,
    bethe_c_cutoff: float = 200.0,
    dbdiff_sg_cutoff: float = 1.0,
    bethe_c_strong: float = 20.0,
    bethe_c_weak: float = 40.0,
) -> PrescanResult:
    total_rows = kvecs.size // 3
    if total_rows != batch_count:
        raise ValueError(
            f"prescan_bethe_beam_counts_gpu: batch_count ({batch_count}) must match k-vector rows ({total_rows})"
        )
    strong_needed = np.zeros(total_rows, dtype=np.int32)
    weak_needed = np.zeros(total_rows, dtype=np.int32)
    candidate_needed = np.zeros(total_rows, dtype=np.int32)
    n_strong = 1
    n_weak = 0
    max_candidates = 1
    if total_rows == 0:
        return PrescanResult(n_strong, n_weak, max_candidates, strong_needed, weak_needed, candidate_needed)

    max_rows_per_pass = max(1, max_compute_workgroups_per_dimension(kernels.device))
    metric_buf = metric if isinstance(metric, StorageBuffer) else kernels.create_metric_buffer(metric)
    owns_metric = not isinstance(metric, StorageBuffer)
    try:
        for row_start in range(0, total_rows, max_rows_per_pass):
            row_count = min(max_rows_per_pass, total_rows - row_start)
            k_chunk = kernels.create_k_chunk_buffer(kvecs[row_start * 3 : (row_start + row_count) * 3])
            row_counts = kernels._storage("ebsd:prescanCounts", u32_bytes(row_count * 4), True)
            try:
                params = struct.pack(
                    "<IIIIffff",
                    row_count,
                    n_g,
                    0,
                    0,
                    bethe_c_cutoff,
                    dbdiff_sg_cutoff,
                    bethe_c_strong,
                    bethe_c_weak,
                )
                kernels._dispatch_with_params(
                    "ebsd:prescanCounts",
                    "dynamical/prescan_counts.wgsl",
                    params,
                    [
                        k_chunk,
                        persistent.hkl,
                        metric_buf,
                        persistent.coupling,
                        persistent.reflection_dbdiff,
                        row_counts,
                    ],
                    [
                        row_count * 16,
                        n_g * 16,
                        48,
                        f32_bytes(n_g),
                        u32_bytes(n_g),
                        u32_bytes(row_count * 4),
                    ],
                    (row_count, 1, 1),
                    "ebsd:prescanCounts",
                    uniform_size=32,
                )
                sync_device(kernels.device)
                counts = row_counts.read_as(np.uint32, size=u32_bytes(row_count * 4))
                for row in range(row_count):
                    base = row * 4
                    strong = int(counts[base])
                    weak = int(counts[base + 1])
                    candidates = int(counts[base + 2])
                    dst = row_start + row
                    strong_needed[dst] = strong
                    weak_needed[dst] = weak
                    candidate_needed[dst] = candidates
                    n_strong = max(n_strong, strong)
                    n_weak = max(n_weak, weak)
                    max_candidates = max(max_candidates, candidates)
            finally:
                row_counts.destroy()
                k_chunk.destroy()
    finally:
        if owns_metric:
            metric_buf.destroy()

    return PrescanResult(
        n_strong=n_strong,
        n_weak=n_weak,
        max_candidates=max_candidates,
        strong_needed=strong_needed,
        weak_needed=weak_needed,
        candidate_needed=candidate_needed,
    )

