"""pack_w, intensity contract, output writeback (GPU dynamical layer).

Free functions take the kernels façade (device/queue/pipelines/LU) as their first argument.
"""
from __future__ import annotations

import struct

from ebsdsim.gpu.buffers import StorageBuffer, c64_bytes, f32_bytes, u32_bytes
from ebsdsim.gpu.dynamical.workspace import (
    FixedRankWorkspace,
)
from ebsdsim.gpu.pipelines import workgroups_1d


def pack_w(
    kernels,
    w: StorageBuffer,
    ws: FixedRankWorkspace,
    *,
    batch_count: int,
    n: int,
    rank: int,
    iter: int,
    q_offset: int = 0,
    apply_scale: bool = False,
) -> None:
    params = struct.pack("<IIIIIIII", batch_count, n, rank, iter, q_offset, 1 if apply_scale else 0, 0, 0)
    kernels._dispatch_with_params(
        "ebsd:packW",
        "dynamical/pack_w_c64.wgsl",
        params,
        [ws.q_values, w, ws.w_stack],
        [f32_bytes(batch_count * 4), c64_bytes(batch_count * n), c64_bytes(batch_count * n * rank)],
        workgroups_1d(batch_count * n, 256),
        "ebsd:packW",
    )


def contract_intensity(
    kernels,
    ws: FixedRankWorkspace,
    sgh_tables: StorageBuffer,
    *,
    batch_count: int,
    n: int,
    rank: int,
    n_sites: int,
    table_size: int,
    amplitude: float,
) -> None:
    params = struct.pack(
        "<IIIIIIIIffff",
        batch_count,
        n,
        rank,
        n_sites,
        table_size,
        0,
        0,
        0,
        amplitude,
        0.0,
        0.0,
        0.0,
    )
    kernels._dispatch_with_params(
        "ebsd:intensityContract",
        "dynamical/intensity_contract.wgsl",
        params,
        [ws.dh, sgh_tables, ws.w_stack, ws.intensities],
        [
            u32_bytes(batch_count * n * n),
            c64_bytes(n_sites * table_size),
            c64_bytes(batch_count * n * rank),
            f32_bytes(batch_count * n_sites),
        ],
        (batch_count, n_sites, 1),
        "ebsd:intensityContract",
        uniform_size=48,
    )


def writeback_output(
    kernels,
    chunk_values: StorageBuffer,
    output_indices: StorageBuffer,
    output: StorageBuffer,
    *,
    batch_count: int,
    n_sites: int,
    output_count: int | None = None,
) -> None:
    params = struct.pack("<IIII", batch_count, n_sites, 0, 0)
    total = output_count if output_count is not None else batch_count
    kernels._dispatch_with_params(
        "ebsd:outputWriteback",
        "dynamical/output_writeback_f32.wgsl",
        params,
        [chunk_values, output_indices, output],
        [
            f32_bytes(batch_count * n_sites),
            u32_bytes(batch_count),
            f32_bytes(total * n_sites),
        ],
        workgroups_1d(batch_count * n_sites, 256),
        "ebsd:outputWriteback",
    )

