"""Dynamical workspace: descriptors, packing helpers, buffer allocation.

GPU layer. Units: reciprocal lattice nm⁻¹; complex amplitudes as c64 pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.buffers import StorageBuffer, c64_bytes, f32_bytes, u32_bytes
from ebsdsim.gpu.pipelines import BufferResource, load_wgsl

BetheMode = Literal["bloch", "structure", "bethe"]
C64 = tuple[float, float]


@dataclass
class FixedRankChunkDescriptor:
    batch_count: int
    n_g: int
    n_strong: int
    n_weak: int
    rank: int
    table_size: int
    n_sites: int
    debug: bool = False


@dataclass
class PersistentBuffers:
    hkl: StorageBuffer
    hkl_hash: StorageBuffer
    diff_table: StorageBuffer
    coupling: StorageBuffer
    reflection_dbdiff: StorageBuffer
    sgh_tables: StorageBuffer


@dataclass
class FixedRankWorkspace:
    sg: StorageBuffer
    scores: StorageBuffer
    candidate_mask: StorageBuffer
    selected_flags: StorageBuffer
    idx_a: StorageBuffer
    idx_w: StorageBuffer
    v_aa: StorageBuffer
    v_sw: StorageBuffer
    d_a: StorageBuffer
    d_w: StorageBuffer
    weighted_vws: StorageBuffer
    sigma: StorageBuffer
    geff: StorageBuffer
    q_values: StorageBuffer
    q_i_minus_a: StorageBuffer
    q_i_plus_a: StorageBuffer
    e0: StorageBuffer
    pivots: StorageBuffer
    w_a: StorageBuffer
    w_b: StorageBuffer
    rhs: StorageBuffer
    w_stack: StorageBuffer
    dh: StorageBuffer
    intensities: StorageBuffer


@dataclass
class RunChunk:
    kvecs: NDArray[np.float32]
    output_indices: NDArray[np.uint32] | None = None


def _mode_id(mode: BetheMode) -> int:
    if mode == "bloch":
        return 0
    if mode == "structure":
        return 1
    return 2


def _pack_f32_vec3_to_vec4(src: NDArray[np.float32]) -> NDArray[np.float32]:
    count = src.size // 3
    out = np.zeros((count, 4), dtype=np.float32)
    out[:, :3] = src.reshape(count, 3)
    return out.reshape(-1)


def _pack_i32_vec3_to_vec4(src: NDArray[np.int32]) -> NDArray[np.int32]:
    count = src.size // 3
    out = np.zeros((count, 4), dtype=np.int32)
    out[:, :3] = src.reshape(count, 3)
    return out.reshape(-1)


def _to_u32(src: NDArray[np.integer]) -> NDArray[np.uint32]:
    arr = np.asarray(src)
    if arr.dtype == np.uint32:
        return arr
    return (arr != 0).astype(np.uint32)


def create_workspace(kernels, desc: FixedRankChunkDescriptor) -> FixedRankWorkspace:
    b = desc.batch_count
    n_a = desc.n_strong
    n_w = desc.n_weak
    mat_a = b * n_a * n_a
    mat_sw = b * n_a * n_w
    dbg = desc.debug
    return FixedRankWorkspace(
        sg=kernels._storage("ebsd:sg", f32_bytes(b * desc.n_g), dbg),
        scores=kernels._storage("ebsd:scores", f32_bytes(b * desc.n_g), dbg),
        candidate_mask=kernels._storage("ebsd:candidateMask", u32_bytes(b * desc.n_g), dbg),
        selected_flags=kernels._storage("ebsd:selectedFlags", u32_bytes(b * desc.n_g), dbg),
        idx_a=kernels._storage("ebsd:idxA", u32_bytes(b * n_a), dbg),
        idx_w=kernels._storage("ebsd:idxW", u32_bytes(max(1, b * n_w)), dbg),
        v_aa=kernels._storage("ebsd:vAA", c64_bytes(mat_a), dbg),
        v_sw=kernels._storage("ebsd:vSW", c64_bytes(max(1, mat_sw)), dbg),
        d_a=kernels._storage("ebsd:dA", c64_bytes(b * n_a), dbg),
        d_w=kernels._storage("ebsd:dW", c64_bytes(max(1, b * n_w)), dbg),
        weighted_vws=kernels._storage("ebsd:weightedVWS", c64_bytes(max(1, b * n_w * n_a)), dbg),
        sigma=kernels._storage("ebsd:sigma", c64_bytes(mat_a), dbg),
        geff=kernels._storage("ebsd:geff", c64_bytes(mat_a), dbg),
        q_values=kernels._storage("ebsd:qValues", f32_bytes(b * 4), dbg),
        q_i_minus_a=kernels._storage("ebsd:qIMinusA", c64_bytes(mat_a), dbg),
        q_i_plus_a=kernels._storage("ebsd:qIPlusA", c64_bytes(mat_a), dbg),
        e0=kernels._storage("ebsd:e0", c64_bytes(b * n_a), dbg),
        pivots=kernels._storage("ebsd:pivots", u32_bytes(b * n_a), dbg),
        w_a=kernels._storage("ebsd:wA", c64_bytes(b * n_a), dbg),
        w_b=kernels._storage("ebsd:wB", c64_bytes(b * n_a), dbg),
        rhs=kernels._storage("ebsd:rhs", c64_bytes(b * n_a), dbg),
        w_stack=kernels._storage("ebsd:wStack", c64_bytes(b * n_a * desc.rank), dbg),
        dh=kernels._storage("ebsd:dh", u32_bytes(mat_a), dbg),
        intensities=kernels._storage("ebsd:intensities", f32_bytes(b * desc.n_sites), True),
    )


def create_persistent_buffers(
    kernels,
    *,
    hkl: NDArray[np.int32],
    hkl_hash: NDArray[np.int32],
    diff_table: NDArray[np.float32],
    coupling: NDArray[np.float32],
    reflection_dbdiff: NDArray[np.integer],
    sgh_tables: NDArray[np.float32],
    hkl_packed_vec4: bool = False,
) -> PersistentBuffers:
    hkl_buf = hkl if hkl_packed_vec4 else _pack_i32_vec3_to_vec4(hkl)
    return PersistentBuffers(
        hkl=StorageBuffer(kernels.device, kernels.queue, label="ebsd:persistent:hkl", data=hkl_buf),
        hkl_hash=StorageBuffer(kernels.device, kernels.queue, label="ebsd:persistent:hklHash", data=hkl_hash),
        diff_table=StorageBuffer(kernels.device, kernels.queue, label="ebsd:persistent:diffTable", data=diff_table),
        coupling=StorageBuffer(kernels.device, kernels.queue, label="ebsd:persistent:coupling", data=coupling),
        reflection_dbdiff=StorageBuffer(
            kernels.device,
            kernels.queue,
            label="ebsd:persistent:reflectionDbdiff",
            data=_to_u32(reflection_dbdiff),
        ),
        sgh_tables=StorageBuffer(kernels.device, kernels.queue, label="ebsd:persistent:sghTables", data=sgh_tables),
    )


def create_metric_buffer(kernels, metric: NDArray[np.float32]) -> StorageBuffer:
    return StorageBuffer(
        kernels.device,
        kernels.queue,
        label="ebsd:metric",
        data=_pack_f32_vec3_to_vec4(metric.astype(np.float32, copy=False)),
    )


def create_k_chunk_buffer(kernels, kvecs: NDArray[np.float32]) -> StorageBuffer:
    return StorageBuffer(
        kernels.device,
        kernels.queue,
        label="ebsd:kChunk",
        data=_pack_f32_vec3_to_vec4(kvecs.astype(np.float32, copy=False)),
    )


def _storage(kernels, label: str, byte_length: int, copy_src: bool = False) -> StorageBuffer:
    return StorageBuffer(
        kernels.device,
        kernels.queue,
        label=label,
        byte_length=byte_length,
        copy_src=copy_src,
        copy_dst=True,
    )


def _dispatch_with_params(
    kernels,
    key: str,
    wgsl_name: str,
    params_data: bytes,
    resources: list[StorageBuffer],
    resource_sizes: list[int | None],
    workgroups: tuple[int, int, int],
    label: str,
    *,
    uniform_size: int = 16,
    storage_read_only: bool | list[bool] | None = None,
) -> None:
    packed: list[BufferResource | StorageBuffer] = []
    for buf, size in zip(resources, resource_sizes):
        if size is None:
            packed.append(buf)
        else:
            packed.append(BufferResource(buffer=buf.buffer, size=size))

    kernels.pipelines.dispatch_with_params(
        kernels.queue,
        key,
        load_wgsl(wgsl_name),
        params_data,
        packed,
        workgroups,
        label=label,
        n_storage_bindings=len(resources),
        uniform_size=uniform_size,
        storage_read_only=storage_read_only,
    )

