"""EBSD dynamical-theory GPU kernels (fixed-rank Bethe)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.buffers import StorageBuffer, c64_bytes, f32_bytes, u32_bytes
from ebsdsim.gpu.device import sync_device
from ebsdsim.gpu.lu import LuKernels
from ebsdsim.gpu.limits import max_compute_workgroups_per_dimension
from ebsdsim.gpu.pipelines import PipelineCache, load_wgsl, workgroups_1d
from ebsdsim.prescan import PrescanResult

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
    if src.dtype == np.uint32:
        return src
    out = np.zeros(src.size, dtype=np.uint32)
    for i in range(src.size):
        out[i] = 1 if src[i] else 0
    return out


class EBSDDynamicalKernels:
    """GPU kernel orchestration for fixed-rank Bethe dynamical solves."""

    def __init__(
        self,
        device: Any,
        queue: Any,
        lu_kernels: LuKernels | None = None,
    ) -> None:
        self.device = device
        self.queue = queue
        self.pipelines = PipelineCache(device)
        self.lu_kernels = lu_kernels or LuKernels(device, queue, self.pipelines)
        self._owns_lu = lu_kernels is None

    def destroy(self) -> None:
        self.pipelines.clear()
        if self._owns_lu:
            self.lu_kernels.destroy()

    def _storage(self, label: str, byte_length: int, copy_src: bool = False) -> StorageBuffer:
        return StorageBuffer(
            self.device,
            self.queue,
            label=label,
            byte_length=byte_length,
            copy_src=copy_src,
            copy_dst=True,
        )

    def create_workspace(self, desc: FixedRankChunkDescriptor) -> FixedRankWorkspace:
        b = desc.batch_count
        n_a = desc.n_strong
        n_w = desc.n_weak
        mat_a = b * n_a * n_a
        mat_sw = b * n_a * n_w
        dbg = desc.debug
        return FixedRankWorkspace(
            sg=self._storage("ebsd:sg", f32_bytes(b * desc.n_g), dbg),
            scores=self._storage("ebsd:scores", f32_bytes(b * desc.n_g), dbg),
            candidate_mask=self._storage("ebsd:candidateMask", u32_bytes(b * desc.n_g), dbg),
            selected_flags=self._storage("ebsd:selectedFlags", u32_bytes(b * desc.n_g), dbg),
            idx_a=self._storage("ebsd:idxA", u32_bytes(b * n_a), dbg),
            idx_w=self._storage("ebsd:idxW", u32_bytes(max(1, b * n_w)), dbg),
            v_aa=self._storage("ebsd:vAA", c64_bytes(mat_a), dbg),
            v_sw=self._storage("ebsd:vSW", c64_bytes(max(1, mat_sw)), dbg),
            d_a=self._storage("ebsd:dA", c64_bytes(b * n_a), dbg),
            d_w=self._storage("ebsd:dW", c64_bytes(max(1, b * n_w)), dbg),
            weighted_vws=self._storage("ebsd:weightedVWS", c64_bytes(max(1, b * n_w * n_a)), dbg),
            sigma=self._storage("ebsd:sigma", c64_bytes(mat_a), dbg),
            geff=self._storage("ebsd:geff", c64_bytes(mat_a), dbg),
            q_values=self._storage("ebsd:qValues", f32_bytes(b * 4), dbg),
            q_i_minus_a=self._storage("ebsd:qIMinusA", c64_bytes(mat_a), dbg),
            q_i_plus_a=self._storage("ebsd:qIPlusA", c64_bytes(mat_a), dbg),
            e0=self._storage("ebsd:e0", c64_bytes(b * n_a), dbg),
            pivots=self._storage("ebsd:pivots", u32_bytes(b * n_a), dbg),
            w_a=self._storage("ebsd:wA", c64_bytes(b * n_a), dbg),
            w_b=self._storage("ebsd:wB", c64_bytes(b * n_a), dbg),
            rhs=self._storage("ebsd:rhs", c64_bytes(b * n_a), dbg),
            w_stack=self._storage("ebsd:wStack", c64_bytes(b * n_a * desc.rank), dbg),
            dh=self._storage("ebsd:dh", u32_bytes(mat_a), dbg),
            intensities=self._storage("ebsd:intensities", f32_bytes(b * desc.n_sites), True),
        )

    def create_persistent_buffers(
        self,
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
            hkl=StorageBuffer(self.device, self.queue, label="ebsd:persistent:hkl", data=hkl_buf),
            hkl_hash=StorageBuffer(self.device, self.queue, label="ebsd:persistent:hklHash", data=hkl_hash),
            diff_table=StorageBuffer(self.device, self.queue, label="ebsd:persistent:diffTable", data=diff_table),
            coupling=StorageBuffer(self.device, self.queue, label="ebsd:persistent:coupling", data=coupling),
            reflection_dbdiff=StorageBuffer(
                self.device,
                self.queue,
                label="ebsd:persistent:reflectionDbdiff",
                data=_to_u32(reflection_dbdiff),
            ),
            sgh_tables=StorageBuffer(self.device, self.queue, label="ebsd:persistent:sghTables", data=sgh_tables),
        )

    def create_metric_buffer(self, metric: NDArray[np.float32]) -> StorageBuffer:
        return StorageBuffer(
            self.device,
            self.queue,
            label="ebsd:metric",
            data=_pack_f32_vec3_to_vec4(metric.astype(np.float32, copy=False)),
        )

    def create_k_chunk_buffer(self, kvecs: NDArray[np.float32]) -> StorageBuffer:
        return StorageBuffer(
            self.device,
            self.queue,
            label="ebsd:kChunk",
            data=_pack_f32_vec3_to_vec4(kvecs.astype(np.float32, copy=False)),
        )

    def _dispatch_with_params(
        self,
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
        from ebsdsim.gpu.pipelines import BufferResource

        packed: list[BufferResource | StorageBuffer] = []
        for buf, size in zip(resources, resource_sizes):
            if size is None:
                packed.append(buf)
            else:
                packed.append(BufferResource(buffer=buf.buffer, size=size))

        self.pipelines.dispatch_with_params(
            self.queue,
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

    def gemm_c64(
        self,
        a: StorageBuffer,
        b: StorageBuffer,
        c: StorageBuffer,
        *,
        batch_count: int,
        m: int,
        n: int,
        k: int,
        a_stride: int | None = None,
        b_stride: int | None = None,
        c_stride: int | None = None,
        lda: int | None = None,
        ldb: int | None = None,
        ldc: int | None = None,
        alpha: C64 = (1.0, 0.0),
        beta: C64 = (0.0, 0.0),
    ) -> None:
        if k <= 0:
            return
        a_stride = a_stride if a_stride is not None else m * k
        b_stride = b_stride if b_stride is not None else k * n
        c_stride = c_stride if c_stride is not None else m * n
        lda = lda if lda is not None else k
        ldb = ldb if ldb is not None else n
        ldc = ldc if ldc is not None else n
        params = struct.pack(
            "<10Iffff",
            batch_count,
            m,
            n,
            k,
            a_stride,
            b_stride,
            c_stride,
            lda,
            ldb,
            ldc,
            alpha[0],
            alpha[1],
            beta[0],
            beta[1],
        )
        self._dispatch_with_params(
            "ebsd:gemmC64",
            "ebsd_gemm_c64_batched.wgsl",
            params,
            [a, b, c],
            [
                c64_bytes(batch_count * a_stride),
                c64_bytes(batch_count * b_stride),
                c64_bytes(batch_count * c_stride),
            ],
            (max(1, (n + 7) // 8), max(1, (m + 7) // 8), batch_count),
            "ebsd:gemmC64",
            uniform_size=64,
        )

    def gemv_smith_rhs(
        self,
        geff: StorageBuffer,
        w: StorageBuffer,
        out: StorageBuffer,
        *,
        batch_count: int,
        n: int,
        q_offset: int = 2,
    ) -> None:
        params = struct.pack("<IIIIIIII", batch_count, n, n * n, n, n, q_offset, 0, 0)
        self._dispatch_with_params(
            "ebsd:gemvC64",
            "ebsd_gemv_c64_batched.wgsl",
            params,
            [geff, w, out],
            [c64_bytes(batch_count * n * n), c64_bytes(batch_count * n), c64_bytes(batch_count * n)],
            (n, batch_count, 1),
            "ebsd:gemvC64",
        )

    def excitation_score(
        self,
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
        self._dispatch_with_params(
            "ebsd:excitationScore",
            "ebsd_excitation_score.wgsl",
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
        self,
        ws: FixedRankWorkspace,
        *,
        batch_count: int,
        n_g: int,
        n_strong: int,
        n_weak: int,
    ) -> None:
        params = struct.pack("<IIII", batch_count, n_g, n_strong, n_weak)
        self._dispatch_with_params(
            "ebsd:topK",
            "ebsd_topk_indices.wgsl",
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

    def lookup_submatrix(
        self,
        idx_rows: StorageBuffer,
        idx_cols: StorageBuffer,
        hkl_hash: StorageBuffer,
        table: StorageBuffer,
        out: StorageBuffer,
        *,
        batch_count: int,
        n_rows: int,
        n_cols: int,
        table_size: int,
        offset: int,
        prefactor: C64,
        zero_diagonal: bool = False,
    ) -> None:
        buf = bytearray(48)
        struct.pack_into("<I", buf, 0, batch_count)
        struct.pack_into("<I", buf, 4, n_rows)
        struct.pack_into("<I", buf, 8, n_cols)
        struct.pack_into("<I", buf, 12, table_size)
        struct.pack_into("<i", buf, 16, offset)
        struct.pack_into("<f", buf, 24, prefactor[0])
        struct.pack_into("<f", buf, 28, prefactor[1])
        struct.pack_into("<I", buf, 32, 1 if zero_diagonal else 0)
        params = bytes(buf)
        self._dispatch_with_params(
            "ebsd:lookupSubmatrix",
            "ebsd_lookup_submatrix_c64.wgsl",
            params,
            [idx_rows, idx_cols, hkl_hash, table, out],
            [
                u32_bytes(batch_count * n_rows),
                u32_bytes(batch_count * n_cols),
                None,
                c64_bytes(table_size),
                c64_bytes(batch_count * n_rows * n_cols),
            ],
            workgroups_1d(batch_count * n_rows * n_cols, 256),
            "ebsd:lookupSubmatrix",
            uniform_size=48,
        )

    def hash_diff(
        self,
        ws: FixedRankWorkspace,
        hkl_hash: StorageBuffer,
        *,
        batch_count: int,
        n: int,
        table_size: int,
        offset: int,
    ) -> None:
        buf = bytearray(32)
        struct.pack_into("<I", buf, 0, batch_count)
        struct.pack_into("<I", buf, 4, n)
        struct.pack_into("<I", buf, 8, table_size)
        struct.pack_into("<i", buf, 16, offset)
        params = bytes(buf)
        self._dispatch_with_params(
            "ebsd:hashDiff",
            "ebsd_hash_diff_u32.wgsl",
            params,
            [ws.idx_a, hkl_hash, ws.dh],
            [u32_bytes(batch_count * n), None, u32_bytes(batch_count * n * n)],
            workgroups_1d(batch_count * n * n, 256),
            "ebsd:hashDiff",
            uniform_size=32,
        )

    def gather_diagonal(
        self,
        sg: StorageBuffer,
        idx: StorageBuffer,
        out: StorageBuffer,
        *,
        batch_count: int,
        n_g: int,
        n: int,
        mode: BetheMode,
        diag_imag: float,
        mlambda: float,
    ) -> None:
        params = struct.pack("<IIIIf f", batch_count, n_g, n, _mode_id(mode), diag_imag, mlambda)
        self._dispatch_with_params(
            "ebsd:gatherDiagonal",
            "ebsd_gather_diagonal_c64.wgsl",
            params,
            [sg, idx, out],
            [f32_bytes(batch_count * n_g), u32_bytes(batch_count * n), c64_bytes(batch_count * n)],
            workgroups_1d(batch_count * n, 256),
            "ebsd:gatherDiagonal",
            uniform_size=32,
        )

    def weight_bethe_vws(
        self,
        ws: FixedRankWorkspace,
        *,
        batch_count: int,
        n_a: int,
        n_w: int,
        eps: float = 1e-7,
    ) -> None:
        if n_w == 0:
            return
        params = struct.pack("<IIIIf", batch_count, n_a, n_w, 0, eps)
        self._dispatch_with_params(
            "ebsd:betheWeightVws",
            "ebsd_bethe_weight_vws_c64.wgsl",
            params,
            [ws.v_sw, ws.d_w, ws.weighted_vws],
            [
                c64_bytes(batch_count * n_a * n_w),
                c64_bytes(batch_count * n_w),
                c64_bytes(batch_count * n_w * n_a),
            ],
            workgroups_1d(batch_count * n_a * n_w, 256),
            "ebsd:betheWeightVws",
            uniform_size=32,
        )

    def build_sigma_with_bethe_gemm(
        self,
        ws: FixedRankWorkspace,
        *,
        batch_count: int,
        n_a: int,
        n_w: int,
        eps: float = 1e-7,
    ) -> None:
        if n_w == 0:
            return
        self.weight_bethe_vws(ws, batch_count=batch_count, n_a=n_a, n_w=n_w, eps=eps)
        self.gemm_c64(
            ws.v_sw,
            ws.weighted_vws,
            ws.sigma,
            batch_count=batch_count,
            m=n_a,
            n=n_a,
            k=n_w,
            a_stride=n_a * n_w,
            b_stride=n_w * n_a,
            c_stride=n_a * n_a,
            lda=n_w,
            ldb=n_a,
            ldc=n_a,
            beta=(0.0, 0.0),
        )

    def assemble_geff_q(
        self,
        ws: FixedRankWorkspace,
        *,
        batch_count: int,
        n: int,
        use_sigma: bool = True,
        mu_shift: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        params = struct.pack(
            "<IIIIf fff",
            batch_count,
            n,
            0 if not use_sigma else 1,
            0,
            mu_shift,
            eps,
            0.0,
            0.0,
        )
        self._dispatch_with_params(
            "ebsd:assembleGeffQ",
            "ebsd_assemble_geff_q.wgsl",
            params,
            [
                ws.v_aa,
                ws.d_a,
                ws.sigma,
                ws.idx_a,
                ws.q_values,
                ws.q_i_minus_a,
                ws.q_i_plus_a,
                ws.e0,
            ],
            [
                c64_bytes(batch_count * n * n),
                c64_bytes(batch_count * n),
                c64_bytes(batch_count * n * n),
                u32_bytes(batch_count * n),
                f32_bytes(batch_count * 4),
                c64_bytes(batch_count * n * n),
                c64_bytes(batch_count * n * n),
                c64_bytes(batch_count * n),
            ],
            (batch_count, 1, 1),
            "ebsd:assembleGeffQ",
            uniform_size=32,
        )

    def _solve_c64(
        self,
        lu: StorageBuffer,
        pivots: StorageBuffer,
        rhs: StorageBuffer,
        out: StorageBuffer,
        *,
        batch_count: int,
        n: int,
    ) -> None:
        self.lu_kernels.lu_solve_complex64_batched(lu, pivots, rhs, out, batch_count, n)

    def run_fixed_smith_loop(
        self,
        ws: FixedRankWorkspace,
        *,
        batch_count: int,
        n: int,
        rank: int,
    ) -> None:
        self.lu_kernels.lu_factor_complex64_batched(ws.q_i_minus_a, ws.pivots, batch_count, n)
        self._solve_c64(ws.q_i_minus_a, ws.pivots, ws.e0, ws.w_a, batch_count=batch_count, n=n)
        self.pack_w(ws.w_a, ws, batch_count=batch_count, n=n, rank=rank, iter=0, apply_scale=True, q_offset=3)

        current = ws.w_a
        next_buf = ws.w_b
        for iter_idx in range(1, rank):
            self.gemv_smith_rhs(ws.q_i_plus_a, current, ws.rhs, batch_count=batch_count, n=n)
            self._solve_c64(ws.q_i_minus_a, ws.pivots, ws.rhs, next_buf, batch_count=batch_count, n=n)
            self.pack_w(
                next_buf,
                ws,
                batch_count=batch_count,
                n=n,
                rank=rank,
                iter=iter_idx,
                apply_scale=False,
                q_offset=0,
            )
            current, next_buf = next_buf, current

    def pack_w(
        self,
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
        self._dispatch_with_params(
            "ebsd:packW",
            "ebsd_pack_w_c64.wgsl",
            params,
            [ws.q_values, w, ws.w_stack],
            [f32_bytes(batch_count * 4), c64_bytes(batch_count * n), c64_bytes(batch_count * n * rank)],
            workgroups_1d(batch_count * n, 256),
            "ebsd:packW",
        )

    def contract_intensity(
        self,
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
        self._dispatch_with_params(
            "ebsd:intensityContract",
            "ebsd_intensity_contract.wgsl",
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
        self,
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
        self._dispatch_with_params(
            "ebsd:outputWriteback",
            "ebsd_output_writeback_f32.wgsl",
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

    def run_fixed_rank_bethe_chunk(
        self,
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
    ) -> None:
        score_kw = dict(
            batch_count=batch_count,
            n_g=n_g,
            bethe_c_cutoff=bethe_c_cutoff,
            dbdiff_sg_cutoff=dbdiff_sg_cutoff,
            bethe_c_strong=bethe_c_strong,
            bethe_c_weak=bethe_c_weak,
        )
        self.excitation_score(kvecs, persistent, metric, ws, **score_kw)
        self.top_k(ws, batch_count=batch_count, n_g=n_g, n_strong=n_strong, n_weak=n_weak)
        self.lookup_submatrix(
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
        self.gather_diagonal(
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
            self.lookup_submatrix(
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
            self.gather_diagonal(
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
            self.build_sigma_with_bethe_gemm(ws, batch_count=batch_count, n_a=n_strong, n_w=n_weak)
        self.assemble_geff_q(
            ws,
            batch_count=batch_count,
            n=n_strong,
            use_sigma=n_weak > 0,
            mu_shift=mu_shift,
        )
        self.run_fixed_smith_loop(ws, batch_count=batch_count, n=n_strong, rank=rank)
        self.hash_diff(ws, persistent.hkl_hash, batch_count=batch_count, n=n_strong, table_size=table_size, offset=offset)
        self.contract_intensity(
            ws,
            persistent.sgh_tables,
            batch_count=batch_count,
            n=n_strong,
            rank=rank,
            n_sites=n_sites,
            table_size=table_size,
            amplitude=amplitude,
        )

    def run_fixed_rank_chunks(
        self,
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
    ) -> None:
        metric_buf = metric if isinstance(metric, StorageBuffer) else self.create_metric_buffer(metric)
        owns_metric = not isinstance(metric, StorageBuffer)
        chunks_done = 0
        rows_done = 0
        try:
            for chunk in chunks:
                rows = chunk.kvecs.size // 3
                if rows > batch_count:
                    raise ValueError(f"chunk has {rows} k-vectors but batch_count is {batch_count}")
                kvecs = self.create_k_chunk_buffer(chunk.kvecs)
                output_indices: StorageBuffer | None = None
                try:
                    self.run_fixed_rank_bethe_chunk(
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
                    )
                    if output is not None and chunk.output_indices is not None:
                        output_indices = StorageBuffer(
                            self.device,
                            self.queue,
                            label="ebsd:outputIndices",
                            data=chunk.output_indices,
                        )
                        self.writeback_output(
                            workspace.intensities,
                            output_indices,
                            output,
                            batch_count=rows,
                            n_sites=n_sites,
                            output_count=output_count,
                        )
                    sync_device(self.device)
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

    def prescan_bethe_beam_counts_gpu(
        self,
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

        limits = self.device.limits
        max_rows_per_pass = max(1, max_compute_workgroups_per_dimension(self.device))
        metric_buf = metric if isinstance(metric, StorageBuffer) else self.create_metric_buffer(metric)
        owns_metric = not isinstance(metric, StorageBuffer)
        try:
            for row_start in range(0, total_rows, max_rows_per_pass):
                row_count = min(max_rows_per_pass, total_rows - row_start)
                k_chunk = self.create_k_chunk_buffer(kvecs[row_start * 3 : (row_start + row_count) * 3])
                row_counts = self._storage("ebsd:prescanCounts", u32_bytes(row_count * 4), True)
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
                    self._dispatch_with_params(
                        "ebsd:prescanCounts",
                        "ebsd_prescan_counts.wgsl",
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
                    sync_device(self.device)
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
