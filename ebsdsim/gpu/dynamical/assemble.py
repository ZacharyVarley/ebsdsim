"""Dense V / Q assemble, GEMM/GEMV, sigma/geff (GPU dynamical layer).

Free functions take the kernels façade (device/queue/pipelines/LU) as their first argument.
"""
from __future__ import annotations

import struct

from ebsdsim.gpu.buffers import StorageBuffer, c64_bytes, f32_bytes, u32_bytes
from ebsdsim.gpu.dynamical.workspace import (
    C64,
    BetheMode,
    FixedRankWorkspace,
    _mode_id,
)
from ebsdsim.gpu.pipelines import workgroups_1d


def gemm_c64(
    kernels,
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
    kernels._dispatch_with_params(
        "ebsd:gemmC64",
        "dynamical/gemm_c64_batched.wgsl",
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
    kernels,
    geff: StorageBuffer,
    w: StorageBuffer,
    out: StorageBuffer,
    *,
    batch_count: int,
    n: int,
    q_offset: int = 2,
) -> None:
    params = struct.pack("<IIIIIIII", batch_count, n, n * n, n, n, q_offset, 0, 0)
    kernels._dispatch_with_params(
        "ebsd:gemvC64",
        "dynamical/gemv_c64_batched.wgsl",
        params,
        [geff, w, out],
        [c64_bytes(batch_count * n * n), c64_bytes(batch_count * n), c64_bytes(batch_count * n)],
        (n, batch_count, 1),
        "ebsd:gemvC64",
    )


def lookup_submatrix(
    kernels,
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
    kernels._dispatch_with_params(
        "ebsd:lookupSubmatrix",
        "dynamical/lookup_submatrix_c64.wgsl",
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
    kernels,
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
    kernels._dispatch_with_params(
        "ebsd:hashDiff",
        "dynamical/hash_diff_u32.wgsl",
        params,
        [ws.idx_a, hkl_hash, ws.dh],
        [u32_bytes(batch_count * n), None, u32_bytes(batch_count * n * n)],
        workgroups_1d(batch_count * n * n, 256),
        "ebsd:hashDiff",
        uniform_size=32,
    )


def gather_diagonal(
    kernels,
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
    kernels._dispatch_with_params(
        "ebsd:gatherDiagonal",
        "dynamical/gather_diagonal_c64.wgsl",
        params,
        [sg, idx, out],
        [f32_bytes(batch_count * n_g), u32_bytes(batch_count * n), c64_bytes(batch_count * n)],
        workgroups_1d(batch_count * n, 256),
        "ebsd:gatherDiagonal",
        uniform_size=32,
    )


def weight_bethe_vws(
    kernels,
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
    kernels._dispatch_with_params(
        "ebsd:betheWeightVws",
        "dynamical/bethe_weight_vws_c64.wgsl",
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
    kernels,
    ws: FixedRankWorkspace,
    *,
    batch_count: int,
    n_a: int,
    n_w: int,
    eps: float = 1e-7,
) -> None:
    if n_w == 0:
        return
    kernels.weight_bethe_vws(ws, batch_count=batch_count, n_a=n_a, n_w=n_w, eps=eps)
    kernels.gemm_c64(
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
    kernels,
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
    kernels._dispatch_with_params(
        "ebsd:assembleGeffQ",
        "dynamical/assemble_geff_q.wgsl",
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

