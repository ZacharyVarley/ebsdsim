"""Per-tile Smith iterative dispatch packing, mode census, and profile stage names."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.batch import DispatchItem
from ebsdsim.gpu.buffers import StorageBuffer

# WebGPU hard limit: workgroups per dimension.
MAX_WG_PER_DIM = 65535

RANK = 16
MAX_ITER = 256
ATOL = 5e-3

_LABEL_TO_STAGE = {
    "hp:score": "excitation_score",
    "hp:topk": "topk_radix",
    "hp:gatherslim": "gather_slim_q",
    "hp:gather": "gather_diagonal",
    "hp:slimq": "smith_iterative_slim_q",
    "hp:smith_iterative": "smith_iterative_shared_solve",
    "hp:inten": "intensity_fused",
}


def grid_capped(total: int, *, wg: int = 256, cap: int = MAX_WG_PER_DIM) -> tuple[int, int, int]:
    """Bounded 1-D launch; grid-strided kernels loop to cover ``total``."""
    need = (int(total) + wg - 1) // wg
    return (max(1, min(need, cap)), 1, 1)


def smith_iterative_code_digest(code: str) -> str:
    """Stable short digest for pipeline cache keys (not PYTHONHASHSEED-salted)."""
    return hashlib.blake2b(code.encode(), digest_size=8).hexdigest()


def pack_implicit_bicg(batch: int, n: int, lu, pref) -> bytes:
    # Matches scratch/other_viewpoints ImplicitBicgstabSmith.dispatch packing.
    return struct.pack(
        "<IIIIfIiIffff",
        batch,
        n,
        RANK,
        MAX_ITER,
        float(ATOL),
        int(lu.table_size),
        int(lu.offset),
        0,
        float(pref[0]),
        float(pref[1]),
        0.0,
        0.0,
    )


def pack_smith_iterative(batch: int, n: int, bl: np.ndarray, lu, pref) -> bytes:
    b = np.asarray(bl, dtype=np.float32).reshape(3, 3)
    return struct.pack(
        "<IIIIfiiiIIfffffffffff",
        batch,
        n,
        RANK,
        MAX_ITER,
        float(ATOL),
        int(lu.table_size),
        int(lu.offset),
        int(lu.stride_h),
        int(lu.stride_k),
        0,
        float(pref[0]),
        float(pref[1]),
        float(b[0, 0]),
        float(b[0, 1]),
        float(b[0, 2]),
        float(b[1, 0]),
        float(b[1, 1]),
        float(b[1, 2]),
        float(b[2, 0]),
        float(b[2, 1]),
        float(b[2, 2]),
    ) + bytes(12)


def count_mode_flags(st: NDArray[np.uint32]) -> tuple[int, int, int, int, int]:
    """Census smith_iterative mode bits: dense / unique-Δ / unique-seg / BiCGSTAB / fail."""
    st = np.asarray(st, dtype=np.uint32)
    tiled = int(np.count_nonzero((st & np.uint32(0x80000000)) != 0))
    unique = int(np.count_nonzero((st & np.uint32(0x10000000)) != 0))
    useg = int(np.count_nonzero((st & np.uint32(0x40000000)) != 0))
    bicg = int(np.count_nonzero((st & np.uint32(0x20000000)) != 0))
    fail = int(np.count_nonzero(st == np.uint32(0xFFFFFFFF)))
    return tiled, unique, useg, bicg, fail


def profile_stages(items: list[DispatchItem]) -> list[tuple[str, list[DispatchItem]]]:
    """One (stage_name, [item]) per dispatch — profile path submits each alone."""
    return [(_LABEL_TO_STAGE.get(it.label, it.label), [it]) for it in items]


@dataclass(frozen=True)
class TileDispatchCtx:
    """Inputs for :func:`build_tile_dispatch` (explicit; no closure over locals)."""

    rows: int
    n_g: int
    n_use: int
    n_w_use: int
    n_sites: int
    code_score: str
    code_topk: str
    code_gather: str
    code_slim: str
    code_gather_slim: str
    code_smith_iterative: str
    code_inten: str
    use_gather_slim: bool
    uses_implicit_bicg: bool
    bethe_c_cutoff: float
    dbdiff_sg_cutoff: float
    bethe_c_strong: float
    bethe_c_weak: float
    mu: float
    amp: float
    bl: NDArray[np.float32]
    lu: Any
    pref: Any
    kbuf: StorageBuffer
    out_idx: StorageBuffer
    ws: SimpleNamespace
    persistent: Any  # PersistentBuffers
    metric: StorageBuffer
    sgh_tables_delta_major: StorageBuffer
    stats: StorageBuffer
    bin_out: StorageBuffer
    intensity_buf: StorageBuffer
    write_global: int
    smith_iterative_scratch: StorageBuffer | None
    smith_iterative_uniq_vals: StorageBuffer | None
    smith_iterative_uniq_meta: StorageBuffer | None


def build_tile_dispatch(ctx: TileDispatchCtx) -> list[DispatchItem]:
    """Pure score→topk→(gather/slim)→smith_iterative→inten packing for one k-tile."""
    rows = ctx.rows
    n_g = ctx.n_g
    n_use = ctx.n_use
    n_w_use = ctx.n_w_use
    n_sites = ctx.n_sites
    lu = ctx.lu
    pref = ctx.pref
    ws = ctx.ws
    persistent = ctx.persistent

    items: list[DispatchItem] = [
        DispatchItem(
            key="hp:score",
            code=ctx.code_score,
            params_data=struct.pack(
                "<IIIIffff",
                rows,
                n_g,
                0,
                0,
                ctx.bethe_c_cutoff,
                ctx.dbdiff_sg_cutoff,
                ctx.bethe_c_strong,
                ctx.bethe_c_weak,
            ),
            resources=[
                ctx.kbuf,
                persistent.hkl,
                ctx.metric,
                persistent.coupling,
                persistent.reflection_dbdiff,
                ws.sg,
                ws.scores,
                ws.candidate_mask,
            ],
            workgroups=grid_capped(rows * n_g),
            label="hp:score",
            n_storage_bindings=8,
            uniform_size=32,
        ),
        DispatchItem(
            key=f"hp:topk:n{n_use}",
            code=ctx.code_topk,
            params_data=struct.pack("<4I", rows, n_g, n_use, n_w_use),
            resources=[
                ws.scores,
                ws.candidate_mask,
                ws.idx_a,
                ws.idx_w,
                ws.selected_flags,
            ],
            workgroups=(rows, 1, 1),
            label="hp:topk",
            n_storage_bindings=5,
            uniform_size=16,
        ),
    ]
    if ctx.use_gather_slim:
        items.append(
            DispatchItem(
                key=f"hp:gatherslim:n{n_use}",
                code=ctx.code_gather_slim,
                params_data=struct.pack(
                    "<4I4i8f",
                    rows,
                    n_use,
                    int(lu.table_size),
                    n_g,
                    int(lu.stride_h),
                    int(lu.stride_k),
                    int(lu.offset),
                    0,  # bloch gather mode
                    float(pref[0]),
                    float(pref[1]),
                    float(ctx.mu),
                    1e-12,
                    float(lu.diag_imag),
                    float(lu.mlambda),
                    0.0,
                    0.0,
                ),
                resources=[
                    ws.idx_a,
                    ws.sg,
                    persistent.hkl,
                    persistent.diff_table,
                    ws.d_a,
                    ws.q_values,
                    ws.e0,
                ],
                workgroups=(rows, 1, 1),
                label="hp:gatherslim",
                n_storage_bindings=7,
                uniform_size=64,
            )
        )
    else:
        items.extend(
            [
                DispatchItem(
                    key="hp:gather",
                    code=ctx.code_gather,
                    params_data=struct.pack(
                        "<IIIIffff",
                        rows,
                        n_g,
                        n_use,
                        0,
                        float(lu.diag_imag),
                        float(lu.mlambda),
                        0.0,
                        0.0,
                    ),
                    resources=[ws.sg, ws.idx_a, ws.d_a],
                    workgroups=grid_capped(rows * n_use),
                    label="hp:gather",
                    n_storage_bindings=3,
                    uniform_size=32,
                ),
                DispatchItem(
                    key=f"hp:slimq:n{n_use}",
                    code=ctx.code_slim,
                    params_data=struct.pack(
                        "<4I4i4f",
                        rows,
                        n_use,
                        int(lu.table_size),
                        0,
                        int(lu.stride_h),
                        int(lu.stride_k),
                        int(lu.offset),
                        0,
                        float(pref[0]),
                        float(pref[1]),
                        float(ctx.mu),
                        1e-12,
                    ),
                    resources=[
                        ws.idx_a,
                        ws.d_a,
                        persistent.hkl,
                        persistent.diff_table,
                        ws.q_values,
                        ws.e0,
                    ],
                    workgroups=(rows, 1, 1),
                    label="hp:slimq",
                    n_storage_bindings=6,
                    uniform_size=48,
                ),
            ]
        )
    items.extend(
        [
            DispatchItem(
                key=f"hp:smith_iterative:{smith_iterative_code_digest(ctx.code_smith_iterative)}:n{n_use}",
                code=ctx.code_smith_iterative,
                params_data=(
                    pack_implicit_bicg(rows, n_use, lu, pref)
                    if ctx.uses_implicit_bicg
                    else pack_smith_iterative(rows, n_use, ctx.bl, lu, pref)
                ),
                resources=(
                    [
                        persistent.hkl_hash,
                        persistent.diff_table,
                        ws.idx_a,
                        ws.d_a,
                        ws.e0,
                        ws.q_values,
                        ws.w_stack,
                        ctx.stats,
                    ]
                    if ctx.uses_implicit_bicg
                    else [
                        persistent.hkl,
                        persistent.diff_table,
                        ws.idx_a,
                        ws.d_a,
                        ws.e0,
                        ws.q_values,
                        ws.w_stack,
                        ctx.stats,
                    ]
                    + ([ctx.smith_iterative_scratch] if ctx.smith_iterative_scratch is not None else [])
                    + ([ctx.smith_iterative_uniq_vals] if ctx.smith_iterative_uniq_vals is not None else [])
                    + ([ctx.smith_iterative_uniq_meta] if ctx.smith_iterative_uniq_meta is not None else [])
                ),
                workgroups=(rows, 1, 1),
                label="hp:smith_iterative",
                n_storage_bindings=(
                    8
                    if ctx.uses_implicit_bicg
                    else (
                        8
                        + (1 if ctx.smith_iterative_scratch is not None else 0)
                        + (1 if ctx.smith_iterative_uniq_vals is not None else 0)
                        + (1 if ctx.smith_iterative_uniq_meta is not None else 0)
                    )
                ),
                uniform_size=48 if ctx.uses_implicit_bicg else 96,
            ),
            DispatchItem(
                key=f"hp:inten:n{n_use}:s{n_sites}",
                code=ctx.code_inten,
                params_data=struct.pack(
                    "<4I4i4I4f",
                    rows,
                    n_use,
                    RANK,
                    n_sites,
                    int(lu.stride_h),
                    int(lu.stride_k),
                    int(lu.offset),
                    int(lu.table_size),
                    n_sites,
                    int(ctx.write_global),
                    1,  # delta-major Sgh
                    0,
                    float(ctx.amp),
                    0.0,
                    0.0,
                    0.0,
                ),
                resources=[
                    ws.idx_a,
                    persistent.hkl,
                    ws.w_stack,
                    ctx.sgh_tables_delta_major,
                    ctx.out_idx,
                    ctx.intensity_buf,
                    ctx.bin_out,
                ],
                workgroups=(rows, 1, 1),
                label="hp:inten",
                n_storage_bindings=7,
                uniform_size=64,
            ),
        ]
    )
    return items
