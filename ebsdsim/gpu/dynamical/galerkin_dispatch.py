"""Per-tile Galerkin dispatch packing, Lyapunov strategy, and uniform packs.

Per k-tile the chain is:

1. Bethe score → top-k → gather/slim (shared with Smith)
2. Galerkin solve — rational Krylov + MGS; writes ``H``, ``b`` (no intensity)
3. Projected Lyapunov — shared Gauss–Jordan (m≤6) or implicit CGNE (7..16)
4. Expand ``W = Q F``
5. Fused intensity on ``w_out`` with ``rank = out_rank``

Smith is only the basis generator; Galerkin is the RKSM coefficient choice
(Druskin–Simoncini, single repeated pole).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.batch import DispatchItem, ResourceBinding
from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.gpu.dynamical.smith_dispatch import (
    MAX_WG_PER_DIM,
    count_mode_flags,
    grid_capped,
    pack_smith,
    profile_stages,
    smith_code_digest,
)

# Default / ceiling Krylov ranks for the Galerkin path.
KRYLOV_RANK = 8
OUT_RANK = 8
MAX_RANK = 16
SHARED_LYAPUNOV_MAX_M = 6
IMPLICIT_LYAPUNOV_MAX_M = 16

LyapunovStrategy = Literal["shared", "implicit"]

_LABEL_TO_STAGE = {
    "hp:score": "excitation_score",
    "hp:topk": "topk_radix",
    "hp:gatherslim": "gather_slim_q",
    "hp:gather": "gather_diagonal",
    "hp:slimq": "smith_slim_q",
    "hp:galerkin": "galerkin_shared_solve",
    "hp:lyap": "galerkin_lyapunov",
    "hp:expand": "galerkin_expand",
    "hp:inten": "intensity_fused",
}


def lyapunov_strategy(krylov_rank: int) -> LyapunovStrategy:
    """Select on-device Lyapunov kernel from Krylov rank."""
    m = int(krylov_rank)
    if m < 1 or m > MAX_RANK:
        raise ValueError(f"krylov_rank={m} out of range [1, {MAX_RANK}]")
    if m <= SHARED_LYAPUNOV_MAX_M:
        return "shared"
    return "implicit"


def lyapunov_f_ld(krylov_rank: int) -> int:
    """Leading dimension / MAX_M for the selected Lyapunov path."""
    return (
        SHARED_LYAPUNOV_MAX_M
        if lyapunov_strategy(krylov_rank) == "shared"
        else IMPLICIT_LYAPUNOV_MAX_M
    )


def pack_lyapunov(batch: int, h_ld: int, f_ld: int) -> bytes:
    return struct.pack("<4I", int(batch), int(h_ld), int(f_ld), 0)


def pack_expand(
    batch: int,
    n: int,
    *,
    in_rank: int,
    out_rank: int,
    max_rank: int,
    f_col_stride: int,
) -> bytes:
    return struct.pack(
        "<8I",
        int(batch),
        int(n),
        int(in_rank),
        int(out_rank),
        int(max_rank),
        int(f_col_stride),
        0,
        0,
    )


def galerkin_profile_stages(
    items: list[DispatchItem],
) -> list[tuple[str, list[DispatchItem]]]:
    """One (stage_name, [item]) per dispatch — profile path submits each alone."""
    return [(_LABEL_TO_STAGE.get(it.label, it.label), [it]) for it in items]


@dataclass(frozen=True)
class GalerkinTileDispatchCtx:
    """Inputs for :func:`build_galerkin_tile_dispatch`."""

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
    code_galerkin: str
    code_lyapunov: str
    code_expand: str
    code_inten: str
    use_gather_slim: bool
    bethe_c_cutoff: float
    dbdiff_sg_cutoff: float
    bethe_c_strong: float
    bethe_c_weak: float
    mu: float
    amp: float
    krylov_rank: int
    out_rank: int
    max_rank: int
    f_ld: int
    bl: NDArray[np.float32]
    lu: Any
    pref: Any
    kbuf: StorageBuffer
    out_idx: StorageBuffer
    ws: SimpleNamespace
    persistent: Any
    metric: StorageBuffer
    sgh_tables_delta_major: StorageBuffer
    stats: StorageBuffer
    bin_out: StorageBuffer
    intensity_buf: StorageBuffer
    write_global: int
    smith_scratch: StorageBuffer | None
    smith_uniq_vals: StorageBuffer | None
    smith_uniq_meta: StorageBuffer | None
    galerkin_h: StorageBuffer
    galerkin_b: StorageBuffer
    galerkin_f: StorageBuffer
    w_out: StorageBuffer


def build_galerkin_tile_dispatch(ctx: GalerkinTileDispatchCtx) -> list[DispatchItem]:
    """score→topk→(gather/slim)→galerkin→lyapunov→expand→inten for one k-tile."""
    rows = ctx.rows
    n_g = ctx.n_g
    n_use = ctx.n_use
    n_w_use = ctx.n_w_use
    n_sites = ctx.n_sites
    lu = ctx.lu
    pref = ctx.pref
    ws = ctx.ws
    persistent = ctx.persistent
    krylov = int(ctx.krylov_rank)
    out_rank = int(ctx.out_rank)

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
                    0,
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

    # Galerkin solve: bindings 1–11 contiguous, 12/13 reserved, 14/15 = H/b.
    solve_resources: list = [
        persistent.hkl,
        persistent.diff_table,
        ws.idx_a,
        ws.d_a,
        ws.e0,
        ws.q_values,
        ws.w_stack,
        ctx.stats,
    ]
    if ctx.smith_scratch is not None:
        solve_resources.append(ctx.smith_scratch)
    if ctx.smith_uniq_vals is not None:
        solve_resources.append(ctx.smith_uniq_vals)
    if ctx.smith_uniq_meta is not None:
        solve_resources.append(ctx.smith_uniq_meta)
    solve_resources.extend(
        [
            ResourceBinding(ctx.galerkin_h, binding=14),
            ResourceBinding(ctx.galerkin_b, binding=15),
        ]
    )
    n_solve_storage = len(solve_resources)

    items.extend(
        [
            DispatchItem(
                key=(
                    f"hp:galerkin:{smith_code_digest(ctx.code_galerkin)}"
                    f":n{n_use}:r{krylov}"
                ),
                code=ctx.code_galerkin,
                params_data=pack_smith(
                    rows, n_use, ctx.bl, lu, pref, rank=krylov
                ),
                resources=solve_resources,
                workgroups=(rows, 1, 1),
                label="hp:galerkin",
                n_storage_bindings=n_solve_storage,
                uniform_size=96,
            ),
            DispatchItem(
                key=f"hp:lyap:m{krylov}:f{ctx.f_ld}",
                code=ctx.code_lyapunov,
                params_data=pack_lyapunov(rows, krylov, ctx.f_ld),
                resources=[
                    ctx.galerkin_h,
                    ctx.galerkin_b,
                    ctx.stats,
                    ctx.galerkin_f,
                ],
                workgroups=(rows, 1, 1),
                label="hp:lyap",
                n_storage_bindings=4,
                uniform_size=16,
            ),
            DispatchItem(
                key=f"hp:expand:n{n_use}:in{krylov}:out{out_rank}",
                code=ctx.code_expand,
                params_data=pack_expand(
                    rows,
                    n_use,
                    in_rank=krylov,
                    out_rank=out_rank,
                    # Match on-device Lyapunov packing: F is [batch][f_ld][f_ld].
                    max_rank=ctx.f_ld,
                    f_col_stride=ctx.f_ld,
                ),
                resources=[
                    ws.w_stack,
                    ctx.galerkin_f,
                    ctx.stats,
                    ctx.w_out,
                ],
                workgroups=(rows, 1, 1),
                label="hp:expand",
                n_storage_bindings=4,
                uniform_size=32,
            ),
            DispatchItem(
                key=f"hp:inten:n{n_use}:s{n_sites}:r{out_rank}",
                code=ctx.code_inten,
                params_data=struct.pack(
                    "<4I4i4I4f",
                    rows,
                    n_use,
                    out_rank,
                    n_sites,
                    int(lu.stride_h),
                    int(lu.stride_k),
                    int(lu.offset),
                    int(lu.table_size),
                    n_sites,
                    int(ctx.write_global),
                    1,
                    0,
                    float(ctx.amp),
                    0.0,
                    0.0,
                    0.0,
                ),
                resources=[
                    ws.idx_a,
                    persistent.hkl,
                    ctx.w_out,
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


# Re-exports used by galerkin.py / tests.
__all__ = [
    "KRYLOV_RANK",
    "OUT_RANK",
    "MAX_RANK",
    "MAX_WG_PER_DIM",
    "SHARED_LYAPUNOV_MAX_M",
    "IMPLICIT_LYAPUNOV_MAX_M",
    "GalerkinTileDispatchCtx",
    "build_galerkin_tile_dispatch",
    "count_mode_flags",
    "galerkin_profile_stages",
    "grid_capped",
    "lyapunov_f_ld",
    "lyapunov_strategy",
    "pack_expand",
    "pack_lyapunov",
    "profile_stages",
]
