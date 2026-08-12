"""Optimized Galerkin (RKSM) dynamical loop — persistent buffers, one submit per chunk.

Per k-vector the path builds the same rational Krylov subspace as Smith
(Cayley / BiCGSTAB, repeated pole at q), then solves the projected Lyapunov
equation on-device and expands ``W = Q F`` for intensity. Smith is only the
basis generator; Galerkin is the optimal-over-subspace coefficient choice.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.batch import PersistentSubmitter
from ebsdsim.gpu.buffers import StorageBuffer, c64_bytes, f32_bytes, u32_bytes
from ebsdsim.gpu.device import sync_device
from ebsdsim.gpu.dynamical.galerkin_dispatch import (
    KRYLOV_RANK,
    MAX_RANK,
    MAX_WG_PER_DIM,
    OUT_RANK,
    GalerkinTileDispatchCtx,
    build_galerkin_tile_dispatch,
    count_mode_flags,
    galerkin_profile_stages,
    lyapunov_f_ld,
    lyapunov_strategy,
)
from ebsdsim.gpu.dynamical.galerkin_shader import (
    MAX_N_GALERKIN,
    load_galerkin_expand,
    load_galerkin_lyapunov_implicit,
    load_galerkin_lyapunov_shared,
    load_galerkin_solve_shader,
)
from ebsdsim.gpu.dynamical.smith_prep import (
    SmithCrystalPrep,
    VoltageBin,
    activate_voltage_bin,
)
from ebsdsim.gpu.pipelines import load_wgsl
from ebsdsim.lambert.kgrid import chunk_k_vectors
from ebsdsim.physics.prescan import relative_normalized_view_change

__all__ = [
    "MAX_N_GALERKIN",
    "TILE_VRAM_BUDGET_BYTES",
    "KRYLOV_RANK",
    "OUT_RANK",
    "MAX_RANK",
    "auto_tile_k",
    "auto_queue_depth",
    "load_galerkin_solve_shader",
    "lyapunov_strategy",
    "run_bins_dynamical",
]

TILE_VRAM_BUDGET_BYTES = 1_500_000_000
NOMINAL_SOLVE_ITERS = 96
SUBMIT_WORK_BUDGET = 4_000_000_000
MAX_QUEUE_DEPTH = 64


def _per_k_workspace_bytes(
    n_g: int,
    n: int,
    n_w: int,
    n_sites: int,
    *,
    krylov_rank: int,
    out_rank: int,
    max_rank: int,
) -> int:
    # Slim workspace + Galerkin projection / expand buffers (once per tile width).
    return (
        n_g * 16
        + n * (4 + 8 + 8 + krylov_rank * 8 + out_rank * 8)  # idx/d/e0/w_stack/w_out
        + max(1, n_w) * 4
        + n_sites * 4
        + 16
        + krylov_rank * krylov_rank * 8  # galerkin_h
        + krylov_rank * 8  # galerkin_b
        + max_rank * max_rank * 8  # galerkin_f upper bound (f_ld ≤ max_rank)
    )


def auto_tile_k(
    n_g: int,
    n: int,
    n_w: int,
    n_sites: int,
    *,
    krylov_rank: int = KRYLOV_RANK,
    out_rank: int = OUT_RANK,
    max_rank: int = MAX_RANK,
    budget: int = TILE_VRAM_BUDGET_BYTES,
    cap: int = MAX_WG_PER_DIM,
    submit_work_budget: int | None = None,
) -> int:
    """Largest k-tile whose workspace fits ``budget`` (min 256, capped)."""
    per_k = _per_k_workspace_bytes(
        n_g,
        n,
        n_w,
        n_sites,
        krylov_rank=krylov_rank,
        out_rank=out_rank,
        max_rank=max_rank,
    )
    tile = min(cap, max(256, budget // max(1, per_k)))
    work = SUBMIT_WORK_BUDGET if submit_work_budget is None else int(submit_work_budget)
    if work > 0:
        tile = min(tile, max(16, work // max(1, NOMINAL_SOLVE_ITERS * n * n)))
    return int(tile)


def auto_queue_depth(
    n_g: int,
    n: int,
    n_w: int,
    n_sites: int,
    *,
    tile: int,
    krylov_rank: int = KRYLOV_RANK,
    out_rank: int = OUT_RANK,
    max_rank: int = MAX_RANK,
    budget: int = TILE_VRAM_BUDGET_BYTES,
    cap: int = MAX_WG_PER_DIM,
    min_depth: int = 4,
) -> int:
    """How many k-tiles to pipeline before ``sync_device`` to fill VRAM."""
    if tile <= 0:
        return min_depth
    per_k = _per_k_workspace_bytes(
        n_g,
        n,
        n_w,
        n_sites,
        krylov_rank=krylov_rank,
        out_rank=out_rank,
        max_rank=max_rank,
    )
    vram_tile = min(cap, max(256, budget // max(1, per_k)))
    depth = max(min_depth, (vram_tile + tile - 1) // tile)
    return min(depth, MAX_QUEUE_DEPTH)


class _Prof:
    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    def add(self, name: str, dt: float) -> None:
        self.t[name] = self.t.get(name, 0.0) + float(dt)


def _pack_k_vec4(kvecs: NDArray[np.float32]) -> NDArray[np.float32]:
    src = np.asarray(kvecs, dtype=np.float32).reshape(-1)
    count = src.size // 3
    out = np.zeros((count, 4), dtype=np.float32)
    out[:, :3] = src.reshape(count, 3)
    return out.reshape(-1)


def _slim_workspace(
    kernels,
    *,
    B: int,
    n_g: int,
    n: int,
    n_w: int,
    n_sites: int,
    krylov_rank: int,
) -> SimpleNamespace:
    """Bethe + Smith-basis workspace (Q lives in ``w_stack``)."""
    dev, q = kernels.device, kernels.queue

    def sb(label: str, nbytes: int, *, copy_src: bool = False) -> StorageBuffer:
        return StorageBuffer(
            dev, q, label=label, byte_length=nbytes, copy_src=copy_src, copy_dst=True
        )

    return SimpleNamespace(
        sg=sb("hp:sg", f32_bytes(B * n_g)),
        scores=sb("hp:scores", f32_bytes(B * n_g)),
        candidate_mask=sb("hp:cmask", u32_bytes(B * n_g)),
        selected_flags=sb("hp:flags", u32_bytes(B * n_g)),
        idx_a=sb("hp:idxA", u32_bytes(B * n)),
        idx_w=sb("hp:idxW", u32_bytes(max(1, B * n_w))),
        d_a=sb("hp:dA", c64_bytes(B * n)),
        e0=sb("hp:e0", c64_bytes(B * n)),
        q_values=sb("hp:q", f32_bytes(B * 4)),
        w_stack=sb("hp:w", c64_bytes(B * n * krylov_rank)),
        intensities=sb("hp:I", f32_bytes(B * n_sites), copy_src=True),
    )


def _galerkin_projection_buffers(
    kernels,
    *,
    B: int,
    n: int,
    krylov_rank: int,
    out_rank: int,
    f_ld: int,
) -> SimpleNamespace:
    """Allocate H/b/F/w_out once per workspace (not per tile)."""
    dev, q = kernels.device, kernels.queue

    def sb(label: str, nbytes: int) -> StorageBuffer:
        return StorageBuffer(dev, q, label=label, byte_length=nbytes, copy_dst=True)

    return SimpleNamespace(
        galerkin_h=sb("hp:galH", c64_bytes(B * krylov_rank * krylov_rank)),
        galerkin_b=sb("hp:galB", c64_bytes(B * krylov_rank)),
        galerkin_f=sb("hp:galF", c64_bytes(B * f_ld * f_ld)),
        w_out=sb("hp:wOut", c64_bytes(B * n * out_rank)),
    )


def _destroy_ns(ns: SimpleNamespace) -> None:
    for v in vars(ns).values():
        if hasattr(v, "destroy"):
            v.destroy()


def run_bins_dynamical(
    prep: SmithCrystalPrep,
    bins: list[VoltageBin],
    *,
    chunk: int,
    code_topk: str,
    code_slim: str,
    code_galerkin: str,
    code_inten: str,
    code_lyapunov: str | None = None,
    code_expand: str | None = None,
    profile: bool = False,
    progress_every: int = 0,
    max_chunks: int | None = None,
    code_score: str | None = None,
    code_gather: str | None = None,
    collect_stats: bool = False,
    inter_tile_pause_s: float = 0.0,
    on_chunk_done: Callable[[dict], None] | None = None,
    k_order: NDArray[np.uint32] | None = None,
    first_act: dict | None = None,
    pre_activated: list[dict] | None = None,
    queue_depth: int = 4,
    fuse_gather_slim: bool = True,
    relative_image_stop: float = 0.01,
    krylov_rank: int = KRYLOV_RANK,
    out_rank: int | None = None,
    max_rank: int = MAX_RANK,
) -> dict:
    """Integrate voltage bins with the Galerkin RKSM path."""
    if chunk < 8:
        raise ValueError(f"tile size chunk={chunk} too small; use >= 8")
    if queue_depth < 1:
        raise ValueError(f"queue_depth={queue_depth} must be >= 1")
    krylov_rank = int(krylov_rank)
    if krylov_rank < 1 or krylov_rank > max_rank:
        raise ValueError(f"krylov_rank={krylov_rank} out of range [1, {max_rank}]")
    out_rank = int(out_rank) if out_rank is not None else krylov_rank
    if out_rank < 1 or out_rank > max_rank:
        raise ValueError(f"out_rank={out_rank} out of range [1, {max_rank}]")
    if out_rank > krylov_rank:
        raise ValueError(f"out_rank={out_rank} exceeds krylov_rank={krylov_rank}")
    max_rank = int(max_rank)
    if max_rank < krylov_rank or max_rank > MAX_RANK:
        raise ValueError(f"max_rank={max_rank} must be in [{krylov_rank}, {MAX_RANK}]")

    f_ld = lyapunov_f_ld(krylov_rank)
    strategy = lyapunov_strategy(krylov_rank)

    prof = _Prof()
    bin_prep_timings: list[dict[str, float]] = []
    kernels = prep.kernels
    device, queue = prep.device, prep.queue
    pipelines = prep.pipelines
    bl = prep.bl
    n_sites = prep.n_sites
    submitter = PersistentSubmitter(pipelines, queue)

    if code_score is None:
        code_score = load_wgsl("dynamical/excitation_score_gridstride.wgsl")
    if code_gather is None:
        code_gather = load_wgsl("dynamical/gather_diagonal_c64_gridstride.wgsl")
    if code_expand is None:
        code_expand = load_galerkin_expand()
    if code_lyapunov is None:
        code_lyapunov = (
            load_galerkin_lyapunov_shared()
            if strategy == "shared"
            else load_galerkin_lyapunov_implicit()
        )
    use_gather_slim = bool(fuse_gather_slim)
    code_gather_slim = ""
    if use_gather_slim:
        code_gather_slim = load_wgsl("dynamical/gather_slim_q.wgsl")
    if profile or progress_every > 0 or on_chunk_done is not None or collect_stats:
        queue_depth = 1
    queue_depth = int(queue_depth)

    first = first_act if first_act is not None else activate_voltage_bin(prep, bins[0])
    if first_act is None:
        bin_prep_timings.append(first["timings"])
    elif pre_activated is None:
        bin_prep_timings.append(first.get("timings", {}))
    n = int(first["n_strong"])
    n_w = int(first["n_weak"])
    n_g = int(first["n_g"])
    n_k = first["kvecs"].size // 3
    if pre_activated is not None:
        if len(pre_activated) != len(bins):
            raise ValueError("pre_activated length must match bins")
        n = max(int(a["n_strong"]) for a in pre_activated)
        n_w = max(int(a["n_weak"]) for a in pre_activated)
        bin_prep_timings = [dict(a.get("timings", {})) for a in pre_activated]
    if n > MAX_N_GALERKIN:
        raise RuntimeError(f"n_strong={n} > galerkin MAX_N={MAX_N_GALERKIN}")
    uses_unique_bits = "uniq_bits" in code_galerkin
    uses_uniq_vals = "uniq_vals" in code_galerkin
    uses_uniq_meta = "uniq_meta" in code_galerkin
    chunk = min(int(chunk), int(n_k), MAX_WG_PER_DIM)
    _uv_stride = 8 if "uniq_vals: array<vec2<f32>>" in code_galerkin else 4
    if uses_uniq_vals or uses_uniq_meta:
        max_chunk_uniq = 1_800_000_000 // (16384 * _uv_stride)
        if chunk > max_chunk_uniq:
            chunk = int(max_chunk_uniq)

    ws = _slim_workspace(
        kernels,
        B=chunk,
        n_g=n_g,
        n=n,
        n_w=max(n_w, 1),
        n_sites=n_sites,
        krylov_rank=krylov_rank,
    )
    gal_bufs = _galerkin_projection_buffers(
        kernels,
        B=chunk,
        n=n,
        krylov_rank=krylov_rank,
        out_rank=out_rank,
        f_ld=f_ld,
    )
    stats = StorageBuffer(
        device, queue, label="hp:stats", data=np.zeros(chunk * 2, dtype=np.uint32), copy_src=True
    )
    smith_scratch = None
    smith_uniq_vals = None
    smith_uniq_meta = None
    _max_uwords = 2048
    if uses_unique_bits:
        smith_scratch = StorageBuffer(
            device,
            queue,
            label="hp:uniqueBitsetAtomic",
            byte_length=chunk * _max_uwords * 4,
            copy_dst=True,
        )
    if uses_uniq_vals:
        smith_uniq_vals = StorageBuffer(
            device,
            queue,
            label="hp:uniqueVals",
            byte_length=chunk * 16384 * _uv_stride,
            copy_dst=True,
        )
    if uses_uniq_meta:
        smith_uniq_meta = StorageBuffer(
            device,
            queue,
            label="hp:uniqueMeta",
            byte_length=chunk * _max_uwords * 2 * 4,
            copy_dst=True,
        )
    out_idx = StorageBuffer(
        device, queue, label="hp:oi", data=np.zeros(chunk, dtype=np.uint32), copy_dst=True
    )
    kbuf = StorageBuffer(
        device,
        queue,
        label="hp:k",
        data=np.zeros(chunk * 4, dtype=np.float32),
        copy_dst=True,
    )
    output = StorageBuffer(
        device,
        queue,
        label="hp:out",
        byte_length=n_k * n_sites * 4,
        copy_src=True,
        copy_dst=True,
    )
    output.write(np.zeros(n_k * n_sites, dtype=np.float32))
    inten_ring = [
        StorageBuffer(
            device,
            queue,
            label=f"hp:Iring{i}",
            byte_length=f32_bytes(chunk * n_sites),
            copy_src=True,
            copy_dst=True,
        )
        for i in range(queue_depth)
    ]
    n_bins = len(bins)
    bin_weighted = np.zeros((n_bins, n_k, n_sites), dtype=np.float32)
    bin_weights = np.asarray([float(b.energy_weight) for b in bins], dtype=np.float64)

    tiled_count = 0
    unique_count = 0
    useg_count = 0
    bicg_count = 0
    fail_count = 0
    dyn_wall = 0.0
    k_done = 0
    k_solved = 0
    mode_flags_parts: list[np.ndarray] = []
    bin_dyn_timings: list[float] = []
    want_stats = bool(collect_stats or profile or progress_every > 0)
    integrated_sum_view = np.zeros(n_k, dtype=np.float64)
    last_relative_change = float("inf")
    stopped_by_relative_change = False
    bins_run = 0

    def ensure_workspace(n_s: int, n_wk: int) -> None:
        nonlocal ws, gal_bufs, n, n_w
        if n_s <= n and n_wk <= n_w:
            return
        if n_s > MAX_N_GALERKIN:
            raise RuntimeError(f"n_strong={n_s} > galerkin MAX_N={MAX_N_GALERKIN}")
        _destroy_ns(ws)
        _destroy_ns(gal_bufs)
        n = max(n, n_s)
        n_w = max(n_w, n_wk)
        ws = _slim_workspace(
            kernels,
            B=chunk,
            n_g=n_g,
            n=n,
            n_w=max(n_w, 1),
            n_sites=n_sites,
            krylov_rank=krylov_rank,
        )
        gal_bufs = _galerkin_projection_buffers(
            kernels,
            B=chunk,
            n=n,
            krylov_rank=krylov_rank,
            out_rank=out_rank,
            f_ld=f_ld,
        )
        submitter._bind_groups.clear()

    try:
        saw_weighted_bin = False
        saw_intensity = False
        for bi, vbin in enumerate(bins):
            if pre_activated is not None:
                act = pre_activated[bi]
                t_sw0 = time.perf_counter()
                need_swap = (
                    prep._current_vkv != float(vbin.voltage_kv)
                    or prep._current_lookup is not act["lookup"]
                )
                if need_swap:
                    prep.resident.update(act["lookup"])
                    prep._current_vkv = float(vbin.voltage_kv)
                    prep._current_lookup = act["lookup"]
                    if vbin.next_voltage_kv is not None:
                        prep.prefetcher.prefetch(vbin.next_voltage_kv)
                    bin_prep_timings[bi] = {
                        "lookup_get": 0.0,
                        "persistent_update": time.perf_counter() - t_sw0,
                        "kvecs": 0.0,
                        "prescan": 0.0,
                        "reused_lookup": True,
                    }
                else:
                    bin_prep_timings[bi] = {
                        "lookup_get": 0.0,
                        "persistent_update": 0.0,
                        "kvecs": 0.0,
                        "prescan": 0.0,
                        "reused_lookup": True,
                    }
            elif bi == 0:
                act = first
            else:
                act = activate_voltage_bin(prep, vbin)
                bin_prep_timings.append(act["timings"])

            ensure_workspace(int(act["n_strong"]), int(act["n_weak"]))
            n_bin = int(act["n_strong"])
            n_w_bin = max(int(act["n_weak"]), 1)
            lu = act["lookup"]
            pref = lu.prefactor
            mu = float(act["mu"])
            amp = float(act["amplitude"])
            kvecs = act["kvecs"]
            persistent = prep.resident.buffers
            if k_order is None:
                out_order = np.arange(n_k, dtype=np.uint32)
                kvecs_run = kvecs
            else:
                out_order = np.asarray(k_order, dtype=np.uint32).reshape(-1)
                if out_order.size != n_k:
                    raise ValueError(f"k_order length {out_order.size} != n_k={n_k}")
                kvecs_run = (
                    np.asarray(kvecs, dtype=np.float32).reshape(n_k, 3)[out_order].reshape(-1)
                )
            bin_out = output

            chunks = list(chunk_k_vectors(kvecs_run, out_order, chunk))
            if max_chunks is not None:
                chunks = chunks[: max(0, int(max_chunks))]

            def one_chunk(ch, *, do_profile: bool, inten_buf: StorageBuffer) -> None:
                nonlocal tiled_count, unique_count, useg_count, bicg_count, fail_count, k_done
                rows = ch.kvecs.size // 3
                kbuf.write(_pack_k_vec4(ch.kvecs))
                out_idx.write(np.asarray(ch.output_indices, dtype=np.uint32))
                n_use = n_bin
                n_w_use = n_w_bin

                def mark(name: str, t0: float) -> float:
                    if do_profile:
                        sync_device(device)
                        prof.add(name, time.perf_counter() - t0)
                        return time.perf_counter()
                    return t0

                t = time.perf_counter()
                items = build_galerkin_tile_dispatch(
                    GalerkinTileDispatchCtx(
                        rows=rows,
                        n_g=n_g,
                        n_use=n_use,
                        n_w_use=n_w_use,
                        n_sites=n_sites,
                        code_score=code_score,
                        code_topk=code_topk,
                        code_gather=code_gather,
                        code_slim=code_slim,
                        code_gather_slim=code_gather_slim,
                        code_galerkin=code_galerkin,
                        code_lyapunov=code_lyapunov,
                        code_expand=code_expand,
                        code_inten=code_inten,
                        use_gather_slim=use_gather_slim,
                        bethe_c_cutoff=prep.bethe_c_cutoff,
                        dbdiff_sg_cutoff=prep.dbdiff_sg_cutoff,
                        bethe_c_strong=prep.bethe_c_strong,
                        bethe_c_weak=prep.bethe_c_weak,
                        mu=mu,
                        amp=amp,
                        krylov_rank=krylov_rank,
                        out_rank=out_rank,
                        max_rank=max_rank,
                        f_ld=f_ld,
                        bl=bl,
                        lu=lu,
                        pref=pref,
                        kbuf=kbuf,
                        out_idx=out_idx,
                        ws=ws,
                        persistent=persistent,
                        metric=prep.metric,
                        sgh_tables_delta_major=prep.sgh_tables_delta_major,
                        stats=stats,
                        bin_out=bin_out,
                        intensity_buf=inten_buf,
                        write_global=1,
                        smith_scratch=smith_scratch,
                        smith_uniq_vals=smith_uniq_vals,
                        smith_uniq_meta=smith_uniq_meta,
                        galerkin_h=gal_bufs.galerkin_h,
                        galerkin_b=gal_bufs.galerkin_b,
                        galerkin_f=gal_bufs.galerkin_f,
                        w_out=gal_bufs.w_out,
                    )
                )

                if do_profile:
                    for name, subset in galerkin_profile_stages(items):
                        submitter.submit(subset, label=f"hp:{name}")
                        t = mark(name, t)
                    st = stats.read_as(np.uint32, size=4 * max(chunk, rows) * 2)[:rows]
                    tc, uc, sc, bc, fc = count_mode_flags(st)
                    tiled_count += tc
                    unique_count += uc
                    useg_count += sc
                    bicg_count += bc
                    fail_count += fc
                else:
                    submitter.submit(items, label="hp:chunk")

            def flush_pending(pending: list[tuple[object, int]]) -> None:
                for ch_p, slot in pending:
                    rows_p = ch_p.kvecs.size // 3
                    tile = (
                        inten_ring[slot]
                        .read_as(np.float32, size=4 * rows_p * n_sites)
                        .reshape(rows_p, n_sites)
                    )
                    idx = np.asarray(ch_p.output_indices, dtype=np.intp).reshape(-1)
                    bin_weighted[bi, idx, :] = tile

            if bi == 0 and chunks:
                one_chunk(chunks[0], do_profile=False, inten_buf=inten_ring[0])
                sync_device(device)
                k_done = 0
                output.write(np.zeros(n_k * n_sites, dtype=np.float32))

            n_tiles = len(chunks)
            k_done = 0
            bin_dyn = 0.0
            pending_rows = 0
            pending: list[tuple[object, int]] = []
            ring_slot = 0
            t_batch0 = time.perf_counter()
            for ti, ch in enumerate(chunks):
                rows = ch.kvecs.size // 3
                one_chunk(ch, do_profile=False, inten_buf=inten_ring[ring_slot])
                pending.append((ch, ring_slot))
                ring_slot = (ring_slot + 1) % queue_depth
                pending_rows += rows
                k_done += rows
                k_solved += rows
                flush = (ti + 1) % queue_depth == 0 or (ti + 1) == n_tiles
                if not flush:
                    continue
                sync_device(device)
                flush_pending(pending)
                pending.clear()
                dt = time.perf_counter() - t_batch0
                dyn_wall += dt
                bin_dyn += dt
                if want_stats:
                    st = stats.read_as(np.uint32, size=4 * rows * 2)[:rows]
                    if collect_stats:
                        mode_flags_parts.append(st.copy())
                    tc, uc, sc, bc, fc = count_mode_flags(st)
                    tiled_count += tc
                    unique_count += uc
                    useg_count += sc
                    bicg_count += bc
                    fail_count += fc
                if progress_every > 0 and (
                    ti == 0 or (ti + 1) % progress_every == 0 or ti + 1 == n_tiles
                ):
                    rate = pending_rows / dt if dt > 0 else 0.0
                    print(
                        f"    bin{bi} tile {ti+1}/{n_tiles}  "
                        f"{k_done}/{n_k} k  batch={dt:.2f}s  {rate:.0f} k/s (measured)  "
                        f"res/uniq/useg/bicgstab/denseTile="
                        f"{k_done - unique_count - tiled_count - fail_count}/"
                        f"{unique_count - useg_count}/{useg_count}/{bicg_count}/{tiled_count} "
                        f"fail={fail_count}  qdepth={queue_depth}",
                        flush=True,
                    )
                if inter_tile_pause_s > 0:
                    time.sleep(inter_tile_pause_s)
                if on_chunk_done is not None:
                    on_chunk_done(
                        {
                            "bi": bi,
                            "ti": ti,
                            "n_tiles": n_tiles,
                            "rows": rows,
                            "k_done": k_done,
                            "n_k": n_k,
                            "n_sites": n_sites,
                            "dt_s": dt,
                            "intensities": bin_weighted[bi].copy(),
                        }
                    )
                pending_rows = 0
                t_batch0 = time.perf_counter()

            bin_dyn_timings.append(bin_dyn)
            bins_run = bi + 1

            saw_weighted_bin = saw_weighted_bin or amp > 0.0
            saw_intensity = saw_intensity or bool(bin_weighted[bi].any())

            weighted_delta_sum_view = bin_weighted[bi].astype(np.float64).sum(axis=1)
            last_relative_change = relative_normalized_view_change(
                integrated_sum_view, weighted_delta_sum_view
            )
            integrated_sum_view += weighted_delta_sum_view
            if (
                relative_image_stop > 0
                and bins_run > 1
                and last_relative_change <= relative_image_stop
            ):
                stopped_by_relative_change = True
                break

            if profile and bi == 0:
                snap = output.read_as(np.float32, size=4 * n_k * n_sites)
                for k in list(prof.t):
                    del prof.t[k]
                tiled_count = 0
                unique_count = 0
                useg_count = 0
                bicg_count = 0
                fail_count = 0
                k_saved = k_done
                for ch in chunks[: min(3, len(chunks))]:
                    one_chunk(ch, do_profile=True, inten_buf=inten_ring[0])
                sync_device(device)
                output.write(snap)
                k_done = k_saved

        I = np.nan_to_num(  # noqa: E741
            output.read_as(np.float32, size=4 * n_k * n_sites).reshape(n_k, n_sites),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        bin_weighted = bin_weighted[:bins_run]
        w = bin_weights[:bins_run].reshape(bins_run, 1, 1)
        safe_w = np.where(w != 0.0, w, 1.0)
        bin_intensities = (bin_weighted.astype(np.float64) / safe_w).astype(np.float32)
        bin_intensities = np.where(w != 0.0, bin_intensities, 0.0).astype(np.float32)
        if saw_weighted_bin and not saw_intensity:
            raise RuntimeError(
                f"galerkin solved {bins_run} weighted bin(s) but every intensity is "
                f"zero (k tile {chunk}, n_strong {n}). The GPU discarded the submits, which "
                f"happens when a command buffer runs too long; retry with a smaller k tile."
            )
    finally:
        _destroy_ns(ws)
        _destroy_ns(gal_bufs)
        stats.destroy()
        if smith_scratch is not None:
            smith_scratch.destroy()
        if smith_uniq_vals is not None:
            smith_uniq_vals.destroy()
        if smith_uniq_meta is not None:
            smith_uniq_meta.destroy()
        out_idx.destroy()
        kbuf.destroy()
        output.destroy()
        for buf in inten_ring:
            buf.destroy()
        submitter.destroy()

    k_denom = k_solved if k_solved > 0 else n_k
    return {
        "intensities": I,
        "bin_intensities": bin_intensities,
        "n_k": n_k,
        "n_strong": n,
        "n_g": n_g,
        "n_bins": bins_run,
        "stopped_by_relative_change": stopped_by_relative_change,
        "last_relative_change": float(last_relative_change),
        "dyn_wall_s": dyn_wall,
        "k_per_s": k_denom / dyn_wall if dyn_wall > 0 else 0.0,
        "k_done": k_done,
        "k_solved": k_solved,
        "tiled_k": tiled_count,
        "unique_k": unique_count,
        "useg_k": useg_count,
        "bicg_k": bicg_count,
        "fail_k": fail_count,
        "profile": dict(prof.t),
        "bin_prep_timings": bin_prep_timings,
        "bin_dyn_timings": bin_dyn_timings,
        "crystal_prep_timings": dict(prep.prep_timings),
        "krylov_rank": krylov_rank,
        "out_rank": out_rank,
        "lyapunov_strategy": strategy,
        "mode_flags": (
            np.concatenate(mode_flags_parts)
            if mode_flags_parts
            else np.empty(0, dtype=np.uint32)
        ),
    }
