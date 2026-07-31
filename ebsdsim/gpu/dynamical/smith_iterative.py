"""Optimized Smith iterative dynamical loop — persistent buffers, one submit per chunk."""

from __future__ import annotations

import time
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray

from ebsdsim.gpu.batch import PersistentSubmitter
from ebsdsim.gpu.buffers import StorageBuffer, c64_bytes, f32_bytes, u32_bytes
from ebsdsim.gpu.device import sync_device
from ebsdsim.gpu.dynamical.smith_iterative_dispatch import (
    MAX_WG_PER_DIM,
    RANK,
    TileDispatchCtx,
    build_tile_dispatch,
    count_mode_flags,
    profile_stages,
)
from ebsdsim.gpu.dynamical.smith_iterative_prep import (
    SmithIterativeCrystalPrep,
    VoltageBin,
    activate_voltage_bin,
)
from ebsdsim.gpu.dynamical.smith_iterative_shader import (
    MAX_N_SMITH_ITERATIVE,
    load_smith_iterative_shader,
)
from ebsdsim.gpu.pipelines import load_wgsl
from ebsdsim.lambert.kgrid import chunk_k_vectors
from ebsdsim.physics.prescan import relative_normalized_view_change

# Re-export for engine / callers that import from this module.
__all__ = [
    "MAX_N_SMITH_ITERATIVE",
    "TILE_VRAM_BUDGET_BYTES",
    "auto_tile_k",
    "load_smith_iterative_shader",
    "run_bins_dynamical",
]

# VRAM budget for the per-tile k workspace. Tiling over k is forced by memory
# (w_stack alone is n_k * n * RANK * 8 bytes ≈ 17 GB for a full 500k grid), so we
# size ONE large tile to this budget and stream a handful of tiles with backpressure.
TILE_VRAM_BUDGET_BYTES = 1_500_000_000


def _per_k_workspace_bytes(n_g: int, n: int, n_w: int, n_sites: int) -> int:
    # Mirrors _slim_workspace allocations (dominant terms).
    return (
        n_g * 16  # sg + scores + candidate_mask + selected_flags (f32/u32)
        + n * (4 + 8 + 8 + RANK * 8)  # idx_a + d_a + e0 + w_stack
        + max(1, n_w) * 4  # idx_w
        + n_sites * 4  # intensities
        + 16  # q_values
    )


def auto_tile_k(
    n_g: int, n: int, n_w: int, n_sites: int, *, budget: int = TILE_VRAM_BUDGET_BYTES, cap: int = MAX_WG_PER_DIM
) -> int:
    """Largest k-tile whose workspace fits ``budget`` (min 256, capped).

    Capped at ``MAX_WG_PER_DIM`` because the per-k kernels (smith_iterative/topk/slim/inten)
    launch one workgroup per k. The score/gather kernels are grid-strided, so they
    do not constrain the tile.
    """
    per_k = _per_k_workspace_bytes(n_g, n, n_w, n_sites)
    tile = max(256, budget // max(1, per_k))
    return int(min(cap, tile))


class _Prof:
    def __init__(self) -> None:
        self.t: dict[str, float] = {}

    def add(self, name: str, dt: float) -> None:
        self.t[name] = self.t.get(name, 0.0) + float(dt)


def _pack_k_vec4(kvecs: NDArray[np.float32]) -> NDArray[np.float32]:
    """std430-friendly vec3→vec4 pad (matches create_k_chunk_buffer)."""
    src = np.asarray(kvecs, dtype=np.float32).reshape(-1)
    count = src.size // 3
    out = np.zeros((count, 4), dtype=np.float32)
    out[:, :3] = src.reshape(count, 3)
    return out.reshape(-1)


def _slim_workspace(kernels, *, B: int, n_g: int, n: int, n_w: int, n_sites: int) -> SimpleNamespace:
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
        w_stack=sb("hp:w", c64_bytes(B * n * RANK)),
        intensities=sb("hp:I", f32_bytes(B * n_sites), copy_src=True),
    )


def _destroy_ws(ws: SimpleNamespace) -> None:
    for v in vars(ws).values():
        if hasattr(v, "destroy"):
            v.destroy()


def run_bins_dynamical(
    prep: SmithIterativeCrystalPrep,
    bins: list[VoltageBin],
    *,
    chunk: int,
    code_topk: str,
    code_slim: str,
    code_smith_iterative: str,
    code_inten: str,
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
) -> dict:
    """Integrate voltage bins; return integrated + per-bin intensity stacks.

    ``chunk`` is the k-tile size (one GPU submit per tile). Prefer
    :func:`auto_tile_k` for throughput runs.     ``queue_depth`` controls how many
    tiles stay in flight before ``sync_device`` (default 4; forced to 1 for
    profile / progress / callbacks / ``collect_stats``).

    ``relative_image_stop`` (default 0.01) enables per-bin early stop: after
    each fully-solved bin, the relative change in the min/max-normalized
    voltage-integrated pattern (flattened fundamental-sector ``n_k`` view) is
    compared against this threshold and the loop stops once it drops below it.

    Device keeps one ``(n_k, n_sites)`` accumulator for the energy-integrated
    total (bitwise-stable amp-add). Per-bin patterns are read back from a
    ``queue_depth`` ring of ``chunk × n_sites`` intensity scratches at each
    flush and scattered into a host stack ``(n_bins, n_k, n_sites)``.

    Parameters
    ----------
    max_chunks
        If set, only process this many tiles (debug/smoke). Dyn wall and
        intensities cover only those rows; remaining rows stay 0.
    on_chunk_done
        Optional callback after each awaited tile with keys
        ``bi, ti, n_tiles, rows, k_done, n_k, n_sites, dt_s, intensities``
        (``intensities`` is the current-bin host copy, zeros where unfilled).
    k_order
        Optional permutation of ``0..n_k-1`` (e.g. hardest-first by prescan
        ``strong_needed``). Dyn processes k in that order; intensities are
        still written to the original row indices for rasterize.
    first_act / pre_activated
        Optional results from :func:`activate_voltage_bin` to avoid a second
        Bethe prescan. ``pre_activated[i]`` must match ``bins[i]``; when set,
        bin switches reuse those dicts (lookup already resident).
    queue_depth
        How many k-tiles to enqueue before ``sync_device``. ``1`` = honest
        per-tile timing (benchmarks). ``>=2`` pipelines GPU work (throughput).
        Forced to 1 when ``profile``, ``progress_every>0``, or ``on_chunk_done``.
    fuse_gather_slim
        If True, replace separate gather+slim with one fused kernel.
    """
    if chunk < 8:
        raise ValueError(f"tile size chunk={chunk} too small; use >= 8")
    if queue_depth < 1:
        raise ValueError(f"queue_depth={queue_depth} must be >= 1")

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
    use_gather_slim = bool(fuse_gather_slim)
    code_gather_slim = ""
    if use_gather_slim:
        code_gather_slim = load_wgsl("dynamical/gather_slim_q.wgsl")
    # Honest per-tile timing / mode census / callbacks need depth 1
    # (stats buffer is overwritten each tile; multi-tile census would be wrong).
    if profile or progress_every > 0 or on_chunk_done is not None or collect_stats:
        queue_depth = 1
    queue_depth = int(queue_depth)

    first = first_act if first_act is not None else activate_voltage_bin(prep, bins[0])
    if first_act is None:
        bin_prep_timings.append(first["timings"])
    elif pre_activated is None:
        # Caller activated bin0 already; record its timings once.
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
        # Size-pass already paid activate; dyn loop reuses acts (bookkeep≈0).
        bin_prep_timings = [dict(a.get("timings", {})) for a in pre_activated]
    if n > MAX_N_SMITH_ITERATIVE:
        raise RuntimeError(f"n_strong={n} > smith_iterative MAX_N={MAX_N_SMITH_ITERATIVE}")
    uses_unique_bits = "uniq_bits" in code_smith_iterative
    uses_uniq_vals = "uniq_vals" in code_smith_iterative
    uses_uniq_meta = "uniq_meta" in code_smith_iterative
    uses_implicit_bicg = "hkl_hash" in code_smith_iterative and "bicgstab_solve" in code_smith_iterative
    # Clamp tile: auto_tile_k ignores n_k; global uniq_vals/meta are O(chunk*16k).
    chunk = min(int(chunk), int(n_k), MAX_WG_PER_DIM)
    # vec2<f16> = 4 B/entry; vec2<f32> storage variant = 8 B/entry.
    _uv_stride = 8 if "uniq_vals: array<vec2<f32>>" in code_smith_iterative else 4
    if uses_uniq_vals or uses_uniq_meta:
        max_chunk_uniq = 1_800_000_000 // (16384 * _uv_stride)
        if chunk > max_chunk_uniq:
            chunk = int(max_chunk_uniq)

    ws = _slim_workspace(
        kernels, B=chunk, n_g=n_g, n=n, n_w=max(n_w, 1), n_sites=n_sites
    )
    stats = StorageBuffer(
        device, queue, label="hp:stats", data=np.zeros(chunk * 2, dtype=np.uint32), copy_src=True
    )
    smith_iterative_scratch = None
    smith_iterative_uniq_vals = None
    smith_iterative_uniq_meta = None
    # MAX_UWORDS = 2048 for global bitset/prefix (plen up to 65536).
    _max_uwords = 2048
    if uses_unique_bits:
        smith_iterative_scratch = StorageBuffer(
            device,
            queue,
            label="hp:uniqueBitsetAtomic",
            byte_length=chunk * _max_uwords * 4,
            copy_dst=True,
        )
    if uses_uniq_vals:
        smith_iterative_uniq_vals = StorageBuffer(
            device,
            queue,
            label="hp:uniqueVals",
            byte_length=chunk * 16384 * _uv_stride,
            copy_dst=True,
        )
    if uses_uniq_meta:
        smith_iterative_uniq_meta = StorageBuffer(
            device,
            queue,
            label="hp:uniqueMeta",
            byte_length=chunk * _max_uwords * 2 * 4,
            copy_dst=True,
        )
    out_idx = StorageBuffer(
        device, queue, label="hp:oi", data=np.zeros(chunk, dtype=np.uint32), copy_dst=True
    )
    # Persistent k-chunk (vec4-padded); rewritten in place each chunk.
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
    # Ring of per-tile intensity scratches so queue_depth>1 tiles do not overwrite
    # before flush readback. Device cost: queue_depth × chunk × n_sites (≪ n_k).
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
    # Host per-bin stack: GPU intensity writes bake amp=amplitude*energy_weight,
    # so these are weighted bin contributions (same values amp-added into output).
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
        nonlocal ws, n, n_w
        if n_s <= n and n_wk <= n_w:
            return
        if n_s > MAX_N_SMITH_ITERATIVE:
            raise RuntimeError(f"n_strong={n_s} > smith_iterative MAX_N={MAX_N_SMITH_ITERATIVE}")
        _destroy_ws(ws)
        n = max(n, n_s)
        n_w = max(n_w, n_wk)
        ws = _slim_workspace(
            kernels, B=chunk, n_g=n_g, n=n, n_w=max(n_w, 1), n_sites=n_sites
        )
        # Resource set changed → drop cached bind groups.
        submitter._bind_groups.clear()

    try:
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
            # Resident integrated accumulator (method a: bitwise-stable totals).
            bin_out = output

            chunks = list(chunk_k_vectors(kvecs_run, out_order, chunk))
            if max_chunks is not None:
                chunks = chunks[: max(0, int(max_chunks))]

            def one_chunk(ch, *, do_profile: bool, inten_buf: StorageBuffer) -> None:
                nonlocal tiled_count, unique_count, useg_count, bicg_count, fail_count, k_done
                rows = ch.kvecs.size // 3
                kbuf.write(_pack_k_vec4(ch.kvecs))
                out_idx.write(np.asarray(ch.output_indices, dtype=np.uint32))
                # Workspace is sized to max beams across bins; dispatches use this bin's n.
                n_use = n_bin
                n_w_use = n_w_bin

                def mark(name: str, t0: float) -> float:
                    if do_profile:
                        sync_device(device)
                        prof.add(name, time.perf_counter() - t0)
                        return time.perf_counter()
                    return t0

                t = time.perf_counter()
                # Full pipeline in ONE queue.submit (score → topk → gather → slim → smith_iterative → inten).
                items = build_tile_dispatch(
                    TileDispatchCtx(
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
                        code_smith_iterative=code_smith_iterative,
                        code_inten=code_inten,
                        use_gather_slim=use_gather_slim,
                        uses_implicit_bicg=uses_implicit_bicg,
                        bethe_c_cutoff=prep.bethe_c_cutoff,
                        dbdiff_sg_cutoff=prep.dbdiff_sg_cutoff,
                        bethe_c_strong=prep.bethe_c_strong,
                        bethe_c_weak=prep.bethe_c_weak,
                        mu=mu,
                        amp=amp,
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
                        smith_iterative_scratch=smith_iterative_scratch,
                        smith_iterative_uniq_vals=smith_iterative_uniq_vals,
                        smith_iterative_uniq_meta=smith_iterative_uniq_meta,
                    )
                )

                if do_profile:
                    for name, subset in profile_stages(items):
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
                """Scatter awaited tile intensities into host bin_weighted[bi]."""
                for ch_p, slot in pending:
                    rows_p = ch_p.kvecs.size // 3
                    tile = (
                        inten_ring[slot]
                        .read_as(np.float32, size=4 * rows_p * n_sites)
                        .reshape(rows_p, n_sites)
                    )
                    idx = np.asarray(ch_p.output_indices, dtype=np.intp).reshape(-1)
                    bin_weighted[bi, idx, :] = tile

            # Compile / bind-group warmup (excluded from dyn wall and k_done).
            if bi == 0 and chunks:
                one_chunk(chunks[0], do_profile=False, inten_buf=inten_ring[0])
                sync_device(device)
                k_done = 0
                # Warmup wrote into the accumulator — clear before real integrate.
                output.write(np.zeros(n_k * n_sites, dtype=np.float32))

            # Pipeline up to queue_depth tiles before sync (depth 1 = per-tile timing).
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
                # Cheap mode census after sync.
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
                    # Live-preview: current-bin host copy (zeros where unfilled).
                    # queue_depth is forced to 1 above when this callback is set.
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

            # Early stop: relative change of the voltage-integrated pattern in
            # the flattened fundamental-sector n_k view (mirrors the LU path).
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
                # Profile only the first few chunks — never re-run the full grid.
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
                output.write(snap)  # undo profile amp-adds into accumulator
                k_done = k_saved

        I = np.nan_to_num(  # noqa: E741 — I is site intensity
            output.read_as(np.float32, size=4 * n_k * n_sites).reshape(n_k, n_sites),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        # LU-compatible unweighted bin patterns: GPU wrote amplitude*weight*raw.
        # Only the first ``bins_run`` bins were solved (early stop may cut the rest).
        bin_weighted = bin_weighted[:bins_run]
        w = bin_weights[:bins_run].reshape(bins_run, 1, 1)
        safe_w = np.where(w != 0.0, w, 1.0)
        bin_intensities = (bin_weighted.astype(np.float64) / safe_w).astype(np.float32)
        bin_intensities = np.where(w != 0.0, bin_intensities, 0.0).astype(np.float32)
    finally:
        _destroy_ws(ws)
        stats.destroy()
        if smith_iterative_scratch is not None:
            smith_iterative_scratch.destroy()
        if smith_iterative_uniq_vals is not None:
            smith_iterative_uniq_vals.destroy()
        if smith_iterative_uniq_meta is not None:
            smith_iterative_uniq_meta.destroy()
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
        "mode_flags": (
            np.concatenate(mode_flags_parts)
            if mode_flags_parts
            else np.empty(0, dtype=np.uint32)
        ),
    }
