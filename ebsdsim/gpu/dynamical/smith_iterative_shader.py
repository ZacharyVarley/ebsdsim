"""Smith iterative WGSL bucketing: MAX_N / MAX_PACK / MAX_UNIQ / BPT templates."""

from __future__ import annotations

from pathlib import Path

from ebsdsim.gpu.pipelines import load_wgsl

# Krylov still keeps ss/sord/sp in workgroup memory sized MAX_N. Pack TILE has
# no plen ceiling, but beam count is capped by ~48 KiB shared (ss+sord+sp+spack).
MAX_N_SMITH_ITERATIVE = 2048


def bucket_uniq_smith_iterative(
    code: str,
    n: int,
    *,
    solver: str,
    force_unique_seg_tile: bool = False,
    unique_seg_tile: int | None = None,
    force_dense_tile: bool = False,
    use_global_uniq_vals: bool = True,
) -> tuple[str, str]:
    """String-replace MAX_N / MAX_PACK / MAX_UNIQ / BPT for ~48 KB buckets."""
    n = int(n)

    def _sync_seg(c: str, max_uniq: int, max_pack: int) -> str:
        # With global bitset/prefix, SEG may be up to MAX_PACK (full shared values).
        seg = int(unique_seg_tile) if unique_seg_tile is not None else max_pack
        seg = max(1, min(seg, max_pack))
        c = c.replace("const UNIQUE_SEG_TILE: u32 = 8600u;", f"const UNIQUE_SEG_TILE: u32 = {seg}u;")
        if force_unique_seg_tile:
            c = c.replace(
                "const FORCE_UNIQUE_SEG_TILE: u32 = 0u;",
                "const FORCE_UNIQUE_SEG_TILE: u32 = 1u;",
            )
        if force_dense_tile:
            c = c.replace(
                "const FORCE_DENSE_TILE: u32 = 0u;",
                "const FORCE_DENSE_TILE: u32 = 1u;",
            )
        if not use_global_uniq_vals:
            c = c.replace(
                "const USE_GLOBAL_UNIQ_VALS: u32 = 1u;",
                "const USE_GLOBAL_UNIQ_VALS: u32 = 0u;",
            )
        return c

    if n <= 384:
        code = _sync_seg(code, 8600, 10200)
        tag = f"{solver}_uniq_384"
    elif n <= 512:
        code = code.replace("const MAX_N: u32 = 384u;", "const MAX_N: u32 = 512u;")
        code = code.replace("const MAX_PACK: u32 = 10200u;", "const MAX_PACK: u32 = 9690u;")
        code = code.replace("const MAX_UNIQ: u32 = 8600u;", "const MAX_UNIQ: u32 = 8100u;")
        code = _sync_seg(code, 8100, 9690)
        tag = f"{solver}_uniq_512"
    elif n <= 768:
        code = code.replace("const MAX_N: u32 = 384u;", "const MAX_N: u32 = 768u;")
        code = code.replace("const MAX_PACK: u32 = 10200u;", "const MAX_PACK: u32 = 8600u;")
        code = code.replace("const MAX_UNIQ: u32 = 8600u;", "const MAX_UNIQ: u32 = 7000u;")
        code = code.replace("const BPT: u32 = 2u;", "const BPT: u32 = 3u;")
        code = _sync_seg(code, 7000, 8600)
        tag = f"{solver}_uniq_768"
    elif n <= 1024:
        code = code.replace("const MAX_N: u32 = 384u;", "const MAX_N: u32 = 1024u;")
        code = code.replace("const MAX_PACK: u32 = 10200u;", "const MAX_PACK: u32 = 7600u;")
        code = code.replace("const MAX_UNIQ: u32 = 8600u;", "const MAX_UNIQ: u32 = 6000u;")
        code = code.replace("const BPT: u32 = 2u;", "const BPT: u32 = 4u;")
        code = _sync_seg(code, 6000, 7600)
        tag = f"{solver}_uniq_1024"
    elif n <= MAX_N_SMITH_ITERATIVE:
        # Overflow: pad to WG multiple, shrink pack to keep ~48 KiB shared.
        # Shared unique meta needs ~1564 slots beyond MAX_UNIQ; remainder is TILE/useg.
        max_n = ((n + 255) // 256) * 256
        bpt = max_n // 256
        max_pack = max(1024, (48000 - max_n * 16 - 2200) // 4)
        max_uniq = max(0, max_pack - 1564)
        code = code.replace("const MAX_N: u32 = 384u;", f"const MAX_N: u32 = {max_n}u;")
        code = code.replace("const MAX_PACK: u32 = 10200u;", f"const MAX_PACK: u32 = {max_pack}u;")
        code = code.replace("const MAX_UNIQ: u32 = 8600u;", f"const MAX_UNIQ: u32 = {max_uniq}u;")
        code = code.replace("const BPT: u32 = 2u;", f"const BPT: u32 = {bpt}u;")
        code = _sync_seg(code, max_uniq, max_pack)
        tag = f"{solver}_uniq_{max_n}"
    else:
        raise ValueError(
            f"smith_iterative shader supports at most {MAX_N_SMITH_ITERATIVE} beams (Krylov workgroup arrays), got {n}"
        )
    if force_dense_tile:
        tag = tag + "_denseforce"
    elif force_unique_seg_tile:
        tag = tag + "_segforce"
    if not use_global_uniq_vals and "uniq_vals" in code:
        tag = tag + "_legacyfill"
    return code, tag


def load_smith_iterative_shader(
    n: int,
    wgsl_dir: Path | None = None,
    *,
    force_unique_seg_tile: bool = False,
    unique_seg_tile: int | None = None,
    force_dense_tile: bool = False,
    use_global_uniq_vals: bool = True,
) -> tuple[str, str]:
    """Load the beam-bucketed BiCGSTAB smith_iterative kernel (resident / unique-Δ / tile)."""
    name = "dynamical/smith_iterative_build_uniq_shared_bicgstab_f16.wgsl"
    solver = "bicgstab"
    if wgsl_dir is not None:
        code = Path(wgsl_dir).joinpath(Path(name).name).read_text(encoding="utf-8")
    else:
        code = load_wgsl(name)
    return bucket_uniq_smith_iterative(
        code,
        n,
        solver=solver,
        force_unique_seg_tile=force_unique_seg_tile,
        unique_seg_tile=unique_seg_tile,
        force_dense_tile=force_dense_tile,
        use_global_uniq_vals=use_global_uniq_vals,
    )
