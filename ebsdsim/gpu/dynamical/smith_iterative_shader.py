"""Smith iterative WGSL bucketing: MAX_N / MAX_PACK / MAX_UNIQ / BPT templates."""

from __future__ import annotations

from pathlib import Path

from ebsdsim.gpu.dynamical.ladder import DEFAULT_SHARED_BUDGET, bucket_params_for_n
from ebsdsim.gpu.pipelines import load_wgsl

# Krylov still keeps ss/sord/sp in workgroup memory sized MAX_N. Pack TILE has
# no plen ceiling, but beam count is capped by the device workgroup-storage
# budget (ss+sord+sp+spack).
MAX_N_SMITH_ITERATIVE = 2048


def bucket_uniq_smith_iterative(
    code: str,
    n: int,
    *,
    solver: str,
    shared_budget: int = DEFAULT_SHARED_BUDGET,
    force_unique_seg_tile: bool = False,
    unique_seg_tile: int | None = None,
    force_dense_tile: bool = False,
    use_global_uniq_vals: bool = True,
) -> tuple[str, str]:
    """String-replace MAX_N / MAX_PACK / MAX_UNIQ / BPT for the device budget."""
    max_n, max_pack, max_uniq, bpt = bucket_params_for_n(n, shared_budget=shared_budget)

    def _sync_seg(c: str) -> str:
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

    code = code.replace("const MAX_N: u32 = 384u;", f"const MAX_N: u32 = {max_n}u;")
    code = code.replace("const MAX_PACK: u32 = 10200u;", f"const MAX_PACK: u32 = {max_pack}u;")
    code = code.replace("const MAX_UNIQ: u32 = 8600u;", f"const MAX_UNIQ: u32 = {max_uniq}u;")
    code = code.replace("const BPT: u32 = 2u;", f"const BPT: u32 = {bpt}u;")
    code = _sync_seg(code)
    tag = f"{solver}_uniq_{max_n}_{max_pack}"
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
    shared_budget: int = DEFAULT_SHARED_BUDGET,
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
        shared_budget=shared_budget,
        force_unique_seg_tile=force_unique_seg_tile,
        unique_seg_tile=unique_seg_tile,
        force_dense_tile=force_dense_tile,
        use_global_uniq_vals=use_global_uniq_vals,
    )
