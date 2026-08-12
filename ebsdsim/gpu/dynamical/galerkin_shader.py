"""Galerkin solve WGSL bucketing: MAX_N / MAX_PACK / MAX_UNIQ / BPT templates.

The solve shader builds the same rational Krylov subspace as Smith (repeated
pole at q via Cayley / BiCGSTAB) and writes the projected pencil ``H``, ``b``
for the on-device Lyapunov stage. Bucketing reuses the Smith string-replace
helpers with :func:`~ebsdsim.gpu.dynamical.ladder.galerkin_bucket_params_for_n`
so the epilogue's ``g_rdiag`` / ``g_kept`` workgroup storage is reserved
without shrinking the shared budget ad hoc.
"""

from __future__ import annotations

from pathlib import Path

from ebsdsim.gpu.dynamical.ladder import (
    DEFAULT_SHARED_BUDGET,
    galerkin_bucket_params_for_n,
)
from ebsdsim.gpu.dynamical.smith_shader import bucket_uniq_smith
from ebsdsim.gpu.pipelines import load_wgsl

# Same beam ceiling as smith (Krylov private arrays sized MAX_N).
MAX_N_GALERKIN = 2048

GALERKIN_SOLVE_WGSL = "dynamical/galerkin_build_uniq_shared_bicgstab_f16.wgsl"
GALERKIN_LYAPUNOV_SHARED_WGSL = "dynamical/galerkin_lyapunov_shared.wgsl"
GALERKIN_LYAPUNOV_IMPLICIT_WGSL = "dynamical/galerkin_lyapunov_implicit.wgsl"
GALERKIN_EXPAND_WGSL = "dynamical/galerkin_expand.wgsl"


def load_galerkin_solve_shader(
    n: int,
    wgsl_dir: Path | None = None,
    *,
    shared_budget: int = DEFAULT_SHARED_BUDGET,
    force_unique_seg_tile: bool = False,
    unique_seg_tile: int | None = None,
    force_dense_tile: bool = False,
    use_global_uniq_vals: bool = True,
    use_f32: bool = False,
) -> tuple[str, str]:
    """Load the beam-bucketed Galerkin BiCGSTAB + projection kernel."""
    pack_entry = 8 if use_f32 else 4
    bucket = galerkin_bucket_params_for_n(
        n, shared_budget=shared_budget, pack_entry_bytes=pack_entry
    )
    if wgsl_dir is not None:
        code = Path(wgsl_dir).joinpath(Path(GALERKIN_SOLVE_WGSL).name).read_text(
            encoding="utf-8"
        )
    else:
        code = load_wgsl(GALERKIN_SOLVE_WGSL)
    return bucket_uniq_smith(
        code,
        n,
        solver="galerkin",
        shared_budget=shared_budget,
        force_unique_seg_tile=force_unique_seg_tile,
        unique_seg_tile=unique_seg_tile,
        force_dense_tile=force_dense_tile,
        use_global_uniq_vals=use_global_uniq_vals,
        use_f32=use_f32,
        bucket_params=bucket,
    )


def load_galerkin_lyapunov_shared() -> str:
    return load_wgsl(GALERKIN_LYAPUNOV_SHARED_WGSL)


def load_galerkin_lyapunov_implicit() -> str:
    return load_wgsl(GALERKIN_LYAPUNOV_IMPLICIT_WGSL)


def load_galerkin_expand() -> str:
    return load_wgsl(GALERKIN_EXPAND_WGSL)
