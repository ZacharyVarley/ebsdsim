"""Galerkin / Smith accuracy against exact CPU Lyapunov (ground truth).

Compares site-weighted FS intensities to ``exact_slow_cpu`` (batched eig
Lyapunov on the LU Bethe path). Galerkin at the product default rank should
match or beat Smith@16 against that reference.

Rank-monotonicity of the *intensity* error is NOT required: repeated-pole
Krylov in f32 is ill-conditioned, and trailing Galerkin modes of Y can move
Tr(S X) away from the full-space truth even while the projected Lyapunov
residual stays tiny (see scratch/galerkin_accuracy/). Pairwise Galerkin-vs-
Smith spread is also a poor metric — two approximations can diverge while
both approach the truth.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.energy.weights import reduce_over_sites, site_weights_from_meta_cell
from ebsdsim.engine.params import SimParams
from ebsdsim.gpu.device import require_gpu

_SCRATCH_GAN = Path(__file__).resolve().parents[2] / "scratch" / "GaN.cif"
_NI = Path(str(importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")))
_GAN = (
    _SCRATCH_GAN
    if _SCRATCH_GAN.is_file()
    else Path(str(importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif")))
)

_COMPARE = [
    (_NI, "Ni"),
    (_GAN, "GaN"),
]


def _gpu_f16_available() -> bool:
    try:
        require_gpu(required_features=("shader-f16",))
        return True
    except RuntimeError:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _gpu_f16_available(),
        reason="WebGPU adapter with shader-f16 unavailable",
    ),
]


def _run(
    cif: Path,
    *,
    solver: str,
    rank: int,
    halfw: int = 3,
    exact: bool = False,
) -> es.MasterPattern:
    return es.master_pattern_from_cif(
        cif,
        voltage_kv=20.0,
        halfw=halfw,
        dmin=0.05,
        energy_binwidth_keV=20.0,
        marginal_coverage=1.0,
        mc_backend="surrogate",
        solver=solver,  # type: ignore[arg-type]
        rank=rank,
        exact_slow_cpu=exact,
        verbosity=0,
    )


def _site_weighted_fs(mp: es.MasterPattern) -> np.ndarray:
    n_k = int(mp.n_k)
    n_sites = int(mp.n_sites)
    fs = np.asarray(mp.integrated, dtype=np.float64).reshape(n_k, n_sites)
    weights = site_weights_from_meta_cell(mp.metadata.get("cell", {}))
    return reduce_over_sites(fs, weights).astype(np.float64, copy=False)


def _rel_err(approx: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(approx - ref) / np.maximum(np.abs(ref), 1e-30)


@pytest.mark.parametrize("cif_path,label", _COMPARE, ids=[label for _, label in _COMPARE])
def test_galerkin_vs_exact_beats_smith(cif_path: Path, label: str) -> None:
    """Galerkin at the default rank should beat Smith@16 vs exact (median, p95)."""
    assert cif_path.is_file(), f"missing CIF fixture: {cif_path}"

    default_rank = SimParams().rank
    exact = _run(cif_path, solver="smith", rank=16, exact=True)
    smith = _run(cif_path, solver="smith", rank=16)
    gal = _run(cif_path, solver="galerkin", rank=default_rank)

    assert exact.metadata.get("exact_slow_cpu") is True
    assert smith.metadata["solver"] == "smith"
    assert gal.metadata["solver"] == "galerkin"
    # Ranks above 6 outgrow the shared Gauss-Jordan Kronecker workspace.
    assert gal.metadata.get("lyapunov_strategy") == (
        "shared" if default_rank <= 6 else "implicit"
    )
    assert int(smith.metadata.get("fail_k", 0)) == 0
    assert int(gal.metadata.get("fail_k", 0)) == 0
    assert gal.n_k == smith.n_k == exact.n_k

    ref = _site_weighted_fs(exact)
    e_smith = _rel_err(_site_weighted_fs(smith), ref)
    e_gal = _rel_err(_site_weighted_fs(gal), ref)

    assert np.all(np.isfinite(e_gal))
    assert np.all(np.isfinite(e_smith))
    assert np.any(ref > 0)

    med_g, med_s = float(np.median(e_gal)), float(np.median(e_smith))
    p95_g, p95_s = float(np.percentile(e_gal, 95)), float(np.percentile(e_smith, 95))

    # Galerkin is optimal over the same rational Krylov space Smith uses with
    # fixed ADI weights, so at the default rank it should not lose to Smith@16
    # vs exact. Allow slack for Bethe path differences (exact uses the LU runner).
    slack = 1.05
    assert med_g <= med_s * slack, (
        f"{label}: gal@{default_rank} median {med_g:.3e} > smith@16 median {med_s:.3e}"
    )
    assert p95_g <= p95_s * slack, (
        f"{label}: gal@{default_rank} p95 {p95_g:.3e} > smith@16 p95 {p95_s:.3e}"
    )
    # Absolute sanity vs exact (Krylov truncation, not pairwise spread).
    assert med_g < 3e-2, f"{label}: gal@{default_rank} median vs exact {med_g:.3e}"
    assert p95_g < 6e-2, f"{label}: gal@{default_rank} p95 vs exact {p95_g:.3e}"


@pytest.mark.parametrize("rank,strategy", [(6, "shared"), (8, "implicit")])
def test_galerkin_reproducible_on_ni(rank: int, strategy: str) -> None:
    """Same-solver rerun must bit-match (Bethe top-k is deterministic).

    Parametrized to cover both projected-Lyapunov kernels.
    """
    first = _run(_NI, solver="galerkin", rank=rank)
    assert first.metadata.get("lyapunov_strategy") == strategy
    a = _site_weighted_fs(first)
    b = _site_weighted_fs(_run(_NI, solver="galerkin", rank=rank))
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) == 0.0


def test_galerkin_per_bin_stack_matches_integrated() -> None:
    mp = _run(_NI, solver="galerkin", rank=6)
    n_bins = len(mp.bin_voltages_kv)
    assert n_bins >= 1
    assert len(mp.bin_patterns) == n_bins
    assert mp.metadata.get("galerkin_mode")
    assert mp.metadata.get("lyapunov_strategy") == "shared"

    from ebsdsim.engine.results import stack_bins

    n_k, n_sites = int(mp.n_k), int(mp.n_sites)
    stack = stack_bins(list(mp.bin_patterns), n_k, n_sites)
    weights = np.asarray(mp.bin_weights, dtype=np.float64).reshape(n_bins, 1, 1)
    host_sum = (stack.astype(np.float64) * weights).sum(axis=0).astype(np.float32)
    integrated = np.asarray(mp.integrated, dtype=np.float32).reshape(n_k, n_sites)
    abs_err = float(np.max(np.abs(host_sum - integrated)))
    rel_err = float(
        np.max(np.abs(host_sum - integrated) / np.maximum(np.abs(integrated), 1e-30))
    )
    assert abs_err < 1e-3 or rel_err < 1e-5
