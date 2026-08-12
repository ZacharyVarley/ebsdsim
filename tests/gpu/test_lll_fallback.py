"""Regression: Non-LLL geometry fallback stays load-bearing on BCC Fe.

At halfw=3, BCC Fe has one pathological k (Lambert index 1, near the
sector corner — *not* [001]) whose LLL Toeplitz pack is invalid
(``geom[29]=0``). Without the Non-LLL ``U=I`` fallback in
``*_build_uniq_shared_bicgstab_f16.wgsl``, that k is rejected
(``stats=0xFFFFFFFF``, zero intensity). With the fallback it solves and
matches ``exact_slow_cpu``.
"""

from __future__ import annotations

from pathlib import Path

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.energy.weights import reduce_over_sites, site_weights_from_meta_cell
from ebsdsim.gpu.device import require_gpu

_FE_BCC = Path(__file__).resolve().parents[1] / "data" / "cif" / "Fe_bcc.cif"

# Empirically stable on this fixture: the LLL-invalid k at halfw=3.
_FALLBACK_K_INDEX = 1
_HALFW = 3
# Smith@16 / Galerkin@6 vs exact on Fe halfw=3: observed max rel ~5e-2.
_REL_TOL_ALL = 0.08
_REL_TOL_FALLBACK_K = 0.08


def _gpu_f16_available() -> bool:
    try:
        require_gpu(required_features=("shader-f16",))
        return True
    except RuntimeError:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _gpu_f16_available(),
        reason="WebGPU adapter with shader-f16 unavailable",
    ),
]


def _run(solver: str, *, exact: bool = False) -> es.MasterPattern:
    rank = 6 if solver == "galerkin" else 16
    return es.master_pattern_from_cif(
        _FE_BCC,
        voltage_kv=20.0,
        halfw=_HALFW,
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
    n_k, n_sites = int(mp.n_k), int(mp.n_sites)
    fs = np.asarray(mp.integrated, dtype=np.float64).reshape(n_k, n_sites)
    weights = site_weights_from_meta_cell(mp.metadata.get("cell", {}))
    return reduce_over_sites(fs, weights).astype(np.float64, copy=False)


@pytest.mark.parametrize("solver", ["smith", "galerkin"])
def test_lll_fallback_fe_bcc_matches_exact(solver: str) -> None:
    """Fe BCC halfw=3: zero fails, fallback k nonzero, matches exact."""
    assert _FE_BCC.is_file(), f"missing CIF fixture: {_FE_BCC}"

    exact = _run(solver, exact=True)
    gpu = _run(solver, exact=False)

    assert gpu.n_k == exact.n_k == 19
    assert _FALLBACK_K_INDEX < gpu.n_k

    ref = _site_weighted_fs(exact)
    approx = _site_weighted_fs(gpu)
    assert np.all(np.isfinite(approx))
    assert np.all(np.isfinite(ref))
    assert np.any(ref > 0)

    # Without the Non-LLL fallback this index is rejected → intensity 0.
    # (fail_k metadata can undercount when the dyn census is off, so intensity
    # vs exact is the load-bearing assertion.)
    assert approx[_FALLBACK_K_INDEX] > 0.0
    assert ref[_FALLBACK_K_INDEX] > 0.0

    rel = np.abs(approx - ref) / np.maximum(np.abs(ref), 1e-30)
    assert float(np.max(rel)) <= _REL_TOL_ALL, float(np.max(rel))
    assert float(rel[_FALLBACK_K_INDEX]) <= _REL_TOL_FALLBACK_K
    assert int(gpu.metadata.get("fail_k", -1)) == 0
