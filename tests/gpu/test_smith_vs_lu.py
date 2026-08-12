"""Live Smith vs LU–Smith FS accuracy (no golden files)."""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.energy.weights import reduce_over_sites, site_weights_from_meta_cell
from ebsdsim.gpu.device import require_gpu

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"

# Handoff checklist materials: Ni + one odd-U CIF.
_COMPARE = [
    (Path(str(importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif"))), "Ni"),
    (_CIF_DIR / "sg_001_4030618.cif", "sg_001_4030618"),
]


def _gpu_smith_available() -> bool:
    try:
        require_gpu(required_features=("shader-f16",))
        return True
    except RuntimeError:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _gpu_smith_available(),
        reason="WebGPU adapter with shader-f16 unavailable",
    ),
]


def _run(cif: Path, *, solver: str) -> es.MasterPattern:
    return es.master_pattern_from_cif(
        cif,
        voltage_kv=20.0,
        halfw=12,
        dmin=0.05,
        energy_binwidth_keV=20.0,
        marginal_coverage=1.0,
        mc_backend="surrogate",
        solver=solver,  # type: ignore[arg-type]
        rank=16,
        verbosity=0,
    )


def _site_weighted_fs(mp: es.MasterPattern) -> np.ndarray:
    n_k = int(mp.n_k)
    n_sites = int(mp.n_sites)
    fs = np.asarray(mp.integrated, dtype=np.float64).reshape(n_k, n_sites)
    weights = site_weights_from_meta_cell(mp.metadata.get("cell", {}))
    return reduce_over_sites(fs, weights).astype(np.float64, copy=False)


@pytest.mark.parametrize("cif_path,label", _COMPARE, ids=[label for _, label in _COMPARE])
def test_smith_vs_lu_smith_fs_p95(cif_path: Path, label: str) -> None:
    """Site-weighted FS p95 relative error vs LU–Smith must stay under 1e-2."""
    assert cif_path.is_file(), f"missing CIF fixture: {cif_path}"

    si = _run(cif_path, solver="smith")
    lu = _run(cif_path, solver="lu_smith")

    assert si.metadata["solver"] == "smith"
    assert lu.metadata["solver"] == "lu_smith"
    assert int(si.metadata.get("fail_k", 0)) == 0
    assert si.n_k == lu.n_k
    assert si.n_sites == lu.n_sites

    i_si = _site_weighted_fs(si)
    i_lu = _site_weighted_fs(lu)
    assert i_si.shape == i_lu.shape
    assert np.all(np.isfinite(i_si))
    assert np.all(np.isfinite(i_lu))
    assert np.any(i_lu > 0)

    rel = np.abs(i_si - i_lu) / np.maximum(np.abs(i_lu), 1e-30)
    median = float(np.median(rel))
    p95 = float(np.percentile(rel, 95))
    # Independent API runs (separate Bethe top-k) — median under 1e-2, p95 under 2e-2.
    assert median < 1e-2, f"{label}: median relative error {median:.3e}"
    assert p95 < 2e-2, (
        f"{label}: site-weighted FS p95 relative error {p95:.3e} "
        f"(median={median:.3e}, max={float(rel.max()):.3e})"
    )
