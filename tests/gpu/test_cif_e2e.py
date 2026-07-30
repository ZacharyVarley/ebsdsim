"""End-to-end master-pattern runs on the space-group CIF suite (halfw=10)."""

from __future__ import annotations

from pathlib import Path

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.gpu.device import require_gpu

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
_CIF_PATHS = sorted(_CIF_DIR.glob("sg_*.cif"))
assert _CIF_PATHS, f"no CIF fixtures under {_CIF_DIR}"


def _gpu_smith_iterative_available() -> bool:
    try:
        require_gpu(required_features=("shader-f16",))
        return True
    except RuntimeError:
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _gpu_smith_iterative_available(),
        reason="WebGPU adapter with shader-f16 unavailable",
    ),
]


@pytest.mark.parametrize("cif_path", _CIF_PATHS, ids=[p.stem for p in _CIF_PATHS])
def test_master_pattern_cif_suite_halfw10(cif_path: Path) -> None:
    """Smith iterative E2E on each suite CIF at halfw=10 (21×21 Lambert)."""
    mp = es.master_pattern_from_cif(
        cif_path,
        voltage_kv=20.0,
        halfw=10,
        dmin=0.05,
        energy_binwidth_keV=5.0,
        marginal_coverage=1.0,
        mc_backend="surrogate",
        solver="smith_iterative",
        verbosity=0,
    )
    side = 21
    assert mp.pattern.shape == (side, side)
    assert mp.metadata["grid_size"] == side
    assert mp.metadata["halfw"] == 10
    assert mp.metadata["solver"] == "smith_iterative"
    assert int(mp.metadata.get("fail_k", 0)) == 0
    assert np.all(np.isfinite(mp.pattern))
    assert np.any(mp.pattern > 0)
    assert np.all(np.isfinite(mp.integrated))
    assert np.any(mp.integrated > 0)
    assert mp.n_k > 0
    assert mp.n_sites >= 1
