"""End-to-end master-pattern runs on the space-group CIF suite (halfw=10)."""

from __future__ import annotations

import sys
from pathlib import Path

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.gpu.device import require_gpu

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
_CIF_PATHS = sorted(_CIF_DIR.glob("sg_*.cif"))
assert _CIF_PATHS, f"no CIF fixtures under {_CIF_DIR}"

# Cells that deterministically produce an all-zero pattern on Apple Silicon
# Metal (upstream wgpu-native pre-v30 class: 6/6 fresh-device draws zero while
# the identical shader constants are bit-correct on D3D12). Repro archived at
# scratch/metal_sg004_repro.py. xfail is macOS-only and non-strict so an xpass
# (e.g. fixed wgpu-native) is visible without breaking the suite.
_METAL_ZERO_CIFS = {"sg_004_1004038"}


def _suite_params() -> list:
    params = []
    for p in _CIF_PATHS:
        marks = []
        if sys.platform == "darwin" and p.stem in _METAL_ZERO_CIFS:
            marks.append(
                pytest.mark.xfail(
                    reason=(
                        "deterministic all-zero output on Apple Silicon Metal "
                        "(upstream wgpu-native pre-v30); correct on D3D12"
                    ),
                    strict=False,
                )
            )
        params.append(pytest.param(p, marks=marks, id=p.stem))
    return params


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


@pytest.mark.parametrize("cif_path", _suite_params())
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
    ctx = (
        f"mode={mp.metadata.get('smith_iterative_mode')} "
        f"fail_k={mp.metadata.get('fail_k')} "
        f"k_solved={mp.metadata.get('k_solved')} "
        f"k_per_s={mp.metadata.get('k_per_s')}"
    )
    side = 21
    assert mp.pattern.shape == (side, side)
    assert mp.metadata["grid_size"] == side
    assert mp.metadata["halfw"] == 10
    assert mp.metadata["solver"] == "smith_iterative"
    assert int(mp.metadata.get("fail_k", 0)) == 0, ctx
    assert np.all(np.isfinite(mp.pattern)), ctx
    assert np.any(mp.pattern > 0), ctx
    assert np.all(np.isfinite(mp.integrated)), ctx
    assert np.any(mp.integrated > 0), ctx
    assert mp.n_k > 0
    assert mp.n_sites >= 1
