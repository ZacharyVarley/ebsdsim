"""TEMPORARY Metal stage bisect for the two deterministic-zero cells.

For each of sg_004_1004038 / sg_168_2232341, run the e2e call on a fresh
device and print per-bin sums (dynamical stage), integrated sum, and the
post-Lambert pattern sum, so the [debug] annotations show which stage
produces the zeros. Always asserts false at the end to surface the report.
Delete once the Metal behavior is understood.
"""

from __future__ import annotations

from pathlib import Path

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.gpu.device import get_device, require_gpu

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
_CIFS = ["sg_004_1004038", "sg_168_2232341"]

pytestmark = pytest.mark.gpu


def test_metal_stage_bisect() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    for stem in _CIFS:
        for attempt in range(2):
            get_device(force=True, required_features=("shader-f16",))
            mp = es.master_pattern_from_cif(
                _CIF_DIR / f"{stem}.cif",
                voltage_kv=20.0,
                halfw=10,
                dmin=0.05,
                energy_binwidth_keV=5.0,
                marginal_coverage=1.0,
                mc_backend="surrogate",
                solver="smith_iterative",
                verbosity=0,
            )
            bin_sums = (
                [float(np.sum(b)) for b in mp.bin_patterns]
                if getattr(mp, "bin_patterns", None) is not None
                else None
            )
            print(
                f"[debug] {stem} attempt={attempt} "
                f"bin_sums={bin_sums} "
                f"integrated_sum={float(np.sum(mp.integrated)):.6e} "
                f"pattern_sum={float(np.sum(mp.pattern)):.6e} "
                f"mode={mp.metadata.get('smith_iterative_mode')}",
                flush=True,
            )
    raise AssertionError("probe complete - see [debug] annotations")
