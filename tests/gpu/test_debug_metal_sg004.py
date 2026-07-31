"""TEMPORARY Metal probe: is the sg_004_1004038 all-zero failure deterministic?

Runs the exact e2e call up to 6 times, each on a fresh WebGPU device, and
prints per-attempt dynamical integrated sum vs post-Lambert pattern sum
(stage bisect + draw statistics in one shot). Fails only if every draw is
zero, with all attempt details in the assertion message.
Delete this file once the Metal behavior is understood.
"""

from __future__ import annotations

from pathlib import Path

import ebsdsim as es
import numpy as np
import pytest
from ebsdsim.gpu.device import get_device, require_gpu

_CIF = Path(__file__).resolve().parents[1] / "data" / "cif" / "sg_004_1004038.cif"

pytestmark = pytest.mark.gpu


def test_sg004_determinism_and_stage() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    lines: list[str] = []
    clean = 0
    for attempt in range(6):
        get_device(force=True, required_features=("shader-f16",))
        mp = es.master_pattern_from_cif(
            _CIF,
            voltage_kv=20.0,
            halfw=10,
            dmin=0.05,
            energy_binwidth_keV=5.0,
            marginal_coverage=1.0,
            mc_backend="surrogate",
            solver="smith_iterative",
            verbosity=0,
        )
        integ = float(np.sum(mp.integrated))
        pat = float(np.sum(mp.pattern))
        clean += int(pat > 0)
        lines.append(
            f"attempt={attempt} integrated_sum={integ:.6e} pattern_sum={pat:.6e} "
            f"mode={mp.metadata.get('smith_iterative_mode')} fail_k={mp.metadata.get('fail_k')}"
        )
        print(f"[debug] {lines[-1]}", flush=True)
    report = "\n".join(lines)
    assert clean > 0, f"sg_004_1004038 all-zero on every fresh device:\n{report}"
