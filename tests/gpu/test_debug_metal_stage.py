"""TEMPORARY Metal stage/mode bisect for the two deterministic-zero cells.

Stage bisect showed bin_sums all zero (dynamical stage, not Lambert).
Now: for each cell, one default-mode draw and one FORCE_DENSE_TILE draw
(bypasses the unique-seg bitset/prefix/uniq_vals machinery entirely).
If dense-tile is correct on Metal, the bug lives in the unique-seg path;
if it is also zero, the problem is upstream of all matvec modes.
Always asserts false at the end to surface the report as annotations.
Delete once the Metal behavior is understood.
"""

from __future__ import annotations

from pathlib import Path

import ebsdsim as es
import ebsdsim.engine.smith_iterative_runner as runner
import numpy as np
import pytest
from ebsdsim.gpu.device import get_device, require_gpu

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
_CIFS = ["sg_004_1004038", "sg_168_2232341"]

pytestmark = pytest.mark.gpu

_orig_loader = runner.load_smith_iterative_shader


def _force_dense_loader(*args, **kwargs):
    kwargs["force_dense_tile"] = True
    return _orig_loader(*args, **kwargs)


def _run(cif: Path) -> tuple[float, float, object]:
    mp = es.master_pattern_from_cif(
        cif,
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
    return float(np.sum(mp.integrated)), float(np.sum(mp.pattern)), bin_sums


def test_metal_mode_bisect() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    for stem in _CIFS:
        cif = _CIF_DIR / f"{stem}.cif"
        get_device(force=True, required_features=("shader-f16",))
        runner.load_smith_iterative_shader = _orig_loader
        integ, pat, bins = _run(cif)
        print(
            f"[debug] {stem} DEFAULT   integrated={integ:.4e} pattern={pat:.4e} bin_sums={bins}",
            flush=True,
        )
        get_device(force=True, required_features=("shader-f16",))
        runner.load_smith_iterative_shader = _force_dense_loader
        try:
            integ, pat, bins = _run(cif)
        finally:
            runner.load_smith_iterative_shader = _orig_loader
        print(
            f"[debug] {stem} DENSE     integrated={integ:.4e} pattern={pat:.4e} bin_sums={bins}",
            flush=True,
        )
    raise AssertionError("probe complete - see [debug] annotations")
