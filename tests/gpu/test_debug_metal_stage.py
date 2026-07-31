"""TEMPORARY Metal queue-depth probe for the two deterministic-zero cells.

Single-chunk solves are bit-correct on Metal (queue_depth=1, one chunk),
but the full e2e (queue_depth=4, all chunks, 4 bins) is deterministically
all-zero. Run the full multi-bin integration at queue_depth 1 and 2 for
the failing cells + control; if qd=1 is correct, the in-flight pipeline
(ring of chunks + RMW accumulator) is what Metal corrupts.
Always asserts false to surface the report.
Delete once the Metal behavior is understood.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ebsdsim.crystal.build import build_cell_from_cif_path
from ebsdsim.energy.surrogate import infer_direct_exp_from_cell_rebinned
from ebsdsim.engine.integrate import surrogate_to_multi_voltage_mc
from ebsdsim.engine.smith_iterative_runner import run_smith_iterative_voltage_integrated
from ebsdsim.gpu.device import get_device, require_gpu
from ebsdsim.gpu.dynamical.kernels import EBSDDynamicalKernels

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
_CELLS = [("sg_004_1004038", "FAIL"), ("sg_168_2232341", "FAIL"), ("sg_003_1536903", "CTRL")]

pytestmark = pytest.mark.gpu


def test_metal_queue_depth() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    for stem, tag in _CELLS:
        cell = build_cell_from_cif_path(_CIF_DIR / f"{stem}.cif")
        direct = infer_direct_exp_from_cell_rebinned(
            cell=cell, sigma_deg=70.0, beam_kv=20.0, energy_binwidth_keV=5.0, n_energy_bins=4
        )
        mc = surrogate_to_multi_voltage_mc(direct, 20.0)
        for qd in (1, 2):
            ctx = get_device(force=True, required_features=("shader-f16",))
            kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
            try:
                result, meta = run_smith_iterative_voltage_integrated(
                    cell=cell,
                    mc=mc,
                    halfw=10,
                    dmin=0.05,
                    voltage_kv=20.0,
                    bethe_c_strong=20.0,
                    bethe_c_weak=40.0,
                    bethe_c_cutoff=200.0,
                    dbdiff_sg_cutoff=1.0,
                    kernels=kernels,
                    marginal_coverage=1.0,
                    queue_depth=qd,
                )
            finally:
                kernels.destroy()
            integ = np.asarray(result.integrated, dtype=np.float64)
            bins = [np.asarray(b, dtype=np.float64) for b in result.bin_patterns]
            host = np.zeros_like(integ)
            for b, w in zip(bins, result.bin_weights, strict=False):
                host += float(w) * np.asarray(b, dtype=np.float64).reshape(-1)
            match = bool(np.allclose(integ, host, rtol=1e-4, atol=1e-3))
            print(
                f"[debug] {stem}({tag}) qd={qd} n_bins={result.n_bins_run} "
                f"bin_sums={[float(np.sum(b)) for b in bins]}\n"
                f"[debug]   gpu_sum={float(integ.sum()):.6e} host_sum={float(host.sum()):.6e} "
                f"MATCH={match} mode={meta.get('smith_iterative_mode')}",
                flush=True,
            )
    raise AssertionError("probe complete - see [debug] annotations")
