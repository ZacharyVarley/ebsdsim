"""TEMPORARY Metal debug probe v3: bisect the integrated accumulator per bin.

Probe v2 showed on Metal: per-bin patterns correct for bins 0/1, zero for
bin 2, and the GPU `output` accumulator (RMW adds in intensity_fused_exact)
reads back ALL ZERO. Run the production runner with max_bins_run=1,2,3 and
compare GPU accumulator vs host-recombined weighted sum after each prefix.
Delete this file once the Metal failure is root-caused.
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

_CIF = Path(__file__).resolve().parents[1] / "data" / "cif" / "sg_003_1536903.cif"

pytestmark = pytest.mark.gpu


def test_metal_accumulator_bisect() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    cell = build_cell_from_cif_path(_CIF)
    direct = infer_direct_exp_from_cell_rebinned(
        cell=cell,
        sigma_deg=70.0,
        beam_kv=20.0,
        energy_binwidth_keV=5.0,
        n_energy_bins=4,
    )
    mc = surrogate_to_multi_voltage_mc(direct, 20.0)
    for iteration in range(5):
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
                max_bins_run=3,
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
            f"[debug] iter={iteration}: n_bins_run={result.n_bins_run} "
            f"mode={meta.get('smith_iterative_mode')} "
            f"bin_sums={[float(np.sum(b)) for b in bins]}\n"
            f"[debug]   gpu_sum={float(integ.sum()):.6e} host_sum={float(host.sum()):.6e} "
            f"MATCH={match} fail_k={meta.get('fail_k')}",
            flush=True,
        )
