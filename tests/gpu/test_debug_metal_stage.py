"""TEMPORARY scalar-field bisect for the Metal zero on sg_004.

The mc VoltageBin scalar fields deterministically zero the dynamical output
on Metal (hand bin with all-1.0 scalars is bit-correct). Split the two
suspects: beta (mu_shift into the f16 Krylov solve) vs amp=amplitude x
energy_weight (f16 multiply in intensity writeback; f16 subnormal flush on
Metal). Prints solver iteration stats to distinguish a degenerate solve
(mu path) from a healthy solve with flushed output (amp path).
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
from ebsdsim.engine.smith_iterative_runner import _voltage_bins_from_mc
from ebsdsim.gpu.device import get_device, require_gpu
from ebsdsim.gpu.dynamical.kernels import EBSDDynamicalKernels
from ebsdsim.gpu.dynamical.smith_iterative import (
    auto_tile_k,
    load_smith_iterative_shader,
    run_bins_dynamical,
)
from ebsdsim.gpu.dynamical.smith_iterative_prep import (
    VoltageBin,
    activate_voltage_bin,
    open_smith_iterative_prep,
)
from ebsdsim.gpu.pipelines import load_wgsl

_STEM = "sg_004_1004038"
_CIF = Path(__file__).resolve().parents[1] / "data" / "cif" / f"{_STEM}.cif"

pytestmark = pytest.mark.gpu


def _direct(cell, bins):
    ctx = get_device(force=True, required_features=("shader-f16",))
    kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
    prep, bins = open_smith_iterative_prep(
        _STEM, halfw=10, voltage_kv=20.0, dmin=0.05, cell=cell, bins=bins
    )
    try:
        act = activate_voltage_bin(prep, bins[0])
        budget = int(prep.device.limits["max-compute-workgroup-storage-size"])
        code, mode = load_smith_iterative_shader(int(act["n_strong"]), shared_budget=budget)
        tile = auto_tile_k(int(act["n_g"]), int(act["n_strong"]), int(act["n_weak"]), prep.n_sites)
        dyn = run_bins_dynamical(
            prep,
            bins,
            chunk=min(tile, 2048),
            code_topk=load_wgsl("dynamical/topk_radix_exact.wgsl"),
            code_slim=load_wgsl("dynamical/smith_iterative_slim_q.wgsl"),
            code_smith_iterative=code,
            code_inten=load_wgsl("dynamical/intensity_fused_exact.wgsl"),
            first_act=act,
            queue_depth=1,
        )
        iters = sum(int(cs.get("iters", 0)) for cs in dyn.get("chunk_stats", []))
        return (
            float(np.sum(dyn["intensities"])),
            mode,
            int(dyn.get("fail_k", -1)),
            iters,
            float(act["mu"]),
            float(act["amplitude"]),
        )
    finally:
        prep.close()
        kernels.destroy()


def test_metal_scalar_bisect() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    cell = build_cell_from_cif_path(_CIF)
    direct = infer_direct_exp_from_cell_rebinned(
        cell=cell, sigma_deg=70.0, beam_kv=20.0, energy_binwidth_keV=5.0, n_energy_bins=4
    )
    mc = surrogate_to_multi_voltage_mc(direct, 20.0)
    b0 = _voltage_bins_from_mc(mc)[0]
    configs = [
        ("ALL1   ", VoltageBin(20.0, None, 1.0, 1.0, 1.0, 0)),
        ("BETA   ", VoltageBin(20.0, None, b0.beta, 1.0, 1.0, 0)),
        ("AMP    ", VoltageBin(20.0, None, 1.0, b0.amplitude, b0.energy_weight, 0)),
        ("FULLMC ", VoltageBin(20.0, None, b0.beta, b0.amplitude, b0.energy_weight, 0)),
    ]
    for name, vb in configs:
        s, mode, fail_k, iters, mu, amp = _direct(cell, [vb])
        print(
            f"[debug] {name} sum={s:.6e} mu={mu:.6f} amp={amp:.6e} "
            f"fail_k={fail_k} iters={iters} mode={mode}",
            flush=True,
        )
    raise AssertionError("probe complete - see [debug] annotations")
