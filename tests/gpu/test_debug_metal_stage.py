"""TEMPORARY final discriminator bisect for the Metal zero on sg_004.

Known: direct run_bins_dynamical with one hand-made VoltageBin(20.0) is
bit-correct on Metal; the runner path is all-zero even at max_bins_run=1.
Isolate the discriminator among {queue_depth, bin construction/prefetch,
runner wrapper}:
  A  direct, hand bin v=20 (known good anchor)
  B1 runner, mrb=1, qd=1
  B2 runner, mrb=1, qd=4 (known bad anchor)
  A2 direct with the runner-constructed mc bin (tests bin/prefetch inputs)
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
from ebsdsim.engine.smith_iterative_runner import (
    _voltage_bins_from_mc,
    run_smith_iterative_voltage_integrated,
)
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


def _direct(cell, bins, qd):
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
            queue_depth=qd,
        )
        return float(np.sum(dyn["intensities"])), mode
    finally:
        prep.close()
        kernels.destroy()


def _runner(cell, mc, qd):
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
            max_bins_run=1,
        )
        return float(np.sum(result.integrated)), str(meta.get("smith_iterative_mode"))
    finally:
        kernels.destroy()


def test_metal_final_discriminator() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    cell = build_cell_from_cif_path(_CIF)
    direct = infer_direct_exp_from_cell_rebinned(
        cell=cell, sigma_deg=70.0, beam_kv=20.0, energy_binwidth_keV=5.0, n_energy_bins=4
    )
    mc = surrogate_to_multi_voltage_mc(direct, 20.0)
    mc_bin = _voltage_bins_from_mc(mc)[0]
    print(
        f"[debug] mc bin0: v={mc_bin.voltage_kv} next={mc_bin.next_voltage_kv} "
        f"amp={mc_bin.amplitude} w={mc_bin.energy_weight}",
        flush=True,
    )
    s, m = _direct(cell, [VoltageBin(20.0, None, 1.0, 1.0, 1.0, 0)], qd=1)
    print(f"[debug] A  direct-handbin-qd1   sum={s:.6e} mode={m}", flush=True)
    s, m = _runner(cell, mc, qd=1)
    print(f"[debug] B1 runner-mrb1-qd1      sum={s:.6e} mode={m}", flush=True)
    s, m = _runner(cell, mc, qd=4)
    print(f"[debug] B2 runner-mrb1-qd4      sum={s:.6e} mode={m}", flush=True)
    s, m = _direct(cell, [mc_bin], qd=1)
    print(f"[debug] A2 direct-mcbin-qd1     sum={s:.6e} mode={m}", flush=True)
    scalars_no_prefetch = VoltageBin(
        mc_bin.voltage_kv,
        None,
        mc_bin.beta,
        mc_bin.amplitude,
        mc_bin.energy_weight,
        mc_bin.bin_index,
    )
    s, m = _direct(cell, [scalars_no_prefetch], qd=1)
    print(f"[debug] A3 direct-mcscalars-noprefetch sum={s:.6e} mode={m}", flush=True)
    prefetch_only = VoltageBin(20.0, 15.0, 1.0, 1.0, 1.0, 0)
    s, m = _direct(cell, [prefetch_only], qd=1)
    print(f"[debug] A4 direct-prefetch-only sum={s:.6e} mode={m}", flush=True)
    raise AssertionError("probe complete - see [debug] annotations")
