"""TEMPORARY Metal bin-threshold + upload-sync probe.

At halfw=10 each bin is a single k-chunk; the e2e repeats per-bin:
resident.update (write_buffer of a multi-MB table) -> prescan -> solve ->
readback. One bin is bit-correct on Metal; four bins are all-zero. Find
the bin threshold (max_bins_run=1,2) and test whether syncing the device
right after the table upload (ResidentTables.update) repairs multi-bin
runs. Always asserts false to surface the report.
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
from ebsdsim.gpu.device import get_device, require_gpu, sync_device
from ebsdsim.gpu.dynamical.kernels import EBSDDynamicalKernels
from ebsdsim.gpu.resident import ResidentTables

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
_CELLS = [("sg_004_1004038", "FAIL"), ("sg_003_1536903", "CTRL")]

pytestmark = pytest.mark.gpu

_orig_update = ResidentTables.update


def _update_with_sync(self, lookup):
    _orig_update(self, lookup)
    sync_device(self.buffers.diff_table.device)


def _run(cell, mc, kernels, max_bins_run):
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
        max_bins_run=max_bins_run,
    )
    bins = [float(np.sum(b)) for b in result.bin_patterns]
    return float(np.sum(result.integrated)), bins


def test_metal_bin_threshold_and_upload_sync() -> None:
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
        configs = [(1, False), (2, False), (4, False), (4, True)]
        for mrb, with_sync in configs:
            ctx = get_device(force=True, required_features=("shader-f16",))
            kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
            if with_sync:
                ResidentTables.update = _update_with_sync
            try:
                integ, bins = _run(cell, mc, kernels, mrb)
            finally:
                ResidentTables.update = _orig_update
                kernels.destroy()
            print(
                f"[debug] {stem}({tag}) mrb={mrb} upload_sync={with_sync} "
                f"integrated={integ:.6e} bin_sums={bins}",
                flush=True,
            )
    raise AssertionError("probe complete - see [debug] annotations")
