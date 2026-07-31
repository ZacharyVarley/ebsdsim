"""TEMPORARY Metal buffer-integrity probe for the two deterministic-zero cells.

Both matvec modes (unique-seg and dense-tile) return all-zero on Metal, and
prescan picks the correct bucket - so the lookup table is read correctly at
prescan time but something is zero by the time intensities come back.
This probe keeps prep handles: read the resident GPU tables back AFTER
prescan (with COPY_SRC forced on all StorageBuffers), then run one solve
chunk and print its intensity sum. Control cell sg_003_1536903 (passes on
Metal) included. Always asserts false to surface the report.
Delete once the Metal behavior is understood.
"""

from __future__ import annotations

from pathlib import Path

import ebsdsim.gpu.buffers as gpu_buffers
import numpy as np
import pytest
from ebsdsim.crystal.build import build_cell_from_cif_path
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

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
_CELLS = [("sg_004_1004038", "FAIL"), ("sg_168_2232341", "FAIL"), ("sg_003_1536903", "CTRL")]
_BINS = [VoltageBin(20.0, None, 1.0, 1.0, 1.0, 0)]

pytestmark = pytest.mark.gpu

_orig_sb_init = gpu_buffers.StorageBuffer.__init__


def _sb_init_copy_src(self, device, queue, **kw):
    kw["copy_src"] = True
    _orig_sb_init(self, device, queue, **kw)


def _table_report(prep, lookup) -> str:
    parts = []
    host = {
        "diff_table": np.asarray(lookup.diff_table, dtype=np.float32),
        "coupling": np.asarray(lookup.coupling, dtype=np.float32),
    }
    for name in ("diff_table", "coupling", "sgh_tables", "hkl_hash"):
        buf = getattr(prep.resident.buffers, name)
        dt = np.float32 if name != "hkl_hash" else np.int32
        back = buf.read_as(dt)
        if name in host:
            exp = host[name]
            agree = bool(np.allclose(back, exp, rtol=1e-5, atol=1e-7))
            parts.append(
                f"{name}[{back.size}] sum={float(np.sum(np.abs(back))):.4e} "
                f"host_sum={float(np.sum(np.abs(exp))):.4e} AGREE={agree}"
            )
        else:
            parts.append(f"{name}[{back.size}] sum={float(np.sum(np.abs(back))):.4e}")
    return " | ".join(parts)


def test_metal_buffer_integrity() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    gpu_buffers.StorageBuffer.__init__ = _sb_init_copy_src
    try:
        for stem, tag in _CELLS:
            get_device(force=True, required_features=("shader-f16",))
            ctx = get_device(required_features=("shader-f16",))
            kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
            cell = build_cell_from_cif_path(_CIF_DIR / f"{stem}.cif")
            prep, bins = open_smith_iterative_prep(
                stem, halfw=10, voltage_kv=20.0, dmin=0.05, cell=cell, bins=list(_BINS)
            )
            try:
                act = activate_voltage_bin(prep, bins[0])
                lookup = act["lookup"]
                print(
                    f"[debug] {stem}({tag}) n_strong={act['n_strong']} n_g={act['n_g']} "
                    f"tables: {_table_report(prep, lookup)}",
                    flush=True,
                )
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
                    max_chunks=1,
                    queue_depth=1,
                )
                inten = np.asarray(dyn["intensities"], dtype=np.float64)
                print(
                    f"[debug] {stem}({tag}) mode={mode} chunk_solve: "
                    f"intensity_sum={float(inten.sum()):.6e} k_solved={dyn['k_solved']} "
                    f"fail_k={dyn['fail_k']}",
                    flush=True,
                )
            finally:
                prep.close()
                kernels.destroy()
    finally:
        gpu_buffers.StorageBuffer.__init__ = _orig_sb_init
    raise AssertionError("probe complete - see [debug] annotations")
