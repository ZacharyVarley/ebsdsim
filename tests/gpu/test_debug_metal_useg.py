"""TEMPORARY Metal debug probe: sg_003_1536903 unique-seg zeros on CI.

A/B: default shader (unique-seg heavy at 32K budget) vs force_dense_tile
(bypasses all uniq bitset/meta/vals machinery; reads f32 table directly).
Delete this file once the Metal failure is root-caused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ebsdsim.crystal.build import build_cell_from_cif_path
from ebsdsim.gpu.device import require_gpu
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

_CIF = Path(__file__).resolve().parents[1] / "data" / "cif" / "sg_003_1536903.cif"

pytestmark = pytest.mark.gpu


def _scalars(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)):
            out[k] = v
    return out


def test_metal_useg_ab() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    cell = build_cell_from_cif_path(_CIF)
    one_bin = [VoltageBin(20.0, None, 1.0, 1.0, 1.0, 0)]
    prep, bins = open_smith_iterative_prep(
        "sg_003_1536903", halfw=10, voltage_kv=20.0, dmin=0.05, cell=cell, bins=list(one_bin)
    )
    try:
        budget = int(prep.device.limits["max-compute-workgroup-storage-size"])
        print(f"\n[debug] device workgroup storage budget = {budget}")
        act = activate_voltage_bin(prep, bins[0])
        n, nw, n_g = int(act["n_strong"]), int(act["n_weak"]), int(act["n_g"])
        tile = auto_tile_k(n_g, n, nw, prep.n_sites)
        for label, kw in (
            ("default", {}),
            ("denseforce", {"force_dense_tile": True}),
        ):
            code, mode = load_smith_iterative_shader(n, shared_budget=budget, **kw)
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
                collect_stats=True,
            )
            inten = np.asarray(dyn["bin_intensities"][0], dtype=np.float64)
            print(
                f"[debug] {label}: mode={mode} "
                f"sum={inten.sum():.6e} max={inten.max():.6e} "
                f"nonzero={np.count_nonzero(inten)}/{inten.size}\n"
                f"[debug] {label} scalars={_scalars(dyn)}",
                flush=True,
            )
    finally:
        prep.close()
