"""TEMPORARY Metal conditioning probe.

Established: the Metal zeros are triggered solely by the mu (absorption)
scalar, which enters only via slim_q's Smith shift q. With real absorption
q collapses (sg_004: 0.144 -> 0.023) leaving a near-singular (q1*I - A).
Windows survives; Metal returns exact zeros, which the shader can only
produce via the divergence guard (r_n NaN or > 1e20 -> yd = 0).

Floor q (pure conditioning change, same code path) and see whether Metal
comes back to life. Reports q, |w|max, BiCGSTAB iterations and fail flags
(note: fail flags require collect_stats, absent from earlier probes).
Always asserts false to surface the report. Delete once understood.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ebsdsim.gpu.dynamical.smith_iterative as si
import numpy as np
import pytest
from ebsdsim.crystal.build import build_cell_from_cif_path
from ebsdsim.energy.surrogate import infer_direct_exp_from_cell_rebinned
from ebsdsim.engine.integrate import surrogate_to_multi_voltage_mc
from ebsdsim.engine.smith_iterative_runner import _voltage_bins_from_mc
from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.gpu.device import get_device, require_gpu
from ebsdsim.gpu.dynamical.kernels import EBSDDynamicalKernels
from ebsdsim.gpu.dynamical.smith_iterative_prep import (
    VoltageBin,
    activate_voltage_bin,
    open_smith_iterative_prep,
)

_CIF_DIR = Path(__file__).resolve().parents[1] / "data" / "cif"
_CELLS = [("sg_004_1004038", "FAIL"), ("sg_168_2232341", "FAIL"), ("sg_003_1536903", "CTRL")]
_Q_LINE = "let q = sqrt(max(fro_per_dim * max(mean_decay, eps), eps));"

pytestmark = pytest.mark.gpu

_orig_load = si.load_wgsl
_orig_ws = si._slim_workspace
_orig_destroy = si._destroy_ws
_cap: dict[str, object] = {}


def _floor_q(code: str, floor: float) -> str:
    return code.replace(
        _Q_LINE,
        f"let q = max(sqrt(max(fro_per_dim * max(mean_decay, eps), eps)), {floor:.6f});",
    )


def _ws_capture(kernels, *, B, n_g, n, n_w, n_sites):
    dev, q = kernels.device, kernels.queue

    def sb(label, nbytes):
        return StorageBuffer(dev, q, label=label, byte_length=nbytes, copy_src=True, copy_dst=True)

    ws = SimpleNamespace(
        sg=sb("hp:sg", 4 * B * n_g),
        scores=sb("hp:scores", 4 * B * n_g),
        candidate_mask=sb("hp:cmask", 4 * B * n_g),
        selected_flags=sb("hp:flags", 4 * B * n_g),
        idx_a=sb("hp:idxA", 4 * B * n),
        idx_w=sb("hp:idxW", max(4, 4 * B * n_w)),
        d_a=sb("hp:dA", 8 * B * n),
        e0=sb("hp:e0", 8 * B * n),
        q_values=sb("hp:q", 4 * B * 4),
        w_stack=sb("hp:w", 8 * B * n * si.RANK),
        intensities=sb("hp:I", 4 * B * n_sites),
    )
    _cap["ws"], _cap["n"], _cap["B"] = ws, n, B
    return ws


def _run(stem: str, vb: VoltageBin, floor: float | None, chunk_cap: int = 2048) -> str:
    cell = build_cell_from_cif_path(_CIF_DIR / f"{stem}.cif")
    get_device(force=True, required_features=("shader-f16",))
    ctx = get_device(required_features=("shader-f16",))
    kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
    prep, bins = open_smith_iterative_prep(
        stem, halfw=10, voltage_kv=20.0, dmin=0.05, cell=cell, bins=[vb]
    )
    si._slim_workspace = _ws_capture
    si._destroy_ws = lambda ws: None
    if floor is not None:
        si.load_wgsl = lambda name: _floor_q(_orig_load(name), floor)
    try:
        act = activate_voltage_bin(prep, bins[0])
        budget = int(prep.device.limits["max-compute-workgroup-storage-size"])
        code, _ = si.load_smith_iterative_shader(int(act["n_strong"]), shared_budget=budget)
        tile = si.auto_tile_k(
            int(act["n_g"]), int(act["n_strong"]), int(act["n_weak"]), prep.n_sites
        )
        dyn = si.run_bins_dynamical(
            prep,
            bins,
            chunk=min(tile, chunk_cap),
            code_topk=_orig_load("dynamical/topk_radix_exact.wgsl"),
            code_slim=si.load_wgsl("dynamical/smith_iterative_slim_q.wgsl"),
            code_smith_iterative=code,
            code_inten=_orig_load("dynamical/intensity_fused_exact.wgsl"),
            first_act=act,
            queue_depth=1,
            collect_stats=True,
        )
        ws, n, B = _cap["ws"], _cap["n"], _cap["B"]
        rows = min(int(B), 528)
        qv = ws.q_values.read_as(np.float32).reshape(int(B), 4)[:rows]
        w = ws.w_stack.read_as(np.float32).reshape(int(B), si.RANK, int(n), 2)[:rows]
        st = np.asarray(dyn["mode_flags"], dtype=np.uint32).ravel()[: int(B)]
        fail = int(np.sum(st == 0xFFFFFFFF))
        it = (st[st != 0xFFFFFFFF] & 0x0FFFFFFF).astype(np.int64)
        it_txt = f"max={it.max()} med={int(np.median(it))}" if it.size else "n/a"
        _orig_destroy(ws)
        return (
            f"q=[{qv[:,0].min():.3e},{qv[:,0].max():.3e}] q1={qv[0,1]:+.3e} "
            f"scale={qv[0,3]:.3e} |w|max={np.abs(w).max():.4e} iters {it_txt} "
            f"fail={fail}/{rows} I={float(np.sum(dyn['intensities'])):.6e}"
        )
    finally:
        si.load_wgsl = _orig_load
        si._slim_workspace = _orig_ws
        si._destroy_ws = _orig_destroy
        prep.close()
        kernels.destroy()


def test_metal_submit_size() -> None:
    """q_values/stats read back as zero-init => the submit was aborted, not miscomputed.

    Shrinking the k-chunk shortens each command buffer without changing any
    numerics. If small chunks land, the Metal command buffer is being killed
    on long submits.
    """
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    for stem, tag in _CELLS:
        cell = build_cell_from_cif_path(_CIF_DIR / f"{stem}.cif")
        d = infer_direct_exp_from_cell_rebinned(
            cell=cell, sigma_deg=70.0, beam_kv=20.0, energy_binwidth_keV=5.0, n_energy_bins=4
        )
        b0 = _voltage_bins_from_mc(surrogate_to_multi_voltage_mc(d, 20.0))[0]
        vb = VoltageBin(20.0, None, b0.beta, 1.0, 1.0, 0)
        for chunk in (2048, 128, 32):
            print(
                f"[debug] {stem}({tag}) chunk={chunk:<5d} {_run(stem, vb, None, chunk)}",
                flush=True,
            )
    raise AssertionError("probe complete - see [debug] annotations")
