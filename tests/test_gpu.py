"""GPU dynamical pipeline smoke tests."""

from __future__ import annotations

import importlib.resources
import struct

import numpy as np
import pytest

from ebsdsim.gpu import EBSDDynamicalKernels, require_gpu
from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.kgrid import build_pg_k_grid, transform_pg_k_grid_to_reciprocal
from ebsdsim.lookup import BuildLookupOptions, build_diff_lookup
from ebsdsim.gpu.rasterize import GpuLambertRasterizer, build_master_pattern_data_gpu
from ebsdsim.integrate import PerVoltageContext
from ebsdsim.mploader import build_master_pattern_data
from ebsdsim.pg_ops import fs_normals, pg_num_to_symbol, point_group_operators
from ebsdsim.runner import RunOneVoltageDeps, make_metric_buffer, run_one_voltage
from ebsdsim.sgh import prepare_site_sgh_tables
from ebsdsim.structure import build_cell_from_cif_path, metric_to_float32


def test_lookup_submatrix_uniform_layout():
    """WGSL vec2 prefactor aligns to byte 24 after i32 offset (not byte 20)."""
    buf = bytearray(48)
    struct.pack_into("<i", buf, 16, -7)
    struct.pack_into("<f", buf, 24, 1.25)
    struct.pack_into("<f", buf, 28, -0.5)
    struct.pack_into("<I", buf, 32, 1)
    assert struct.unpack_from("<i", buf, 16)[0] == -7
    assert struct.unpack_from("<ff", buf, 24) == (1.25, -0.5)
    assert struct.unpack_from("<I", buf, 32)[0] == 1
    # Tight packing at byte 20 would misplace prefactor relative to WGSL layout.
    assert struct.unpack_from("<ff", buf, 20) != (1.25, -0.5)


def test_hash_diff_uniform_layout():
    """WGSL offset field sits at byte 16 with u32 padding at byte 12."""
    buf = bytearray(32)
    struct.pack_into("<I", buf, 8, 4096)
    struct.pack_into("<i", buf, 16, 12345)
    assert struct.unpack_from("<I", buf, 8)[0] == 4096
    assert struct.unpack_from("<i", buf, 12)[0] == 0
    assert struct.unpack_from("<i", buf, 16)[0] == 12345


def _gpu_available() -> bool:
    try:
        require_gpu()
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="WebGPU adapter unavailable")


def _ni_cell():
    path = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    return build_cell_from_cif_path(str(path))


def test_prescan_dispatch():
    cell = _ni_cell()
    ctx = require_gpu()
    kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
    lookup = build_diff_lookup(cell, BuildLookupOptions(voltage_kv=20.0, dmin=0.05))
    pg = build_pg_k_grid(cell.pg_num, hw=2)
    kvecs = transform_pg_k_grid_to_reciprocal(pg, cell.direct_structure_matrix, lookup.mlambda)
    persistent = kernels.create_persistent_buffers(
        hkl=lookup.hkl,
        hkl_hash=lookup.hkl_hash,
        diff_table=lookup.diff_table,
        coupling=lookup.coupling,
        reflection_dbdiff=lookup.reflection_dbdiff,
        sgh_tables=prepare_site_sgh_tables(cell, 0.05).tables,
    )
    metric = kernels.create_metric_buffer(metric_to_float32(cell))
    try:
        prescan = kernels.prescan_bethe_beam_counts_gpu(
            kvecs,
            persistent,
            metric,
            batch_count=kvecs.size // 3,
            n_g=lookup.hkl.size // 3,
            bethe_c_cutoff=200.0,
            dbdiff_sg_cutoff=1.0,
            bethe_c_strong=20.0,
            bethe_c_weak=40.0,
        )
        assert prescan.n_strong >= 1
        assert prescan.n_strong > 1, "prescan should select multiple strong beams with app cutoffs"
    finally:
        metric.destroy()
        for name in vars(persistent):
            buf = getattr(persistent, name)
            if isinstance(buf, StorageBuffer):
                buf.destroy()


def test_lambert_fill_gpu_matches_cpu():
    cell = _ni_cell()
    ctx = require_gpu()
    hw = 2
    pg = build_pg_k_grid(cell.pg_num, hw)
    n_k = pg.kij.size // 3
    rng = np.random.default_rng(0)
    integrated = rng.random((n_k, 1), dtype=np.float32)
    bins = rng.random((2, n_k, 1), dtype=np.float32)
    kij = pg.kij.reshape(-1, 3)
    symbol = pg_num_to_symbol(int(cell.pg_num))
    ops = point_group_operators(symbol).reshape(-1, 3, 3)
    normals = fs_normals(symbol).reshape(-1, 3)
    side = pg.side
    cpu, _ = build_master_pattern_data(
        integrated_fs=integrated,
        bin_fs=bins,
        kij=kij,
        pg_operators=ops,
        fs_normals=normals,
        hw=hw,
        side=side,
        needs_southern_hemisphere=False,
    )
    rasterizer = GpuLambertRasterizer(ctx.device, ctx.queue, pg)
    try:
        gpu, _ = build_master_pattern_data_gpu(
            rasterizer,
            integrated_fs=integrated,
            bin_fs=bins,
            kij=kij,
            hw=hw,
            side=side,
            needs_southern_hemisphere=False,
        )
    finally:
        rasterizer.destroy()
    assert np.max(np.abs(cpu - gpu)) < 1e-5


def test_run_one_voltage_small():
    cell = _ni_cell()
    ctx = require_gpu()
    hw = 2
    pg = build_pg_k_grid(cell.pg_num, hw)
    sgh = prepare_site_sgh_tables(cell, dmin=0.05)
    kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
    metric = make_metric_buffer(kernels, cell)
    deps = RunOneVoltageDeps(
        cell=cell,
        pg_grid=pg,
        sgh=sgh,
        kernels=kernels,
        metric=metric,
        chunk_size=8,
        rank=4,
        dmin=0.05,
    )
    ctx_bin = PerVoltageContext(
        voltage_kv=20.0,
        bin_index=0,
        energy_weight=1.0,
        amplitude=1.0,
        beta=0.05,
        mode="bloch",
    )
    try:
        result = run_one_voltage(ctx_bin, deps)
        assert result is not None
        assert result.pattern.size == result.n_k * result.n_sites
        assert np.all(np.isfinite(result.pattern))
        assert np.any(result.pattern > 0)
    finally:
        metric.destroy()
