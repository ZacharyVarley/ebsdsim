"""GPU Lambert raster expansion (parity with ebsdsim-web worker)."""

from __future__ import annotations

import struct
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from wgpu import BufferUsage

from ebsdsim.gpu.buffers import StorageBuffer, f32_bytes, u32_bytes
from ebsdsim.gpu.pipelines import PipelineCache, load_wgsl, workgroups_1d
from ebsdsim.kgrid import PgKGrid
from ebsdsim.normalize import NormalizeMode, scale_fs_channel
from ebsdsim.rasterize import build_rasterize_pixel_maps
from ebsdsim.weights import reduce_over_sites


LAMBERT_FILL_WORKGROUP_SIZE = 64
_MAX_DISPATCH_WORKGROUPS = 65535


def _workgroups_1d_limited(count: int, wg_size: int) -> tuple[int, int, int, int]:
    """Return ``(wx, wy, wz, threads_per_row)`` for large 1-D dispatches."""
    if count <= 0:
        return (0, 1, 1, 0)
    total_wg = (count + wg_size - 1) // wg_size
    if total_wg <= _MAX_DISPATCH_WORKGROUPS:
        return (total_wg, 1, 1, total_wg * wg_size)
    wx = _MAX_DISPATCH_WORKGROUPS
    wy = (total_wg + wx - 1) // wx
    return (wx, wy, 1, wx * wg_size)


def _to_u32_flags(values: NDArray[np.uint8]) -> NDArray[np.uint32]:
    return values.astype(np.uint32, copy=False)


def _build_per_direction_views(
    values_fs: NDArray[np.floating],
    n_k: int,
    n_sites: int,
    site_weights: NDArray[np.floating] | None = None,
) -> list[NDArray[np.float32]]:
    """Site-marginal view first, then per-site views (matches ``build_master_pattern_data``)."""
    arr = np.asarray(values_fs, dtype=np.float32).reshape(n_k, n_sites)
    if n_sites > 1:
        views = [reduce_over_sites(arr, site_weights)]
        for s in range(n_sites):
            views.append(arr[:, s].astype(np.float32, copy=False))
        return views
    return [arr[:, 0].astype(np.float32, copy=False)]


def _scatter_views_to_sheet_stack(
    views: list[NDArray[np.float32]],
    kij: NDArray[np.integer],
    hw: int,
    side: int,
    *,
    normalize: NormalizeMode | None,
    robust_p_low: float,
    robust_p_high: float,
) -> NDArray[np.float32]:
    plane = side * side
    kij = np.asarray(kij).reshape(-1, 3)
    ix = kij[:, 0].astype(np.intp) + hw
    iy = kij[:, 1].astype(np.intp) + hw
    sign = kij[:, 2]
    dst = iy * side + ix
    north = sign > 0
    stacked = np.zeros(len(views) * 2 * plane, dtype=np.float32)
    for v, per_dir in enumerate(views):
        vals = scale_fs_channel(
            per_dir, normalize, robust_p_low=robust_p_low, robust_p_high=robust_p_high
        )
        sheet_nh = np.zeros(plane, dtype=np.float32)
        sheet_sh = np.zeros(plane, dtype=np.float32)
        sheet_nh[dst[north]] = vals[north]
        sheet_sh[dst[~north]] = vals[~north]
        base = v * 2 * plane
        stacked[base : base + plane] = sheet_nh
        stacked[base + plane : base + 2 * plane] = sheet_sh
    return stacked


def _reshape_lambert_stack(
    stack: NDArray[np.float32],
    *,
    view_count: int,
    side: int,
    e_dim: int,
    s_dim: int,
    h_dim: int,
    normalize: NormalizeMode | None,
) -> NDArray[np.float32]:
    plane = side * side
    view_stride = plane * 2

    def _plane(slice_: NDArray[np.float32]) -> NDArray[np.float32]:
        arr = slice_.reshape(side, side)
        if normalize is not None:
            return np.clip(arr, 0.0, 1.0)
        return arr

    data = np.empty((e_dim, s_dim, h_dim, side, side), dtype=np.float32)
    for c in range(view_count):
        base = c * view_stride
        nh = _plane(stack[base : base + plane])
        if h_dim == 1:
            e_idx = c // s_dim
            s_idx = c % s_dim
            data[e_idx, s_idx, 0] = nh
        else:
            sh = _plane(stack[base + plane : base + view_stride])
            e_idx = c // s_dim
            s_idx = c % s_dim
            data[e_idx, s_idx, 0] = nh
            data[e_idx, s_idx, 1] = sh
    return data


class GpuLambertRasterizer:
    """Persistent GPU Lambert fill matching ``ebsdsim-web`` ``GpuLambertRasterizer``."""

    def __init__(
        self,
        device: Any,
        queue: Any,
        pg_grid: PgKGrid,
        *,
        pipelines: PipelineCache | None = None,
    ) -> None:
        self.device = device
        self.queue = queue
        self.side = int(pg_grid.side)
        self.plane_size = self.side * self.side
        self.is_centro = bool(pg_grid.is_centro)
        pixel_maps = build_rasterize_pixel_maps(pg_grid)
        sh_map = pixel_maps.sh or pixel_maps.nh
        self._nh_src_x = StorageBuffer(
            device, queue, label="lambert:nh-src-x", data=pixel_maps.nh.src_x, copy_dst=False
        )
        self._nh_src_y = StorageBuffer(
            device, queue, label="lambert:nh-src-y", data=pixel_maps.nh.src_y, copy_dst=False
        )
        self._nh_from_sh = StorageBuffer(
            device,
            queue,
            label="lambert:nh-from-sh",
            data=_to_u32_flags(pixel_maps.nh.from_sh),
            copy_dst=False,
        )
        self._sh_src_x = StorageBuffer(
            device, queue, label="lambert:sh-src-x", data=sh_map.src_x, copy_dst=False
        )
        self._sh_src_y = StorageBuffer(
            device, queue, label="lambert:sh-src-y", data=sh_map.src_y, copy_dst=False
        )
        self._sh_from_sh = StorageBuffer(
            device,
            queue,
            label="lambert:sh-from-sh",
            data=_to_u32_flags(sh_map.from_sh),
            copy_dst=False,
        )
        self._pipelines = pipelines or PipelineCache(device)
        self._wgsl = load_wgsl("ebsd_lambert_fill.wgsl")
        self._pipeline, self._layout = self._pipelines.get_pipeline(
            "ebsd:lambert-fill",
            self._wgsl,
            n_storage_bindings=8,
            uniform_size=32,
        )
        self._sheet_buffer: StorageBuffer | None = None
        self._out_buffer: StorageBuffer | None = None
        self._capacity_views = 0

    def _ensure_capacity(self, view_count: int) -> None:
        if view_count <= self._capacity_views and self._sheet_buffer and self._out_buffer:
            return
        if self._sheet_buffer is not None:
            self._sheet_buffer.destroy()
        if self._out_buffer is not None:
            self._out_buffer.destroy()
        byte_length = max(4, view_count * 2 * self.plane_size * 4)
        self._sheet_buffer = StorageBuffer(
            self.device,
            self.queue,
            label="lambert:sheets",
            byte_length=byte_length,
            copy_dst=True,
        )
        self._out_buffer = StorageBuffer(
            self.device,
            self.queue,
            label="lambert:out",
            byte_length=byte_length,
            copy_src=True,
        )
        self._capacity_views = view_count

    def _dispatch(self, view_count: int) -> None:
        assert self._sheet_buffer is not None
        assert self._out_buffer is not None
        params_buf = self.device.create_buffer(
            size=32,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="lambert:params",
        )
        total = view_count * 2 * self.plane_size
        wx, wy, _, threads_per_row = _workgroups_1d_limited(
            total, LAMBERT_FILL_WORKGROUP_SIZE
        )
        params = struct.pack(
            "<IIIII",
            self.side,
            view_count,
            self.plane_size,
            1 if self.is_centro else 0,
            threads_per_row,
        )
        self.queue.write_buffer(params_buf, 0, params)
        entries = [
            {"binding": 0, "resource": {"buffer": self._sheet_buffer.buffer, "size": self._sheet_buffer.byte_length}},
            {"binding": 1, "resource": {"buffer": self._nh_src_x.buffer, "size": self._nh_src_x.byte_length}},
            {"binding": 2, "resource": {"buffer": self._nh_src_y.buffer, "size": self._nh_src_y.byte_length}},
            {"binding": 3, "resource": {"buffer": self._nh_from_sh.buffer, "size": self._nh_from_sh.byte_length}},
            {"binding": 4, "resource": {"buffer": self._sh_src_x.buffer, "size": self._sh_src_x.byte_length}},
            {"binding": 5, "resource": {"buffer": self._sh_src_y.buffer, "size": self._sh_src_y.byte_length}},
            {"binding": 6, "resource": {"buffer": self._sh_from_sh.buffer, "size": self._sh_from_sh.byte_length}},
            {"binding": 7, "resource": {"buffer": self._out_buffer.buffer, "size": self._out_buffer.byte_length}},
            {"binding": 8, "resource": {"buffer": params_buf, "size": 32}},
        ]
        bind_group = self.device.create_bind_group(
            layout=self._layout,
            entries=entries,
            label="ebsd:lambert-fill:bg",
        )
        encoder = self.device.create_command_encoder(label="ebsd:lambert-fill")
        pass_ = encoder.begin_compute_pass(label="ebsd:lambert-fill")
        pass_.set_pipeline(self._pipeline)
        pass_.set_bind_group(0, bind_group)
        pass_.dispatch_workgroups(wx, wy, 1)
        pass_.end()
        self.queue.submit([encoder.finish()])
        params_buf.destroy()

    def expand_sheets(
        self,
        stacked_sheets: NDArray[np.floating],
        view_count: int,
        *,
        readback: bool = True,
    ) -> NDArray[np.float32]:
        if view_count <= 0:
            return np.zeros(0, dtype=np.float32)
        self._ensure_capacity(view_count)
        assert self._sheet_buffer is not None
        assert self._out_buffer is not None
        stacked = np.asarray(stacked_sheets, dtype=np.float32)
        nbytes = view_count * 2 * self.plane_size * 4
        self._sheet_buffer.write(stacked.reshape(-1)[: view_count * 2 * self.plane_size])
        self._dispatch(view_count)
        if not readback:
            return np.zeros(0, dtype=np.float32)
        return self._out_buffer.read_as(np.float32, size=nbytes)

    def rasterize_fs_values(
        self,
        values_fs: NDArray[np.floating],
        n_k: int,
        n_sites: int,
        kij: NDArray[np.integer],
        hw: int,
        *,
        normalize: NormalizeMode | None = None,
        robust_p_low: float = 0.01,
        robust_p_high: float = 0.99,
        site_weights: NDArray[np.floating] | None = None,
        readback: bool = True,
    ) -> NDArray[np.float32]:
        """GPU Lambert fill for integrated FS values (per-bin preview parity with web)."""
        views = _build_per_direction_views(values_fs, n_k, n_sites, site_weights)
        stack = _scatter_views_to_sheet_stack(
            views,
            kij,
            hw,
            self.side,
            normalize=normalize,
            robust_p_low=robust_p_low,
            robust_p_high=robust_p_high,
        )
        return self.expand_sheets(stack, len(views), readback=readback)

    def destroy(self) -> None:
        if self._sheet_buffer is not None:
            self._sheet_buffer.destroy()
        if self._out_buffer is not None:
            self._out_buffer.destroy()
        self._nh_src_x.destroy()
        self._nh_src_y.destroy()
        self._nh_from_sh.destroy()
        self._sh_src_x.destroy()
        self._sh_src_y.destroy()
        self._sh_from_sh.destroy()


def build_master_pattern_data_gpu(
    rasterizer: GpuLambertRasterizer,
    *,
    integrated_fs: NDArray[np.floating],
    bin_fs: NDArray[np.floating],
    kij: NDArray[np.integer],
    hw: int,
    side: int,
    needs_southern_hemisphere: bool,
    normalize: NormalizeMode | None = None,
    robust_p_low: float = 0.01,
    robust_p_high: float = 0.99,
    site_weights: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Expand FS intensities on GPU into ``(E, S, H, side, side)`` (CPU sheet scatter)."""
    integrated_fs = np.asarray(integrated_fs, dtype=np.float32)
    bin_fs = np.asarray(bin_fs, dtype=np.float32)
    kij = np.asarray(kij)
    n_k, n_sites = int(integrated_fs.shape[0]), int(integrated_fs.shape[1])
    n_bins = int(bin_fs.shape[0])

    if n_bins > 1:
        energy_sources = [integrated_fs] + [bin_fs[b] for b in range(n_bins)]
        energy_integrated_index = 0
        bin_to_energy_index = [1 + b for b in range(n_bins)]
    else:
        energy_sources = [integrated_fs]
        energy_integrated_index = 0
        bin_to_energy_index = [0] * n_bins
    e_dim = len(energy_sources)

    multi_site = n_sites > 1
    s_dim = 1 + n_sites if multi_site else 1
    site_integrated_index = 0
    site_to_index = [1 + s for s in range(n_sites)] if multi_site else [0]

    all_views: list[NDArray[np.float32]] = []
    for esrc in energy_sources:
        all_views.extend(_build_per_direction_views(esrc, n_k, n_sites, site_weights))

    sheet_stack = _scatter_views_to_sheet_stack(
        all_views,
        kij,
        hw,
        side,
        normalize=normalize,
        robust_p_low=robust_p_low,
        robust_p_high=robust_p_high,
    )
    raster_stack = rasterizer.expand_sheets(sheet_stack, len(all_views))
    h_dim = 2 if needs_southern_hemisphere else 1
    data = _reshape_lambert_stack(
        raster_stack,
        view_count=len(all_views),
        side=side,
        e_dim=e_dim,
        s_dim=s_dim,
        h_dim=h_dim,
        normalize=normalize,
    )
    axes = {
        "dims": ["energy", "site", "hemisphere", "height", "width"],
        "energy_dim": e_dim,
        "site_dim": s_dim,
        "hemisphere_dim": h_dim,
        "energy_integrated_index": energy_integrated_index,
        "bin_to_energy_index": bin_to_energy_index,
        "site_integrated_index": site_integrated_index,
        "site_to_index": site_to_index,
        "hemispheres": ["north"] if h_dim == 1 else ["north", "south"],
    }
    return data, axes


def rasterize_nh_gpu(
    device: Any,
    queue: Any,
    pattern: NDArray[np.floating],
    grid: PgKGrid,
    *,
    n_sites: int,
    pipelines: PipelineCache | None = None,
) -> NDArray[np.float32]:
    """Rasterize one north-hemisphere Lambert image (site-mean view)."""
    n_k = grid.kij.size // 3
    rasterizer = GpuLambertRasterizer(device, queue, grid, pipelines=pipelines)
    try:
        stack = rasterizer.rasterize_fs_values(
            pattern,
            n_k,
            n_sites,
            grid.kij.reshape(-1, 3),
            grid.hw,
        )
        plane = grid.side * grid.side
        return np.clip(stack[:plane], 0.0, 1.0).reshape(grid.side, grid.side)
    finally:
        rasterizer.destroy()
