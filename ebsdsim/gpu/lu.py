"""GPU layer: batched complex64 LU factorization/solve (WasmGPU-compatible)."""

from __future__ import annotations

import struct
from typing import Any

from wgpu import BufferUsage

from ebsdsim.gpu.buffers import StorageBuffer, c64_bytes, u32_bytes
from ebsdsim.gpu.pipelines import PipelineCache, load_wgsl, workgroups_1d


class LuKernels:
    """GPU LU kernels for batched complex64 systems.

    Each dispatch submits its own command buffer, matching @zushah/wasmgpu
    (LU kernels do not support batched encoders).
    """

    def __init__(self, device: Any, queue: Any, pipelines: PipelineCache | None = None) -> None:
        self.device = device
        self.queue = queue
        self.pipelines = pipelines or PipelineCache(device)
        self._lu_batched_params = device.create_buffer(
            size=16,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="lu:batchedParams",
        )
        self._lu_blocked_params = device.create_buffer(
            size=32,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="lu:blockedParams",
        )

    def destroy(self) -> None:
        self._lu_batched_params.destroy()
        self._lu_blocked_params.destroy()

    def _write_batched_params(self, batch_count: int, n: int, elems_per_matrix: int) -> None:
        data = struct.pack("<IIII", batch_count, n, elems_per_matrix, 0)
        self.queue.write_buffer(self._lu_batched_params, 0, data)

    def _write_blocked_params(
        self,
        batch_count: int,
        n: int,
        elems_per_matrix: int,
        kk: int,
        pw: int,
    ) -> None:
        data = struct.pack("<IIIIIIII", batch_count, n, elems_per_matrix, kk, pw, 0, 0, 0)
        self.queue.write_buffer(self._lu_blocked_params, 0, data)

    def _submit_compute(self, pipeline: Any, bind_group: Any, workgroups: tuple[int, int, int], label: str) -> None:
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass(label=label)
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind_group)
        cp.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2])
        cp.end()
        self.queue.submit([enc.finish()])

    def lu_factor_complex64_batched(
        self,
        matrices: StorageBuffer,
        ipiv: StorageBuffer,
        batch_count: int,
        n: int,
    ) -> None:
        if batch_count <= 0 or n <= 0:
            return
        elems_per_matrix = n * n
        need_bytes = c64_bytes(batch_count * elems_per_matrix)
        ipiv_bytes = u32_bytes(batch_count * n)

        if n < 160:
            self._write_batched_params(batch_count, n, elems_per_matrix)
            code = load_wgsl("lu/factor_complex64.wgsl")
            pipeline, layout = self.pipelines.get_pipeline(
                "kernels:lu:factorComplexSmall",
                code,
                n_storage_bindings=2,
                uniform_size=16,
                storage_read_only=[False, False],
            )
            bg = self.device.create_bind_group(
                layout=layout,
                entries=[
                    {"binding": 0, "resource": {"buffer": self._lu_batched_params, "size": 16}},
                    {"binding": 1, "resource": matrices.buffer_binding(need_bytes)},
                    {"binding": 2, "resource": ipiv.buffer_binding(ipiv_bytes)},
                ],
            )
            self._submit_compute(pipeline, bg, (batch_count, 1, 1), "luFactorComplex64Batched")
            return

        lead_code = load_wgsl("lu/factor_lead_complex64.wgsl")
        upper_code = load_wgsl("lu/factor_upper_complex64.wgsl")
        trail_code = load_wgsl("lu/factor_trailing_complex64.wgsl")
        lead_pipe, lead_layout = self.pipelines.get_pipeline(
            "kernels:lu:factorComplexLead",
            lead_code,
            n_storage_bindings=2,
            uniform_size=32,
            storage_read_only=[False, False],
        )
        upper_pipe, upper_layout = self.pipelines.get_pipeline(
            "kernels:lu:factorComplexUpper",
            upper_code,
            n_storage_bindings=1,
            uniform_size=32,
            storage_read_only=[False],
        )
        trail_pipe, trail_layout = self.pipelines.get_pipeline(
            "kernels:lu:factorComplexTrailing",
            trail_code,
            n_storage_bindings=1,
            uniform_size=32,
            storage_read_only=[False],
        )

        bg_lead = self.device.create_bind_group(
            layout=lead_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._lu_blocked_params, "size": 32}},
                {"binding": 1, "resource": matrices.buffer_binding(need_bytes)},
                {"binding": 2, "resource": ipiv.buffer_binding(ipiv_bytes)},
            ],
        )
        bg_upper = self.device.create_bind_group(
            layout=upper_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._lu_blocked_params, "size": 32}},
                {"binding": 1, "resource": matrices.buffer_binding(need_bytes)},
            ],
        )
        bg_trail = self.device.create_bind_group(
            layout=trail_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._lu_blocked_params, "size": 32}},
                {"binding": 1, "resource": matrices.buffer_binding(need_bytes)},
            ],
        )

        panel_b = 16
        for kk in range(0, n, panel_b):
            pw = min(panel_b, n - kk)
            trail = n - (kk + pw)
            self._write_blocked_params(batch_count, n, elems_per_matrix, kk, pw)
            enc = self.device.create_command_encoder()
            cp = enc.begin_compute_pass(label=f"luFactorComplex64:kk{kk}")
            cp.set_pipeline(lead_pipe)
            cp.set_bind_group(0, bg_lead)
            cp.dispatch_workgroups(batch_count, 1, 1)
            if trail > 0:
                total_trsm = batch_count * trail
                wg = workgroups_1d(total_trsm, 256)
                cp.set_pipeline(upper_pipe)
                cp.set_bind_group(0, bg_upper)
                cp.dispatch_workgroups(wg[0], wg[1], wg[2])
                tile_m = 16
                tile_n = 8
                m_tiles = (trail + tile_m - 1) // tile_m
                n_tiles = (trail + tile_n - 1) // tile_n
                cp.set_pipeline(trail_pipe)
                cp.set_bind_group(0, bg_trail)
                cp.dispatch_workgroups(n_tiles, m_tiles, batch_count)
            cp.end()
            self.queue.submit([enc.finish()])

    def lu_solve_complex64_batched(
        self,
        lu: StorageBuffer,
        ipiv: StorageBuffer,
        rhs: StorageBuffer,
        out_x: StorageBuffer,
        batch_count: int,
        n: int,
    ) -> None:
        if batch_count <= 0 or n <= 0:
            return
        elems_per_matrix = n * n
        lu_bytes = c64_bytes(batch_count * elems_per_matrix)
        rhs_bytes = c64_bytes(batch_count * n)
        ipiv_bytes = u32_bytes(batch_count * n)

        self._write_batched_params(batch_count, n, elems_per_matrix)
        key = "kernels:lu:solveComplex" if n <= 512 else "kernels:lu:solveComplexLarge"
        wgsl_name = "lu/solve_shared_complex64.wgsl" if n <= 512 else "lu/solve_large_complex64.wgsl"
        code = load_wgsl(wgsl_name)
        pipeline, layout = self.pipelines.get_pipeline(
            key,
            code,
            n_storage_bindings=4,
            uniform_size=16,
            storage_read_only=[True, True, False, True],
        )
        bg = self.device.create_bind_group(
            layout=layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._lu_batched_params, "size": 16}},
                {"binding": 1, "resource": lu.buffer_binding(lu_bytes)},
                {"binding": 2, "resource": rhs.buffer_binding(rhs_bytes)},
                {"binding": 3, "resource": out_x.buffer_binding(rhs_bytes)},
                {"binding": 4, "resource": ipiv.buffer_binding(ipiv_bytes)},
            ],
        )
        self._submit_compute(pipeline, bg, (batch_count, 1, 1), "luSolveComplex64Batched")
