"""GPU layer: WGSL load + compute pipeline cache (paths under gpu/shaders/)."""

from __future__ import annotations

import importlib.resources
import re
import struct
from dataclasses import dataclass
from typing import Any

from ebsdsim.gpu.buffers import StorageBuffer

_WGSL_CACHE: dict[str, str] = {}


def load_wgsl(name: str) -> str:
    """Load a WGSL source file from ``ebsdsim/gpu/shaders/`` (cached after first read).

    ``name`` is a path relative to ``shaders/``, e.g. ``dynamical/excitation_score.wgsl``.
    """
    cached = _WGSL_CACHE.get(name)
    if cached is not None:
        return cached
    parts = name.replace("\\", "/").split("/")
    path = importlib.resources.files("ebsdsim").joinpath("gpu", "shaders", *parts)
    code = path.read_text(encoding="utf-8")
    _WGSL_CACHE[name] = code
    return code


def infer_storage_read_only(wgsl: str, n_storage_bindings: int) -> list[bool]:
    """Infer read-only flags from ``@binding(N) var<storage, read>`` declarations."""
    ro = [False] * n_storage_bindings
    for match in re.finditer(
        r"@binding\((\d+)\)\s+var<storage,\s*(read|read_write)>",
        wgsl,
    ):
        binding = int(match.group(1))
        if 1 <= binding <= n_storage_bindings:
            ro[binding - 1] = match.group(2) == "read"
    return ro


def resolve_storage_read_only(
    wgsl: str,
    n_storage_bindings: int,
    explicit: bool | list[bool] | None,
) -> list[bool]:
    if explicit is not None:
        if isinstance(explicit, bool):
            return [explicit] * n_storage_bindings
        return list(explicit)
    return infer_storage_read_only(wgsl, n_storage_bindings)


def workgroups_1d(count: int, wg_size: int) -> tuple[int, int, int]:
    count = int(count)
    if count <= 0:
        return (0, 1, 1)
    return ((count + wg_size - 1) // wg_size, 1, 1)


@dataclass
class BufferResource:
    buffer: Any
    size: int | None = None


class PipelineCache:
    """Lazy WGSL compute pipeline cache."""

    def __init__(self, device: Any) -> None:
        self.device = device
        self._pipelines: dict[str, Any] = {}
        self._layouts: dict[str, Any] = {}
        # Reusable uniform buffers by dispatch label (bind groups stay per-call —
        # LU chunk workspaces are short-lived and must not be retained in a BG cache).
        self._params: dict[str, Any] = {}
        self._param_sizes: dict[str, int] = {}

    def clear(self) -> None:
        self._pipelines.clear()
        self._layouts.clear()
        for buf in self._params.values():
            buf.destroy()
        self._params.clear()
        self._param_sizes.clear()

    def __del__(self) -> None:
        try:
            self.clear()
        except Exception:
            pass

    def _params_buf(self, label: str, size: int) -> Any:
        from wgpu import BufferUsage

        size = max(int(size), 16)
        existing = self._params.get(label)
        if existing is not None and self._param_sizes[label] >= size:
            return existing
        if existing is not None:
            existing.destroy()
        buf = self.device.create_buffer(
            size=size,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label=f"{label}:params",
        )
        self._params[label] = buf
        self._param_sizes[label] = size
        return buf

    def get_pipeline(
        self,
        key: str,
        code: str,
        *,
        n_storage_bindings: int,
        uniform_size: int = 16,
        storage_read_only: bool | list[bool] | None = None,
    ) -> tuple[Any, Any]:
        cache_key = key
        layout = self._layouts.get(cache_key)
        pipeline = self._pipelines.get(cache_key)
        if pipeline is not None and layout is not None:
            return pipeline, layout

        shader = self.device.create_shader_module(code=code, label=key)
        pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
            label=key,
        )
        bind_layout = pipeline.get_bind_group_layout(0)
        self._layouts[cache_key] = bind_layout
        self._pipelines[cache_key] = pipeline
        return pipeline, bind_layout

    def dispatch_with_params(
        self,
        queue: Any,
        key: str,
        code: str,
        params_data: bytes | memoryview,
        resources: list[BufferResource | StorageBuffer | dict[str, Any]],
        workgroups: tuple[int, int, int],
        *,
        label: str,
        n_storage_bindings: int,
        uniform_size: int = 16,
        storage_read_only: bool | list[bool] | None = None,
    ) -> None:
        # One submit per call (callers may read back immediately). Uniform buffers
        # are reused by label; bind groups are created fresh each call.
        x, y, z = workgroups
        if x == 0 or y == 0 or z == 0:
            return

        params_size = max(uniform_size, 16, len(params_data))
        pdata = bytes(params_data)
        if len(pdata) < params_size:
            pdata = pdata + b"\x00" * (params_size - len(pdata))
        params_buf = self._params_buf(label, params_size)
        queue.write_buffer(params_buf, 0, pdata)

        pipeline, layout = self.get_pipeline(
            key,
            code,
            n_storage_bindings=n_storage_bindings,
            uniform_size=uniform_size,
            storage_read_only=storage_read_only,
        )

        entries: list[dict[str, Any]] = [
            {"binding": 0, "resource": {"buffer": params_buf, "size": params_size}}
        ]
        for i, resource in enumerate(resources):
            if isinstance(resource, StorageBuffer):
                binding = resource.buffer_binding()
            elif isinstance(resource, BufferResource):
                size = resource.size if resource.size is not None else None
                binding = (
                    {"buffer": resource.buffer, "size": size}
                    if size is not None
                    else {"buffer": resource.buffer}
                )
            else:
                binding = resource
            entries.append({"binding": i + 1, "resource": binding})

        bind_group = self.device.create_bind_group(
            layout=layout, entries=entries, label=f"{label}:bg"
        )
        encoder = self.device.create_command_encoder(label=label)
        pass_ = encoder.begin_compute_pass(label=label)
        pass_.set_pipeline(pipeline)
        pass_.set_bind_group(0, bind_group)
        pass_.dispatch_workgroups(x, y, z)
        pass_.end()
        queue.submit([encoder.finish()])


def make_mixed_params(size: int, writer) -> bytes:
    buf = bytearray(size)
    writer(struct.Struct("<").pack_into, buf)
    return bytes(buf)
