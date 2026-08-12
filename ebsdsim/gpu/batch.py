"""Persistent GPU dispatch: rewrite uniforms in place, cache bind groups, one submit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wgpu import BufferUsage

from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.gpu.pipelines import BufferResource, PipelineCache, resolve_storage_read_only


@dataclass
class ResourceBinding:
    """Storage resource with an optional explicit WebGPU binding index.

    When ``binding`` is ``None``, assignment is positional (``list index + 1``),
    matching historical ``DispatchItem.resources`` behaviour. Set ``binding`` to
    skip reserved slots (e.g. galerkin solve uses 14/15 while leaving 12/13 free).
    """

    resource: Any
    binding: int | None = None


@dataclass
class DispatchItem:
    key: str
    code: str
    params_data: bytes
    resources: list
    workgroups: tuple[int, int, int]
    label: str
    n_storage_bindings: int
    uniform_size: int = 16


def _unwrap_resource(resource: Any) -> tuple[Any, int | None]:
    if isinstance(resource, ResourceBinding):
        return resource.resource, resource.binding
    return resource, None


def resource_binding_index(resource: Any, position: int) -> int:
    """Resolve the WebGPU storage binding for ``resources[position]``."""
    _res, explicit = _unwrap_resource(resource)
    if explicit is not None:
        return int(explicit)
    return position + 1


def binding_indices_for(resources: list) -> list[int]:
    """Binding numbers for each entry in a resources list (sparse-aware)."""
    return [resource_binding_index(r, i) for i, r in enumerate(resources)]


def _resource_binding(resource: Any) -> dict[str, Any]:
    if isinstance(resource, StorageBuffer):
        return resource.buffer_binding()
    if isinstance(resource, BufferResource):
        size = resource.size if resource.size is not None else None
        return (
            {"buffer": resource.buffer, "size": size}
            if size is not None
            else {"buffer": resource.buffer}
        )
    return resource


def _resource_token(resource: Any) -> int:
    if isinstance(resource, StorageBuffer):
        return id(resource.buffer)
    if isinstance(resource, BufferResource):
        return id(resource.buffer)
    if isinstance(resource, dict) and "buffer" in resource:
        return id(resource["buffer"])
    return id(resource)


class PersistentSubmitter:
    """Reuse uniform buffers + bind groups; record many passes into one queue.submit."""

    def __init__(self, pipelines: PipelineCache, queue: Any) -> None:
        self.pipelines = pipelines
        self.queue = queue
        self.device = pipelines.device
        self._params: dict[str, Any] = {}
        self._param_sizes: dict[str, int] = {}
        self._bind_groups: dict[tuple, Any] = {}

    def _params_buf(self, label: str, size: int) -> Any:
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
        # Bind groups that referenced the old buffer are invalid.
        self._bind_groups = {
            k: v for k, v in self._bind_groups.items() if k[0] != label
        }
        return buf

    def _bind_group(
        self,
        *,
        label: str,
        layout: Any,
        params_buf: Any,
        params_size: int,
        resources: list,
    ) -> Any:
        tokens = tuple(
            (_resource_token(_unwrap_resource(r)[0]), resource_binding_index(r, i))
            for i, r in enumerate(resources)
        )
        token = (label, id(layout), id(params_buf), tokens)
        cached = self._bind_groups.get(token)
        if cached is not None:
            return cached
        # Drop other bind groups for this label so we do not retain refs to
        # destroyed storage buffers (LU chunk path recreates workspace often).
        self._bind_groups = {k: v for k, v in self._bind_groups.items() if k[0] != label}
        entries: list[dict[str, Any]] = [
            {"binding": 0, "resource": {"buffer": params_buf, "size": params_size}}
        ]
        for i, resource in enumerate(resources):
            raw, _explicit = _unwrap_resource(resource)
            entries.append(
                {
                    "binding": resource_binding_index(resource, i),
                    "resource": _resource_binding(raw),
                }
            )
        bg = self.device.create_bind_group(
            layout=layout, entries=entries, label=f"{label}:bg"
        )
        self._bind_groups[token] = bg
        return bg

    def submit(self, items: list[DispatchItem], *, label: str = "batch") -> None:
        if not items:
            return
        # 1) Rewrite all uniforms (host→GPU copies, no buffer alloc).
        prepared: list[tuple[DispatchItem, Any, int, Any, Any]] = []
        for it in items:
            params_size = max(it.uniform_size, 16, len(it.params_data))
            pdata = it.params_data
            if len(pdata) < params_size:
                pdata = bytes(pdata) + b"\x00" * (params_size - len(pdata))
            params_buf = self._params_buf(it.label, params_size)
            self.queue.write_buffer(params_buf, 0, pdata)
            indices = binding_indices_for(it.resources)
            pipeline, layout = self.pipelines.get_pipeline(
                it.key,
                it.code,
                n_storage_bindings=it.n_storage_bindings,
                uniform_size=it.uniform_size,
                storage_read_only=resolve_storage_read_only(
                    it.code,
                    it.n_storage_bindings,
                    None,
                    binding_indices=indices,
                ),
            )
            bg = self._bind_group(
                label=it.label,
                layout=layout,
                params_buf=params_buf,
                params_size=params_size,
                resources=it.resources,
            )
            prepared.append((it, pipeline, params_size, params_buf, bg))

        # 2) One command encoder / one submit for the whole stage list.
        encoder = self.device.create_command_encoder(label=label)
        for it, pipeline, _ps, _pb, bg in prepared:
            x, y, z = it.workgroups
            if x == 0 or y == 0 or z == 0:
                continue
            pass_ = encoder.begin_compute_pass(label=it.label)
            pass_.set_pipeline(pipeline)
            pass_.set_bind_group(0, bg)
            pass_.dispatch_workgroups(x, y, z)
            pass_.end()
        self.queue.submit([encoder.finish()])

    def destroy(self) -> None:
        for buf in self._params.values():
            buf.destroy()
        self._params.clear()
        self._param_sizes.clear()
        self._bind_groups.clear()


def submit_dispatches(
    pipelines: PipelineCache,
    queue: Any,
    items: list[DispatchItem],
    *,
    label: str = "batch",
    submitter: PersistentSubmitter | None = None,
) -> PersistentSubmitter:
    """Submit items via a PersistentSubmitter (created on demand)."""
    if submitter is None:
        submitter = PersistentSubmitter(pipelines, queue)
    submitter.submit(items, label=label)
    return submitter
