"""GPU layer: storage-buffer helpers (bytes sizes for f32/c64/u32)."""

from __future__ import annotations

from typing import Any

import numpy as np
import wgpu
from numpy.typing import NDArray
from wgpu import BufferUsage


def f32_bytes(count: int) -> int:
    return int(count) * 4


def u32_bytes(count: int) -> int:
    return int(count) * 4


def c64_bytes(count: int) -> int:
    return int(count) * 8


def _align_up(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


class StorageBuffer:
    """Thin wrapper around a wgpu storage buffer with NumPy I/O."""

    def __init__(
        self,
        device: Any,
        queue: Any,
        *,
        label: str = "",
        data: NDArray[np.generic] | memoryview | None = None,
        byte_length: int | None = None,
        copy_src: bool = False,
        copy_dst: bool = True,
    ) -> None:
        self.device = device
        self.queue = queue
        usage = BufferUsage.STORAGE
        if copy_src:
            usage |= BufferUsage.COPY_SRC
        if copy_dst:
            usage |= BufferUsage.COPY_DST

        if data is not None:
            view = np.asarray(data)
            if not view.flags.c_contiguous:
                view = np.ascontiguousarray(view)
            nbytes = view.nbytes
            self._buffer = device.create_buffer_with_data(data=view, usage=usage, label=label)
            self.byte_length = nbytes
        else:
            if byte_length is None or byte_length < 4:
                raise ValueError("byte_length must be >= 4 when data is not provided")
            self.byte_length = int(byte_length)
            self._buffer = device.create_buffer(size=self.byte_length, usage=usage, label=label)

    @property
    def buffer(self) -> Any:
        return self._buffer

    def write(self, data: NDArray[np.generic] | memoryview, offset: int = 0) -> None:
        view = np.asarray(data)
        if not view.flags.c_contiguous:
            view = np.ascontiguousarray(view)
        self.queue.write_buffer(self._buffer, offset, view)

    def read(self, offset: int = 0, size: int | None = None) -> bytes:
        if size is None:
            size = self.byte_length - offset
        size = int(size)
        if size <= 0:
            return b""
        staging = self.device.create_buffer(
            size=size,
            usage=BufferUsage.MAP_READ | BufferUsage.COPY_DST,
            label="staging:read",
        )
        encoder = self.device.create_command_encoder()
        encoder.copy_buffer_to_buffer(self._buffer, offset, staging, 0, size)
        self.queue.submit([encoder.finish()])
        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            return bytes(staging.read_mapped())
        finally:
            staging.unmap()
            staging.destroy()

    def read_as(self, dtype: np.dtype, offset: int = 0, size: int | None = None) -> NDArray[Any]:
        raw = self.read(offset=offset, size=size)
        return np.frombuffer(raw, dtype=dtype).copy()

    def buffer_binding(self, size: int | None = None) -> dict[str, Any]:
        binding_size = self.byte_length if size is None else max(4, int(size))
        return {"buffer": self._buffer, "size": binding_size}

    def destroy(self) -> None:
        self._buffer.destroy()
