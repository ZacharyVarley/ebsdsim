"""GPU layer: resident voltage-table buffers (not a submitter — see batch.PersistentSubmitter)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ebsdsim.crystal.build import metric_to_float32
from ebsdsim.crystal.simcell import SimCell
from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.gpu.dynamical import EBSDDynamicalKernels, PersistentBuffers, _to_u32
from ebsdsim.physics.lookup import DiffLookupData


@dataclass
class ResidentTables:
    """Reuse GPU voltage-table buffers; only voltage-dependent tables update per bin."""

    buffers: PersistentBuffers

    @classmethod
    def create(
        cls,
        kernels: EBSDDynamicalKernels,
        lookup: DiffLookupData,
        sgh_tables: NDArray[np.float32],
    ) -> ResidentTables:
        return cls(
            buffers=kernels.create_persistent_buffers(
                hkl=lookup.hkl,
                hkl_hash=lookup.hkl_hash,
                diff_table=lookup.diff_table,
                coupling=lookup.coupling,
                reflection_dbdiff=lookup.reflection_dbdiff,
                sgh_tables=sgh_tables,
            )
        )

    def update(self, lookup: DiffLookupData) -> None:
        self.buffers.diff_table.write(lookup.diff_table)
        self.buffers.coupling.write(lookup.coupling)
        self.buffers.reflection_dbdiff.write(_to_u32(lookup.reflection_dbdiff))

    def destroy(self) -> None:
        for name in vars(self.buffers):
            buf = getattr(self.buffers, name)
            if isinstance(buf, StorageBuffer):
                buf.destroy()


def make_metric_buffer(kernels: EBSDDynamicalKernels, cell: SimCell) -> StorageBuffer:
    return kernels.create_metric_buffer(metric_to_float32(cell))
