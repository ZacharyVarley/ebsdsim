"""GPU dynamical-theory kernels (LU–Smith and Smith iterative).

:class:`EBSDDynamicalKernels` is the LU–Smith façade used by the engine.
Smith iterative BiCGSTAB lives in :mod:`ebsdsim.gpu.dynamical.smith_iterative`.
"""

from __future__ import annotations

from ebsdsim.gpu.dynamical.kernels import EBSDDynamicalKernels
from ebsdsim.gpu.dynamical.workspace import (
    C64,
    BetheMode,
    FixedRankChunkDescriptor,
    FixedRankWorkspace,
    PersistentBuffers,
    RunChunk,
    _to_u32,
)

__all__ = [
    "BetheMode",
    "C64",
    "EBSDDynamicalKernels",
    "FixedRankChunkDescriptor",
    "FixedRankWorkspace",
    "PersistentBuffers",
    "RunChunk",
    "_to_u32",
]
