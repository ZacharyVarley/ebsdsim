"""Compatibility shim — prefer :mod:`ebsdsim.gpu.dynamical.kernels`."""

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
