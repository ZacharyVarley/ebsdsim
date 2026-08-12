"""GPU dynamical-theory kernels (LU–Smith, Smith, and Galerkin RKSM).

:class:`EBSDDynamicalKernels` is the LU–Smith façade used by the engine.
Smith BiCGSTAB lives in :mod:`ebsdsim.gpu.dynamical.smith`.
Galerkin (rational Krylov / projected Lyapunov) lives in
:mod:`ebsdsim.gpu.dynamical.galerkin`.
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
