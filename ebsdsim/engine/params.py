"""Simulation parameter bundle for master-pattern runs."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

from ebsdsim.engine.progress import validate_verbosity

SolverName = Literal["galerkin", "smith", "lu_smith"]
McBackend = Literal["surrogate", "gpu"]
VALID_SOLVERS: tuple[str, ...] = ("galerkin", "smith", "lu_smith")
_DEPRECATED_SOLVER_ALIASES: dict[str, SolverName] = {
    "smith_iterative": "smith",
}


def normalize_solver_name(solver: str, *, stacklevel: int = 2) -> str:
    """Map deprecated solver aliases to their canonical names.

    Emits :class:`DeprecationWarning` when an alias is used.
    """
    canonical = _DEPRECATED_SOLVER_ALIASES.get(solver)
    if canonical is None:
        return solver
    warnings.warn(
        f"solver={solver!r} is deprecated; use solver={canonical!r} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return canonical


@dataclass(frozen=True)
class SimParams:
    """Keyword arguments shared by :func:`~ebsdsim.master_pattern` entry points."""

    voltage_kv: float = 20.0
    halfw: int = 500
    dmin: float = 0.05
    energy_binwidth_keV: float = 1.0
    n_trajectories: int = 1_048_576
    sigma_deg: float = 70.0
    omega_deg: float = 0.0
    solver: SolverName = "galerkin"
    rank: int = 8
    exact_slow_cpu: bool = False
    verbosity: int = 0
    chunk_size: int = 256
    marginal_coverage: float = 1.0
    relative_image_stop: float = 0.01
    mc_backend: McBackend = "surrogate"
    bethe_c_strong: float = 20.0
    bethe_c_weak: float = 40.0
    bethe_c_cutoff: float = 200.0
    dbdiff_sg_cutoff: float = 1.0
    mc_auto_stop: bool = True
    mc_relative_tol: float = 0.01
    mc_min_trajectories: int = 1_048_576
    mc_max_trajectories: int = 16_777_216

    def __post_init__(self) -> None:
        halfw = int(self.halfw)
        if halfw < 1:
            raise ValueError("halfw must be >= 1")
        object.__setattr__(self, "halfw", halfw)
        object.__setattr__(self, "verbosity", validate_verbosity(self.verbosity))
        # Accept deprecated aliases (e.g. ``smith_iterative`` → ``smith``) before
        # validating against the canonical set. Cast through ``str`` so callers
        # that still pass the alias at runtime are not rejected by the Literal.
        # stacklevel=3: normalize → __post_init__ → SimParams caller
        solver = normalize_solver_name(str(self.solver), stacklevel=3)
        object.__setattr__(self, "solver", solver)
        if self.solver not in VALID_SOLVERS:
            raise ValueError(
                f"unknown solver: {self.solver!r}; "
                f"expected one of {list(VALID_SOLVERS)}"
            )


def resolve_solver_choice(params: SimParams) -> str:
    """Normalize solver + ``exact_slow_cpu`` into one dispatch key.

    Returns one of ``\"galerkin\"``, ``\"smith\"``, ``\"lu_smith\"``,
    ``\"exact_slow_cpu\"``.
    """
    if params.exact_slow_cpu:
        return "exact_slow_cpu"
    if params.solver == "galerkin":
        return "galerkin"
    if params.solver == "smith":
        return "smith"
    if params.solver == "lu_smith":
        return "lu_smith"
    raise ValueError(
        f"unknown solver: {params.solver!r}; expected one of {list(VALID_SOLVERS)}"
    )
