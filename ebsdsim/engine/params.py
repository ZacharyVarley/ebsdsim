"""Simulation parameter bundle for master-pattern runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ebsdsim.engine.progress import validate_verbosity

SolverName = Literal["smith_iterative", "lu_smith"]
McBackend = Literal["surrogate", "gpu"]
VALID_SOLVERS: tuple[str, ...] = ("smith_iterative", "lu_smith")


@dataclass(frozen=True)
class SimParams:
    """Keyword arguments shared by :func:`~ebsdsim.master_pattern` entry points."""

    voltage_kv: float = 20.0
    halfw: int = 250
    dmin: float = 0.05
    energy_binwidth_keV: float = 1.0
    n_trajectories: int = 1_048_576
    sigma_deg: float = 70.0
    omega_deg: float = 0.0
    solver: SolverName = "smith_iterative"
    rank: int = 20
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
        if self.solver not in VALID_SOLVERS:
            raise ValueError(
                f"unknown solver: {self.solver!r}; "
                f"expected one of {list(VALID_SOLVERS)}"
            )


def resolve_solver_choice(params: SimParams) -> str:
    """Normalize solver + ``exact_slow_cpu`` into one dispatch key.

    Returns one of ``\"smith_iterative\"``, ``\"lu_smith\"``, ``\"exact_slow_cpu\"``.
    """
    if params.exact_slow_cpu:
        return "exact_slow_cpu"
    if params.solver == "smith_iterative":
        return "smith_iterative"
    if params.solver == "lu_smith":
        return "lu_smith"
    raise ValueError(
        f"unknown solver: {params.solver!r}; expected one of {list(VALID_SOLVERS)}"
    )
