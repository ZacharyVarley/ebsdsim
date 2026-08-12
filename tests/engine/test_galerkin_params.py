"""CPU tests for Galerkin solver registration, packs, and Lyapunov strategy."""

from __future__ import annotations

import struct

from ebsdsim.engine.params import VALID_SOLVERS, SimParams, resolve_solver_choice
from ebsdsim.gpu.dynamical.galerkin_dispatch import (
    KRYLOV_RANK,
    MAX_RANK,
    lyapunov_f_ld,
    lyapunov_strategy,
    pack_expand,
    pack_lyapunov,
)


def test_galerkin_is_default_solver():
    params = SimParams()
    assert params.solver == "galerkin"
    assert params.rank == KRYLOV_RANK
    assert resolve_solver_choice(params) == "galerkin"


def test_galerkin_registered_in_valid_solvers():
    assert "galerkin" in VALID_SOLVERS
    assert VALID_SOLVERS[0] == "galerkin"
    assert SimParams(solver="galerkin").solver == "galerkin"
    assert SimParams(solver="smith").solver == "smith"
    assert SimParams(solver="lu_smith").solver == "lu_smith"


def test_lyapunov_strategy_selection():
    assert lyapunov_strategy(1) == "shared"
    assert lyapunov_strategy(6) == "shared"
    assert lyapunov_strategy(7) == "implicit"
    assert lyapunov_strategy(16) == "implicit"
    assert lyapunov_f_ld(6) == 6
    assert lyapunov_f_ld(8) == MAX_RANK


def test_pack_lyapunov_and_expand_layouts():
    lyap = pack_lyapunov(32, 6, 6)
    assert len(lyap) == 16
    assert struct.unpack("<4I", lyap) == (32, 6, 6, 0)

    exp = pack_expand(
        32, 100, in_rank=6, out_rank=6, max_rank=6, f_col_stride=6
    )
    assert len(exp) == 32
    assert struct.unpack("<8I", exp) == (32, 100, 6, 6, 6, 6, 0, 0)
