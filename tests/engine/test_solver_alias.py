"""CPU tests for solver name aliases and saved-metadata normalization."""

from __future__ import annotations

import warnings

import pytest
from ebsdsim.engine.params import SimParams, normalize_solver_name, resolve_solver_choice
from ebsdsim.io.load import _normalize_solver_metadata


def test_normalize_solver_name_alias_warns():
    with pytest.warns(DeprecationWarning, match="smith_iterative.*smith"):
        assert normalize_solver_name("smith_iterative") == "smith"


def test_normalize_solver_name_canonical_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert normalize_solver_name("galerkin") == "galerkin"
        assert normalize_solver_name("smith") == "smith"
        assert normalize_solver_name("lu_smith") == "lu_smith"


def test_simparams_accepts_deprecated_smith_iterative_alias():
    with pytest.warns(DeprecationWarning, match="smith_iterative"):
        params = SimParams(solver="smith_iterative")  # type: ignore[arg-type]
    assert params.solver == "smith"
    assert resolve_solver_choice(params) == "smith"


def test_simparams_rejects_unknown_solver():
    with pytest.raises(ValueError, match="unknown solver"):
        SimParams(solver="not_a_solver")  # type: ignore[arg-type]


def test_normalize_loaded_solver_metadata_legacy_keys():
    meta = {
        "solver": "smith_iterative",
        "smith_iterative_mode": "bicgstab_uniq_384_10200",
        "rank": 16,
    }
    _normalize_solver_metadata(meta)
    assert meta["solver"] == "smith"
    assert meta["smith_mode"] == "bicgstab_uniq_384_10200"
    assert "smith_iterative_mode" not in meta


def test_normalize_loaded_solver_metadata_already_canonical():
    meta = {"solver": "smith", "smith_mode": "bicgstab"}
    _normalize_solver_metadata(meta)
    assert meta == {"solver": "smith", "smith_mode": "bicgstab"}
