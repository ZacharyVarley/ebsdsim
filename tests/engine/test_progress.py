"""Unit tests for verbosity helpers."""

from __future__ import annotations

import pytest
from ebsdsim.engine.progress import MasterPatternProgress, validate_verbosity


def test_validate_verbosity_accepts_012():
    assert validate_verbosity(0) == 0
    assert validate_verbosity(1) == 1
    assert validate_verbosity(2) == 2


def test_validate_verbosity_rejects_other():
    with pytest.raises(ValueError, match="0, 1, or 2"):
        validate_verbosity(3)


def _reporter(verbosity: int, *, solver: str = "smith_iterative") -> MasterPatternProgress:
    return MasterPatternProgress(
        verbosity=verbosity,
        source="Ni",
        halfw=20,
        dmin=0.05,
        exact_slow_cpu=False,
        rank=20,
        chunk_size=256,
        solver=solver,
    )


def test_progress_banner_smith_iterative(capsys):
    _reporter(1, solver="smith_iterative").run_banner(mc_backend="surrogate", n_bins=10, n_k=1681)
    out = capsys.readouterr().out
    assert "[ebsdsim]" in out
    assert "Smith iterative" in out
    assert "1681 k-pts/bin" in out


def test_progress_banner_smith(capsys):
    _reporter(1, solver="lu_smith").run_banner(mc_backend="surrogate", n_bins=10, n_k=1681)
    out = capsys.readouterr().out
    assert "Smith rank 20" in out


def test_progress_silent_at_zero(capsys):
    _reporter(0).run_banner(mc_backend="surrogate", n_bins=10, n_k=100)
    assert capsys.readouterr().out == ""


def test_progress_bin_and_chunk_lines(capsys):
    p = _reporter(2, solver="lu_smith")
    p.run_banner(mc_backend="gpu_fly_first", n_bins=2, n_k=100)
    p.on_bin_start(0, 2, 20.0, 0.5, 1.0)
    p.dynamical_start(
        voltage_kv=20.0,
        n_k=100,
        n_chunks=4,
        n_strong=30,
        n_weak=10,
        n_g=500,
        effective_chunk=32,
    )
    p.on_chunk(1, 32, 100, 4)
    p.dynamical_finished()
    p.on_bin_complete(0, 2, 20.0)
    out = capsys.readouterr().out
    assert "weight=0.50000" in out
    assert "30 strong + 10 weak" in out
    assert "chunk 1/4" in out
    assert "k-pts/s" in out
    assert "finished" in out
