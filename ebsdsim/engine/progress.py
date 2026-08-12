"""Engine layer: progress reporting for long-running jobs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def validate_verbosity(verbosity: int) -> int:
    v = int(verbosity)
    if v not in (0, 1, 2):
        raise ValueError("verbosity must be 0, 1, or 2")
    return v


@dataclass
class MasterPatternProgress:
    """Structured stderr progress for ``master_pattern`` runs."""

    verbosity: int
    source: str
    halfw: int
    dmin: float
    exact_slow_cpu: bool
    rank: int
    chunk_size: int
    solver: str = "galerkin"
    _bin_t0: float = field(default=0.0, repr=False)
    _dyn_t0: float = field(default=0.0, repr=False)
    _chunk_t0: float = field(default=0.0, repr=False)
    _prev_rows: int = field(default=0, repr=False)
    _dyn_elapsed: float = field(default=0.0, repr=False)
    _last_n_k: int = field(default=0, repr=False)
    _last_n_chunks: int = field(default=0, repr=False)

    @property
    def enabled(self) -> bool:
        return self.verbosity >= 1

    @property
    def detailed(self) -> bool:
        return self.verbosity >= 2

    def _solver_label(self) -> str:
        if self.exact_slow_cpu or self.solver == "exact_slow_cpu":
            return "exact CPU Lyapunov"
        if self.solver == "galerkin":
            return f"Galerkin rank {self.rank}"
        if self.solver == "smith":
            return "Smith"
        return f"Smith rank {self.rank}"

    def run_banner(self, *, mc_backend: str, n_bins: int, n_k: int) -> None:
        if not self.enabled:
            return
        grid = 1 + 2 * self.halfw
        print(
            f"[ebsdsim] {self.source}  Lambert {grid}x{grid}  dmin={self.dmin:.3f}  "
            f"{n_bins} voltage bins  {n_k} k-pts/bin  MC={mc_backend}  "
            f"{self._solver_label()}  chunk={self.chunk_size}",
            flush=True,
        )

    def on_bin_start(
        self,
        bin_index: int,
        n_bins: int,
        voltage_kv: float,
        weight: float,
        amplitude: float,
    ) -> None:
        if not self.enabled:
            return
        self._bin_t0 = time.perf_counter()
        self._dyn_elapsed = 0.0
        print(
            f"[ebsdsim] voltage bin {bin_index + 1}/{n_bins}: {voltage_kv:.2f} kV  "
            f"weight={weight:.5f}  amplitude={amplitude:.4f}",
            flush=True,
        )

    def on_bin_complete(self, bin_index: int, n_bins: int, voltage_kv: float) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._bin_t0
        if self.detailed and self._dyn_elapsed > 0 and self._last_n_k > 0:
            rate = self._last_n_k / self._dyn_elapsed
            print(
                f"[ebsdsim]   dynamical: {self._dyn_elapsed:.2f}s  "
                f"({rate:.0f} k-pts/s over {self._last_n_chunks} chunks)",
                flush=True,
            )
        print(
            f"[ebsdsim] voltage bin {bin_index + 1}/{n_bins} finished  "
            f"{voltage_kv:.2f} kV  wall {elapsed:.2f}s",
            flush=True,
        )

    def dynamical_start(
        self,
        *,
        voltage_kv: float,
        n_k: int,
        n_chunks: int,
        n_strong: int,
        n_weak: int,
        n_g: int,
        effective_chunk: int,
    ) -> None:
        if not self.detailed:
            return
        self._dyn_t0 = time.perf_counter()
        self._chunk_t0 = self._dyn_t0
        self._prev_rows = 0
        self._last_n_k = n_k
        self._last_n_chunks = n_chunks
        print(
            f"[ebsdsim]   {voltage_kv:.2f} kV dynamical ({self._solver_label()}): "
            f"{n_k} k-vectors in {n_chunks} chunks (size {effective_chunk}); "
            f"{n_strong} strong + {n_weak} weak selected from {n_g} reflections",
            flush=True,
        )

    def on_chunk(self, chunks_done: int, rows_done: int, n_k: int, n_chunks: int) -> None:
        if not self.detailed:
            return
        batch_rows = rows_done - self._prev_rows
        self._prev_rows = rows_done
        chunk_elapsed = time.perf_counter() - self._chunk_t0
        self._chunk_t0 = time.perf_counter()
        rate = batch_rows / chunk_elapsed if chunk_elapsed > 0 else 0.0
        pct = 100.0 * rows_done / n_k if n_k > 0 else 100.0
        print(
            f"[ebsdsim]   chunk {chunks_done}/{n_chunks}: "
            f"{rows_done}/{n_k} k-vectors ({pct:.0f}%)  "
            f"{chunk_elapsed:.2f}s  {rate:.0f} k-pts/s",
            flush=True,
        )

    def dynamical_finished(self) -> None:
        if not self.detailed:
            return
        self._dyn_elapsed = time.perf_counter() - self._dyn_t0

    def integration_stopped(self, *, last_relative_change: float, n_bins_run: int) -> None:
        if not self.enabled:
            return
        print(
            f"[ebsdsim] early stop after {n_bins_run} bins  "
            f"relative image change {last_relative_change:.4f}",
            flush=True,
        )
