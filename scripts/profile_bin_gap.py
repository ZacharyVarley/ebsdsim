"""Profile per-bin timing for the first two GaN voltage bins (defaults)."""

from __future__ import annotations

import importlib.resources
import time

from ebsdsim.crystal.build import build_cell_from_cif_path
from ebsdsim.engine.master_pattern import run_master_pattern
from ebsdsim.engine.params import SimParams

_last = time.perf_counter()


def _bin_cb(bin_index: int, total_bins: int, voltage_kv: float, _weight: float, _amp: float) -> None:
    global _last
    now = time.perf_counter()
    print(f"  bin {bin_index + 1}/{total_bins} @ {voltage_kv:.2f} kV  gap_since_last={now - _last:.3f}s")
    _last = now


def main() -> None:
    gan = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif")
    cell = build_cell_from_cif_path(gan)
    print("GaN 501x501 — first 2 bins, dmin=0.05, energy_binwidth_keV=1.0")
    global _last
    _last = time.perf_counter()
    t0 = time.perf_counter()
    # max_bins_run / bin_callback are LU-path only.
    params = SimParams(
        voltage_kv=20.0,
        halfw=250,
        dmin=0.05,
        energy_binwidth_keV=1.0,
        n_trajectories=0,
        sigma_deg=70.0,
        omega_deg=0.0,
        solver="lu_smith",
        rank=20,
        chunk_size=256,
        marginal_coverage=1.0,
        relative_image_stop=0.0,
        mc_backend="surrogate",
    )
    run_master_pattern(
        cell,
        params,
        source="GaN.cif",
        mode="bloch",
        max_bins_run=2,
        bin_callback=_bin_cb,
    )
    print(f"total={time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
