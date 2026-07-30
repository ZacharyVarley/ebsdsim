"""Run the GaN master-pattern example (501x501, API defaults)."""

from __future__ import annotations

import importlib.resources
import time
from pathlib import Path

import ebsdsim as es
from ebsdsim.io.load import save_png_gray, to_uint8

OUT = Path("output")
OUT.mkdir(exist_ok=True)


def main() -> None:
    gan_cif = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif")
    print("GaN master pattern (501x501, dmin=0.05, energy_binwidth_keV=1.0)")
    t0 = time.perf_counter()
    mp = es.master_pattern_from_cif(
        gan_cif,
        voltage_kv=20.0,
        halfw=250,
        # dmin=0.05, energy_binwidth_keV=1.0 — package defaults (match ebsdsim-web)
    )
    elapsed = time.perf_counter() - t0
    meta = mp.metadata
    print(
        f"Done in {elapsed:.1f}s  bins_run={meta['n_bins_run']}  "
        f"stopped={meta['stopped_by_relative_change']}  "
        f"rel_change={meta['last_relative_change']:.4g}"
    )
    print(
        f"pattern {mp.pattern.shape}  n_sites={mp.n_sites}  "
        f"southern={meta['needs_southern_hemisphere']}  "
        f"dmin={meta['dmin']}  energy_binwidth_keV={meta['energy_binwidth_keV']}"
    )

    png = OUT / "gan_master_pattern_lambert_nh_501.png"
    save_png_gray(to_uint8(mp.pattern), png)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
