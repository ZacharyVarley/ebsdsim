"""End-to-end example: simulate GaN, save the .npz, reload it with the
NumPy-only loader, verify the reconstruction against the GPU output, and
export PNGs of the integrated pattern and per-energy-bin intermediates.

Run from the repository root:

    python scripts/validate_save_load.py
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import numpy as np

import ebsdsim as es
from ebsdsim.mploader import load_master_pattern, save_png_gray, to_uint8

OUT = Path("output")
OUT.mkdir(exist_ok=True)


def main() -> None:
    gan_cif = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/GaN.cif")

    # 1. Simulate (GaN: two sites, non-centrosymmetric -> southern hemisphere present).
    mp = es.master_pattern_from_cif(gan_cif, voltage_kv=20.0, halfw=250)
    print(f"pattern {mp.pattern.shape}  n_sites={mp.n_sites}  bins={len(mp.bin_patterns)}")
    print(
        f"needs_southern={mp.metadata['needs_southern_hemisphere']}  "
        f"pg={mp.metadata['pg_symbol']}"
    )
    print(
        "cell sites:",
        [(s["symbol"], s["b_iso_angstrom_sq"]) for s in mp.metadata["cell"]["sites"]],
    )

    # 2. Save (always compressed, always with intermediates, fundamental sector only).
    npz_path = mp.save(OUT / "GaN-master-pattern.npz")
    print(f"saved {npz_path} ({npz_path.stat().st_size} bytes)")

    # 3. Reload with the NumPy-only loader and reconstruct full hemispheres.
    loaded = load_master_pattern(npz_path)
    nh, sh = loaded.reconstruct_integrated()
    diff = float(np.max(np.abs(nh - mp.pattern)))
    print(f"loader-vs-GPU NH max abs diff: {diff:.3e}")

    # 4. Export PNGs.
    save_png_gray(to_uint8(nh), OUT / "GaN_integrated_nh.png")
    if loaded.needs_southern_hemisphere:
        save_png_gray(to_uint8(sh), OUT / "GaN_integrated_sh.png")
    for b in range(min(loaded.n_bins, 3)):
        nh_b, _ = loaded.reconstruct_bin(b)
        kv = float(loaded.bin_voltages_kv[b])
        save_png_gray(to_uint8(nh_b), OUT / f"GaN_bin{b:02d}_{kv:.1f}kV_nh.png")
    print("PNGs written to", OUT)


if __name__ == "__main__":
    main()
