"""Generate and save a Ni Lambert master pattern PNG (northern hemisphere)."""

from __future__ import annotations

import argparse
import importlib.resources
import struct
import zlib
from pathlib import Path

import numpy as np

from ebsdsim.api import _run_master_pattern
from ebsdsim.cif import parse_cif_crystal
from ebsdsim.kgrid import build_pg_k_grid
from ebsdsim.rasterize import RasterizeOptions, float01_to_uint8, rasterize_pattern
from ebsdsim.structure import build_cell_from_cif


def write_grayscale_png(path: Path, gray: np.ndarray) -> None:
    """Write a single-channel uint8 image as PNG (stdlib only)."""
    if gray.ndim != 2:
        raise ValueError("expected 2D grayscale array")
    if gray.dtype != np.uint8:
        gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    height, width = gray.shape
    raw_rows = [b"\x00" + gray[y].tobytes() for y in range(height)]
    compressed = zlib.compress(b"".join(raw_rows), level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--halfw",
        type=int,
        default=8,
        help="Lambert half-width (default 8 → 17×17; app default halfw is 250 → 501×501)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use halfw=250 (501×501). Slow — only run when benchmarking correctness.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default depends on grid size)",
    )
    args = parser.parse_args()

    halfw = 250 if args.full else args.halfw
    if halfw < 1:
        raise SystemExit("halfw must be >= 1")
    grid_size = 1 + 2 * halfw

    ni_cif = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
    out_path = args.output or (
        Path(__file__).resolve().parents[1]
        / "output"
        / (f"ni_master_pattern_lambert_nh_{grid_size}.png" if not args.full else "ni_master_pattern_lambert_nh_501.png")
    )
    cell = build_cell_from_cif(parse_cif_crystal(ni_cif.read_text(encoding="utf-8")))

    print(f"Running Ni master pattern ({grid_size}x{grid_size} Lambert NH, app beam cutoffs)...")
    result = _run_master_pattern(
        cell=cell,
        voltage_kv=20.0,
        halfw=halfw,
        dmin=0.05,
        energy_binwidth_keV=1.0,
        n_trajectories=1_048_576,
        sigma_deg=70.0,
        omega_deg=0.0,
        rank=20 if args.full else 8,
        chunk_size=256,
        mode="bloch",
        marginal_coverage=1.0,
        relative_image_stop=0.01,
        mc_backend="surrogate",
        source=str(ni_cif),
        bethe_c_strong=20.0,
        bethe_c_weak=40.0,
        bethe_c_cutoff=200.0,
        dbdiff_sg_cutoff=1.0,
    )

    pg_grid = build_pg_k_grid(int(result.metadata["pg_num"]), halfw)
    raster = rasterize_pattern(
        result.integrated,
        pg_grid,
        n_sites=result.n_sites,
        opts=RasterizeOptions(normalize="robust", interp_mode="bilinear"),
    )
    pattern = raster.nh.astype(np.float32).reshape(grid_size, grid_size)
    gray = float01_to_uint8(pattern)
    write_grayscale_png(out_path, gray)

    print(f"Saved {out_path} ({gray.shape[1]}x{gray.shape[0]})")
    print(f"  integrated min/max: {float(result.integrated.min()):.6g} / {float(result.integrated.max()):.6g}")
    print(f"  pattern min/max: {float(pattern.min()):.6g} / {float(pattern.max()):.6g}")
    print(f"  n_k: {result.n_k}, n_mc_bins: {result.metadata.get('n_mc_bins')}")


if __name__ == "__main__":
    main()
