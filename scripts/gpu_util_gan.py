"""Master-pattern run with per-bin lookup wait timing and nvidia-smi GPU sampling."""

from __future__ import annotations

import importlib.resources
import subprocess
import sys
import threading
import time

from ebsdsim.api import _run_master_pattern
from ebsdsim.structure import build_cell_from_cif_path
import ebsdsim.runner as runner

SAMPLE_MS = 200
MAX_BINS = 6
HALFW = 250
DMIN = 0.05
PRESET = (sys.argv[1] if len(sys.argv) > 1 else "GaN").removesuffix(".cif")


def _gpu_sampler(samples: list[tuple[float, int]], stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            util = int(out.strip().split("\n")[0].strip())
            samples.append((time.perf_counter(), util))
        except Exception:
            pass
        stop.wait(SAMPLE_MS / 1000.0)


def main() -> None:
    lookup_waits_ms: list[float] = []
    bin_starts: list[float] = []
    t_run0 = time.perf_counter()

    orig_resolve = runner._resolve_diff_lookup

    def timed_resolve(ctx, deps):
        t0 = time.perf_counter()
        out = orig_resolve(ctx, deps)
        lookup_waits_ms.append((time.perf_counter() - t0) * 1000.0)
        return out

    runner._resolve_diff_lookup = timed_resolve

    gpu_samples: list[tuple[float, int]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_gpu_sampler, args=(gpu_samples, stop), daemon=True)
    sampler.start()

    def on_bin(bin_index: int, total_bins: int, voltage_kv: float) -> None:
        bin_starts.append(time.perf_counter() - t_run0)

    cif_path = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs", f"{PRESET}.cif")
    cell = build_cell_from_cif_path(cif_path)

    print(f"{PRESET} {2 * HALFW + 1}x{2 * HALFW + 1}  dmin={DMIN}  max_bins={MAX_BINS}")
    print(f"GPU samples every {SAMPLE_MS} ms via nvidia-smi")
    print()

    t0 = time.perf_counter()
    _run_master_pattern(
        cell=cell,
        voltage_kv=20.0,
        halfw=HALFW,
        dmin=DMIN,
        energy_binwidth_keV=1.0,
        n_trajectories=0,
        sigma_deg=70.0,
        omega_deg=0.0,
        rank=20,
        chunk_size=256,
        mode="bloch",
        marginal_coverage=1.0,
        relative_image_stop=0.0,
        mc_backend="surrogate",
        source=f"{PRESET}.cif",
        max_bins_run=MAX_BINS,
        bin_callback=on_bin,
    )
    total_s = time.perf_counter() - t0
    stop.set()
    sampler.join(timeout=2.0)

    print(f"total wall time: {total_s:.2f}s")
    print()
    print("Per-bin lookup wait at bin start (ms) — should stay low if prefetch hides CPU work:")
    for i, ms in enumerate(lookup_waits_ms):
        print(f"  bin {i + 1}: {ms:6.1f} ms")

    if len(bin_starts) > 1:
        print()
        print("Gap between bin callbacks (s) — large gaps suggest GPU idle between bins:")
        for i in range(1, len(bin_starts)):
            print(f"  bin {i + 1} start gap: {bin_starts[i] - bin_starts[i - 1]:.3f}s")

    if gpu_samples:
        utils = [u for _, u in gpu_samples]
        t0s = gpu_samples[0][0]
        rel = [(t - t0s, u) for t, u in gpu_samples]
        print()
        print(f"GPU utilization samples (n={len(utils)}):")
        print(f"  min={min(utils)}%  mean={sum(utils) / len(utils):.0f}%  max={max(utils)}%")
        low = sum(1 for u in utils if u < 10)
        print(f"  samples <10%: {low} ({100 * low / len(utils):.0f}%)")
        print()
        print("Timeline (t+s, util%):")
        for t, u in rel[:: max(1, len(rel) // 20)]:
            bar = "#" * (u // 5)
            print(f"  {t:6.2f}  {u:3d}%  {bar}")
    else:
        print("No nvidia-smi samples (non-NVIDIA GPU or wgpu on other device).")


if __name__ == "__main__":
    main()
