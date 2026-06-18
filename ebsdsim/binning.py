"""Monte Carlo histogram binning and depth-axis resampling (NumPy only)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def depth_bin_edges(
    n_depth_bins: int, binsize_exit_depth: float, depth_mode: str
) -> NDArray[np.float64]:
    edges = np.arange(n_depth_bins + 1, dtype=np.float64) * float(binsize_exit_depth)
    if depth_mode == "log":
        return np.expm1(edges)
    return edges


def reconstruct_histogram_from_exp_tally(
    counts_3d: NDArray[np.floating],
    depth_sum_3d: NDArray[np.floating],
    depth_edges_nm: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    counts = np.asarray(counts_3d, dtype=np.float64)
    depth_sums = np.asarray(depth_sum_3d, dtype=np.float64)
    rates = np.zeros_like(counts, dtype=np.float64)

    unbiased_mask = (counts > 1.0) & (depth_sums > 0.0)
    fallback_mask = (counts > 0.0) & (depth_sums > 0.0) & (~unbiased_mask)
    rates[unbiased_mask] = (counts[unbiased_mask] - 1.0) / depth_sums[unbiased_mask]
    rates[fallback_mask] = counts[fallback_mask] / depth_sums[fallback_mask]

    z_lo = depth_edges_nm[:-1]
    z_hi = depth_edges_nm[1:]
    raw = np.zeros(
        (counts.shape[0], z_lo.size, counts.shape[1], counts.shape[2]),
        dtype=np.float64,
    )
    valid = rates > 0.0
    if np.any(valid):
        rate_expanded = rates[..., None]
        bin_mass = np.exp(-rate_expanded * z_lo[None, None, None, :]) - np.exp(
            -rate_expanded * z_hi[None, None, None, :]
        )
        raw = np.moveaxis(counts[..., None] * bin_mass, -1, 1)
        raw *= valid[:, None, :, :]

    stats = {
        "escaped": float(counts.sum()),
        "bins_unbiased": float(np.count_nonzero(unbiased_mask)),
        "bins_mle_fallback": float(np.count_nonzero(fallback_mask)),
        "reconstructed_mass_fraction": (
            float(raw.sum() / counts.sum()) if counts.sum() > 0.0 else float("nan")
        ),
    }
    return raw, rates, stats


def dynamical_voltages_kv(
    beam_kv: float,
    n_bins: int,
    binsize_keV: float,
) -> NDArray[np.float64]:
    """Dynamical beam voltage for each energy bin (kV).

    Returns ``beam_kv - i * binsize_keV`` for ``i = 0, 1, …, n_bins - 1``.
    """
    return beam_kv - np.arange(int(n_bins), dtype=np.float64) * float(binsize_keV)


def extra_energy_bin_params(
    voltage_kv: float, binsize_exit_energy: float, extra_energy_kv: float
) -> dict[str, Any]:
    n_extra_bins = int(round(extra_energy_kv / binsize_exit_energy))
    n_extra_bins = max(1, n_extra_bins) if extra_energy_kv > 0 else 0
    n_fillable_bins = int(voltage_kv // binsize_exit_energy)
    n_sim_energy_bins = n_fillable_bins + n_extra_bins
    starting_e_keV = voltage_kv + extra_energy_kv
    return {
        "starting_E_keV": starting_e_keV,
        "n_extra_bins": n_extra_bins,
        "n_fillable_bins": n_fillable_bins,
        "n_sim_energy_bins": n_sim_energy_bins,
        "binsize_exit_energy": binsize_exit_energy,
    }


def shave_and_renormalize_4d(
    hist_4d: NDArray[np.floating], n_extra_bins: int, n_fillable_bins: int
) -> NDArray[np.float64]:
    out = hist_4d[n_extra_bins : n_extra_bins + n_fillable_bins].astype(np.float64, copy=False)
    s = out.sum()
    if s > 0:
        out = out / s
    return out


def make_batch_sizes(total_trajectories: int, batch_size: int) -> list[int]:
    batches: list[int] = []
    remaining = int(total_trajectories)
    while remaining > 0:
        n = min(remaining, batch_size)
        batches.append(n)
        remaining -= n
    return batches


def resample_pdf_depth_linear(
    hist_4d: NDArray[np.floating],
    *,
    n_depth_bins_log: int,
    binsize_exit_depth: float,
    depth_mode: str,
    n_linear_bins: int,
    linear_binsize_nm: float,
) -> NDArray[np.float64]:
    hist_4d = np.asarray(hist_4d, dtype=np.float64)
    edges = depth_bin_edges(n_depth_bins_log, binsize_exit_depth, depth_mode)
    bin_widths = np.diff(edges)
    bin_widths = np.maximum(bin_widths, 1e-30)
    log_centers = 0.5 * (edges[:-1] + edges[1:])

    density = hist_4d / bin_widths[None, :, None, None]

    lin_edges = np.arange(n_linear_bins + 1, dtype=np.float64) * linear_binsize_nm
    lin_centers = 0.5 * (lin_edges[:-1] + lin_edges[1:])

    n_e, n_d, n_dir, n_dir2 = hist_4d.shape
    out = np.empty((n_e, n_linear_bins, n_dir, n_dir2), dtype=np.float64)
    for e in range(n_e):
        for p in range(n_dir):
            for q in range(n_dir2):
                out[e, :, p, q] = np.interp(
                    lin_centers,
                    log_centers,
                    density[e, :, p, q],
                    left=0.0,
                    right=0.0,
                )
    np.clip(out, 0.0, None, out=out)
    s = out.sum()
    if s > 0:
        out /= s
    return out
