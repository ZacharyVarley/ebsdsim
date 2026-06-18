"""Multi-voltage master-pattern integration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from ebsdsim.binning import dynamical_voltages_kv
from ebsdsim.surrogate import SurrogateDirectExp
from ebsdsim.types import MasterPatternMode, MasterPatternResult, MultiVoltageMC

DEFAULT_ENERGY_MARGINAL_COVERAGE = 0.90


def _clamp_coverage(coverage: float) -> float:
    if not math.isfinite(coverage):
        return 1.0
    if coverage <= 0:
        return 0.0
    if coverage >= 1:
        return 1.0
    return coverage


def energy_bins_for_marginal_coverage(mc: MultiVoltageMC, coverage: float) -> int:
    n = mc.energy_weights.size
    if n <= 1:
        return n
    cov = _clamp_coverage(coverage)
    if cov >= 1:
        return n
    total = float(np.sum(mc.energy_weights[np.isfinite(mc.energy_weights) & (mc.energy_weights > 0)]))
    if total <= 0:
        return n
    target = cov * total
    acc = 0.0
    for i in range(n):
        w = mc.energy_weights[i]
        if math.isfinite(float(w)) and w > 0:
            acc += float(w)
        if acc >= target:
            return i + 1
    return n


def trim_multi_voltage_mc_by_coverage(mc: MultiVoltageMC, coverage: float) -> MultiVoltageMC:
    keep = energy_bins_for_marginal_coverage(mc, coverage)
    if keep >= mc.voltages_kv.size:
        return MultiVoltageMC(
            binsize_energy_keV=mc.binsize_energy_keV,
            voltages_kv=mc.voltages_kv.copy(),
            energy_weights=mc.energy_weights.copy(),
            amplitudes=mc.amplitudes.copy(),
            betas=mc.betas.copy(),
        )
    voltages = mc.voltages_kv[:keep].copy()
    amplitudes = mc.amplitudes[:keep].copy()
    betas = mc.betas[:keep].copy()
    weights = mc.energy_weights[:keep].copy()
    wsum = float(np.sum(weights[np.isfinite(weights) & (weights > 0)]))
    if wsum > 0:
        weights /= wsum
    return MultiVoltageMC(
        binsize_energy_keV=mc.binsize_energy_keV,
        voltages_kv=voltages,
        energy_weights=weights,
        amplitudes=amplitudes,
        betas=betas,
    )


def surrogate_to_multi_voltage_mc(de: SurrogateDirectExp, beam_kv: float) -> MultiVoltageMC:
    """Map surrogate depth/energy marginals to a :class:`MultiVoltageMC`."""
    n = de.amplitudes.size
    centers = de.energy_centers_keV
    if centers.size >= 2:
        d_e = float(centers[1] - centers[0])
    elif centers.size == 1:
        d_e = float(centers[0]) * 2.0
    else:
        d_e = 1.0
    voltages = dynamical_voltages_kv(beam_kv, n, d_e)
    return MultiVoltageMC(
        binsize_energy_keV=d_e,
        voltages_kv=voltages,
        energy_weights=de.energy_weights.astype(np.float64, copy=True),
        amplitudes=de.amplitudes.astype(np.float64, copy=True),
        betas=de.betas.astype(np.float64, copy=True),
    )


def _normalize_min_max_with_delta(
    values: NDArray[np.float64],
    delta: NDArray[np.float64] | None,
    out: NDArray[np.float64],
) -> None:
    if delta is None:
        combined = values
    else:
        combined = values + delta
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    span = hi - lo
    if span <= 1e-30:
        out.fill(0.0)
        return
    if delta is None:
        out[:] = (values - lo) / span
    else:
        out[:] = (values + delta - lo) / span


def relative_normalized_view_change(
    current_sum_view: NDArray[np.float64],
    delta_sum_view: NDArray[np.float64],
) -> float:
    """Match web-app Δ/image on min/max normalized sum views."""
    normalized_before = np.empty_like(current_sum_view)
    normalized_after = np.empty_like(current_sum_view)
    _normalize_min_max_with_delta(current_sum_view, None, normalized_before)
    _normalize_min_max_with_delta(current_sum_view, delta_sum_view, normalized_after)
    diff = normalized_after - normalized_before
    delta_sq = float(np.dot(diff, diff))
    total_sq = float(np.dot(normalized_after, normalized_after))
    denom = total_sq**0.5
    if denom <= 1e-30:
        return float("inf")
    return (delta_sq**0.5) / denom


def compute_mu_eff(
    beta: float,
    mlambda: float,
    diag_imag: float,
    mode: MasterPatternMode,
) -> float:
    two_pi = 2 * math.pi
    if mode == "structure":
        return beta - two_pi / diag_imag
    return beta - two_pi * mlambda * abs(diag_imag)


def next_active_voltage_kv(
    mc: MultiVoltageMC,
    after_index: int,
    *,
    min_weight: float = 1e-12,
    min_amplitude: float = 1e-12,
) -> float | None:
    """Return the next active bin voltage after *after_index*, or ``None``."""
    for j in range(after_index + 1, mc.voltages_kv.size):
        w = float(mc.energy_weights[j])
        a = float(mc.amplitudes[j])
        vkv = float(mc.voltages_kv[j])
        if w >= min_weight and a >= min_amplitude and vkv > 0:
            return vkv
    return None


@dataclass
class PerVoltageContext:
    voltage_kv: float
    bin_index: int
    energy_weight: float
    amplitude: float
    beta: float
    mode: MasterPatternMode
    next_voltage_kv: float | None = None


@dataclass
class PerVoltageResult:
    pattern: NDArray[np.float32]
    n_k: int
    n_sites: int


class PerVoltageRunner(Protocol):
    def __call__(self, ctx: PerVoltageContext) -> PerVoltageResult | None: ...


@dataclass
class RunMasterPatternIntegratedOptions:
    mc: MultiVoltageMC
    run_one_voltage: PerVoltageRunner
    mode: MasterPatternMode = "bloch"
    bin_callback: Callable[[int, int, float], None] | None = None
    on_bin_integrated: Callable[[NDArray[np.float32], int, int], None] | None = None
    min_weight: float = 1e-12
    min_amplitude: float = 1e-12
    marginal_coverage: float = 1.0
    relative_image_stop: float = 0.01
    max_bins_run: int | None = None


@dataclass
class MasterPatternIntegratedResult(MasterPatternResult):
    n_bins_run: int = 0
    stopped_by_relative_change: bool = False
    last_relative_change: float = float("inf")
    bin_voltages_kv: list[float] = field(default_factory=list)
    bin_weights: list[float] = field(default_factory=list)
    bin_indices: list[int] = field(default_factory=list)


def run_master_pattern_voltage_integrated(
    opts: RunMasterPatternIntegratedOptions,
) -> MasterPatternIntegratedResult:
    mc = trim_multi_voltage_mc_by_coverage(opts.mc, opts.marginal_coverage)
    mode = opts.mode
    min_w = opts.min_weight
    min_a = opts.min_amplitude
    n_bins = mc.voltages_kv.size

    integrated: NDArray[np.float32] | None = None
    n_k = 0
    n_sites = 0
    bin_patterns: list[NDArray[np.float32]] = []
    bin_voltages_kv: list[float] = []
    bin_weights: list[float] = []
    bin_indices: list[int] = []
    integrated_sum_view: NDArray[np.float64] | None = None
    weighted_delta_sum_view: NDArray[np.float64] | None = None
    active_bins = 0
    stopped_by_relative_change = False
    last_relative_change = float("inf")

    for v in range(n_bins):
        w = float(mc.energy_weights[v])
        a = float(mc.amplitudes[v])
        beta = float(mc.betas[v])
        vkv = float(mc.voltages_kv[v])
        if opts.bin_callback:
            opts.bin_callback(v, n_bins, vkv)
        if w < min_w or a < min_a or vkv <= 0:
            continue
        active_bins += 1
        ctx = PerVoltageContext(
            voltage_kv=vkv,
            bin_index=v,
            energy_weight=w,
            amplitude=a,
            beta=beta,
            mode=mode,
            next_voltage_kv=next_active_voltage_kv(mc, v, min_weight=min_w, min_amplitude=min_a),
        )
        res = opts.run_one_voltage(ctx)
        if res is None:
            continue
        if integrated is None:
            n_k = res.n_k
            n_sites = res.n_sites
            integrated = np.zeros(res.pattern.size, dtype=np.float32)
            integrated_sum_view = np.zeros(n_k, dtype=np.float64)
            weighted_delta_sum_view = np.zeros(n_k, dtype=np.float64)
        if res.pattern.size != integrated.size:
            raise ValueError(
                f"pattern length mismatch at bin {v}: got {res.pattern.size}, expected {integrated.size}"
            )
        assert integrated_sum_view is not None
        assert weighted_delta_sum_view is not None
        weighted_delta_sum_view.fill(0.0)
        for r in range(n_k):
            row_base = r * n_sites
            row_delta = 0.0
            for s in range(n_sites):
                idx = row_base + s
                delta = w * float(res.pattern[idx])
                integrated[idx] += np.float32(delta)
                row_delta += delta
            weighted_delta_sum_view[r] = row_delta
        last_relative_change = relative_normalized_view_change(integrated_sum_view, weighted_delta_sum_view)
        integrated_sum_view += weighted_delta_sum_view
        bin_patterns.append(res.pattern)
        bin_voltages_kv.append(vkv)
        bin_weights.append(w)
        bin_indices.append(v)
        if opts.on_bin_integrated is not None:
            opts.on_bin_integrated(integrated, n_k, n_sites)
        if opts.max_bins_run is not None and active_bins >= opts.max_bins_run:
            break
        if (
            opts.relative_image_stop > 0
            and active_bins > 1
            and last_relative_change <= opts.relative_image_stop
        ):
            stopped_by_relative_change = True
            break

    if integrated is None:
        integrated = np.zeros(0, dtype=np.float32)
    return MasterPatternIntegratedResult(
        integrated=integrated,
        n_k=n_k,
        n_sites=n_sites,
        bin_patterns=bin_patterns,
        bin_voltages_kv=bin_voltages_kv,
        bin_weights=bin_weights,
        bin_indices=bin_indices,
        n_bins_run=active_bins,
        stopped_by_relative_change=stopped_by_relative_change,
        last_relative_change=last_relative_change,
    )
