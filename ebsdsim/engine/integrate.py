"""Engine layer: multi-voltage master-pattern integration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from ebsdsim.energy.mc import MultiVoltageMC, surrogate_to_multi_voltage_mc
from ebsdsim.engine.results import MasterPatternResult
from ebsdsim.physics.mode import MasterPatternMode
from ebsdsim.physics.prescan import compute_mu_eff, relative_normalized_view_change

__all__ = [
    "ActiveVoltageBin",
    "MasterPatternIntegratedResult",
    "MultiVoltageMC",
    "active_voltage_bins",
    "compute_mu_eff",
    "next_active_voltage_kv",
    "relative_normalized_view_change",
    "surrogate_to_multi_voltage_mc",
]

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


def next_active_voltage_kv(
    mc: MultiVoltageMC,
    after_index: int,
    *,
    min_weight: float = 1e-12,
    min_amplitude: float = 1e-12,
) -> float | None:
    """Return the next active bin voltage after *after_index*, or ``None``."""
    w = np.asarray(mc.energy_weights, dtype=np.float64)
    a = np.asarray(mc.amplitudes, dtype=np.float64)
    v = np.asarray(mc.voltages_kv, dtype=np.float64)
    if after_index + 1 >= v.size:
        return None
    mask = (
        (w[after_index + 1 :] >= min_weight)
        & (a[after_index + 1 :] >= min_amplitude)
        & (v[after_index + 1 :] > 0)
    )
    hits = np.flatnonzero(mask)
    if hits.size == 0:
        return None
    return float(v[after_index + 1 + int(hits[0])])


@dataclass(frozen=True)
class ActiveVoltageBin:
    """One energy bin with nonzero weight/amplitude and a positive voltage."""

    bin_index: int
    voltage_kv: float
    energy_weight: float
    amplitude: float
    beta: float
    next_voltage_kv: float | None


def active_voltage_bins(
    mc: MultiVoltageMC,
    *,
    min_weight: float = 1e-12,
    min_amplitude: float = 1e-12,
) -> list[ActiveVoltageBin]:
    """Filter MC bins to those that participate in dynamical integration."""
    w = np.asarray(mc.energy_weights, dtype=np.float64)
    a = np.asarray(mc.amplitudes, dtype=np.float64)
    v = np.asarray(mc.voltages_kv, dtype=np.float64)
    beta = np.asarray(mc.betas, dtype=np.float64)
    mask = (w >= min_weight) & (a >= min_amplitude) & (v > 0)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    next_v = np.full(idx.size, np.nan, dtype=np.float64)
    if idx.size > 1:
        next_v[:-1] = v[idx[1:]]
    out: list[ActiveVoltageBin] = []
    for k, i in enumerate(idx):
        nxt = float(next_v[k]) if np.isfinite(next_v[k]) else None
        out.append(
            ActiveVoltageBin(
                bin_index=int(i),
                voltage_kv=float(v[i]),
                energy_weight=float(w[i]),
                amplitude=float(a[i]),
                beta=float(beta[i]),
                next_voltage_kv=nxt,
            )
        )
    return out


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
    bin_callback: Callable[[int, int, float, float, float], None] | None = None
    on_bin_complete: Callable[[int, int, float], None] | None = None
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
    active = active_voltage_bins(mc, min_weight=min_w, min_amplitude=min_a)
    active_by_index = {b.bin_index: b for b in active}

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

    # Callbacks fire for every MC bin (including inactive); solve only actives.
    for v in range(n_bins):
        w = float(mc.energy_weights[v])
        a = float(mc.amplitudes[v])
        vkv = float(mc.voltages_kv[v])
        if opts.bin_callback:
            opts.bin_callback(v, n_bins, vkv, w, a)
        bin_info = active_by_index.get(v)
        if bin_info is None:
            continue
        active_bins += 1
        ctx = PerVoltageContext(
            voltage_kv=bin_info.voltage_kv,
            bin_index=bin_info.bin_index,
            energy_weight=bin_info.energy_weight,
            amplitude=bin_info.amplitude,
            beta=bin_info.beta,
            mode=mode,
            next_voltage_kv=bin_info.next_voltage_kv,
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
        # Weighted accumulation: pattern is (n_k * n_sites,); sum view is per-k.
        pat = np.asarray(res.pattern, dtype=np.float64).reshape(n_k, n_sites)
        delta = bin_info.energy_weight * pat
        integrated_2d = integrated.reshape(n_k, n_sites)
        integrated_2d += delta.astype(np.float32)
        weighted_delta_sum_view[:] = delta.sum(axis=1)
        last_relative_change = relative_normalized_view_change(integrated_sum_view, weighted_delta_sum_view)
        integrated_sum_view += weighted_delta_sum_view
        bin_patterns.append(res.pattern)
        bin_voltages_kv.append(bin_info.voltage_kv)
        bin_weights.append(bin_info.energy_weight)
        bin_indices.append(bin_info.bin_index)
        if opts.on_bin_integrated is not None:
            opts.on_bin_integrated(integrated, n_k, n_sites)
        if opts.on_bin_complete is not None:
            opts.on_bin_complete(v, n_bins, bin_info.voltage_kv)
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
