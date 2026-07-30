"""Physics layer: Bethe excitation-error prescan (beam counts)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ebsdsim.physics.mode import MasterPatternMode


@dataclass
class BetheThresholds:
    c_strong: float
    c_weak: float
    c_cutoff: float
    dbdiff_sg_cutoff: float


@dataclass
class PrescanResult:
    n_strong: int
    n_weak: int
    max_candidates: int
    strong_needed: NDArray[np.int32]
    weak_needed: NDArray[np.int32]
    candidate_needed: NDArray[np.int32]


def _normalize_min_max_with_delta(
    values: NDArray[np.float64],
    delta: NDArray[np.float64] | None,
    out: NDArray[np.float64],
) -> None:
    combined = values if delta is None else values + delta
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    span = hi - lo
    if span <= 1e-30:
        out.fill(0.0)
        return
    out[:] = (combined - lo) / span if delta is not None else (values - lo) / span


def relative_normalized_view_change(
    current_sum_view: NDArray[np.float64],
    delta_sum_view: NDArray[np.float64],
) -> float:
    """Relative change between min/max-normalized sum views before/after a bin.

    Used by both the LU and smith_iterative integration paths to decide early stop on the
    voltage-integrated pattern (flattened fundamental-sector ``n_k`` view).
    """
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


def _dot_metric(
    a0: float, a1: float, a2: float,
    m: NDArray[np.floating],
    b0: float, b1: float, b2: float,
) -> float:
    mb0 = m[0] * b0 + m[1] * b1 + m[2] * b2
    mb1 = m[3] * b0 + m[4] * b1 + m[5] * b2
    mb2 = m[6] * b0 + m[7] * b1 + m[8] * b2
    return a0 * mb0 + a1 * mb1 + a2 * mb2


def excitation_error(
    kx: float,
    ky: float,
    kz: float,
    h: int,
    k: int,
    l: int,  # noqa: E741 — Miller l
    metric: NDArray[np.floating],
) -> float:
    if h == 0 and k == 0 and l == 0:
        return 0.0
    kpg0 = kx + h
    kpg1 = ky + k
    kpg2 = kz + l
    tkpg0 = 2 * kx + h
    tkpg1 = 2 * ky + k
    tkpg2 = 2 * kz + l
    xnom = -_dot_metric(h, k, l, metric, tkpg0, tkpg1, tkpg2)
    q1 = np.sqrt(max(_dot_metric(kpg0, kpg1, kpg2, metric, kpg0, kpg1, kpg2), 0.0))
    klen = np.sqrt(max(_dot_metric(kx, ky, kz, metric, kx, ky, kz), 0.0))
    kpg_k = _dot_metric(kpg0, kpg1, kpg2, metric, kx, ky, kz)
    xden = 2 * q1 * kpg_k / (q1 * klen + 1e-16)
    return xnom / (xden + 1e-16)


def prescan_bethe_beam_counts(
    kvecs: NDArray[np.float32],
    hkl: NDArray[np.int32],
    metric: NDArray[np.floating],
    coupling: NDArray[np.float32],
    reflection_dbdiff: NDArray[np.integer],
    thresholds: BetheThresholds,
) -> PrescanResult:
    rows = kvecs.size // 3
    n_g = hkl.size // 3
    strong_needed = np.zeros(rows, dtype=np.int32)
    weak_needed = np.zeros(rows, dtype=np.int32)
    candidate_needed = np.zeros(rows, dtype=np.int32)
    n_strong = 1
    n_weak = 0
    max_candidates = 1

    for row in range(rows):
        kb = row * 3
        strong = 0
        weak = 0
        candidate_count = 0
        for gi in range(n_g):
            hb = gi * 3
            sg = excitation_error(
                float(kvecs[kb]), float(kvecs[kb + 1]), float(kvecs[kb + 2]),
                int(hkl[hb]), int(hkl[hb + 1]), int(hkl[hb + 2]),
                metric,
            )
            abs_sg = abs(sg)
            ratio = abs_sg / max(float(coupling[gi]), 1e-12)
            candidate = (
                abs_sg <= thresholds.dbdiff_sg_cutoff
                if reflection_dbdiff[gi]
                else ratio <= thresholds.c_cutoff
            )
            is_strong = ratio <= thresholds.c_strong and candidate
            is_weak = thresholds.c_strong < ratio <= thresholds.c_weak and candidate
            if gi == 0:
                candidate = True
                is_strong = True
            if candidate:
                candidate_count += 1
            if is_strong:
                strong += 1
            if is_weak:
                weak += 1
        strong_needed[row] = strong
        weak_needed[row] = weak
        candidate_needed[row] = candidate_count
        n_strong = max(n_strong, strong)
        n_weak = max(n_weak, weak)
        max_candidates = max(max_candidates, candidate_count)

    return PrescanResult(
        n_strong=n_strong,
        n_weak=n_weak,
        max_candidates=max_candidates,
        strong_needed=strong_needed,
        weak_needed=weak_needed,
        candidate_needed=candidate_needed,
    )


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
