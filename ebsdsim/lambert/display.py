"""Lambert layer: display normalization (minmax / robust percentiles)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

NormalizeMode = Literal["minmax", "robust"]


def _percentile(sorted_vals: NDArray[np.float32], p: float) -> float:
    idx = min(sorted_vals.size - 1, max(0, int(np.floor(p * (sorted_vals.size - 1)))))
    return float(sorted_vals[idx])


def scale_fs_channel(
    vals: NDArray[np.floating],
    normalize: NormalizeMode | None,
    *,
    robust_p_low: float = 0.01,
    robust_p_high: float = 0.99,
) -> NDArray[np.float32]:
    """Scale one fundamental-sector channel to ``[0, 1]`` for display.

    Parameters
    ----------
    normalize:
        ``None`` — return raw values (no scaling).
        ``"minmax"`` — divide by per-channel min/max (lossless up to float precision).
        ``"robust"`` — IQR outlier gate, then percentile window (lossy by default
        uses ``robust_p_low`` / ``robust_p_high``).
    robust_p_low, robust_p_high:
        Percentiles in ``[0, 1]`` used only when ``normalize="robust"``.
    """
    arr = np.asarray(vals, dtype=np.float32)
    if normalize is None:
        return arr.copy() if arr.dtype != np.float32 else arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr.copy()
    if normalize == "minmax":
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    elif normalize == "robust":
        if not (0.0 <= robust_p_low < robust_p_high <= 1.0):
            raise ValueError("robust_p_low and robust_p_high must satisfy 0 <= low < high <= 1")
        sorted_vals = np.sort(arr)
        q1 = _percentile(sorted_vals, 0.25)
        q3 = _percentile(sorted_vals, 0.75)
        iqr = q3 - q1
        if iqr > 0:
            lo_bound = q1 - 3 * iqr
            hi_bound = q3 + 3 * iqr
            good = sorted_vals[(sorted_vals >= lo_bound) & (sorted_vals <= hi_bound)]
            if good.size > 0:
                lo = _percentile(good, robust_p_low)
                hi = _percentile(good, robust_p_high)
            else:
                lo = float(sorted_vals[0])
                hi = float(sorted_vals[-1])
        else:
            lo = float(sorted_vals[0])
            hi = float(sorted_vals[-1])
    else:
        raise ValueError(f"unknown normalize mode: {normalize!r}")
    span = hi - lo
    out = np.zeros_like(arr)
    if span <= 1e-15:
        return out
    out[:] = np.clip((arr - lo) / span, 0.0, 1.0)
    return out
