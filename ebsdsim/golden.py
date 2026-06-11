"""Golden-file float32 comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class GoldenComparison:
    max_abs_error: float
    max_rel_error: float
    max_index: int
    passed: bool


def compare_float32_golden(
    actual: NDArray[np.float32],
    expected: NDArray[np.float32],
    *,
    atol: float = 5e-4,
    rtol: float = 5e-3,
) -> GoldenComparison:
    if actual.size != expected.size:
        raise ValueError(f"golden length mismatch: {actual.size} != {expected.size}")
    max_abs_error = 0.0
    max_rel_error = 0.0
    max_index = -1
    passed = True
    for i in range(actual.size):
        abs_error = abs(float(actual[i]) - float(expected[i]))
        rel_error = abs_error / max(abs(float(expected[i])), 1e-12)
        if abs_error > max_abs_error or rel_error > max_rel_error:
            max_abs_error = max(max_abs_error, abs_error)
            max_rel_error = max(max_rel_error, rel_error)
            max_index = i
        if abs_error > atol + rtol * abs(float(expected[i])):
            passed = False
    return GoldenComparison(
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        max_index=max_index,
        passed=passed,
    )
