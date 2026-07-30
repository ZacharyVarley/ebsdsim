"""Engine layer: integrated master-pattern result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class MasterPatternResult:
    integrated: NDArray[np.float32]
    n_k: int
    n_sites: int
    bin_patterns: list[NDArray[np.float32]] = field(default_factory=list)


def validate_bin_contract(
    bin_patterns: Sequence[Any],
    bin_voltages_kv: Sequence[float],
    bin_weights: Sequence[float],
) -> None:
    """Enforce the per-bin intensity vs energy-model metadata count invariant.

    Legal states
    ------------
    * ``len(bin_patterns) == 0``: integrated-only edge case.
      ``bin_voltages_kv`` / ``bin_weights`` may still describe the energy model.
    * ``len(bin_patterns) == len(bin_voltages_kv) == len(bin_weights)`` with
      length ≥ 1: real per-bin intensity slices (``lu_smith`` and Smith iterative).

    Any other combination raises ``ValueError`` so save/load cannot silently
    fabricate one intensity pattern per voltage.
    """
    n_pat = len(bin_patterns)
    n_v = len(bin_voltages_kv)
    n_w = len(bin_weights)
    if n_v != n_w:
        raise ValueError(
            f"bin_voltages_kv length ({n_v}) must match bin_weights length ({n_w})"
        )
    if n_pat == 0:
        return
    if n_pat != n_v:
        raise ValueError(
            f"bin_patterns length ({n_pat}) must be 0 (integrated-only) or "
            f"match bin_voltages_kv / bin_weights ({n_v})"
        )


def stack_bins(bin_patterns: list[np.ndarray], n_k: int, n_sites: int) -> np.ndarray:
    """Stack flat per-bin FS patterns into ``(n_bins, n_k, n_sites)`` float32.

    Empty ``bin_patterns`` yields shape ``(0, n_k, n_sites)`` (integrated-only).
    """
    if not bin_patterns:
        return np.zeros((0, n_k, n_sites), dtype=np.float32)
    stacked = np.stack(
        [np.asarray(pat, dtype=np.float32).reshape(n_k, n_sites) for pat in bin_patterns],
        axis=0,
    )
    return np.asarray(stacked, dtype=np.float32)


@dataclass
class MasterPattern:
    """Rasterized master pattern and integration metadata.

    :attr:`pattern` is the north-hemisphere Lambert raster ``(side, side)`` with
    ``side = 1 + 2 * halfw``, equal to ``data[energy_int, site_int, 0]``. It
    holds raw dynamical intensities; use :meth:`lambert_data` for display scaling.

    :attr:`data` is the dense Lambert tensor ``(E, S, H, side, side)`` of raw
    intensities. :attr:`integrated` and :attr:`bin_patterns` store
    fundamental-sector values (flattened ``(n_k * n_sites,)``). Both
    ``lu_smith`` and Smith iterative return ``len(bin_patterns) ==
    len(bin_voltages_kv)``; the integrated total is a separate accumulator.
    Empty ``bin_patterns`` is the integrated-only edge case; voltages and
    weights remain energy-model metadata. :attr:`kij` and :attr:`khat` give
    fundamental-sector pixel indices and unit directions.

    ``save`` / ``lambert_data`` are attached by :mod:`ebsdsim.api` so the engine
    layer does not import :mod:`ebsdsim.io`.
    """

    pattern: NDArray[np.float32]
    integrated: NDArray[np.float32]
    n_k: int
    n_sites: int
    metadata: dict[str, Any] = field(default_factory=dict)
    bin_patterns: list[NDArray[np.float32]] = field(default_factory=list)
    bin_voltages_kv: list[float] = field(default_factory=list)
    bin_weights: list[float] = field(default_factory=list)
    kij: NDArray[np.int32] | None = None
    khat: NDArray[np.float32] | None = None
    pg_num: int | None = None
    pg_symbol: str | None = None
    data: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0,), np.float32))
    axes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_bin_contract(self.bin_patterns, self.bin_voltages_kv, self.bin_weights)
