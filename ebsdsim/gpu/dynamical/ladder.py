"""Portable fallback ladder for smith_iterative Toeplitz Smith (shared-memory matvecs).

Decision per k (after LLL smith_iterative reindex of strong beams):

```text
plen = d0·d1·d2          # dense Toeplitz box size (f16c2 slots)
nu   = |{ s_i − s_j }|   # unique pack indices actually gathered

bucket by n_strong → (MAX_N, MAX_PACK, MAX_UNIQ, BPT)

if plen <= MAX_PACK:
    RESIDENT             # full dense box in shared
elif nu <= MAX_UNIQ and plen <= MAX_PLEN_CAP:
    UNIQUE_Δ             # compact touched deltas into shared (~2× smaller)
elif nu <= MAX_NU_CAP and plen <= MAX_PLEN_CAP:
    UNIQUE_SEG           # global compact once + shared windows of MAX_UNIQ
else:
    DENSE_TILE           # stream AABB windows through shared
```

Bench (BiCGSTAB, ~48 KB):
  sg_003 n=566: unique resident ~466 k/s; dense TILE ~500 k/s
  sg_004 n=654, plen~25k, ν~11k (formerly dense-only):
    global-meta unique-seg ~337 k/s  vs  dense TILE ~297 k/s  (~13% win)

Cost is dominated by **# of n² matvec passes**, not fill style.
Prefer **few large windows** (SEG ≈ MAX_UNIQ / MAX_PACK). Global compact
helps little when pass count is already 1; it matters when you would otherwise
rescan plen every segment.

48 KiB constellations (host string-replaces WGSL consts):

| n_strong | MAX_PACK | MAX_UNIQ | BPT |
|---------:|---------:|---------:|----:|
| ≤384     | 10200    | 8600     | 2   |
| ≤512     | 9690     | 8100     | 2   |
| ≤768     | 8600     | 7000     | 3   |
| ≤1024    | 7600     | 6000     | 4   |
| ≤2048    | shrink   | pack−1564| n/256 |

Shaders
-------
- BiCGSTAB: ``wgsl/smith_iterative_build_uniq_shared_bicgstab_f16.wgsl``
  bit28 = unique, bit29 = BiCGSTAB, bit30 = unique-seg, bit31 = dense tile
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

MAX_PLEN_FOR_UNIQ_TRY = 65536
MAX_PLEN_SHARED_META = 20000
MAX_NU_CAP = 16384


def max_pack_for_n(n: int) -> int:
    if n <= 384:
        return 10200
    if n <= 512:
        return 9690
    if n <= 768:
        return 8600
    if n <= 1024:
        return 7600
    if n <= 2048:
        max_n = ((int(n) + 255) // 256) * 256
        return max(1024, (48000 - max_n * 16 - 2200) // 4)
    return 0


def max_uniq_for_n(n: int) -> int:
    if n <= 384:
        return 8600
    if n <= 512:
        return 8100
    if n <= 768:
        return 7000
    if n <= 1024:
        return 6000
    if n <= 2048:
        return max(0, max_pack_for_n(n) - 1564)
    return 0


# Back-compat aliases
MAX_PACK = 8600
MAX_PACK_BUCKET_384 = 10200


class PackMode(Enum):
    RESIDENT = "resident"
    UNIQUE_DELTA = "unique_delta"
    UNIQUE_SEG = "unique_seg"
    TILE = "tile"


def choose_mode(
    plen: int, n_unique: int, *, n: int = 768, max_pack: int | None = None
) -> PackMode:
    """Host-side mirror of the WGSL pack ladder (for diagnostics)."""
    if max_pack is None:
        max_pack = max_pack_for_n(n)
    mu = max_uniq_for_n(n)
    if plen <= max_pack:
        return PackMode.RESIDENT
    if n_unique <= mu and plen <= MAX_PLEN_SHARED_META:
        return PackMode.UNIQUE_DELTA
    if n_unique <= MAX_NU_CAP and plen <= MAX_PLEN_FOR_UNIQ_TRY:
        # Global meta → value window = MAX_PACK (also covers ν≤μ with fat plen).
        useg_passes = (n_unique + max_pack - 1) // max(max_pack, 1)
        dense_passes = (plen + max_pack - 1) // max(max_pack, 1)
        if useg_passes <= dense_passes:
            return PackMode.UNIQUE_SEG
    return PackMode.TILE


def pack_stats(
    s: NDArray[np.integer],
    *,
    plen: int,
    off: int = 0,
    n: int | None = None,
) -> dict:
    """Footprint stats for a packed index vector ``s`` and dense box length ``plen``."""
    s64 = np.asarray(s, dtype=np.int64)
    keys = np.unique((s64[:, None] - s64[None, :]).ravel() + int(off))
    keys = keys[(keys >= 0) & (keys < plen)]
    nu = int(keys.size)
    beam_count = int(s64.size if n is None else n)
    mode = choose_mode(int(plen), nu, n=beam_count)
    mp, mu = max_pack_for_n(beam_count), max_uniq_for_n(beam_count)
    return {
        "plen": int(plen),
        "n_unique": nu,
        "occupancy": float(nu / max(plen, 1)),
        "mode": mode.value,
        "fits_resident": plen <= mp,
        "fits_unique": nu <= mu,
        "dense_passes": 1 if plen <= mp else (plen + mp - 1) // max(mp, 1),
        "useg_passes": 1 if nu <= mu else (nu + mu - 1) // max(mu, 1),
    }
