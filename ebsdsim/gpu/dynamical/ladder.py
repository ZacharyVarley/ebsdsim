"""Portable fallback ladder for Smith Toeplitz packs (shared-memory matvecs).

Decision per k (after LLL Smith reindex of strong beams):

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
- BiCGSTAB: ``wgsl/smith_build_uniq_shared_bicgstab_f16.wgsl``
  bit28 = unique, bit29 = BiCGSTAB, bit30 = unique-seg, bit31 = dense tile
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

MAX_PLEN_FOR_UNIQ_TRY = 65536
MAX_PLEN_SHARED_META = 20000
MAX_NU_CAP = 16384
MAX_BEAMS = 2048

# Bucket table: (MAX_N, MAX_PACK, MAX_UNIQ, BPT). Tuned so ss/sord/sp
# (MAX_N*16 B) + spack (MAX_PACK*4 B) + fixed arrays (~2.2 KB) fit a
# 48 KiB workgroup-storage budget — the D3D12-class device limit. The real
# budget comes from device.limits["max-compute-workgroup-storage-size"];
# buckets shrink to fit smaller budgets (e.g. 32 KiB Metal devices).
_BUCKET_TABLE = (
    (384, 10200, 8600, 2),
    (512, 9690, 8100, 2),
    (768, 8600, 7000, 3),
    (1024, 7600, 6000, 4),
)
# red_re/red_im (2*256*4 B) + geom (128 B) + scalars.
_OVERHEAD_BYTES = 2200
# Galerkin solve adds g_rdiag (16×4) + g_kept (+ alignment) ≈ +68 B workgroup
# storage. Keep smith at 2200 so its shared-memory tiling / pack buckets are
# unchanged; callers of the galerkin path must pass GALERKIN_OVERHEAD_BYTES.
GALERKIN_OVERHEAD_BYTES = 2400
_MIN_PACK = 1024
# Shared unique meta (META_BITS_SLOTS 939 + MAX_UWORDS_SHARED 625) packed into
# the spack tail beyond MAX_UNIQ.
_META_SLOTS = 1564
DEFAULT_SHARED_BUDGET = 49152


def bucket_params_for_n(
    n: int,
    *,
    shared_budget: int = DEFAULT_SHARED_BUDGET,
    pack_entry_bytes: int = 4,
    overhead_bytes: int = _OVERHEAD_BYTES,
) -> tuple[int, int, int, int]:
    """(MAX_N, MAX_PACK, MAX_UNIQ, BPT) for beam count ``n``.

    Pack sizes are capped so the workgroup arrays fit ``shared_budget`` bytes
    (the device limit), so small-budget devices slide down the pack ladder
    (resident → unique-Δ → unique-seg → dense tile) instead of failing
    pipeline creation. ``pack_entry_bytes`` is 4 for the f16 spack variant
    and 8 for the f32 variant (used to A/B Metal f16 behavior).

    Pass ``overhead_bytes=GALERKIN_OVERHEAD_BYTES`` when sizing buckets for the
    Galerkin solve shader (extra ``g_rdiag`` / ``g_kept`` workgroup storage).
    """
    n = int(n)
    if n <= 1024:
        for max_n, pack, uniq, bpt in _BUCKET_TABLE:
            if n <= max_n:
                break
    elif n <= MAX_BEAMS:
        max_n = ((n + 255) // 256) * 256
        pack, uniq, bpt = 0, 0, max_n // 256
    else:
        raise ValueError(
            f"Smith shader supports at most {MAX_BEAMS} beams "
            f"(Krylov workgroup arrays), got {n}"
        )
    entry = max(4, int(pack_entry_bytes))
    overhead = int(overhead_bytes)
    avail = (int(shared_budget) - max_n * 16 - overhead) // entry
    if avail < _MIN_PACK:
        raise ValueError(
            f"n={n} beams needs {max_n * 16 + overhead + _MIN_PACK * entry} B "
            f"of workgroup storage but the device allows {shared_budget} B"
        )
    if pack:
        max_pack = max(_MIN_PACK, min(pack, avail))
        max_uniq = max(0, min(uniq, max_pack - _META_SLOTS))
    else:
        max_pack = max(_MIN_PACK, avail)
        max_uniq = max(0, max_pack - _META_SLOTS)
    return max_n, max_pack, max_uniq, bpt


def galerkin_bucket_params_for_n(
    n: int, *, shared_budget: int = DEFAULT_SHARED_BUDGET, pack_entry_bytes: int = 4
) -> tuple[int, int, int, int]:
    """Like ``bucket_params_for_n`` but reserves Galerkin epilogue shared overhead."""
    return bucket_params_for_n(
        n,
        shared_budget=shared_budget,
        pack_entry_bytes=pack_entry_bytes,
        overhead_bytes=GALERKIN_OVERHEAD_BYTES,
    )


def max_pack_for_n(n: int, *, shared_budget: int = DEFAULT_SHARED_BUDGET) -> int:
    return bucket_params_for_n(n, shared_budget=shared_budget)[1]


def max_uniq_for_n(n: int, *, shared_budget: int = DEFAULT_SHARED_BUDGET) -> int:
    return bucket_params_for_n(n, shared_budget=shared_budget)[2]


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
