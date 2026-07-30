# Generated crystal tables



Do **not** edit these files by hand.



| File | Contents | Produced by |

|---|---|---|

| `sg_point_groups.py` | `SG_PG_FOR_SG` — space-group number (1–230) → crystallographic point-group number | Extracted from the retired `sg_ops.py` blob (`SG_PG_FOR_SG` only) |

| `pg_ops.py` | Point-group operator tables | `scripts/regenerate_pg_ops.py` |

| `wk_params.py` | Weickenmeier–Kohl atomic form-factor parameters | Ported from `wk-params.ts` |



The legacy `sg_ops.py` operator blob (`SG_OP_DATA` / `SG_OP_OFFSETS`) was removed:

orbit expansion now uses Hall operators from `ebsdsim.crystal.reader` exclusively.



Hand-written crystal code lives in the parent `crystal/` package.

`ebsdsim/data/` holds packaged binary assets (surrogate weights, preset CIFs).

