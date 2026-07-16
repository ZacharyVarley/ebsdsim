# cif_reader

Turn a CIF into an IT-standard crystal: space-group number, conventional cell,
asymmetric-unit sites, and the operators that go with that setting.

Dependency: NumPy.

```python
from ebsdsim.cif_reader import read_cif

s = read_cif("example.cif")  # SymmetryError if the file is inconsistent
print(s.log())
```

## What you get back

`Structure` fields used for simulation (IT standard frame already applied):

| Field | Meaning |
|-------|---------|
| `s.number` | IT space-group number 1…230 |
| `s.cell` | `(a, b, c, α, β, γ)` in Å / degrees |
| `s.species` | asymmetric-unit element symbols |
| `s.coords` | `(N, 3)` fractional, same frame as `s.cell` |
| `s.occupancies` | site occupancies |
| `s.uiso` | Uiso (`B` in the file is stored as `B / 8π²`) |
| `s.setting` | the `(P, p)` that was applied |
| `s.provenance` | which CIF tags / recovery path supplied the symmetry |

Do not apply `s.setting` again. Cell and coords are already transformed.

Also kept for bookkeeping:

- `s.cif_input`: lattice, sites, and declared symmetry tags **before** any
  transform (what the author wrote)
- `s.metadata()`: `{symmetry_provenance, cif_input, cell}` for master-pattern JSON

Operators for the standardized group:

```python
from ebsdsim.cif_reader import hall_ops, STD_HALL

ops = hall_ops(STD_HALL[s.number])  # list[Op]; x -> W @ x + w/24
```

For the 24 dual-origin groups, that Hall symbol is origin choice 2.

## Pipeline inside `read_cif`

### 1. Parse

A small extractor walks the text and keeps only the tags we care about (cell
lengths/angles, space-group / Hall / H-M tags, symop xyz loop, atom_site loop
with fract/occ/Uiso or Biso). Result is a `CifBlock`: lowercase tag → string or
list of strings.

### 2. Read cell + sites as deposited

Pull `a,b,c,α,β,γ` and the asymmetric atom list. Element symbols are cleaned
from `_atom_site_type_symbol` (or the label). Occupancy defaults to 1. Thermal
factors: prefer Uiso; if only Biso is present, convert with `U = B / 8π²`.

Before touching symmetry, the reader also records `cif_input` (source, declared
IT/Hall/H-M, cell, sites). That copy stays put: later steps may rewrite the
working cell and coordinates, but `cif_input` still matches the file as read.

### 3. Resolve symmetry

`symmetry_from_block` looks at the CIF in this order:

1. **Symop xyz loop** (or Hall symbol) → those operators are authoritative.
2. Else **H-M and/or IT number** (non-P1) → trust the group number `n`, but do
   not yet trust the setting. The deposited cell may be a non-standard unique
   axis, cell choice, or origin.
3. Else **nothing / P1** → try to recover a higher group from the sites alone
   (subgroup descent against Hall tables).

### 4. Map onto the IT standard setting

Goal: a change of basis `(P, p)` such that conjugating the CIF operators gives
the standard Hall group for space group `n`.

- Authoritative ops: `find_setting(ops, …)` fingerprints the closed group and
  looks up `(n, P, p)` in `_settings.npz`, or searches metric-legal `P` and a
  1/24-grid origin shift `p` if needed.
- Trusted IT/H-M only: try a short list of common conjugations of the standard
  ops (monoclinic unique-axis / cell-choice, orthorhombic axis perms, origin-1
  variants), keep the one that leaves the sites invariant and yields a
  conventional metric.
- Sites-only recovery: walk a subgroup DAG (`_descent.npz`) from the largest
  groups compatible with the cell metric down until sites stay invariant;
  then run `find_setting` on that operator set.

`P` is required to have `det = +1`. A mirror would swap enantiomorphic pairs
(e.g. `P4₁` ↔ `P4₃`), so those are rejected.

### 5. Apply the transform

- Cell ← `transform_cell(cell, P)` (and, if the input was rhombohedral `:R`,
  then the fixed hex←rhomb matrix). Metadata sets `rhombohedral_input=True` in
  that case; the lattice changes even when `transformed=False`.
- Coords ← `transform_coords(coords, P, p)`.
- Dual-origin groups always land on origin 2 via `STD_HALL[n]`.

### 6. Asymmetric unit

If sites were expanded or recovered from a full orbit, fold back to an
asymmetric unit under the standard operators.

Return a `Structure` whose `cell` / `coords` / `number` are ready to simulate.

## Files

| File | Role |
|------|------|
| `cif.py` | Parse, symmetry policy, `Structure` / `read_cif` |
| `sym.py` | `Op` algebra, Hall expand, `STD_HALL` / IT tables |
| `setting.py` | `(P, p)` search, sites descent, H-M salvage |
| `verify.py` | `python -m ebsdsim.cif_reader.verify` |
| `_settings.npz` | fingerprint → `(SG, P, p)` |
| `_descent.npz` | subgroup DAG + packed Hall ops |

Both `.npz` files load with `allow_pickle=False`. Rebuild after editing Hall
tables:

```bash
python -m ebsdsim.cif_reader.setting --rebuild
python -m ebsdsim.cif_reader.verify
```

The tables are opened lazily on first need. `warm_caches()` only loads them
earlier; it does not make a batch of reads faster than letting the first
`read_cif` pay that cost.

## How ebsdsim uses this

`ebsdsim.cif.load_structure` calls `read_cif`. The simulation cell is built from
the standardized `Structure`, and site orbits use `hall_ops(STD_HALL[n])`.
Master-pattern metadata stores both `cif_input` (as authored) and the used
IT-standard cell with setting stamps (`transformed`, `origin_choice`, `P`, `p`,
`rhombohedral_input`, …).

Manual `Material` construction still goes through the older `SG_OP_DATA` path;
only CIF ingest uses this package.
