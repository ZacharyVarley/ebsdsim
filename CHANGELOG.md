# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- macOS/Metal: work around the wgpu-hal blocking-poll deadlock
  ([gfx-rs/wgpu#9531](https://github.com/gfx-rs/wgpu/issues/9531)) that made
  GPU synchronisation hang forever after command buffers longer than
  ~hundreds of ms — long dynamical-solve dispatches could wedge any wait on
  Apple Silicon. The upstream fix has not shipped in any wgpu-py release
  (0.31.1 bundles wgpu-native v27, 0.32.0 bundles v29.0.0.0; both predate
  it), so on macOS the device poll thread now polls non-blocking.
- macOS/Metal: smith_iterative shader buckets are sized to the device's
  reported `max-compute-workgroup-storage-size` instead of a hardcoded
  ~48 KiB budget, so devices with smaller workgroup storage slide down the
  pack ladder instead of failing pipeline creation. Buckets on 48 KiB
  devices (D3D12-class) are unchanged.
- macOS/Metal: the unique-segment tile path stored compacted deltas in an
  f16 storage buffer; Metal miscompiles f16 storage writes (values read
  back as zeros), producing silently all-zero master patterns for
  plen-heavy cells. The compacted-value buffer is now f32 (f16 remains
  workgroup-only, where it is correct on all tested backends).
- `wgpu==0.32.0` is supported again: the 0.2.1 exclusion is replaced by a
  runtime shim that corrects that release's queue work-done CFFI callback
  signature (its bundled wgpu-native v29 added a `WGPUStringView`
  parameter the Python codegen missed). The shim activates only on
  wgpu-py 0.32.0.

## [0.2.1] - 2026-07-30

### Fixed

- Exclude `wgpu==0.32.0`: that upstream release breaks
  `queue.on_submitted_work_done_sync()` (TypeError from a callback signature
  mismatch against wgpu-native v27) on every platform, which crashed all GPU
  entry points on a fresh install. `wgpu>=0.29,!=0.32.0` resolves to a working
  build until upstream ships a fix.

## [0.2.0] - 2026-07-30

### Changed

- Package layout now mirrors the simulation pipeline (`crystal` → `physics` →
  `{lambert, energy}` → `gpu` → `engine` → `io` → `api`).
- Default dynamical solver is Smith iterative (`solver="smith_iterative"`): a
  BiCGSTAB Krylov solver over fixed-rank Smith iterates (rank 16), producing
  per-bin patterns plus the voltage-integrated total. Requires WebGPU
  `shader-f16`. Pass `solver="lu_smith"` for the previous dense LU–Smith path
  (`rank` applies only to `lu_smith`).
- Dense LU–Smith host code is split into stage modules under
  `ebsdsim.gpu.dynamical` (`workspace` / `score` / `assemble` / `solve` /
  `intensity` / `chunks`) with a thin `kernels` façade.
- Internal simulation cell renamed to `SimCell` / `SimAtom` (nm); public
  `api.Cell` / `api.Atom` remain ångström specs.
- Master-pattern loader lives at `ebsdsim.io.load` (Lambert math shared with
  `ebsdsim.lambert`).
- Simulation keywords are one `SimParams` dataclass (`ebsdsim.engine.params`)
  instead of being re-declared on each entry point; `master_pattern_from_cif`
  now forwards `**kwargs` to `master_pattern`.
- `Atom` / `Cell` / `Material` are defined in `ebsdsim.crystal.material` and
  `MasterPattern` in `ebsdsim.engine.results`; both are re-exported from
  `ebsdsim` and `ebsdsim.api`, so existing imports are unaffected.
- Orbit expansion for hand-built `Material` inputs now uses the same
  International Tables Hall-operator path as CIF ingest. **This changes
  symmetry-expanded sites for 24 dual-origin space groups** (48, 50, 59, 68,
  70, 85, 86, 88, 125, 126, 129, 130, 133, 134, 137, 138, 141, 142, 201, 203,
  222, 224, 227, 228) — see *Fixed*. Other space groups are unaffected.
- `folding_symbol` and `build_pg_k_grid` now require the space group, and the
  number-only symbol lookup (`pg_num_to_symbol` / `unoriented_pg_symbol`) has
  been removed entirely, so an orientation-ambiguous point group can no longer
  be resolved from its number alone.

### Fixed

- Hand-built `Material` inputs used a legacy operator table that produced wrong
  symmetry orbits for the 24 dual-origin space groups, including incorrect site
  multiplicities. Space group 227 with an atom at (⅛, ⅛, ⅛) — the diamond site —
  expanded to 16 sites instead of 8. Both ingest paths now use Hall operators.
  Orbit expansion is also canonically sorted, so results no longer depend on the
  order symmetry operators happen to be enumerated in.
- CIF ingest now applies the documented default Debye–Waller factor
  **B_iso = 0.5 Å² (0.005 nm²)** when `_atom_site_B_iso_or_equiv` /
  `_atom_site_U_iso_or_equiv` are missing or non-positive (including explicit
  `0`, which is unphysical at finite temperature), matching the Material path.
  The CIF reader warns and substitutes the default before simulation cell build.
- LU–Smith chunk planning now respects
  `max_compute_workgroups_per_dimension` (in addition to storage-buffer size),
  so large Bethe systems no longer dispatch illegal 1-D workgroup counts.
- The GPU Lambert rasterizer fills sheets in bounded per-view bands instead of
  stacking every energy bin into one dispatch, so the storage-buffer binding
  limit is no longer hit at large `halfw` with many bins (bitwise-identical
  output).

### Notes

- Patterns from CIFs that previously ran with B=0 (no U/B tags, or explicit
  non-positive U/B) should be regenerated if Debye–Waller matters for the use
  case.

## [0.1.10] - 2026-07-16

### Fixed

- Structure-factor (and SGH) phase sums no longer treat zero-padded orbit slots as
  real atoms at the origin. Dense stacking of per-site orbits left short
  multiplicities padded with `(0,0,0)`; summing `cos`/`sin` over those pads
  inflated `|U|`, Bethe beam counts, and dynamical intensities whenever site
  multiplicities differed within a cell. Equal-multiplicity / single-orbit
  crystals (e.g. Ni) were unaffected. Present since 0.1.0.
- `expi_vec` no longer triggers a spurious `log1p` divide-by-zero warning when
  evaluating the exponential integral for tiny arguments (vectorized `np.where`
  was computing both branches).
- CIF ingest now applies the documented default Debye–Waller factor
  **B_iso = 0.5 Å² (0.005 nm²)** when `_atom_site_B_iso_or_equiv` /
  `_atom_site_U_iso_or_equiv` are missing or non-positive, matching the Material
  path. A console message reports how many sites received the default.

### Notes

- Master patterns saved before 0.1.10 for crystals with **unequal** site
  multiplicities should be regenerated. Equal-multiplicity cells are unchanged.
  Patterns from CIFs that previously ran with B=0 (no U/B tags) should also be
  regenerated if Debye–Waller matters for the use case.

## [0.1.9] - 2026-07-16

### Changed

- CIF ingest now uses the vendored `ebsdsim.cif_reader` package (NumPy only)
  instead of PyCifRW. Structures are standardized to the International Tables
  setting on load (dual-origin groups use origin choice 2) and expanded with
  matching Hall operators, not legacy `SG_OP_DATA`.
- Master-pattern metadata from CIF includes `cif_input` (as-read snapshot) and
  IT-standard setting stamps on `cell` (`setting`, `origin_choice`,
  `transformed`, `P`, `p`, …).

### Fixed

- Corrected Lambert fundamental-sector folding for several point groups whose
  orientation was wrong. These are grouped by cause below. In every case the
  fundamental sector (and, for the orientation bugs, the point-group operators)
  used to fold the sphere did not match the crystal, so the affected master
  patterns were sampled and expanded over the wrong region.
  - Monoclinic `2`, `m`, `2/m` now fold in the unique-b setting (matching the IT
    cells ebsdsim simulates), not unique-c. Affects space groups 3-15.
  - Monoclinic `2/m` fundamental sector was `{z>=0, x>=0}`, which left half the
    sphere uncovered because the `m_y` mirror flips only y. Corrected to
    `{z>=0, y>=0}`. Affects space groups 10-15.
  - Trigonal and secondary-axis orientations that `pg_from_sg` cannot
    distinguish are now selected from the space-group number: `312`/`31m`/`-31m`
    (space groups 149, 151, 153, 157, 159, 162, 163), `-4m2` (115-120), and
    `-62m` (189, 190). Previously these folded with the point-group axes rotated
    30 or 45 degrees.
- Verified the folding point-group selection against each crystal's actual
  Cartesian symmetry (derived from its space-group operators) for all 230 space
  groups across multiple cell metrics per system.

### Notes

- Master patterns saved before 0.1.9 for the affected space groups (3-15,
  115-120, 149, 151, 153, 157, 159, 162, 163, 189, 190) store fundamental-sector
  data sampled over the wrong region and cannot be repaired in place; regenerate
  them. All other space groups are unaffected and their saved files remain valid.

### Removed

- Dependency on PyCifRW.

## [0.1.8] - 2026-06-25

### Added

- `exact_slow_cpu` on `master_pattern` / `master_pattern_from_cif`: full-rank CPU
  Lyapunov via batched `numpy.linalg.eig` (GPU dynamical assembly unchanged).
- `verbosity` (`0`, `1`, or `2`, default `0`): run banner, per-bin MC
  weights/timing, and (at level 2) dynamical beam counts and chunk throughput.
- `ebsdsim.lyapunov_cpu` helpers and unit tests.

### Changed

- Shortened `README.md`; development, releases, and parameter tables moved to
  `CONTRIBUTING.md` and Sphinx (`docs/installation.rst`, `docs/parameters.rst`).

## [0.1.7] - 2026-06-15

### Changed

- Shared `dynamical_voltages_kv` helper for surrogate and GPU MC voltage assignment.

## [0.1.6] - 2026-06-15

### Added

- Sphinx API documentation (`docs/`), Read the Docs config (`.readthedocs.yaml`),
  and a CI docs build job. Hosted at [ebsdsim.readthedocs.io](https://ebsdsim.readthedocs.io/).
- NumPy-style docstrings on the public API.
- `Fe_gamma.cif` regression fixture (high-temperature γ-Fe, spaced `F m 3 m` symbol).

### Fixed

- CIF tag lookup is case-insensitive (e.g. `_symmetry_space_group_name_H-M`).
- Hermann–Mauguin symbols without an IT number (e.g. `F m 3 m`) resolve to the
  correct space group and point group when building a cell from CIF.
- `bin_voltages_kv` in saved master patterns uses ``beam_kv - i * energy_binwidth``
  (e.g. 20, 19, 18 kV at 1 keV width for a 20 kV beam).

## [0.1.5] - 2026-06-14

### Fixed

- P1 (triclinic) master patterns: point group `1` now uses an empty fundamental-sector
  normal set so the Lambert grid covers the full hemisphere(s) instead of failing with
  “No FS normals for PG symbol '1'” ([#3](https://github.com/ZacharyVarley/ebsdsim/issues/3)).

### Changed

- `LookupPrefetcher` builds the next voltage's diff lookup on **one background thread**
  (`ThreadPoolExecutor`, `max_workers=1`) instead of a process pool. Process pools
  required OS-specific `fork`/`spawn` handling ([#1](https://github.com/ZacharyVarley/ebsdsim/issues/1),
  [#2](https://github.com/ZacharyVarley/ebsdsim/pull/2)) and caused `BrokenProcessPool`
  on macOS when forking after wgpu/threads were already active ([#3](https://github.com/ZacharyVarley/ebsdsim/issues/3)).
  Threading is the same on macOS, Windows, and Linux: NumPy lookup work overlaps GPU
  bins without forking or pickling `DiffLookupGeometry`.
- `build_lookup_cache` is sequential only (removed unused parallel pool path).
- GPU/API tests use `dmin=0.05` (package default); avoid `dmin > 0.05` in tests.

### Added

- `tests/test_pg_grid.py` for P1 FS normals and k-grid coverage.
- `scripts/gpu_util_gan.py` — optional per-bin lookup-wait and `nvidia-smi` utilization
  profiling (`python scripts/gpu_util_gan.py Ni` or `GaN`).

## [0.1.4] - 2026-06-14

### Changed

- CIF parsing now uses [PyCifRW](https://pypi.org/project/PyCifRW/) instead of a
  custom tokenizer/parser.

### Fixed

- On macOS/Linux, diff-lookup multiprocessing uses the `fork` start method so
  `LookupPrefetcher` and parallel lookup builds work from plain scripts (fixes
  spawn re-import of `__main__`). Thanks to
  [Håkon Wiik Ånes](https://github.com/hakonanes) ([#2](https://github.com/ZacharyVarley/ebsdsim/pull/2)).

## [0.1.3] - 2026-06-11

### Changed

- Master pattern output (`pattern`, `data`, `integrated`, and saved `.npz` arrays) is
  always **raw** dynamical intensities. Display scaling is applied only on demand via
  `lambert_data()` / `reconstruct()` / `RasterizeOptions`, never baked into simulation
  results.
- Display normalization API: `normalize="minmax"` | `"robust"` | `None` replaces the
  old boolean `normalize` / `robust` flags. Robust mode uses configurable percentiles
  (`robust_p_low` / `robust_p_high`, default 1st–99th).
- `ebsdsim.__version__` is read from `ebsdsim._version` (single source of truth for
  releases; works in editable checkouts, not only when installed from PyPI).

### Added

- `ebsdsim.normalize` with `NormalizeMode` and `scale_fs_channel` for on-demand display
  scaling.
- `ebsdsim.weights` with site weights (occupancy × multiplicity) for site marginals.
- Saved `.npz` files include `site_weights`; `bin_weights` are the exact Monte Carlo
  energy marginal weights used when forming the energy-integrated pattern.
- Tests for normalization, site weights, and Lambert display scaling.

### Fixed

- `MasterPattern.lambert_data(normalize=...)` reshapes embedded point-group operators
  correctly (flat PG tables from `pg_ops` are now accepted).
- Standalone `mploader` remains NumPy-only (scaling helpers are duplicated inline).
- Release workflow and setuptools version discovery: lazy `ebsdsim` package `__init__` so
  reading `__version__` does not import NumPy/wgpu before dependencies are installed.

## [0.1.2] - 2026-06-11

### Fixed

- PyPI README: markdown banner image with absolute `raw.githubusercontent.com` URL (PyPI strips HTML `<img>` and does not resolve relative paths). Drop redundant PyPI version badge; restore CI badge now that the repo is public.

## [0.1.1] - 2026-06-11

### Fixed

- PyPI README: use PNG banner (PyPI blocks SVG) and shields.io badges (GitHub Actions badges do not render for private repos).

## [0.1.0] - 2026-06-08

### Added

- GPU-accelerated dynamical EBSD master-pattern simulation via WebGPU/wgpu-py.
- `master_pattern` / `master_pattern_from_cif` public API with surrogate MC (default) and optional GPU Monte Carlo.
- Compressed `.npz` export with embedded point-group operators for offline expansion.
- NumPy-only `mploader` for reading saved patterns without a GPU.
- Preset CIFs (Ni, GaN), examples notebooks, and regression tests.

[0.1.3]: https://github.com/ZacharyVarley/ebsdsim/releases/tag/v0.1.3
[0.1.2]: https://github.com/ZacharyVarley/ebsdsim/releases/tag/v0.1.2
[0.1.1]: https://github.com/ZacharyVarley/ebsdsim/releases/tag/v0.1.1
[0.1.0]: https://github.com/ZacharyVarley/ebsdsim/releases/tag/v0.1.0
