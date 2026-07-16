# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
