# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Use "fork" multiprocessing start method for macOS/Linux to allow simulating a master
  pattern from a script.

## [0.1.3] - 2026-06-08

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
