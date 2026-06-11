# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.1]: https://github.com/ZacharyVarley/ebsdsim/releases/tag/v0.1.1
[0.1.0]: https://github.com/ZacharyVarley/ebsdsim/releases/tag/v0.1.0
