![ebsdsim — dynamical EBSD master patterns](https://raw.githubusercontent.com/ZacharyVarley/ebsdsim/main/docs/ebsdsim-banner.png)

# ebsdsim

[![CI](https://github.com/ZacharyVarley/ebsdsim/actions/workflows/ci.yml/badge.svg)](https://github.com/ZacharyVarley/ebsdsim/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/ebsdsim/badge/?version=latest)](https://ebsdsim.readthedocs.io/)
[![Python](https://img.shields.io/pypi/pyversions/ebsdsim)](https://pypi.org/project/ebsdsim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ebsdsim computes Lambert-projected Kikuchi master patterns from crystal structures
using multi-beam dynamical electron diffraction on the GPU (WebGPU). Patterns are
written as raw intensities in compressed `.npz` files; display scaling is applied
when you load them.

User documentation is at [ebsdsim.readthedocs.io](https://ebsdsim.readthedocs.io/).

## Installation

```bash
pip install ebsdsim
```

Simulations require a WebGPU adapter. See the
[installation guide](https://ebsdsim.readthedocs.io/en/latest/installation.html)
for Python version support, GPU drivers, and editable installs.

## Quick start

```python
import ebsdsim as es

mp = es.master_pattern_from_cif(
    "GaN.cif",
    voltage_kv=20.0,
    halfw=250,  # 501×501 raster; the default 500 gives 1001×1001
    sigma_deg=70.0,
)
mp.save("GaN-master-pattern.npz")
```

Bundled presets: `GaN.cif`, `Ni.cif`. See
[`examples/01_quick_start.ipynb`](examples/01_quick_start.ipynb).

## Documentation

- [Quick start](https://ebsdsim.readthedocs.io/en/latest/quickstart.html)
- [Simulation parameters](https://ebsdsim.readthedocs.io/en/latest/parameters.html)
- [API reference](https://ebsdsim.readthedocs.io/en/latest/api.html)
- [Changelog](CHANGELOG.md)

Development and releases: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
