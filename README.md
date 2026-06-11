<p align="center">
  <img src="docs/ebsdsim-banner.png" alt="ebsdsim — dynamical EBSD master patterns" width="100%">
</p>

# ebsdsim

[![PyPI](https://img.shields.io/pypi/v/ebsdsim)](https://pypi.org/project/ebsdsim/)
[![Python](https://img.shields.io/pypi/pyversions/ebsdsim)](https://pypi.org/project/ebsdsim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Dynamical EBSD master-pattern simulation for Python.**

`ebsdsim` computes Lambert-projected Kikuchi master patterns from a crystal structure
using multi-beam dynamical electron diffraction (Bloch formulation). Beams are classified
with **Bethe perturbation theory** (`bethe_c_*` cutoffs); the pattern is shaped equally by
the **beam and Monte Carlo model** (energy, tilt, depth / energy distribution) and by
those **dynamical settings** (Bethe cutoffs, rank, `dmin`).

The default Lambert raster uses `halfw=250`, i.e. **501×501** pixels
(`side = 1 + 2 × halfw`, always odd).

---

## Features

- **Dynamical diffraction** — Bloch-theory multi-beam solve with fixed-rank Smith /
  Lyapunov iteration; Bethe theory selects which reflections enter the system.
- **Experiment-aware weighting** — Monte Carlo depth / energy distributions (fast
  surrogate by default, or full GPU Monte Carlo) keyed to beam voltage and specimen
  tilt.
- **Automatic stopping** — energy-bin integration and MC both stop when their outputs
  stop changing.
- **Portable output** — compressed `.npz` files with symmetry-reduced fundamental-sector
  data; expand to full Lambert hemispheres with NumPy only via `mploader`.

---

## What drives the pattern?

| Input family | Examples | Typical use |
| --- | --- | --- |
| **Beam & specimen** | `voltage_kv`, `sigma_deg`, `omega_deg`, `energy_binwidth_keV` | Match microscope settings and sample tilt — usually the first knobs to tune against experiment. |
| **Monte Carlo** | `mc_backend`, `mc_auto_stop`, `mc_relative_tol`, `n_trajectories` | Control how depth and energy loss are sampled (or approximated by the surrogate). |
| **Dynamical diffraction** | `bethe_c_strong`, `bethe_c_weak`, `bethe_c_cutoff`, `dbdiff_sg_cutoff`, `rank`, `dmin` | Bethe beam selection and multi-beam solve settings. |
| **Raster** | `halfw` | Lambert half-width; output size is `(1 + 2×halfw)²`. |
| **Structure** | lattice, sites, `b_iso` | Crystal geometry and thermal motion (structure factors). |

All of the above are keyword arguments to `master_pattern` and `master_pattern_from_cif`.

---

## Debye–Waller factor

Per-site isotropic Debye–Waller factors are often **unknown** when only a lattice and
composition are available. When `b_iso` is omitted on an `Atom`, or when a CIF lacks
`_atom_site_B_iso_or_equiv` / `_atom_site_U_iso_or_equiv`, `ebsdsim` defaults to
**0.5 Å²** (stored internally as **0.005 nm²**) — a room-temperature isotropic
fallback.

Override per site when you have better data:

```python
es.Atom("Ga", x=1/3, y=2/3, z=0.0, b_iso=0.45)  # Å²
```

In the future, defaults should come from the CIF when present (e.g. values refined
against neutron diffraction), from DFT, or from modern surrogates that predict
site-resolved thermal parameters reliably. Until then, treat the default as a
placeholder and set `b_iso` explicitly when it matters for your comparison.

---

## Requirements

| | |
| --- | --- |
| Python | 3.10 – 3.13 |
| Runtime deps | NumPy, [wgpu](https://pypi.org/project/wgpu/) ≥ 0.29 |
| GPU | WebGPU-capable adapter (see platforms below) |

### Supported platforms

| OS | Backend | Notes |
| --- | --- | --- |
| **macOS** | Metal | |
| **Windows** | Vulkan or DirectX 12 | Requires recent GPU drivers. |
| **Linux** | Vulkan | Install Mesa/Vulkan drivers for your GPU. |

Simulation requires a working WebGPU adapter. Headless servers without GPU passthrough
will not be able to run `master_pattern*`.

---

## Install

```bash
pip install ebsdsim
```

From source (editable), with dev and notebook extras:

```bash
git clone https://github.com/ZacharyVarley/ebsdsim.git
cd ebsdsim
pip install -e ".[dev,docs]"
```

---

## Quick start

```python
import ebsdsim as es

mp = es.master_pattern_from_cif(
    "GaN.cif",                # bundled preset, or any path to a .cif file
    halfw=250,                # Lambert half-width → 501×501 output
    voltage_kv=20.0,
    sigma_deg=70.0,           # specimen tilt (degrees)
    omega_deg=0.0,            # azimuthal sample rotation (degrees)
    energy_binwidth_keV=1.0,
    mc_backend="surrogate",   # or "gpu"
    bethe_c_strong=20.0,
    bethe_c_weak=40.0,
    bethe_c_cutoff=200.0,
    dbdiff_sg_cutoff=1.0,
    rank=20,
    dmin=0.05,                # nm
)

print(mp.pattern.shape)       # (501, 501)
mp.save("GaN-master-pattern.npz")
```

Bundled presets: `"GaN.cif"`, `"Ni.cif"` (also accepts a full filesystem path).

Manual lattice + sites (lengths in Å). Omit `b_iso` to use the 0.5 Å² default:

```python
ni = es.Material(
    cell=es.Cell(a=3.52, b=3.52, c=3.52, space_group="Fm-3m"),
    atoms=[es.Atom("Ni", x=0.0, y=0.0, z=0.0)],
    name="Ni",
)
mp = es.master_pattern(ni, voltage_kv=20.0, sigma_deg=70.0, halfw=250)
```

---

## Beam & Monte Carlo parameters

These set the incident beam and how electrons lose energy and penetrate the sample before
diffraction. They are passed directly to `master_pattern*`:

| Parameter | Default | Role |
| --- | --- | --- |
| `voltage_kv` | `20.0` | Nominal beam energy (kV) |
| `sigma_deg` | `70.0` | Specimen tilt angle (degrees) fed to the MC / surrogate model |
| `omega_deg` | `0.0` | Azimuthal sample rotation (degrees); used by GPU MC |
| `energy_binwidth_keV` | `1.0` | Width of energy-loss bins integrated into the master pattern |
| `relative_image_stop` | `0.01` | Stop adding bins when Δimage/image falls below this |
| `marginal_coverage` | `1.0` | Fraction of MC energy bins to include (1.0 = all) |
| `mc_backend` | `"surrogate"` | `"gpu"` for full boundary Monte Carlo |
| `mc_auto_stop` | `True` | GPU MC: stop when depth/energy fits converge |
| `mc_relative_tol` | `0.01` | GPU MC convergence tolerance |
| `n_trajectories` | `1048576` | GPU MC trajectory budget when `mc_auto_stop=False` |

GPU Monte Carlo example:

```python
mp = es.master_pattern_from_cif(
    "GaN.cif",
    voltage_kv=20.0,
    sigma_deg=70.0,
    omega_deg=0.0,
    mc_backend="gpu",
    mc_auto_stop=True,
    mc_relative_tol=0.01,
)
print(mp.metadata["mc_n_trajectories"], mp.metadata["mc_converged"])
```

---

## Dynamical diffraction parameters

The solve always uses the **Bloch** formulation. **Bethe perturbation theory** ranks
reflections and decides which beams are strong, weak, or excluded (`bethe_c_*`). Those
cutoffs usually matter more than fine-tuning `dmin` or `rank` when matching reference
patterns:

| Parameter | Default | Role |
| --- | --- | --- |
| `bethe_c_strong` | `20.0` | Excitation-score threshold for strong beams |
| `bethe_c_weak` | `40.0` | Threshold band for weak beams |
| `bethe_c_cutoff` | `200.0` | Upper cutoff — beams above this are excluded |
| `dbdiff_sg_cutoff` | `1.0` | Structure-factor difference cutoff for beam coupling |
| `rank` | `20` | Truncation rank for the fixed-rank Lyapunov (Smith) solve |
| `dmin` | `0.05` nm | Minimum d-spacing — sets the reflection sphere (smaller → more beams, slower) |
| `halfw` | `250` | Lambert half-width; pattern size is `(1 + 2×halfw) × (1 + 2×halfw)` |

---

## Loading a saved pattern (NumPy only)

[`ebsdsim/mploader.py`](ebsdsim/mploader.py) is self-contained (NumPy only). Copy it
into any project that only needs to read `.npz` output.

```python
from ebsdsim.mploader import load_master_pattern, to_uint8, save_png_gray

mp = load_master_pattern("GaN-master-pattern.npz")

nh = mp.data[0, 0, 0]   # energy-integrated, site-integrated, north hemisphere
save_png_gray(to_uint8(nh), "GaN_integrated_nh.png")
```

### `.npz` layout (summary)

| Array | Meaning |
| --- | --- |
| `fundamental_sector` | Symmetry-reduced intensities `(E, S, n_k)` |
| `fundamental_kij`, `fundamental_khat` | Lambert indices and unit directions per FS pixel |
| `pg_operators`, `fs_normals` | Point-group matrices and sector bounding normals |
| `bin_voltages_kv`, `bin_weights` | Per-bin beam voltage and energy weight |
| `meta_json` | UTF-8 JSON simulation metadata (includes beam, MC, and dynamical settings) |

See [`examples/02_save_and_load.ipynb`](examples/02_save_and_load.ipynb) for a full
walkthrough.

---

## Examples

| Resource | Description |
| --- | --- |
| [`examples/01_quick_start.ipynb`](examples/01_quick_start.ipynb) | GaN and Ni patterns |
| [`examples/02_save_and_load.ipynb`](examples/02_save_and_load.ipynb) | Save, reload, export PNGs |
| [`scripts/run_gan_example.py`](scripts/run_gan_example.py) | Full-resolution GaN script |

Preset CIFs ship under `ebsdsim/data/preset_cifs/` (Ni, GaN).

---

## Development

```bash
pip install -e ".[dev,docs]"
pytest -m "not slow"          # CPU tests; GPU tests skip if no adapter
pytest -m slow                # end-to-end GPU runs (needs WebGPU)
python -m build               # sdist + wheel (requires `build` extra)
```

CI runs CPU tests on Ubuntu (Python 3.11–3.12) and full GPU tests on macOS
(Metal). See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### Releasing (automated PyPI)

Releases are published by pushing a version tag. The
[`.github/workflows/release.yml`](.github/workflows/release.yml) workflow builds the
sdist/wheel, uploads them to the GitHub release, and publishes to PyPI via **trusted
publishing** (no API token in the repo).

**One-time setup**

1. On [PyPI](https://pypi.org/manage/account/publishing/): add a **pending publisher**
   - PyPI project name: `ebsdsim`
   - Owner: `ZacharyVarley`, repository: `ebsdsim`
   - Workflow: `release.yml`, environment: `pypi`
2. On GitHub: repo **Settings → Environments** → create environment `pypi` (no secrets
   required for trusted publishing).

**Each release**

1. Set `version` in `pyproject.toml` and move notes in `CHANGELOG.md` out of
   **Unreleased**.
2. Commit and push to `main`.
3. Tag and push (tag must match `pyproject.toml`, without the `v` prefix):

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

4. Watch the **Release** workflow; the package appears on
   [pypi.org/project/ebsdsim](https://pypi.org/project/ebsdsim/) when it finishes.

---

## Contributing

1. Fork the repo and create a branch from `main`.
2. `pip install -e ".[dev]"` and run `pytest -m "not slow"` before opening a PR.
3. If your change touches GPU paths or save/load, also run `pytest -m slow` on a
   machine with WebGPU (macOS Metal works).
4. Keep PRs focused; include a short note in `CHANGELOG.md` under **Unreleased**
   when user-visible behavior changes.

Bug reports and feature requests: [GitHub Issues](https://github.com/ZacharyVarley/ebsdsim/issues).

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) Zachary Varley.
