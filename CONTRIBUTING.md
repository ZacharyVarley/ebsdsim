# Contributing

Development is managed on the `main` branch of
[github.com/ZacharyVarley/ebsdsim](https://github.com/ZacharyVarley/ebsdsim).

- [GitHub Issues](https://github.com/ZacharyVarley/ebsdsim/issues) for bugs and
  feature requests.
- Pull requests against `main` for code changes.

## Setup

```bash
git clone https://github.com/ZacharyVarley/ebsdsim.git
cd ebsdsim
pip install -e ".[dev,docs]"
```

## Tests

```bash
pytest -m "not slow"    # CPU tests; GPU tests skip without an adapter
pytest -m slow          # end-to-end GPU runs (WebGPU required)
```

CI runs CPU tests on Ubuntu (Python 3.11–3.12) and GPU tests on macOS (Metal).
See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Documentation

```bash
cd docs && make html
```

API docs are hosted on [Read the Docs](https://ebsdsim.readthedocs.io/) from
[`.readthedocs.yaml`](.readthedocs.yaml).

## Pull requests

Keep changes focused. Note user-visible changes under **Unreleased** in
[`CHANGELOG.md`](CHANGELOG.md).

If your change touches GPU paths or save/load, run `pytest -m slow` on a machine
with WebGPU before opening the PR.

## Releasing

Releases are published by pushing a version tag. The
[release workflow](.github/workflows/release.yml) builds sdist/wheel artifacts
and publishes to PyPI via trusted publishing.

**One-time PyPI setup**

1. [PyPI publishing](https://pypi.org/manage/account/publishing/): pending
   publisher for `ebsdsim`, owner `ZacharyVarley`, repository `ebsdsim`, workflow
   `release.yml`, environment `pypi`.
2. GitHub **Settings → Environments**: create environment `pypi`.

**Each release**

1. Bump `__version__` in `ebsdsim/_version.py` and add notes to `CHANGELOG.md`.
2. Commit and push to `main`.
3. Tag and push (tag must match `ebsdsim._version`, with a `v` prefix):

   ```bash
   git tag v0.1.8
   git push origin v0.1.8
   ```
