"""GPU-accelerated dynamical EBSD master-pattern simulation.

Public entry points are re-exported from this package for convenience:

* :func:`~ebsdsim.master_pattern` / :func:`~ebsdsim.master_pattern_from_cif`
  — run a full GPU simulation
* :class:`~ebsdsim.MasterPattern` — in-memory result with Lambert raster
* :func:`~ebsdsim.save_master_pattern` / :func:`~ebsdsim.load_master_pattern`
  — compressed ``.npz`` I/O

See the `documentation <https://ebsdsim.readthedocs.io/>`_ for a quick start
and full API reference.
"""

from __future__ import annotations

from typing import Any

from ebsdsim._version import __version__

__all__ = [
    "Atom",
    "Cell",
    "LoadedMasterPattern",
    "Material",
    "MasterPattern",
    "NormalizeMode",
    "__version__",
    "load_master_pattern",
    "master_pattern",
    "master_pattern_from_cif",
    "save_master_pattern",
    "save_png_gray",
    "to_uint8",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Atom": ("ebsdsim.api", "Atom"),
    "Cell": ("ebsdsim.api", "Cell"),
    "Material": ("ebsdsim.api", "Material"),
    "MasterPattern": ("ebsdsim.api", "MasterPattern"),
    "master_pattern": ("ebsdsim.api", "master_pattern"),
    "master_pattern_from_cif": ("ebsdsim.api", "master_pattern_from_cif"),
    "NormalizeMode": ("ebsdsim.normalize", "NormalizeMode"),
    "LoadedMasterPattern": ("ebsdsim.mploader", "LoadedMasterPattern"),
    "load_master_pattern": ("ebsdsim.mploader", "load_master_pattern"),
    "save_png_gray": ("ebsdsim.mploader", "save_png_gray"),
    "to_uint8": ("ebsdsim.mploader", "to_uint8"),
    "save_master_pattern": ("ebsdsim.save", "save_master_pattern"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
