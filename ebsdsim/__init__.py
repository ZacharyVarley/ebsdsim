"""GPU-accelerated dynamical EBSD master-pattern simulation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ebsdsim.api import Atom, Cell, Material, MasterPattern, master_pattern, master_pattern_from_cif
from ebsdsim.mploader import LoadedMasterPattern, load_master_pattern, save_png_gray, to_uint8
from ebsdsim.save import save_master_pattern

try:
    __version__ = version("ebsdsim")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "Atom",
    "Cell",
    "LoadedMasterPattern",
    "Material",
    "MasterPattern",
    "__version__",
    "load_master_pattern",
    "master_pattern",
    "master_pattern_from_cif",
    "save_master_pattern",
    "save_png_gray",
    "to_uint8",
]
