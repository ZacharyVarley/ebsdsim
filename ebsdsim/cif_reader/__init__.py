"""
cif_reader -- load any reasonable CIF into the IT standard setting.

Dependencies: numpy.  Nothing else.

    from ebsdsim.cif_reader import read_cif
    s = read_cif("9011677.cif")
    print(s.log())
    s.number, s.cell, s.species, s.coords, s.occupancies, s.uiso
    s.metadata()   # cif_input (as-read) + cell (IT standard, as-used)

Layout:
  sym.py      operators, Hall, IT tables
  setting.py  (P,p) search, sites descent, H-M salvage
  cif.py      CIF extract + Structure / read_cif
  verify.py   self-check (python -m ebsdsim.cif_reader.verify)

Run `python -m ebsdsim.cif_reader.verify` after touching STD_HALL.
"""
from .cif import (
    CifBlock, Structure, read_cif, read_cif_file, read_cif_text,
    structure_from_block, symmetry_from_block,
)
from .setting import (
    DEFAULT_EPS, Setting, find_setting, metric_family,
    recover_ops_from_sites, sites_invariant, transform_cell, transform_coords,
    warm_descent_cache, warm_settings_cache,
)
from .sym import (
    Op, STD_HALL, SGNAME, SymmetryError, close_group, close_packed,
    crystal_system, hall_ops, op_from_xyz,
)


def warm_caches(prefill_hall=False):
    """Load settings table + descent DAG. Call once on a long-lived reader."""
    warm_settings_cache()
    warm_descent_cache(prefill_hall=prefill_hall)


__all__ = [
    "Op", "SymmetryError", "close_group", "close_packed", "op_from_xyz", "hall_ops",
    "Setting", "find_setting", "transform_cell", "transform_coords",
    "Structure", "read_cif", "structure_from_block", "symmetry_from_block",
    "recover_ops_from_sites", "sites_invariant", "metric_family", "DEFAULT_EPS",
    "warm_descent_cache", "warm_settings_cache", "warm_caches",
    "CifBlock", "read_cif_file", "read_cif_text",
    "STD_HALL", "SGNAME", "crystal_system",
]
