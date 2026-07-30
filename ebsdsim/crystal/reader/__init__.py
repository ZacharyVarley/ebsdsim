"""CIF reader: load any reasonable CIF into the IT standard setting.

    from ebsdsim.crystal.reader import read_cif
    s = read_cif("9011677.cif")

Modules: ``sym`` (Hall/IT tables), ``setting`` ((P,p) search), ``cif`` (Structure),
``verify`` (self-check). Warm caches once on long-lived processes.
"""

from .cif import (
    DEFAULT_B_ISO_ANGSTROM_SQ,
    DEFAULT_B_ISO_NM_SQ,
    DEFAULT_U_ISO_ANGSTROM_SQ,
    CifBlock,
    Structure,
    read_cif,
    read_cif_file,
    read_cif_text,
    structure_from_block,
    symmetry_from_block,
)
from .setting import (
    DEFAULT_EPS,
    Setting,
    find_setting,
    metric_family,
    recover_ops_from_sites,
    sites_invariant,
    transform_cell,
    transform_coords,
    warm_descent_cache,
    warm_settings_cache,
)
from .sym import (
    SGNAME,
    STD_HALL,
    Op,
    SymmetryError,
    close_group,
    close_packed,
    crystal_system,
    hall_ops,
    op_from_xyz,
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
    "DEFAULT_B_ISO_ANGSTROM_SQ", "DEFAULT_B_ISO_NM_SQ", "DEFAULT_U_ISO_ANGSTROM_SQ",
    "warm_descent_cache", "warm_settings_cache", "warm_caches",
    "CifBlock", "read_cif_file", "read_cif_text",
    "STD_HALL", "SGNAME", "crystal_system",
]
