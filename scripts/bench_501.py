import importlib.resources
import time

from ebsdsim.api import _run_master_pattern
from ebsdsim.structure import build_cell_from_cif_path

ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
cell = build_cell_from_cif_path(ni)
t0 = time.perf_counter()
r = _run_master_pattern(
    cell=cell,
    voltage_kv=20.0,
    halfw=250,
    dmin=0.05,
    energy_binwidth_keV=1.0,
    n_trajectories=0,
    sigma_deg=70.0,
    omega_deg=0.0,
    rank=20,
    chunk_size=1280,
    mode="bloch",
    marginal_coverage=1.0,
    relative_image_stop=0.01,
    mc_backend="surrogate",
    source="Ni",
)
dt = time.perf_counter() - t0
meta = r.metadata
print(
    f"total={dt:.2f}s bins_run={meta['n_bins_run']} "
    f"stopped={meta['stopped_by_relative_change']} "
    f"rel={meta['last_relative_change']:.4g} pattern_max={float(r.pattern.max()):.4g}"
)
