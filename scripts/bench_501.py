import importlib.resources
import time

from ebsdsim.crystal.build import build_cell_from_cif_path
from ebsdsim.engine.master_pattern import run_master_pattern
from ebsdsim.engine.params import SimParams

ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
cell = build_cell_from_cif_path(ni)
t0 = time.perf_counter()
params = SimParams(
    voltage_kv=20.0,
    halfw=250,
    dmin=0.05,
    energy_binwidth_keV=1.0,
    n_trajectories=0,
    sigma_deg=70.0,
    omega_deg=0.0,
    rank=20,
    chunk_size=1280,
    marginal_coverage=1.0,
    relative_image_stop=0.01,
    mc_backend="surrogate",
)
r = run_master_pattern(cell, params, source="Ni", mode="bloch")
dt = time.perf_counter() - t0
meta = r.metadata
print(
    f"total={dt:.2f}s bins_run={meta['n_bins_run']} "
    f"stopped={meta['stopped_by_relative_change']} "
    f"rel={meta['last_relative_change']:.4g} pattern_max={float(r.pattern.max()):.4g}"
)
