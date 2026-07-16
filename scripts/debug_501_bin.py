import importlib.resources
import numpy as np

from ebsdsim.integrate import PerVoltageContext
from ebsdsim.kgrid import build_pg_k_grid, transform_pg_k_grid_to_reciprocal
from ebsdsim.lookup import BuildLookupOptions, build_diff_lookup
from ebsdsim.runner import RunOneVoltageDeps, make_metric_buffer, run_one_voltage
from ebsdsim.sgh import prepare_site_sgh_tables
from ebsdsim.structure import build_cell_from_cif_path
from ebsdsim.gpu import EBSDDynamicalKernels, require_gpu
from ebsdsim.integrate import surrogate_to_multi_voltage_mc
from ebsdsim.surrogate import infer_direct_exp_from_cell_rebinned

ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
cell = build_cell_from_cif_path(ni)
direct = infer_direct_exp_from_cell_rebinned(
    cell=cell, sigma_deg=70.0, beam_kv=20.0, energy_binwidth_keV=1.0, n_energy_bins=20
)
mc = surrogate_to_multi_voltage_mc(direct, 20.0)
ctx = require_gpu()
hw = 250
pg = build_pg_k_grid(cell.pg_num, hw)
sgh = prepare_site_sgh_tables(cell, 0.05)
kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
metric = make_metric_buffer(kernels, cell)
lookup = build_diff_lookup(cell, BuildLookupOptions(voltage_kv=19.5, dmin=0.05))
kvecs = transform_pg_k_grid_to_reciprocal(pg, cell.direct_structure_matrix, lookup.mlambda)
persistent = kernels.create_persistent_buffers(
    hkl=lookup.hkl,
    hkl_hash=lookup.hkl_hash,
    diff_table=lookup.diff_table,
    coupling=lookup.coupling,
    reflection_dbdiff=lookup.reflection_dbdiff,
    sgh_tables=sgh.tables,
)
prescan = kernels.prescan_bethe_beam_counts_gpu(
    kvecs,
    persistent,
    metric,
    batch_count=kvecs.size // 3,
    n_g=lookup.hkl.size // 3,
    bethe_c_cutoff=200,
    dbdiff_sg_cutoff=1.0,
    bethe_c_strong=20,
    bethe_c_weak=40,
)
print("n_strong", prescan.n_strong, "n_weak", prescan.n_weak, "n_k", kvecs.size // 3)
deps = RunOneVoltageDeps(
    cell=cell,
    pg_grid=pg,
    sgh=sgh,
    kernels=kernels,
    metric=metric,
    chunk_size=256,
    rank=20,
    dmin=0.05,
)
res = run_one_voltage(
    PerVoltageContext(
        voltage_kv=19.5,
        bin_index=0,
        energy_weight=1.0,
        amplitude=float(mc.amplitudes[0]),
        beta=float(mc.betas[0]),
        mode="bloch",
    ),
    deps,
)
if res is None:
    print("None result")
else:
    p = res.pattern
    print("finite", np.all(np.isfinite(p)), "max", float(np.nanmax(p)), "bad", int(np.sum(~np.isfinite(p))))
