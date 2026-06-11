import importlib.resources

from ebsdsim.cif import parse_cif_crystal
from ebsdsim.kgrid import build_pg_k_grid
from ebsdsim.lookup import BuildLookupOptions, build_diff_lookup
from ebsdsim.runner import plan_safe_chunk_size, RunOneVoltageDeps
from ebsdsim.sgh import prepare_site_sgh_tables
from ebsdsim.structure import build_cell_from_cif
from ebsdsim.gpu import EBSDDynamicalKernels, require_gpu
from ebsdsim.runner import make_metric_buffer

ni = importlib.resources.files("ebsdsim").joinpath("data/preset_cifs/Ni.cif")
cell = build_cell_from_cif(parse_cif_crystal(ni.read_text(encoding="utf-8")))
ctx = require_gpu()
kernels = EBSDDynamicalKernels(ctx.device, ctx.queue)
lookup = build_diff_lookup(cell, BuildLookupOptions(voltage_kv=19.5, dmin=0.05))
sgh = prepare_site_sgh_tables(cell, 0.05)
pg = build_pg_k_grid(cell.pg_num, 250)
metric = make_metric_buffer(kernels, cell)
from ebsdsim.kgrid import transform_pg_k_grid_to_reciprocal

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
    kvecs, persistent, metric, batch_count=kvecs.size // 3, n_g=lookup.hkl.size // 3,
    bethe_c_cutoff=200, dbdiff_sg_cutoff=1.0, bethe_c_strong=20, bethe_c_weak=40,
)
chunk = plan_safe_chunk_size(
    256, ctx.device.limits, n_g=lookup.hkl.size // 3,
    n_strong=max(1, prescan.n_strong), n_weak=max(0, prescan.n_weak),
    rank=20, n_sites=sgh.n_sites,
)
print("n_strong", prescan.n_strong, "n_weak", prescan.n_weak, "effective_chunk", chunk)
