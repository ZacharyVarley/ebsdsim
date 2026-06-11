"""GPU Monte Carlo backscatter (fly-first boundary mode)."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
from numpy.typing import NDArray
import wgpu
from wgpu import BufferBindingType, BufferUsage, ShaderStage

from ebsdsim.binning import extra_energy_bin_params, make_batch_sizes, shave_and_renormalize_4d
from ebsdsim.gpu.buffers import StorageBuffer
from ebsdsim.gpu.device import require_gpu, sync_device
from ebsdsim.gpu.pipelines import load_wgsl
from ebsdsim.types import Cell, MultiVoltageMC

_DEPTH_MODE_LINEAR = 0
_DEPTH_BINS = 128
_DEPTH_BIN_SIZE_NM = 1.0
_MAX_STEPS = 1000
_DEFAULT_BATCH_SIZE = 262_144
_WORKGROUP_SIZE = 128

_FLY_FIRST_EXIT_MODEL = 1
_FLY_FIRST_DEPTH_METRIC = 0
_FLY_FIRST_BINNING_MODE = 1


def _fit_direct_exp_from_ed_pdf(
    ed_pdf: NDArray[np.floating],
    *,
    depth_bin_size_nm: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit per-energy direct exp-decay parameters from an E-D PDF."""
    ed = np.asarray(ed_pdf, dtype=np.float64)
    if ed.ndim != 2:
        raise ValueError("Expected ed_pdf shaped (E, D)")

    n_energy, n_depth = ed.shape
    energy_mass = np.clip(ed.sum(axis=1), 0.0, np.inf)
    mass_total = float(energy_mass.sum())
    if mass_total > 0.0:
        energy_weights = energy_mass / mass_total
    else:
        energy_weights = np.zeros_like(energy_mass)

    z = np.arange(n_depth, dtype=np.float64) * float(depth_bin_size_nm)
    z_max = float(n_depth) * float(depth_bin_size_nm)
    betas = np.zeros(n_energy, dtype=np.float64)

    for energy_idx in range(n_energy):
        mass_e = float(energy_mass[energy_idx])
        if mass_e <= 0.0:
            continue
        profile = np.clip(ed[energy_idx] / mass_e, 0.0, np.inf)
        valid = profile > 0.0
        if int(np.count_nonzero(valid)) < 3:
            continue
        x_fit = z[valid]
        y_fit = profile[valid]
        log_y = np.log(y_fit)
        weights = np.sqrt(np.maximum(y_fit, 1.0e-30))
        slope, _intercept = np.polyfit(x_fit, log_y, 1, w=weights)
        betas[energy_idx] = max(float(-slope), 0.0)

    amplitudes = np.zeros_like(energy_mass)
    positive = betas > 0.0
    if np.any(positive):
        beta = betas[positive]
        norm = (1.0 - np.exp(-beta * z_max)) / beta
        amplitudes[positive] = np.divide(
            energy_mass[positive],
            norm,
            out=np.zeros_like(beta),
            where=norm > 0.0,
        )
    return amplitudes, betas, energy_weights


def _weighted_relative_beta_change(
    prev_betas: NDArray[np.float64],
    prev_weights: NDArray[np.float64],
    betas: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Energy-weighted relative L2 change of the depth-decay parameters.

    Bins are weighted by the larger of the two energy weights so that bins
    carrying meaningful backscatter mass dominate the convergence test while
    near-empty bins (whose fits are noisy) are ignored.
    """
    n = min(prev_betas.size, betas.size)
    if n == 0:
        return float("inf")
    pb = prev_betas[:n]
    cb = betas[:n]
    w = np.maximum(prev_weights[:n], weights[:n])
    finite = np.isfinite(pb) & np.isfinite(cb) & np.isfinite(w) & (w > 0)
    if not np.any(finite):
        return float("inf")
    w = w[finite]
    diff = cb[finite] - pb[finite]
    num = float(np.sqrt(np.sum(w * diff * diff)))
    den = float(np.sqrt(np.sum(w * cb[finite] * cb[finite])))
    if den <= 1e-30:
        return float("inf")
    return num / den



def _pack_mc_params(
    *,
    n_trajectories: int,
    starting_e_kev: float,
    n_exit_energy_bins: int,
    n_exit_depth_bins: int,
    n_exit_direction_bins: int,
    binsize_exit_energy: float,
    binsize_exit_depth: float,
    atom_num: float,
    rho_gcc: float,
    atomic_weight_a: float,
    n_max_steps: int,
    sigma_deg: float,
    omega_deg: float,
    depth_mode: int,
    e_min_kev: float,
    exit_model: int,
    depth_metric: int,
    binning_mode: int,
) -> bytes:
    return struct.pack(
        "<IfIIIfffffIffIfIIIxxxxxxxx",
        n_trajectories,
        starting_e_kev,
        n_exit_energy_bins,
        n_exit_depth_bins,
        n_exit_direction_bins,
        binsize_exit_energy,
        binsize_exit_depth,
        atom_num,
        rho_gcc,
        atomic_weight_a,
        n_max_steps,
        sigma_deg,
        omega_deg,
        depth_mode,
        e_min_kev,
        exit_model,
        depth_metric,
        binning_mode,
    )


class _McBoundaryRunner:
    def __init__(self, ctx: Any) -> None:
        self.device = ctx.device
        self.queue = ctx.queue
        code = load_wgsl("mc_boundary_modes.wgsl")
        layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": ShaderStage.COMPUTE,
                    "buffer": {"type": BufferBindingType.storage},
                },
                {
                    "binding": 1,
                    "visibility": ShaderStage.COMPUTE,
                    "buffer": {"type": BufferBindingType.read_only_storage},
                },
                {
                    "binding": 2,
                    "visibility": ShaderStage.COMPUTE,
                    "buffer": {
                        "type": BufferBindingType.uniform,
                        "min_binding_size": 80,
                    },
                },
            ]
        )
        shader = self.device.create_shader_module(code=code, label="mc:boundary")
        self.pipeline = self.device.create_compute_pipeline(
            layout=self.device.create_pipeline_layout(bind_group_layouts=[layout]),
            compute={"module": shader, "entry_point": "main"},
            label="mc:boundary",
        )
        self.layout = layout
        self.params_buf = self.device.create_buffer(
            size=80,
            usage=BufferUsage.UNIFORM | BufferUsage.COPY_DST,
            label="mc:params",
        )

    def static_tail(self, params: dict[str, Any]) -> bytes:
        packed = _pack_mc_params(
            n_trajectories=0,
            starting_e_kev=float(params["starting_E_keV"]),
            n_exit_energy_bins=int(params["n_exit_energy_bins"]),
            n_exit_depth_bins=int(params["n_exit_depth_bins"]),
            n_exit_direction_bins=int(params["n_exit_direction_bins"]),
            binsize_exit_energy=float(params["binsize_exit_energy"]),
            binsize_exit_depth=float(params["binsize_exit_depth"]),
            atom_num=float(params["atom_num"]),
            rho_gcc=float(params["unit_cell_density_rho"]),
            atomic_weight_a=float(params["atomic_weight_A"]),
            n_max_steps=int(params["n_max_steps"]),
            sigma_deg=float(params["sigma"]),
            omega_deg=float(params["omega"]),
            depth_mode=0 if params["depth_mode"] == "linear" else 1,
            e_min_kev=float(params["e_min_keV"]),
            exit_model=_FLY_FIRST_EXIT_MODEL,
            depth_metric=_FLY_FIRST_DEPTH_METRIC,
            binning_mode=_FLY_FIRST_BINNING_MODE,
        )
        return packed[4:]

    def make_accumulator(self, params: dict[str, Any]) -> StorageBuffer:
        n_e = int(params["n_exit_energy_bins"])
        n_d = int(params["n_exit_depth_bins"])
        n_dir = int(params["n_exit_direction_bins"])
        acc = StorageBuffer(
            self.device,
            self.queue,
            label="mc:acc4d",
            byte_length=n_e * n_d * n_dir * n_dir * 4,
            copy_src=True,
        )
        acc.write(np.zeros(n_e * n_d * n_dir * n_dir, dtype=np.uint32))
        return acc

    def accumulate_batch(
        self,
        acc: StorageBuffer,
        static_tail: bytes,
        seeds: NDArray[np.uint32],
    ) -> None:
        n_batch = int(seeds.shape[0])
        seeds_buf = StorageBuffer(self.device, self.queue, label="mc:seeds", data=seeds)
        params_data = struct.pack("<I", n_batch) + static_tail
        self.queue.write_buffer(self.params_buf, 0, params_data)
        bg = self.device.create_bind_group(
            layout=self.layout,
            entries=[
                {"binding": 0, "resource": acc.buffer_binding()},
                {"binding": 1, "resource": seeds_buf.buffer_binding()},
                {"binding": 2, "resource": {"buffer": self.params_buf, "size": 80}},
            ],
        )
        global_size = ((n_batch + _WORKGROUP_SIZE - 1) // _WORKGROUP_SIZE) * _WORKGROUP_SIZE
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass(label="mc:boundary")
        cp.set_pipeline(self.pipeline)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(global_size // _WORKGROUP_SIZE, 1, 1)
        cp.end()
        self.queue.submit([enc.finish()])
        seeds_buf.destroy()

    def read_histogram(self, acc: StorageBuffer, params: dict[str, Any]) -> NDArray[np.float64]:
        n_e = int(params["n_exit_energy_bins"])
        n_d = int(params["n_exit_depth_bins"])
        n_dir = int(params["n_exit_direction_bins"])
        sync_device(self.device)
        return acc.read_as(np.uint32).astype(np.float64).reshape(n_e, n_d, n_dir, n_dir)


def _fit_mc_histogram(
    raw_hist: NDArray[np.float64],
    eparams: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n_extra = int(eparams["n_extra_bins"])
    n_fill = int(eparams["n_fillable_bins"])
    pdf_4d = shave_and_renormalize_4d(raw_hist, n_extra, n_fill)
    ed_pdf = np.asarray(pdf_4d.sum(axis=(2, 3)), dtype=np.float64)
    return _fit_direct_exp_from_ed_pdf(ed_pdf, depth_bin_size_nm=_DEPTH_BIN_SIZE_NM)


def run_monte_carlo_gpu(
    cell: Cell,
    voltage_kv: float,
    *,
    energy_binwidth_kev: float = 0.1,
    n_trajectories: int | None = None,
    n_exit_direction_bins: int = 64,
    sigma_deg: float = 20.0,
    omega_deg: float = 0.0,
    seed: int = 42,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    auto_stop: bool = True,
    relative_tol: float = 0.01,
    min_trajectories: int = 1_048_576,
    max_trajectories: int = 16_777_216,
    patience: int = 2,
) -> MultiVoltageMC:
    """Run fly-first GPU Monte Carlo and return a :class:`MultiVoltageMC` view.

    By default the simulation auto-batches: it keeps launching trajectory
    batches and re-fitting the per-voltage depth-decay parameters until those
    estimates stabilize (energy-weighted relative L2 change ``<= relative_tol``
    for ``patience`` consecutive checks), bounded by ``min_trajectories`` and
    ``max_trajectories``.

    Pass an explicit ``n_trajectories`` (or ``auto_stop=False``) to instead run
    a fixed trajectory budget without convergence checking.
    """
    ctx = require_gpu()
    runner = _McBoundaryRunner(ctx)

    eparams = extra_energy_bin_params(voltage_kv, energy_binwidth_kev, extra_energy_kv=0.0)
    params = {
        "atom_num": float(cell.average_atomic_number),
        "unit_cell_density_rho": float(cell.density),
        "atomic_weight_A": float(cell.average_atomic_weight),
        "starting_E_keV": float(eparams["starting_E_keV"]),
        "n_exit_energy_bins": int(eparams["n_sim_energy_bins"]),
        "n_exit_depth_bins": int(_DEPTH_BINS),
        "n_exit_direction_bins": int(n_exit_direction_bins),
        "binsize_exit_energy": float(energy_binwidth_kev),
        "binsize_exit_depth": float(_DEPTH_BIN_SIZE_NM),
        "depth_mode": "linear",
        "n_max_steps": int(_MAX_STEPS),
        "sigma": float(sigma_deg),
        "omega": float(omega_deg),
        "e_min_keV": 0.0,
    }

    fixed_budget = n_trajectories is not None or not auto_stop
    rng = np.random.default_rng(seed)
    acc = runner.make_accumulator(params)
    static_tail = runner.static_tail(params)

    total_trajectories = 0
    converged = False
    n_checks = 0
    last_change = float("inf")
    prev_betas: NDArray[np.float64] | None = None
    prev_weights: NDArray[np.float64] | None = None
    stable_streak = 0
    amplitudes = betas = energy_weights = None

    if fixed_budget:
        budget = int(n_trajectories if n_trajectories is not None else max_trajectories)
        for n_batch in make_batch_sizes(budget, batch_size):
            seeds = rng.integers(1, np.iinfo(np.uint32).max, size=(n_batch, 4), dtype=np.uint32)
            runner.accumulate_batch(acc, static_tail, seeds)
            total_trajectories += n_batch
    else:
        hard_cap = max(int(max_trajectories), int(min_trajectories))
        while total_trajectories < hard_cap:
            n_batch = min(batch_size, hard_cap - total_trajectories)
            seeds = rng.integers(1, np.iinfo(np.uint32).max, size=(n_batch, 4), dtype=np.uint32)
            runner.accumulate_batch(acc, static_tail, seeds)
            total_trajectories += n_batch
            if total_trajectories < min_trajectories:
                continue
            raw_hist = runner.read_histogram(acc, params)
            amplitudes, betas, energy_weights = _fit_mc_histogram(raw_hist, eparams)
            n_checks += 1
            if prev_betas is not None and prev_weights is not None:
                last_change = _weighted_relative_beta_change(
                    prev_betas, prev_weights, betas, energy_weights
                )
                stable_streak = stable_streak + 1 if last_change <= relative_tol else 0
                if stable_streak >= patience:
                    converged = True
                    break
            prev_betas = betas
            prev_weights = energy_weights

    raw_hist = runner.read_histogram(acc, params)
    acc.destroy()
    amplitudes, betas, energy_weights = _fit_mc_histogram(raw_hist, eparams)

    n_energy_bins = int(energy_weights.shape[0])
    voltages_kv = voltage_kv - (
        np.arange(n_energy_bins, dtype=np.float64) + 0.5
    ) * energy_binwidth_kev

    return MultiVoltageMC(
        binsize_energy_keV=float(energy_binwidth_kev),
        voltages_kv=voltages_kv,
        energy_weights=energy_weights.astype(np.float64, copy=False),
        amplitudes=amplitudes.astype(np.float64, copy=False),
        betas=betas.astype(np.float64, copy=False),
        n_trajectories=int(total_trajectories),
        converged=bool(converged),
        n_convergence_checks=int(n_checks),
        last_relative_change=float(last_change),
    )
