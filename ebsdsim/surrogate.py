"""TinyHistNet MC surrogate (matches the web app default energy model)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DEPTH_BIN_SIZE_NM = 1.0
DEFAULT_DEPTH_BINS = 128

_model: "NumpySurrogateModel | None" = None


@dataclass(frozen=True)
class SurrogateDirectExp:
    amplitudes: np.ndarray
    betas: np.ndarray
    energy_mass: np.ndarray
    energy_weights: np.ndarray
    energy_centers_keV: np.ndarray


class NumpySurrogateModel:
    _BATCHNORM_EPS = 1.0e-5

    def __init__(self, weights_file: Path | str) -> None:
        blob = np.load(Path(weights_file), allow_pickle=False)
        self.energy_bin_size_kev = float(self._scalar(blob["energy_bin_size_kev"]))
        if "output_parameterization" in blob:
            raw = self._scalar(blob["output_parameterization"])
            self.output_parameterization = str(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        else:
            self.output_parameterization = "log_amplitude_decay_weighted"

        if "energy_axis_kind" in blob:
            raw_axis = self._scalar(blob["energy_axis_kind"])
            axis_kind = str(raw_axis.decode("utf-8") if isinstance(raw_axis, bytes) else raw_axis)
        else:
            axis_kind = "delta_e_ascending"
        self.reverse_energy_order = axis_kind in {
            "delta_e_descending",
            "loss_descending",
            "legacy_descending",
        }

        self.x_mean = blob["x_mean"].astype(np.float64)
        self.x_std = blob["x_std"].astype(np.float64)
        self.body_ops = self._build_body_ops(blob)
        self.amp_head_weight = blob["amp_head.weight"].astype(np.float64)
        self.amp_head_bias = blob["amp_head.bias"].astype(np.float64)
        self.rate_head_weight = blob["rate_head.weight"].astype(np.float64)
        self.rate_head_bias = blob["rate_head.bias"].astype(np.float64)
        self.max_log = 20.0

    @staticmethod
    def _scalar(arr: np.ndarray) -> Any:
        if arr.ndim == 0:
            return arr.item()
        if arr.size == 1:
            return arr.reshape(-1)[0].item()
        raise ValueError("expected scalar array field")

    @classmethod
    def _build_body_ops(cls, blob: np.lib.npyio.NpzFile) -> list[tuple[str, dict[str, np.ndarray] | None]]:
        body_modules: dict[int, dict[str, np.ndarray]] = {}
        for key in blob.files:
            if not key.startswith("body."):
                continue
            _, idx_s, name = key.split(".", 2)
            body_modules.setdefault(int(idx_s), {})[name] = blob[key].astype(np.float64)

        body_ops: list[tuple[str, dict[str, np.ndarray] | None]] = []
        for module_idx in range(max(body_modules) + 1):
            params = body_modules.get(module_idx)
            if params is None:
                body_ops.append(("silu", None))
                continue
            weight = params.get("weight")
            bias = params.get("bias")
            if weight is None or bias is None:
                raise KeyError(f"missing weight/bias at body.{module_idx}")
            if weight.ndim == 2:
                body_ops.append(("linear", {"weight": weight, "bias": bias}))
            elif weight.ndim == 1 and "running_mean" in params and "running_var" in params:
                body_ops.append(
                    (
                        "batchnorm",
                        {
                            "weight": weight,
                            "bias": bias,
                            "running_mean": params["running_mean"],
                            "running_var": params["running_var"],
                        },
                    )
                )
            else:
                raise KeyError(f"unsupported body module at index {module_idx}")
        body_ops.append(("silu", None))
        return body_ops

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x * (1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    @classmethod
    def _batchnorm_eval(cls, x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
        scale = params["weight"] / np.sqrt(np.maximum(params["running_var"], 0.0) + cls._BATCHNORM_EPS)
        shift = params["bias"] - params["running_mean"] * scale
        return x * scale + shift

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = (x - self.x_mean) / self.x_std
        for op_name, params in self.body_ops:
            if op_name == "linear":
                assert params is not None
                h = h @ params["weight"].T + params["bias"]
            elif op_name == "batchnorm":
                assert params is not None
                h = self._batchnorm_eval(h, params)
            elif op_name == "silu":
                h = self._silu(h)
        amp_raw = (h @ self.amp_head_weight.T + self.amp_head_bias).squeeze(-1)
        rate_raw = (h @ self.rate_head_weight.T + self.rate_head_bias).squeeze(-1)
        if self.output_parameterization.startswith("log_"):
            amp = np.exp(np.clip(amp_raw, -self.max_log, self.max_log))
            rate = np.exp(np.clip(rate_raw, -self.max_log, self.max_log))
        else:
            amp = self._softplus(amp_raw)
            rate = self._softplus(rate_raw)
        return amp, rate


def default_weights_path() -> Path:
    return Path(str(files("ebsdsim").joinpath("data/lightweight_hist_surrogate.npz")))


def get_surrogate_model() -> NumpySurrogateModel:
    global _model
    if _model is None:
        _model = NumpySurrogateModel(default_weights_path())
    return _model


def _combine_direct_exp_by_overlap(
    native: SurrogateDirectExp,
    *,
    src_bin_size_kev: float,
    dst_bin_size_kev: float,
    n_dst_bins: int,
    n_depth_bins: int,
    depth_bin_size_nm: float,
) -> SurrogateDirectExp:
    n_src = int(native.energy_mass.size)
    src_edges = np.arange(n_src + 1, dtype=np.float64) * float(src_bin_size_kev)
    dst_edges = np.arange(int(n_dst_bins) + 1, dtype=np.float64) * float(dst_bin_size_kev)
    dst_centers = dst_edges[:-1] + float(dst_bin_size_kev) / 2.0

    coarse_mass = np.zeros(n_dst_bins, dtype=np.float64)
    coarse_beta_num = np.zeros(n_dst_bins, dtype=np.float64)
    src_idx = 0
    for dst_idx in range(n_dst_bins):
        dst_lo = float(dst_edges[dst_idx])
        dst_hi = float(dst_edges[dst_idx + 1])
        while src_idx < n_src and src_edges[src_idx + 1] <= dst_lo:
            src_idx += 1
        work_idx = src_idx
        while work_idx < n_src and src_edges[work_idx] < dst_hi:
            overlap_lo = max(dst_lo, float(src_edges[work_idx]))
            overlap_hi = min(dst_hi, float(src_edges[work_idx + 1]))
            overlap = overlap_hi - overlap_lo
            if overlap > 0.0:
                frac = overlap / float(src_bin_size_kev)
                m = float(native.energy_mass[work_idx]) * frac
                coarse_mass[dst_idx] += m
                coarse_beta_num[dst_idx] += m * float(native.betas[work_idx])
            work_idx += 1

    coarse_beta = np.zeros(n_dst_bins, dtype=np.float64)
    valid = coarse_mass > 0.0
    coarse_beta[valid] = coarse_beta_num[valid] / coarse_mass[valid]

    z_max = float(n_depth_bins) * float(depth_bin_size_nm)
    coarse_amp = np.zeros(n_dst_bins, dtype=np.float64)
    amp_valid = valid & (coarse_beta > 0.0)
    norm = (1.0 - np.exp(-coarse_beta[amp_valid] * z_max)) / coarse_beta[amp_valid]
    coarse_amp[amp_valid] = np.divide(
        coarse_mass[amp_valid], norm, out=np.zeros_like(norm), where=norm > 0.0
    )

    total = float(coarse_mass.sum())
    coarse_weights = coarse_mass / total if total > 0.0 else np.zeros_like(coarse_mass)
    return SurrogateDirectExp(
        amplitudes=coarse_amp,
        betas=coarse_beta,
        energy_mass=coarse_mass,
        energy_weights=coarse_weights,
        energy_centers_keV=dst_centers,
    )


def infer_direct_exp_from_cell(
    *,
    model: NumpySurrogateModel,
    cell: Any,
    sigma_deg: float,
    beam_kv: float,
    energy_binwidth_keV: float,
    n_energy_bins: int,
    depth_bin_size_nm: float = DEFAULT_DEPTH_BIN_SIZE_NM,
    n_depth_bins: int = DEFAULT_DEPTH_BINS,
) -> SurrogateDirectExp:
    energy_centers_keV = (
        np.arange(n_energy_bins, dtype=np.float64) * float(energy_binwidth_keV)
        + float(energy_binwidth_keV) / 2.0
    )
    x = np.zeros((n_energy_bins, 6), dtype=np.float64)
    x[:, 0] = float(cell.density)
    x[:, 1] = float(cell.average_atomic_number)
    x[:, 2] = float(cell.average_atomic_weight)
    x[:, 3] = float(sigma_deg)
    x[:, 4] = float(beam_kv)
    x[:, 5] = energy_centers_keV

    amplitudes, betas = model.forward(x)
    if model.reverse_energy_order:
        amplitudes = amplitudes[::-1]
        betas = betas[::-1]

    z_max = float(n_depth_bins) * float(depth_bin_size_nm)
    norm_factor = np.zeros_like(betas)
    valid = np.isfinite(betas) & (betas > 0.0)
    norm_factor[valid] = (1.0 - np.exp(-betas[valid] * z_max)) / betas[valid]
    energy_mass = amplitudes * norm_factor
    total_mass = float(np.sum(energy_mass))
    energy_weights = energy_mass / total_mass if total_mass > 0.0 else np.zeros_like(energy_mass)
    return SurrogateDirectExp(
        amplitudes=amplitudes.astype(np.float64),
        betas=betas.astype(np.float64),
        energy_mass=energy_mass.astype(np.float64),
        energy_weights=energy_weights.astype(np.float64),
        energy_centers_keV=energy_centers_keV,
    )


def infer_direct_exp_from_cell_rebinned(
    *,
    model: NumpySurrogateModel | None = None,
    cell: Any,
    sigma_deg: float,
    beam_kv: float,
    energy_binwidth_keV: float,
    n_energy_bins: int,
    depth_bin_size_nm: float = DEFAULT_DEPTH_BIN_SIZE_NM,
    n_depth_bins: int = DEFAULT_DEPTH_BINS,
) -> SurrogateDirectExp:
    model = model or get_surrogate_model()
    native_bin = float(model.energy_bin_size_kev)
    target_bin = float(energy_binwidth_keV)
    if abs(target_bin - native_bin) < 1e-9 * native_bin:
        return infer_direct_exp_from_cell(
            model=model,
            cell=cell,
            sigma_deg=sigma_deg,
            beam_kv=beam_kv,
            energy_binwidth_keV=target_bin,
            n_energy_bins=n_energy_bins,
            depth_bin_size_nm=depth_bin_size_nm,
            n_depth_bins=n_depth_bins,
        )
    n_native_bins = max(1, int(np.floor(float(beam_kv) / native_bin)))
    native = infer_direct_exp_from_cell(
        model=model,
        cell=cell,
        sigma_deg=sigma_deg,
        beam_kv=beam_kv,
        energy_binwidth_keV=native_bin,
        n_energy_bins=n_native_bins,
        depth_bin_size_nm=depth_bin_size_nm,
        n_depth_bins=n_depth_bins,
    )
    return _combine_direct_exp_by_overlap(
        native,
        src_bin_size_kev=native_bin,
        dst_bin_size_kev=target_bin,
        n_dst_bins=n_energy_bins,
        n_depth_bins=n_depth_bins,
        depth_bin_size_nm=depth_bin_size_nm,
    )
