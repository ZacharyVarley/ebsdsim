"""TEMPORARY Metal debug probe: where does sg_003_1536903 zero out on CI?

The dynamical solve alone is CORRECT on Metal (A/B probe: default and
dense-tile both reproduce Windows intensities at the 32K bucket). This probe
runs the exact failing e2e call and dumps every downstream stage: bin
weights, per-bin Lambert sums, raw integrated sums, final pattern sum.
Delete this file once the Metal failure is root-caused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ebsdsim.gpu.device import require_gpu

import ebsdsim as es

_CIF = Path(__file__).resolve().parents[1] / "data" / "cif" / "sg_003_1536903.cif"

pytestmark = pytest.mark.gpu


def test_metal_e2e_stage_dump() -> None:
    try:
        require_gpu(required_features=("shader-f16",))
    except RuntimeError:
        pytest.skip("WebGPU adapter with shader-f16 unavailable")
    mp = es.master_pattern_from_cif(
        _CIF,
        voltage_kv=20.0,
        halfw=10,
        dmin=0.05,
        energy_binwidth_keV=5.0,
        marginal_coverage=1.0,
        mc_backend="surrogate",
        solver="smith_iterative",
        verbosity=0,
    )
    meta = mp.metadata
    print(
        f"\n[debug] bin_voltages_kv={mp.bin_voltages_kv}\n"
        f"[debug] bin_weights={mp.bin_weights}\n"
        f"[debug] bin_pattern sums={[float(np.sum(b)) for b in mp.bin_patterns]}\n"
        f"[debug] bin_pattern nonzeros={[int(np.count_nonzero(b)) for b in mp.bin_patterns]}\n"
        f"[debug] integrated: sum={float(np.sum(mp.integrated)):.6e} "
        f"nonzero={int(np.count_nonzero(mp.integrated))}/{mp.integrated.size}\n"
        f"[debug] pattern: sum={float(np.sum(mp.pattern)):.6e} "
        f"nonzero={int(np.count_nonzero(mp.pattern))}/{mp.pattern.size}\n"
        f"[debug] meta: mode={meta.get('smith_iterative_mode')} "
        f"n_bins_run={meta.get('n_bins_run')} "
        f"stopped={meta.get('stopped_by_relative_change')} "
        f"last_rel={meta.get('last_relative_change')} "
        f"fail_k={meta.get('fail_k')} k_solved={meta.get('k_solved')} "
        f"k_per_s={meta.get('k_per_s')} mc_bins={meta.get('n_mc_bins')}",
        flush=True,
    )
