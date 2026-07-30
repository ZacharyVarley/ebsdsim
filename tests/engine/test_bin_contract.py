"""Unit tests for MasterPattern bin-pattern / voltage metadata contract."""

from __future__ import annotations

import numpy as np
import pytest
from ebsdsim.engine.results import MasterPattern, validate_bin_contract


def test_bin_contract_rejects_mismatch():
    with pytest.raises(ValueError, match="bin_patterns length"):
        validate_bin_contract([np.zeros(4)], [20.0, 16.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="bin_voltages_kv"):
        validate_bin_contract([], [20.0], [0.5, 0.5])
    # Legal: integrated-only with energy-model metadata.
    validate_bin_contract([], [20.0, 16.0], [0.5, 0.5])
    validate_bin_contract([np.zeros(4), np.ones(4)], [20.0, 16.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="bin_patterns length"):
        MasterPattern(
            pattern=np.zeros((3, 3), np.float32),
            integrated=np.zeros(4, np.float32),
            n_k=2,
            n_sites=2,
            bin_patterns=[np.zeros(4, np.float32)],
            bin_voltages_kv=[20.0, 16.0],
            bin_weights=[0.5, 0.5],
        )
