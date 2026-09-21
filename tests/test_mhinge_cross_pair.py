"""Focused tests for the pre-frozen cross-pair Mhinge audit."""

import numpy as np

from scripts.analyze_frozen_candidates import PairData
from scripts.audit_mhinge_cross_pair import hinge_identifiability


def _pair(s: np.ndarray) -> PairData:
    x = -s
    rows, columns = np.indices(s.shape)
    folds = ((rows >= s.shape[0] // 2).astype(int) * 2
             + (columns >= s.shape[1] // 2).astype(int)).astype(np.int8)
    y = 0.001 * x
    prediction = y.copy()
    valid = np.ones_like(s, dtype=bool)
    return PairData(
        label="1->2",
        x=x,
        y=y,
        folds=folds,
        evaluated=valid,
        source_support=valid,
        prediction=prediction,
        residual=y - prediction,
    )


def _parameters(tau: float):
    return [
        {
            "outer_fold": fold,
            "tau": tau,
            "selected_training_quantile": 0.5,
            "boundary_hit": False,
            "base_alpha": 0.001,
            "slope_change": -0.0005,
            "strong_signal_slope": 0.0005,
        }
        for fold in range(4)
    ]


def test_hinge_identifiability_accepts_well_supported_stable_segments():
    rows, columns = np.indices((48, 48))
    local_rows = rows % 24
    local_columns = columns % 24
    s = 100.0 + 900.0 * (local_rows + local_columns) / 46.0
    result = hinge_identifiability(_pair(s), _parameters(550.0))
    assert result["passed"]
    assert result["summary"]["direction"] == "decrease"


def test_hinge_identifiability_rejects_narrow_high_signal_range():
    rows, columns = np.indices((48, 48))
    local_rows = rows % 24
    local_columns = columns % 24
    s = 100.0 + 900.0 * (local_rows + local_columns) / 46.0
    result = hinge_identifiability(_pair(s), _parameters(945.0))
    assert not result["passed"]
    assert not result["checks"]["training_dynamic_range_sufficient"]
