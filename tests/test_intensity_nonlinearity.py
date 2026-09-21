"""Focused tests for the frozen intensity-nonlinearity diagnostics."""

import numpy as np

from scripts.analyze_frozen_candidates import PairData, model_metrics
from scripts.analyze_intensity_nonlinearity import (
    _sat_design,
    nested_parametric_oof,
    source_magnitude,
    spline_oof,
)


def _pair(x: np.ndarray, y: np.ndarray) -> PairData:
    rows, columns = np.indices(x.shape)
    folds = ((rows >= x.shape[0] // 2).astype(int) * 2
             + (columns >= x.shape[1] // 2).astype(int)).astype(np.int8)
    evaluated = np.ones_like(x, dtype=bool)
    support = source_magnitude(x) > np.quantile(source_magnitude(x), 0.25)
    prediction = 0.001 * x
    return PairData(
        label="27->28",
        x=x,
        y=y,
        folds=folds,
        evaluated=evaluated,
        source_support=support,
        prediction=prediction,
        residual=y - prediction,
    )


def test_source_magnitude_clamps_positive_x_and_sat_design_is_finite():
    x = np.array([-100.0, 0.0, 50.0])
    assert np.array_equal(source_magnitude(x), np.array([100.0, 0.0, 0.0]))
    design = _sat_design(x, 25.0)
    assert np.all(np.isfinite(design))
    assert design[1, 1] == 0.0
    assert design[2, 1] == 0.0


def test_nested_saturation_recovers_saturating_conditional_mean():
    rows, columns = np.indices((48, 48))
    local_rows = rows % 24
    local_columns = columns % 24
    s = 100.0 + 4900.0 * (0.55 * local_rows + 0.45 * local_columns) / 23.0
    x = -s
    support = s > np.quantile(s, 0.25)
    true_s0 = float(np.quantile(s[support], 0.5))
    y = _sat_design(x.ravel(), true_s0) @ np.array([0.001, 0.0015, 0.2])
    pair = _pair(x, y.reshape(x.shape))
    result = nested_parametric_oof(
        [pair], "Msat", candidate_quantiles=(0.25, 0.5, 0.75),
    )
    prediction = result["predictions"][pair.label]
    assert model_metrics(pair, prediction)["cv_r2"] > 0.999
    selected = [
        row["selected_training_quantile"]
        for row in result["outer_selections"][pair.label]
    ]
    assert selected == [0.5, 0.5, 0.5, 0.5]


def test_spline_diagnostic_is_strict_oof_and_finite():
    rows, columns = np.indices((48, 48))
    local_rows = rows % 24
    local_columns = columns % 24
    x = -4.0 + 8.0 * (local_rows + local_columns) / (2.0 * 23.0)
    y = np.sin(x) + 0.2 * x
    pair = _pair(x, y)
    result = spline_oof([pair])
    prediction = result["predictions"][pair.label]
    assert np.all(np.isfinite(prediction[pair.evaluated]))
    assert len(result["fold_parameters"][pair.label]) == 4
    assert model_metrics(pair, prediction)["cv_r2"] > 0.98
