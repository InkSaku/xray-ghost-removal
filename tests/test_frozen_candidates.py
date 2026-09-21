"""Focused tests for frozen M1 candidate diagnostics."""

import numpy as np
from scipy.ndimage import gaussian_filter

from scripts.analyze_frozen_candidates import (
    PairData,
    model_metrics,
    nested_blur_oof,
    quadratic_oof,
    regression_metrics,
    residual_trend_summary,
)


def test_m1_metrics_use_stored_oof_residual():
    x = np.arange(64, dtype=float).reshape(8, 8)
    y = 0.5 * x + 2.0
    prediction = y + 0.25
    residual = y - prediction
    pair = PairData(
        label="5->6",
        x=x,
        y=y,
        folds=np.indices(x.shape).sum(axis=0).astype(np.int8) % 4,
        evaluated=np.ones_like(x, dtype=bool),
        source_support=np.zeros_like(x, dtype=bool),
        prediction=prediction,
        residual=residual,
    )
    metrics = regression_metrics(pair)
    assert metrics["mae"] == 0.25
    assert metrics["rmse"] == 0.25
    assert metrics["residual_std"] == 0.0


def _pair(label: str, x: np.ndarray, y: np.ndarray, prediction: np.ndarray) -> PairData:
    rows, columns = np.indices(x.shape)
    folds = ((rows >= x.shape[0] // 2).astype(int) * 2
             + (columns >= x.shape[1] // 2).astype(int)).astype(np.int8)
    evaluated = np.ones_like(x, dtype=bool)
    support = x > np.percentile(x, 65)
    return PairData(
        label=label,
        x=x,
        y=y,
        folds=folds,
        evaluated=evaluated,
        source_support=support,
        prediction=prediction,
        residual=y - prediction,
    )


def test_nested_blur_selects_true_shared_sigma():
    rows, columns = np.indices((48, 48))
    x1 = np.zeros((48, 48), dtype=float)
    x1[8:22, 7:20] = 10.0
    x1[28:43, 25:41] = 18.0
    x2 = np.zeros_like(x1)
    x2[5:19, 27:44] = 14.0
    x2[26:45, 5:22] = 22.0
    x1 += 0.02 * rows + 0.01 * columns
    x2 += 0.01 * rows + 0.02 * columns
    y1 = 2.5 * gaussian_filter(x1, sigma=1.0, mode="reflect") + 3.0
    y2 = 1.7 * gaussian_filter(x2, sigma=1.0, mode="reflect") - 2.0
    pairs = [
        _pair("5->6", x1, y1, 2.5 * x1 + 3.0),
        _pair("27->28", x2, y2, 1.7 * x2 - 2.0),
    ]
    result = nested_blur_oof(pairs, sigma_blocks=(0.0, 0.5, 1.0, 1.5))
    assert [row["selected_sigma_blocks"] for row in result["outer_selections"]] == [
        1.0, 1.0, 1.0, 1.0,
    ]
    for pair in pairs:
        assert model_metrics(pair, result["predictions"][pair.label])["cv_r2"] > 0.999


def test_quadratic_diagnostic_flattens_quadratic_residual_trend():
    rows, columns = np.indices((48, 48))
    x = -3.0 + 6.0 * (rows + columns) / (2.0 * 47.0)
    y = 1.5 + 0.8 * x + 1.2 * x ** 2
    linear = 1.5 + 0.8 * x
    pair = _pair("27->28", x, y, linear)
    prediction = quadratic_oof([pair])["predictions"][pair.label]
    before = residual_trend_summary(pair, linear)
    after = residual_trend_summary(pair, prediction)
    assert model_metrics(pair, prediction)["cv_r2"] > 0.999
    assert after["binned_mean_weighted_rms"] < 0.01 * before["binned_mean_weighted_rms"]
