"""Focused tests for strict spatial OOF pseudo-ghost fitting."""

import numpy as np

from scripts.analyze_pseudo_ghost_mechanism import strict_affine_oof
from scripts.analyze_dark_light_pairs import spatial_folds


def test_strict_affine_oof_recovers_known_model():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 5000, (64, 64))
    y = 0.0008 * x + 2.5 + rng.normal(0, 0.4, x.shape)
    mask = np.ones_like(x, dtype=bool)
    result = strict_affine_oof(x, y, mask, spatial_folds(x.shape))

    assert abs(result["alpha_fold_mean"] - 0.0008) < 0.00003
    assert result["cv_r2"] > 0.98
    assert result["ncc"] > 0.99
    assert np.isfinite(result["prediction"]).all()
    assert np.isfinite(result["residual"]).all()


def test_held_out_targets_do_not_change_their_fold_parameters():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(64, 64))
    y = 0.4 * x + rng.normal(0, 0.1, x.shape)
    folds = spatial_folds(x.shape)
    mask = np.ones_like(x, dtype=bool)
    original = strict_affine_oof(x, y, mask, folds)

    changed_y = y.copy()
    changed_y[folds == 0] += 10000.0
    changed = strict_affine_oof(x, changed_y, mask, folds)

    fold0_original = next(row for row in original["fold_parameters"] if row["fold"] == 0)
    fold0_changed = next(row for row in changed["fold_parameters"] if row["fold"] == 0)
    assert fold0_changed["alpha"] == fold0_original["alpha"]
    assert fold0_changed["intercept"] == fold0_original["intercept"]
