"""Focused tests for strict spatial OOF pseudo-ghost fitting."""

import numpy as np

from scripts.analyze_pseudo_ghost_mechanism import (
    background_spatial_oof,
    detection_gate,
    estimate_hybrid_dark_background,
    estimate_masked_hybrid_dark_background,
    estimate_single_dark_background,
    generate_pseudo_occlusions,
    parse_pairs,
    single_lag_stratum,
    strict_affine_oof,
    validate_background_reconstruction,
)
from scripts.analyze_dark_light_pairs import spatial_folds


def test_parse_all_single_lag_pairs():
    pairs = parse_pairs("all")
    assert len(pairs) == 30
    assert pairs[0] == (1, 2)
    assert pairs[-1] == (30, 31)


def test_single_lag_cohort_is_prestratified():
    pairs = parse_pairs("single-lag")
    strata = [single_lag_stratum(pair) for pair in pairs]
    assert len(pairs) == 13
    assert strata.count("robust_confirmed") == 2
    assert strata.count("parameter_sensitive") == 9
    assert strata.count("not_detected_control") == 2
    assert single_lag_stratum((11, 12)) is None


def test_detection_gate_rejects_null_significant_but_tiny_effect():
    fit = {"alpha_fold_mean": 0.0001, "alpha_fold_cv": 0.05, "cv_r2": 0.02}
    null_rows = [{"kind": "future_light", "status": "ok", "cv_r2": 0.001}]
    null_summary = {"cv_r2_p95": 0.002}
    result = detection_gate(fit, null_rows, null_summary)
    assert not result["passed"]
    assert not result["checks"]["minimum_cv_r2"]
    assert not result["checks"]["minimum_null_margin"]


def test_detection_gate_accepts_stable_practical_effect():
    fit = {"alpha_fold_mean": 0.001, "alpha_fold_cv": 0.05, "cv_r2": 0.5}
    null_rows = [{"kind": "future_light", "status": "ok", "cv_r2": 0.1}]
    null_summary = {"cv_r2_p95": 0.15}
    assert detection_gate(fit, null_rows, null_summary)["passed"]


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


def test_single_dark_surfaces_extrapolate_into_excluded_source_region():
    height, width = 48, 52
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    true_background = 1000.0 + 35.0 * xx - 20.0 * yy + 12.0 * xx * yy + 8.0 * xx ** 2
    source_support = np.zeros((height, width), dtype=bool)
    source_support[14:34, 15:39] = True
    contaminated_dark = true_background.copy()
    contaminated_dark[source_support] += 50.0

    polynomial, _ = estimate_single_dark_background(
        contaminated_dark, ~source_support, "poly2",
    )
    spline, _ = estimate_single_dark_background(
        contaminated_dark, ~source_support, "robust_spline",
    )

    assert np.mean(np.abs(polynomial[source_support] - true_background[source_support])) < 1e-6
    assert np.mean(np.abs(spline[source_support] - true_background[source_support])) < 3.0


def test_hybrid_background_preserves_fixed_pattern_and_fits_current_drift():
    height, width = 48, 52
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    fixed_pattern = 7.0 * np.where((np.indices((height, width)).sum(axis=0) % 2) == 0, 1, -1)
    smooth_drift = 1000.0 + 28.0 * xx - 16.0 * yy + 7.0 * xx * yy
    true_background = fixed_pattern + smooth_drift
    source_support = np.zeros((height, width), dtype=bool)
    source_support[14:34, 15:39] = True
    contaminated = true_background.copy()
    contaminated[source_support] += 80.0
    darks = {
        index: fixed_pattern + 900.0 + index
        for index in range(1, 32)
    }
    darks[6] = contaminated

    estimate, diagnostics = estimate_hybrid_dark_background(
        contaminated, darks, 6, ~source_support, smoothness=20.0,
    )

    assert diagnostics["target_dark_in_fixed_pattern"] is False
    assert np.mean(np.abs(estimate[source_support] - true_background[source_support])) < 3.0


def test_masked_hybrid_fixed_pattern_rejects_repeated_temporal_contamination():
    height, width = 48, 52
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    fixed_pattern = 7.0 * np.where(
        (np.indices((height, width)).sum(axis=0) % 2) == 0, 1, -1,
    )
    source_support = np.zeros((height, width), dtype=bool)
    source_support[14:34, 15:39] = True
    empty_support = np.zeros_like(source_support)
    darks = {}
    contamination_supports = {}
    for index in range(1, 32):
        smooth_drift = 1000.0 + 3.0 * index + 28.0 * xx - 16.0 * yy + 7.0 * xx * yy
        darks[index] = fixed_pattern + smooth_drift
        if 2 <= index <= 20:
            darks[index] = darks[index] + 80.0 * source_support
        if index <= 30:
            contamination_supports[index] = (
                source_support.copy() if index <= 19 or index == 30 else empty_support.copy()
            )

    target_index = 31
    darks[target_index] = darks[target_index] + 50.0 * source_support
    true_background = (
        fixed_pattern + 1000.0 + 3.0 * target_index
        + 28.0 * xx - 16.0 * yy + 7.0 * xx * yy
    )
    masked, diagnostics = estimate_masked_hybrid_dark_background(
        dark=darks[target_index],
        darks=darks,
        contamination_supports=contamination_supports,
        target_index=target_index,
        fit_mask=~source_support,
        smoothness=20.0,
    )
    median_hybrid, _ = estimate_hybrid_dark_background(
        darks[target_index], darks, target_index, ~source_support, smoothness=20.0,
    )

    masked_error = np.mean(np.abs(masked[source_support] - true_background[source_support]))
    median_error = np.mean(
        np.abs(median_hybrid[source_support] - true_background[source_support])
    )
    assert diagnostics["fixed_pattern"]["uses_temporal_pixelwise_median"] is False
    assert diagnostics["fixed_pattern"]["target_dark_in_fixed_pattern"] is False
    assert diagnostics["fixed_pattern"]["coverage_min"] > 0
    assert masked_error < 1.0
    assert masked_error < 0.05 * median_error


def test_background_spatial_oof_never_needs_excluded_source_values():
    height, width = 48, 52
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    background = 800.0 + 18.0 * xx + 9.0 * yy + 4.0 * xx * yy
    source_support = np.zeros_like(background, dtype=bool)
    source_support[12:36, 16:38] = True
    dark = background.copy()
    dark[source_support] += 10000.0

    metrics = background_spatial_oof(
        dark, ~source_support, spatial_folds(dark.shape), "poly2",
    )

    assert metrics["evaluated_blocks"] == int((~source_support).sum())
    assert metrics["mae"] < 1e-6


def test_pseudo_occlusions_stay_inside_trusted_air():
    trusted_air = np.ones((64, 64), dtype=bool)
    trusted_air[20:35, 24:40] = False
    object_shape = np.zeros((14, 18), dtype=bool)
    object_shape[2:12, 3:7] = True
    object_shape[8:12, 3:16] = True

    samples = generate_pseudo_occlusions(
        trusted_air,
        [object_shape],
        seed=23,
        square_sides=(8, 12),
        squares_per_size=1,
        object_count=1,
    )

    assert {row["kind"] for row in samples} == {"square", "object_shape"}
    assert all(np.all(trusted_air[row["mask"]]) for row in samples)


def test_pseudo_occlusion_validation_recovers_known_quadratic_background():
    height, width = 64, 68
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    background = 900.0 + 22.0 * xx - 17.0 * yy + 6.0 * xx * yy + 5.0 * xx ** 2
    trusted_air = np.ones_like(background, dtype=bool)
    object_shape = np.zeros((16, 20), dtype=bool)
    object_shape[2:14, 3:8] = True
    object_shape[10:14, 3:18] = True

    result = validate_background_reconstruction(
        dark=background,
        trusted_air=trusted_air,
        darks={2: background},
        target_index=2,
        background_modes=("poly2", "robust_spline"),
        object_templates=[object_shape],
        smoothness=20.0,
        seed=31,
        square_sides=(8, 12),
        squares_per_size=1,
        object_count=1,
    )

    assert result["generated_sample_count"] == 3
    polynomial = result["summary"]["overall_by_background"]["poly2"]
    spline = result["summary"]["overall_by_background"]["robust_spline"]
    assert polynomial["median_sample_mae"] < 1e-6
    assert spline["median_sample_mae"] < 3.0
