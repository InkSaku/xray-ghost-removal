"""M1 fitting; extracted without changing numerical definitions."""
from __future__ import annotations
from typing import Any
from scipy.ndimage import gaussian_filter
import numpy as np
from xray_ghost_removal.common.spatial import spatial_nulls
from xray_ghost_removal.m1.config import (
    DETECTION_MAX_ALPHA,
    DETECTION_MAX_ALPHA_CV,
    DETECTION_MIN_CV_R2,
    DETECTION_MIN_NULL_MARGIN,
    LOWPASS_SIGMA_BLOCKS,
    TRIM_PERCENTILES,
)

def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x.astype(float), np.ones(x.size, dtype=float)])
    (alpha, intercept), *_ = np.linalg.lstsq(design, y.astype(float), rcond=None)
    return float(alpha), float(intercept)


def _training_keep(
    x: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    trim_percentiles: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float]]:
    """Select training observations; thresholds never inspect held-out values."""
    candidates = base & np.isfinite(x) & np.isfinite(y)
    if candidates.sum() < 20:
        raise ValueError("Too few training blocks for an affine fit")
    xlo, xhi = np.percentile(x[candidates], trim_percentiles)
    ylo, yhi = np.percentile(y[candidates], trim_percentiles)
    keep = candidates & (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    if keep.sum() < 20:
        raise ValueError("Too few training blocks after trimming")
    return keep, {"x_low": float(xlo), "x_high": float(xhi),
                  "y_low": float(ylo), "y_high": float(yhi)}


def _correlation(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    keep = valid & np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return float("nan")
    av, bv = a[keep].astype(float), b[keep].astype(float)
    if np.std(av) <= 1e-12 or np.std(bv) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _weighted_lowpass(values: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    weights = valid.astype(np.float32)
    numerator = gaussian_filter(np.where(valid, values, 0.0), sigma=sigma, mode="reflect")
    denominator = gaussian_filter(weights, sigma=sigma, mode="reflect")
    return numerator / np.maximum(denominator, 1e-6)


def _neighbor_correlation(values: np.ndarray, valid: np.ndarray) -> float:
    left = valid[:, :-1] & valid[:, 1:]
    up = valid[:-1, :] & valid[1:, :]
    a = np.concatenate([values[:, :-1][left], values[:-1, :][up]])
    b = np.concatenate([values[:, 1:][left], values[1:, :][up]])
    if a.size < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def strict_affine_oof(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    folds: np.ndarray,
    trim_percentiles: tuple[float, float] = TRIM_PERCENTILES,
) -> dict[str, Any]:
    """Fit fold-specific affine models without consulting held-out target values.

    Target-based trimming is learned and applied to the training subset only.
    Every finite, pre-eligible held-out block is evaluated; held-out ``y`` never
    decides whether that block is included.
    """
    if not (x.shape == y.shape == mask.shape == folds.shape):
        raise ValueError("x, y, mask, and folds must have the same shape")
    eligible = mask & np.isfinite(x) & np.isfinite(y)
    prediction = np.full(y.shape, np.nan, dtype=np.float64)
    intercept_prediction = np.full(y.shape, np.nan, dtype=np.float64)
    fold_parameters = []

    for fold in sorted(int(v) for v in np.unique(folds[eligible])):
        train_base = eligible & (folds != fold)
        train, thresholds = _training_keep(x, y, train_base, trim_percentiles)
        alpha, intercept = _fit_affine(x[train], y[train])
        test = eligible & (folds == fold)
        prediction[test] = alpha * x[test] + intercept
        intercept_prediction[test] = float(np.mean(y[train]))
        fold_parameters.append({
            "fold": fold,
            "alpha": alpha,
            "intercept": intercept,
            "train_blocks": int(train.sum()),
            "test_blocks": int(test.sum()),
            "training_trim_thresholds": thresholds,
        })

    evaluated = eligible & np.isfinite(prediction)
    if evaluated.sum() < 20:
        raise ValueError("Too few evaluated OOF blocks")
    yv, pv = y[evaluated].astype(float), prediction[evaluated]
    residual = np.full(y.shape, np.nan, dtype=np.float64)
    residual[evaluated] = yv - pv
    ss_total = float(np.sum((yv - yv.mean()) ** 2)) + 1e-12
    ss_residual = float(np.sum((yv - pv) ** 2))
    cv_r2 = 1.0 - ss_residual / ss_total
    m0_residual = yv - intercept_prediction[evaluated]
    m0_cv_r2 = 1.0 - float(np.sum(m0_residual ** 2)) / ss_total

    train_full, full_thresholds = _training_keep(x, y, eligible, trim_percentiles)
    alpha_full, intercept_full = _fit_affine(x[train_full], y[train_full])
    alphas = np.array([row["alpha"] for row in fold_parameters], dtype=float)
    alpha_cv = float(np.std(alphas, ddof=1) / max(abs(np.mean(alphas)), 1e-12)) \
        if alphas.size > 1 else 0.0

    y_low = _weighted_lowpass(y, evaluated, LOWPASS_SIGMA_BLOCKS)
    p_low = _weighted_lowpass(prediction, evaluated, LOWPASS_SIGMA_BLOCKS)
    r_low = _weighted_lowpass(residual, evaluated, LOWPASS_SIGMA_BLOCKS)
    x_low = _weighted_lowpass(x, evaluated, LOWPASS_SIGMA_BLOCKS)
    residual_values = residual[evaluated]
    variance_y = float(np.var(yv))

    return {
        "prediction": prediction,
        "residual": residual,
        "evaluated_mask": evaluated,
        "fold_parameters": fold_parameters,
        "alpha_fold_mean": float(np.mean(alphas)),
        "alpha_fold_median": float(np.median(alphas)),
        "alpha_fold_cv": alpha_cv,
        "alpha_full_descriptive": alpha_full,
        "intercept_full_descriptive": intercept_full,
        "full_fit_trim_thresholds": full_thresholds,
        "evaluated_blocks": int(evaluated.sum()),
        "cv_r2": cv_r2,
        "m0_cv_r2": m0_cv_r2,
        "delta_cv_r2_vs_m0": cv_r2 - m0_cv_r2,
        "ncc": _correlation(y, prediction, evaluated),
        "residual_std": float(np.std(residual_values)),
        "pseudo_ghost_std": float(np.std(yv)),
        "residual_variance_ratio": float(np.var(residual_values) / max(variance_y, 1e-12)),
        "low_frequency_y_source_ncc": _correlation(y_low, x_low, evaluated),
        "low_frequency_y_prediction_ncc": _correlation(y_low, p_low, evaluated),
        "low_frequency_residual_source_ncc": _correlation(r_low, x_low, evaluated),
        "residual_neighbor_correlation": _neighbor_correlation(residual, evaluated),
    }


def _metric_view(fit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fit.items()
            if key not in {"prediction", "residual", "evaluated_mask"}}


def _compact_metric_view(fit: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "alpha_fold_mean",
        "alpha_fold_median",
        "alpha_fold_cv",
        "alpha_full_descriptive",
        "evaluated_blocks",
        "cv_r2",
        "m0_cv_r2",
        "delta_cv_r2_vs_m0",
        "ncc",
        "residual_std",
        "pseudo_ghost_std",
        "residual_variance_ratio",
        "low_frequency_y_source_ncc",
        "low_frequency_y_prediction_ncc",
        "low_frequency_residual_source_ncc",
        "residual_neighbor_correlation",
    )
    return {key: fit[key] for key in keys}


def _null_summary(rows: list[dict[str, Any]], true_fit: dict[str, Any]) -> dict[str, Any]:
    r2 = np.array([row["cv_r2"] for row in rows if np.isfinite(row["cv_r2"])])
    ncc = np.array([row["ncc"] for row in rows if np.isfinite(row["ncc"])])
    if r2.size == 0 or ncc.size == 0:
        raise ValueError("Null analysis produced no finite scores")
    return {
        "total_count": len(rows),
        "valid_count": int(r2.size),
        "skipped_count": int(len(rows) - r2.size),
        "cv_r2_median": float(np.median(r2)),
        "cv_r2_p95": float(np.percentile(r2, 95)),
        "true_minus_null_median_cv_r2": float(true_fit["cv_r2"] - np.median(r2)),
        "true_exceeds_cv_r2_p95": bool(true_fit["cv_r2"] > np.percentile(r2, 95)),
        "cv_r2_empirical_upper_p": float((1 + np.sum(r2 >= true_fit["cv_r2"])) / (1 + r2.size)),
        "ncc_median": float(np.median(ncc)),
        "ncc_p95": float(np.percentile(ncc, 95)),
        "true_minus_null_median_ncc": float(true_fit["ncc"] - np.median(ncc)),
        "true_exceeds_ncc_p95": bool(true_fit["ncc"] > np.percentile(ncc, 95)),
        "ncc_empirical_upper_p": float((1 + np.sum(ncc >= true_fit["ncc"])) / (1 + ncc.size)),
    }


def detection_gate(
    fit: dict[str, Any],
    null_rows: list[dict[str, Any]],
    null_summary: dict[str, Any],
) -> dict[str, Any]:
    """Apply the pre-frozen single-lag evidence and practical-effect gate."""
    future_scores = [
        row["cv_r2"] for row in null_rows
        if row["kind"] == "future_light" and row["status"] == "ok"
        and np.isfinite(row["cv_r2"])
    ]
    checks = {
        "physical_positive_alpha": bool(0 < fit["alpha_fold_mean"] <= DETECTION_MAX_ALPHA),
        "minimum_cv_r2": bool(fit["cv_r2"] >= DETECTION_MIN_CV_R2),
        "minimum_null_margin": bool(
            fit["cv_r2"] - null_summary["cv_r2_p95"] >= DETECTION_MIN_NULL_MARGIN
        ),
        "maximum_alpha_fold_cv": bool(fit["alpha_fold_cv"] <= DETECTION_MAX_ALPHA_CV),
        "true_ranks_first_against_future": bool(
            not future_scores or fit["cv_r2"] > max(future_scores)
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "cv_r2_minus_null_p95": float(fit["cv_r2"] - null_summary["cv_r2_p95"]),
        "thresholds": {
            "min_cv_r2": DETECTION_MIN_CV_R2,
            "min_null_margin": DETECTION_MIN_NULL_MARGIN,
            "max_alpha_fold_cv": DETECTION_MAX_ALPHA_CV,
            "max_alpha": DETECTION_MAX_ALPHA,
        },
    }


def evaluate_with_nulls(
    x: np.ndarray,
    y: np.ndarray,
    source_saturation: np.ndarray,
    mask_mode: str,
    folds: np.ndarray,
    future_sources: list[tuple[int, np.ndarray, np.ndarray]],
    seed: int,
    base_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    mask = (
        np.ones_like(y, dtype=bool)
        if base_mask is None else np.asarray(base_mask, dtype=bool).copy()
    )
    if mask.shape != y.shape:
        raise ValueError("base_mask and y must have the same shape")
    if mask_mode == "exclude_saturated":
        mask &= source_saturation < 0.05
    true_fit = strict_affine_oof(x, y, mask, folds)
    null_rows = []

    for light_index, future_x, future_saturation in future_sources:
        future_mask = mask.copy()
        if mask_mode == "exclude_saturated":
            future_mask &= future_saturation < 0.05
        try:
            fit = strict_affine_oof(future_x, y, future_mask, folds)
            null_rows.append({"kind": "future_light", "label": str(light_index),
                              "status": "ok", "cv_r2": fit["cv_r2"], "ncc": fit["ncc"]})
        except ValueError as error:
            null_rows.append({"kind": "future_light", "label": str(light_index),
                              "status": "skipped", "reason": str(error),
                              "cv_r2": float("nan"), "ncc": float("nan")})

    for null_x, null_saturation, label in spatial_nulls(x, source_saturation, seed):
        null_mask = mask.copy()
        if mask_mode == "exclude_saturated":
            null_mask &= null_saturation < 0.05
        kind = "spatial_shift" if label.startswith("roll") else "block_shuffle"
        try:
            fit = strict_affine_oof(null_x, y, null_mask, folds)
            null_rows.append({"kind": kind, "label": label, "status": "ok",
                              "cv_r2": fit["cv_r2"], "ncc": fit["ncc"]})
        except ValueError as error:
            null_rows.append({"kind": kind, "label": label, "status": "skipped",
                              "reason": str(error), "cv_r2": float("nan"),
                              "ncc": float("nan")})

    return true_fit, null_rows, _null_summary(null_rows, true_fit)
