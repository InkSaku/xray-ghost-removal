"""M1 background; extracted without changing numerical definitions."""
from __future__ import annotations
from typing import Any
from scipy.interpolate import BSpline
from scipy.sparse.linalg import lsqr
import numpy as np
from scipy import sparse
from xray_ghost_removal.m1.config import (
    FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    FIXED_PATTERN_HUBER_ITERATIONS,
    HUBER_DELTA,
    ROBUST_ITERATIONS,
)

def baseline(dark: dict[int, np.ndarray], target: int, mode: str) -> np.ndarray:
    if mode == "intercept_only":
        return np.zeros_like(dark[target])
    indices = [n for n in range(1, 32) if n != target]
    centered = {n: dark[n] - np.median(dark[n]) for n in indices}
    if mode == "low_activity":
        indices = sorted(indices, key=lambda n: float(np.std(centered[n])))[:10]
    return np.median(np.stack([centered[n] for n in indices]), axis=0)

def _huber_weights(residual: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    """Return IRLS weights with a MAD scale estimate."""
    centered = residual - np.median(residual)
    scale = 1.4826 * float(np.median(np.abs(centered)))
    scale = max(scale, np.finfo(float).eps)
    cutoff = delta * scale
    absolute = np.abs(residual)
    weights = np.ones_like(absolute, dtype=np.float64)
    outlier = absolute > cutoff
    weights[outlier] = cutoff / np.maximum(absolute[outlier], np.finfo(float).eps)
    return weights


def _polynomial_design(shape: tuple[int, int], degree: int = 2) -> np.ndarray:
    yy = np.linspace(-1.0, 1.0, shape[0], dtype=np.float64)
    xx = np.linspace(-1.0, 1.0, shape[1], dtype=np.float64)
    y_grid, x_grid = np.meshgrid(yy, xx, indexing="ij")
    columns = [
        (x_grid ** x_power) * (y_grid ** y_power)
        for total in range(degree + 1)
        for y_power in range(total + 1)
        for x_power in (total - y_power,)
    ]
    return np.column_stack([column.ravel() for column in columns])


def _open_uniform_knots(n_basis: int, degree: int = 3) -> np.ndarray:
    if n_basis <= degree:
        raise ValueError("n_basis must be greater than the spline degree")
    interior_count = n_basis - degree - 1
    interior = np.linspace(0.0, 1.0, interior_count + 2)[1:-1]
    return np.concatenate([
        np.zeros(degree + 1, dtype=np.float64),
        interior,
        np.ones(degree + 1, dtype=np.float64),
    ])


def _spline_design(
    shape: tuple[int, int],
    basis_shape: tuple[int, int] | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, tuple[int, int]]:
    """Build a sparse tensor-product cubic B-spline design and curvature penalty."""
    if basis_shape is None:
        basis_shape = (
            min(12, max(6, int(np.ceil(shape[0] / 24)))),
            min(12, max(6, int(np.ceil(shape[1] / 24)))),
        )
    n_y, n_x = basis_shape
    y_basis = BSpline.design_matrix(
        np.linspace(0.0, 1.0, shape[0]), _open_uniform_knots(n_y), 3,
    )
    x_basis = BSpline.design_matrix(
        np.linspace(0.0, 1.0, shape[1]), _open_uniform_knots(n_x), 3,
    )
    design = sparse.kron(y_basis, x_basis, format="csr")

    d2_y = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n_y - 2, n_y), format="csr")
    d2_x = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n_x - 2, n_x), format="csr")
    penalty = sparse.vstack([
        sparse.kron(d2_y, sparse.eye(n_x, format="csr"), format="csr"),
        sparse.kron(sparse.eye(n_y, format="csr"), d2_x, format="csr"),
    ], format="csr")
    return design, penalty, basis_shape


def estimate_single_dark_background(
    dark: np.ndarray,
    fit_mask: np.ndarray,
    mode: str,
    smoothness: float = 20.0,
    huber_delta: float = HUBER_DELTA,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate a full-frame background field from uncontaminated blocks of one dark image.

    The source/ghost support is excluded by ``fit_mask``.  Robust surface fits
    therefore extrapolate into that support without using its target values.
    """
    if dark.shape != fit_mask.shape:
        raise ValueError("dark and fit_mask must have the same shape")
    valid = fit_mask & np.isfinite(dark)
    if valid.sum() < max(30, int(dark.size * 0.05)):
        raise ValueError("Too few uncontaminated blocks for background fitting")
    target = dark.astype(np.float64, copy=False).ravel()
    selected = np.flatnonzero(valid.ravel())

    if mode == "poly2":
        design = _polynomial_design(dark.shape, degree=2)
        train_design = design[selected]
        weights = np.ones(selected.size, dtype=np.float64)
        coefficients = np.zeros(train_design.shape[1], dtype=np.float64)
        for _ in range(ROBUST_ITERATIONS):
            root = np.sqrt(weights)
            coefficients, *_ = np.linalg.lstsq(
                train_design * root[:, None], target[selected] * root, rcond=None,
            )
            weights = _huber_weights(
                target[selected] - train_design @ coefficients, delta=huber_delta,
            )
        estimate = (design @ coefficients).reshape(dark.shape)
        complexity = {"polynomial_degree": 2, "parameter_count": int(coefficients.size)}
    elif mode == "robust_spline":
        design, penalty, basis_shape = _spline_design(dark.shape)
        train_design = design[selected]
        weights = np.ones(selected.size, dtype=np.float64)
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        penalty_weight = np.sqrt(float(smoothness))
        for _ in range(ROBUST_ITERATIONS):
            root = np.sqrt(weights)
            weighted_design = sparse.diags(root, format="csr") @ train_design
            system = sparse.vstack([weighted_design, penalty_weight * penalty], format="csr")
            rhs = np.concatenate([root * target[selected], np.zeros(penalty.shape[0])])
            coefficients = lsqr(system, rhs, atol=1e-7, btol=1e-7, iter_lim=1000)[0]
            weights = _huber_weights(
                target[selected] - train_design @ coefficients, delta=huber_delta,
            )
        estimate = np.asarray(design @ coefficients).reshape(dark.shape)
        complexity = {
            "basis_shape": list(basis_shape),
            "parameter_count": int(coefficients.size),
            "curvature_penalty": float(smoothness),
        }
    else:
        raise ValueError(f"Single-dark background mode does not support {mode!r}")

    residual = dark[valid].astype(np.float64) - estimate[valid]
    diagnostics = {
        "mode": mode,
        "fit_blocks": int(valid.sum()),
        "fit_fraction": float(valid.mean()),
        "fit_residual_mae": float(np.mean(np.abs(residual))),
        "fit_residual_mad": float(np.median(np.abs(residual - np.median(residual)))),
        "huber_delta": float(huber_delta),
        **complexity,
    }
    return estimate, diagnostics


def estimate_hybrid_dark_background(
    dark: np.ndarray,
    darks: dict[int, np.ndarray],
    target_index: int,
    fit_mask: np.ndarray,
    smoothness: float = 20.0,
    huber_delta: float = HUBER_DELTA,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Combine a leave-target-out fixed-pattern template with current-dark drift."""
    fixed_pattern = baseline(darks, target_index, "loo_median")
    smooth_correction, correction_diagnostics = estimate_single_dark_background(
        dark=dark - fixed_pattern,
        fit_mask=fit_mask,
        mode="robust_spline",
        smoothness=smoothness,
        huber_delta=huber_delta,
    )
    return fixed_pattern + smooth_correction, {
        "mode": "hybrid_spline",
        "fixed_pattern_source": "leave-target-out median of median-centered dark acquisitions",
        "target_dark_in_fixed_pattern": False,
        "smooth_correction": correction_diagnostics,
    }


def estimate_masked_fixed_pattern(
    darks: dict[int, np.ndarray],
    contamination_supports: dict[int, np.ndarray],
    target_index: int,
    smoothness: float = 20.0,
    decomposition_iterations: int = FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    huber_iterations: int = FIXED_PATTERN_HUBER_ITERATIONS,
    huber_delta: float = HUBER_DELTA,
    include_support_maps: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate detector-fixed structure without temporally median-combining darks.

    For dark ``n``, the object support from light ``n-1`` is excluded before a
    per-acquisition smooth field is fitted.  The remaining high-frequency
    residuals are combined with a noise-precision-weighted Huber mean.  The
    target dark is never used, so its observable ghost cannot leak into F.

    Dark 1 predates the first light in this dataset and is retained with an
    all-valid mask.  Later darks require a corresponding previous-light mask.
    """
    if target_index not in darks:
        raise KeyError(f"Target dark {target_index} is unavailable")
    if decomposition_iterations < 1 or huber_iterations < 1:
        raise ValueError("Fixed-pattern iteration counts must be positive")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")

    shape = darks[target_index].shape
    training_indices = [index for index in sorted(darks) if index != target_index]
    clean_masks = {}
    for index in training_indices:
        if darks[index].shape != shape:
            raise ValueError("All dark images must have the same shape")
        if index == 1:
            clean_masks[index] = np.ones(shape, dtype=bool)
            continue
        light_index = index - 1
        if light_index not in contamination_supports:
            raise KeyError(f"Missing contamination support for light {light_index}")
        support = np.asarray(contamination_supports[light_index], dtype=bool)
        if support.shape != shape:
            raise ValueError("Contamination supports and dark images must have the same shape")
        clean_masks[index] = ~support

    fixed_pattern = np.zeros(shape, dtype=np.float64)
    coverage = np.zeros(shape, dtype=np.int16)
    frame_scales: dict[int, float] = {}
    for _ in range(decomposition_iterations):
        residual_frames = []
        valid_frames = []
        precisions = []
        for index in training_indices:
            valid = clean_masks[index] & np.isfinite(darks[index])
            smooth_field, _ = estimate_single_dark_background(
                dark=darks[index] - fixed_pattern,
                fit_mask=valid,
                mode="robust_spline",
                smoothness=smoothness,
                huber_delta=huber_delta,
            )
            residual = darks[index].astype(np.float64) - smooth_field
            centered = residual[valid] - np.median(residual[valid])
            scale = max(
                1.4826 * float(np.median(np.abs(centered))),
                np.finfo(float).eps,
            )
            residual_frames.append(residual)
            valid_frames.append(valid)
            precisions.append(1.0 / (scale * scale))
            frame_scales[index] = scale

        residual_stack = np.stack(residual_frames)
        valid_stack = np.stack(valid_frames)
        base_weights = np.asarray(precisions)[:, None, None] * valid_stack
        weights = base_weights.copy()
        location = fixed_pattern
        for _ in range(huber_iterations):
            location_weights = weights
            weight_sum = location_weights.sum(axis=0)
            location = np.divide(
                np.sum(location_weights * residual_stack, axis=0),
                weight_sum,
                out=np.zeros(shape, dtype=np.float64),
                where=weight_sum > 0,
            )
            temporal_residual = residual_stack - location
            temporal_scale = np.sqrt(np.divide(
                np.sum(weights * temporal_residual ** 2, axis=0),
                weight_sum,
                out=np.ones(shape, dtype=np.float64),
                where=weight_sum > 0,
            ))
            temporal_scale = np.maximum(temporal_scale, np.finfo(float).eps)
            huber_weights = np.minimum(
                1.0,
                huber_delta * temporal_scale[None, :, :]
                / np.maximum(np.abs(temporal_residual), np.finfo(float).eps),
            )
            weights = base_weights * huber_weights

        coverage = valid_stack.sum(axis=0).astype(np.int16)
        if np.any(coverage == 0):
            raise ValueError("Contamination masks leave some fixed-pattern pixels unobserved")
        fixed_pattern = location - float(np.mean(location))

    # These are the weights that produced the returned final location.  The
    # subsequent Huber update is intentionally not counted because changing
    # the location after it would alter the frozen estimator itself.
    weight_sum = location_weights.sum(axis=0)
    weight_square_sum = np.sum(location_weights ** 2, axis=0)
    effective_coverage = np.divide(
        weight_sum ** 2,
        weight_square_sum,
        out=np.zeros(shape, dtype=np.float64),
        where=weight_square_sum > 0,
    )
    diagnostics = {
        "aggregation": "contamination-masked noise-precision-weighted Huber mean",
        "uses_temporal_pixelwise_median": False,
        "target_dark_in_fixed_pattern": False,
        "training_dark_indices": training_indices,
        "previous_light_lags_excluded": [1],
        "decomposition_iterations": decomposition_iterations,
        "huber_iterations": huber_iterations,
        "huber_delta": float(huber_delta),
        "coverage_min": int(coverage.min()),
        "coverage_p01": float(np.percentile(coverage, 1)),
        "coverage_median": float(np.median(coverage)),
        "coverage_max": int(coverage.max()),
        "coverage_below_5_fraction": float(np.mean(coverage < 5)),
        "effective_coverage_min": float(effective_coverage.min()),
        "effective_coverage_p01": float(np.percentile(effective_coverage, 1)),
        "effective_coverage_median": float(np.median(effective_coverage)),
        "effective_coverage_max": float(effective_coverage.max()),
        "effective_coverage_below_3_fraction": float(np.mean(effective_coverage < 3)),
        "mean_excluded_fraction": float(np.mean([
            1.0 - clean_masks[index].mean() for index in training_indices
        ])),
        "frame_residual_scales": {
            str(index): float(frame_scales[index]) for index in training_indices
        },
    }
    if include_support_maps:
        diagnostics["coverage_map"] = coverage
        diagnostics["effective_coverage_map"] = effective_coverage
    return fixed_pattern, diagnostics


def estimate_masked_hybrid_dark_background(
    dark: np.ndarray,
    darks: dict[int, np.ndarray],
    contamination_supports: dict[int, np.ndarray],
    target_index: int,
    fit_mask: np.ndarray,
    smoothness: float = 20.0,
    fixed_pattern: np.ndarray | None = None,
    fixed_pattern_diagnostics: dict[str, Any] | None = None,
    decomposition_iterations: int = FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    huber_iterations: int = FIXED_PATTERN_HUBER_ITERATIONS,
    huber_delta: float = HUBER_DELTA,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Combine contamination-aware fixed structure with current-dark drift."""
    if fixed_pattern is None:
        fixed_pattern, fixed_pattern_diagnostics = estimate_masked_fixed_pattern(
            darks=darks,
            contamination_supports=contamination_supports,
            target_index=target_index,
            smoothness=smoothness,
            decomposition_iterations=decomposition_iterations,
            huber_iterations=huber_iterations,
            huber_delta=huber_delta,
        )
    if fixed_pattern.shape != dark.shape:
        raise ValueError("fixed_pattern and dark must have the same shape")
    smooth_correction, correction_diagnostics = estimate_single_dark_background(
        dark=dark - fixed_pattern,
        fit_mask=fit_mask,
        mode="robust_spline",
        smoothness=smoothness,
        huber_delta=huber_delta,
    )
    return fixed_pattern + smooth_correction, {
        "mode": "masked_hybrid_spline",
        "fixed_pattern": fixed_pattern_diagnostics,
        "smooth_correction": correction_diagnostics,
    }


def background_spatial_oof(
    dark: np.ndarray,
    fit_mask: np.ndarray,
    folds: np.ndarray,
    mode: str,
    smoothness: float = 20.0,
    darks: dict[int, np.ndarray] | None = None,
    target_index: int | None = None,
    contamination_supports: dict[int, np.ndarray] | None = None,
    fixed_pattern: np.ndarray | None = None,
    fixed_pattern_diagnostics: dict[str, Any] | None = None,
    decomposition_iterations: int = FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    huber_iterations: int = FIXED_PATTERN_HUBER_ITERATIONS,
    huber_delta: float = HUBER_DELTA,
) -> dict[str, float]:
    """Measure background interpolation on spatially held-out clean blocks."""
    predictions = np.full(dark.shape, np.nan, dtype=np.float64)
    eligible = fit_mask & np.isfinite(dark)
    for fold in sorted(int(value) for value in np.unique(folds[eligible])):
        train = eligible & (folds != fold)
        if mode == "hybrid_spline":
            if darks is None or target_index is None:
                raise ValueError("hybrid_spline spatial OOF requires darks and target_index")
            estimate, _ = estimate_hybrid_dark_background(
                dark, darks, target_index, train, smoothness, huber_delta,
            )
        elif mode == "masked_hybrid_spline":
            if darks is None or target_index is None or contamination_supports is None:
                raise ValueError(
                    "masked_hybrid_spline spatial OOF requires darks, target_index, "
                    "and contamination_supports"
                )
            estimate, _ = estimate_masked_hybrid_dark_background(
                dark=dark,
                darks=darks,
                contamination_supports=contamination_supports,
                target_index=target_index,
                fit_mask=train,
                smoothness=smoothness,
                fixed_pattern=fixed_pattern,
                fixed_pattern_diagnostics=fixed_pattern_diagnostics,
                decomposition_iterations=decomposition_iterations,
                huber_iterations=huber_iterations,
                huber_delta=huber_delta,
            )
        else:
            estimate, _ = estimate_single_dark_background(dark, train, mode, smoothness)
        test = eligible & (folds == fold)
        predictions[test] = estimate[test]
    evaluated = eligible & np.isfinite(predictions)
    residual = dark[evaluated].astype(np.float64) - predictions[evaluated]
    centered = residual - np.median(residual)
    return {
        "evaluated_blocks": int(evaluated.sum()),
        "mae": float(np.mean(np.abs(residual))),
        "centered_rmse": float(np.sqrt(np.mean(centered ** 2))),
        "bias": float(np.mean(residual)),
    }


def crossfit_observable_signal(
    dark: np.ndarray,
    fit_mask: np.ndarray,
    folds: np.ndarray,
    mode: str,
    smoothness: float = 20.0,
    darks: dict[int, np.ndarray] | None = None,
    target_index: int | None = None,
    contamination_supports: dict[int, np.ndarray] | None = None,
    fixed_pattern: np.ndarray | None = None,
    fixed_pattern_diagnostics: dict[str, Any] | None = None,
    decomposition_iterations: int = FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    huber_iterations: int = FIXED_PATTERN_HUBER_ITERATIONS,
    huber_delta: float = HUBER_DELTA,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build Y with a separate BG fit that excludes each block's spatial fold."""
    if not (dark.shape == fit_mask.shape == folds.shape):
        raise ValueError("dark, fit_mask, and folds must have the same shape")
    background = np.full(dark.shape, np.nan, dtype=np.float64)
    fold_rows = []
    for fold in sorted(int(value) for value in np.unique(folds)):
        train = fit_mask & (folds != fold) & np.isfinite(dark)
        if mode == "loo_median":
            if darks is None or target_index is None:
                raise ValueError("loo_median cross-fitting requires darks and target_index")
            centered_baseline = baseline(darks, target_index, "loo_median")
            estimate = np.median(dark[train]) + centered_baseline
        elif mode == "hybrid_spline":
            if darks is None or target_index is None:
                raise ValueError("hybrid_spline cross-fitting requires darks and target_index")
            estimate, _ = estimate_hybrid_dark_background(
                dark=dark,
                darks=darks,
                target_index=target_index,
                fit_mask=train,
                smoothness=smoothness,
                huber_delta=huber_delta,
            )
        elif mode == "masked_hybrid_spline":
            if darks is None or target_index is None or contamination_supports is None:
                raise ValueError(
                    "masked_hybrid_spline cross-fitting requires darks, target_index, "
                    "and contamination_supports"
                )
            estimate, _ = estimate_masked_hybrid_dark_background(
                dark=dark,
                darks=darks,
                contamination_supports=contamination_supports,
                target_index=target_index,
                fit_mask=train,
                smoothness=smoothness,
                fixed_pattern=fixed_pattern,
                fixed_pattern_diagnostics=fixed_pattern_diagnostics,
                decomposition_iterations=decomposition_iterations,
                huber_iterations=huber_iterations,
                huber_delta=huber_delta,
            )
        else:
            estimate, _ = estimate_single_dark_background(
                dark=dark,
                fit_mask=train,
                mode=mode,
                smoothness=smoothness,
                huber_delta=huber_delta,
            )
        test = folds == fold
        background[test] = estimate[test]
        heldout_air = fit_mask & test & np.isfinite(dark)
        errors = dark[heldout_air].astype(np.float64) - estimate[heldout_air]
        fold_rows.append({
            "fold": fold,
            "background_training_blocks": int(train.sum()),
            "heldout_air_blocks": int(heldout_air.sum()),
            "heldout_air_mae": float(np.mean(np.abs(errors))),
            "heldout_air_bias": float(np.mean(errors)),
        })
    if not np.isfinite(background).all():
        raise ValueError("Cross-fitted background did not cover every spatial fold")
    signal = dark.astype(np.float64) - background
    return signal, background, {
        "definition": "Each block uses a BG fit that excluded its entire spatial fold",
        "target_fold_dark_values_used_in_its_background_fit": False,
        "folds": fold_rows,
        "heldout_air_mae": float(np.average(
            [row["heldout_air_mae"] for row in fold_rows],
            weights=[row["heldout_air_blocks"] for row in fold_rows],
        )),
    }
