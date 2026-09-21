"""Mechanism study for observable CR afterglow in dark acquisitions.

This script deliberately does not treat a dark image as clean ground truth. It
compares the existing leave-one-out dark baseline with robust background fields
estimated from the current dark outside an expanded source-object mask:

    Y_t = D_t - estimated_background_t

and asks whether it can be predicted out of sample by

    Y_t = alpha * (L_{t-1} - Q75(L_{t-1})) + intercept + error.

The primary outputs are strict spatial out-of-fold (OOF) predictions, residuals,
null comparisons, and a six-panel diagnostic figure.  Original DICOM files are
read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.ndimage import binary_dilation, find_objects, gaussian_filter, label
from scipy.sparse.linalg import lsqr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.physics_model import detect_object_mask

try:
    from scripts.analyze_dark_light_pairs import (
        BLOCK_SIZES,
        RANDOM_SEED,
        baseline,
        block_mean,
        source,
        spatial_folds,
        spatial_nulls,
    )
except ModuleNotFoundError:  # Allows `python scripts/analyze_...py`.
    from analyze_dark_light_pairs import (  # type: ignore
        BLOCK_SIZES,
        RANDOM_SEED,
        baseline,
        block_mean,
        source,
        spatial_folds,
        spatial_nulls,
    )


DEFAULT_PAIRS = ((5, 6), (27, 28))
PRIMARY_BLOCK = 16
TRIM_PERCENTILES = (0.5, 99.5)
LOWPASS_SIGMA_BLOCKS = 2.0
BACKGROUND_MODES = (
    "loo_median",
    "poly2",
    "robust_spline",
    "hybrid_spline",
    "masked_hybrid_spline",
)
DEFAULT_PRIMARY_BACKGROUND = "masked_hybrid_spline"
HUBER_DELTA = 1.5
ROBUST_ITERATIONS = 5
FIXED_PATTERN_DECOMPOSITION_ITERATIONS = 2
FIXED_PATTERN_HUBER_ITERATIONS = 4
SINGLE_LAG_PAIRS = (
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (22, 23), (23, 24), (25, 26), (27, 28),
)
SINGLE_LAG_ROBUST = frozenset({(5, 6), (27, 28)})
SINGLE_LAG_NOT_DETECTED = frozenset({(4, 5), (6, 7)})
DETECTION_MIN_CV_R2 = 0.05
DETECTION_MIN_NULL_MARGIN = 0.05
DETECTION_MAX_ALPHA_CV = 0.25
DETECTION_MAX_ALPHA = 0.02
PSEUDO_OCCLUSION_SQUARE_SIDES = (12, 24, 40)
PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE = 2
PSEUDO_OBJECT_OCCLUSION_COUNT = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pairs(value: str) -> tuple[tuple[int, int], ...]:
    normalized = value.strip().lower().replace("_", "-")
    if normalized == "all":
        return tuple((light_index, light_index + 1) for light_index in range(1, 31))
    if normalized in {"single-lag", "single-lag-only"}:
        return SINGLE_LAG_PAIRS
    pairs = []
    for item in value.split(","):
        fields = item.strip().replace("->", ":").split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError(f"Invalid pair {item!r}; use light:dark")
        light_index, dark_index = map(int, fields)
        if not (1 <= light_index <= 31 and 1 <= dark_index <= 31):
            raise argparse.ArgumentTypeError("Pair indices must be in 1..31")
        if dark_index != light_index + 1:
            raise argparse.ArgumentTypeError("Primary causal pairs must be light_(t-1):dark_t")
        pairs.append((light_index, dark_index))
    if not pairs:
        raise argparse.ArgumentTypeError("At least one pair is required")
    return tuple(pairs)


def single_lag_stratum(pair: tuple[int, int]) -> str | None:
    if pair not in SINGLE_LAG_PAIRS:
        return None
    if pair in SINGLE_LAG_ROBUST:
        return "robust_confirmed"
    if pair in SINGLE_LAG_NOT_DETECTED:
        return "not_detected_control"
    return "parameter_sensitive"


def parse_background_modes(value: str) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = sorted(set(modes) - set(BACKGROUND_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown background mode(s): {', '.join(unknown)}; "
            f"choose from {', '.join(BACKGROUND_MODES)}"
        )
    if not modes:
        raise argparse.ArgumentTypeError("At least one background mode is required")
    return modes


def parse_block_sizes(value: str) -> tuple[int, ...]:
    """Parse a non-empty subset of the supported block sizes."""
    try:
        blocks = tuple(dict.fromkeys(
            int(item.strip()) for item in value.split(",") if item.strip()
        ))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Block sizes must be comma-separated integers"
        ) from error
    unknown = sorted(set(blocks) - set(BLOCK_SIZES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported block size(s): {unknown}; choose from {BLOCK_SIZES}"
        )
    if not blocks:
        raise argparse.ArgumentTypeError("At least one block size is required")
    return blocks


def validate_frozen_background_config(
    config_path: Path,
    repo_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fail closed if a formal modeling export differs from the frozen BG."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise ValueError(f"Background configuration is not frozen: {config_path}")

    expected = {
        "primary_background": config["background_mode"],
        "mask_dilation_pixels": config["mask"]["dilation_pixels"],
        "spline_smoothness": config["spline_smoothness"],
        "background_huber_delta": config["huber_delta"],
        "fixed_pattern_iterations": config["fixed_pattern"]["decomposition_iterations"],
        "fixed_pattern_huber_iterations": config["fixed_pattern"]["huber_iterations"],
    }
    mismatches = {
        name: {"expected": expected_value, "actual": getattr(args, name)}
        for name, expected_value in expected.items()
        if getattr(args, name) != expected_value
    }
    if config["background_mode"] not in args.background_modes:
        mismatches["background_modes"] = {
            "expected_to_include": config["background_mode"],
            "actual": list(args.background_modes),
        }
    if tuple(args.analysis_blocks) != (int(config["block_size"]),):
        mismatches["analysis_blocks"] = {
            "expected": [int(config["block_size"])],
            "actual": list(args.analysis_blocks),
        }
    if args.summary_only:
        mismatches["summary_only"] = {"expected": False, "actual": True}
    if mismatches:
        raise ValueError(f"Formal export differs from frozen BG: {mismatches}")

    archive_path = repo_root / config["mask"]["archive_relative_path"]
    if args.source_mask_npz is None:
        raise ValueError("Formal frozen export requires --source-mask-npz")
    if args.source_mask_npz.resolve() != archive_path.resolve():
        raise ValueError(
            f"Frozen mask path mismatch: expected {archive_path}, got {args.source_mask_npz}"
        )
    actual_mask_hash = sha256(archive_path)
    if actual_mask_hash != config["mask"]["sha256"]:
        raise ValueError(
            f"Frozen mask SHA-256 mismatch: expected {config['mask']['sha256']}, "
            f"got {actual_mask_hash}"
        )
    return {
        "path": str(config_path.resolve()),
        "sha256": sha256(config_path),
        "mask_archive_sha256": actual_mask_hash,
        "verified": True,
    }


def artifact_record(path: Path, output_dir: Path) -> dict[str, Any]:
    """Return portable provenance for a generated artifact."""
    return {
        "relative_path": str(path.resolve().relative_to(output_dir.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


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


def extract_object_shape_templates(
    source_supports: list[np.ndarray],
    min_blocks: int = 20,
    max_fraction: float = 0.12,
) -> list[np.ndarray]:
    """Extract reusable connected-component shapes from real source masks."""
    templates = []
    for support in source_supports:
        labels, _ = label(support)
        maximum = int(support.size * max_fraction)
        for component_index, slices in enumerate(find_objects(labels), start=1):
            if slices is None:
                continue
            component = labels[slices] == component_index
            area = int(component.sum())
            if min_blocks <= area <= maximum:
                templates.append(component)
    return sorted(templates, key=lambda mask: int(mask.sum()))


def _place_pseudo_occlusion(
    shape: np.ndarray,
    trusted_air: np.ndarray,
    occupied: np.ndarray,
    rng: np.random.Generator,
    attempts: int = 1000,
) -> np.ndarray | None:
    height, width = shape.shape
    if height > trusted_air.shape[0] or width > trusted_air.shape[1]:
        return None
    area = int(shape.sum())
    for _ in range(attempts):
        row = int(rng.integers(0, trusted_air.shape[0] - height + 1))
        column = int(rng.integers(0, trusted_air.shape[1] - width + 1))
        air_window = trusted_air[row:row + height, column:column + width]
        if not np.all(air_window[shape]):
            continue
        occupied_window = occupied[row:row + height, column:column + width]
        if np.logical_and(occupied_window, shape).sum() > 0.25 * area:
            continue
        placed = np.zeros_like(trusted_air, dtype=bool)
        placed[row:row + height, column:column + width] = shape
        return placed
    return None


def generate_pseudo_occlusions(
    trusted_air: np.ndarray,
    object_templates: list[np.ndarray],
    seed: int,
    square_sides: tuple[int, ...] = PSEUDO_OCCLUSION_SQUARE_SIDES,
    squares_per_size: int = PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE,
    object_count: int = PSEUDO_OBJECT_OCCLUSION_COUNT,
) -> list[dict[str, Any]]:
    """Place deterministic square and real-object-shaped holdouts in trusted air."""
    rng = np.random.default_rng(seed)
    occupied = np.zeros_like(trusted_air, dtype=bool)
    occlusions = []

    for side in square_sides:
        shape = np.ones((side, side), dtype=bool)
        for placement in range(squares_per_size):
            mask = _place_pseudo_occlusion(shape, trusted_air, occupied, rng)
            if mask is None:
                continue
            occupied |= mask
            occlusions.append({
                "kind": "square",
                "size_label": f"{side}x{side}_blocks",
                "placement": placement + 1,
                "mask": mask,
            })

    if object_templates and object_count > 0:
        candidate_indices = np.unique(np.round(
            np.linspace(0, len(object_templates) - 1, min(len(object_templates), 9))
        ).astype(int))
        candidates = [object_templates[index] for index in candidate_indices]
        target_indices = np.unique(np.round(
            np.linspace(0, len(candidates) - 1, min(object_count, len(candidates)))
        ).astype(int))
        ordered = [candidates[index] for index in target_indices]
        ordered.extend(mask for index, mask in enumerate(candidates) if index not in target_indices)
        for shape in ordered:
            if sum(row["kind"] == "object_shape" for row in occlusions) >= object_count:
                break
            mask = _place_pseudo_occlusion(shape, trusted_air, occupied, rng)
            if mask is None:
                continue
            occupied |= mask
            occlusions.append({
                "kind": "object_shape",
                "size_label": f"{int(shape.sum())}_blocks",
                "placement": 1,
                "mask": mask,
            })
    return occlusions


def _reconstruction_error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    absolute = np.abs(error)
    return {
        "mae": float(np.mean(absolute)),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
    }


def summarize_reconstruction_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({mode for sample in samples for mode in sample["by_background"]})

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        output = {}
        for mode in modes:
            values = [row["by_background"][mode] for row in rows if mode in row["by_background"]]
            if not values:
                continue
            output[mode] = {
                "sample_count": len(values),
                "evaluated_blocks": int(sum(row["evaluated_blocks"] for row in values)),
                "median_sample_mae": float(np.median([row["mae"] for row in values])),
                "median_absolute_bias": float(np.median([abs(row["bias"]) for row in values])),
                "median_sample_rmse": float(np.median([row["rmse"] for row in values])),
                "median_sample_p95_absolute_error": float(np.median([
                    row["p95_absolute_error"] for row in values
                ])),
            }
        return output

    by_kind = {
        kind: summarize([sample for sample in samples if sample["kind"] == kind])
        for kind in ("square", "object_shape")
    }
    def compare_mae(reference_mode: str) -> dict[str, Any]:
        comparisons = {}
        for mode in modes:
            if mode == reference_mode:
                continue
            deltas = [
                sample["by_background"][mode]["mae"]
                - sample["by_background"][reference_mode]["mae"]
                for sample in samples
                if mode in sample["by_background"]
                and reference_mode in sample["by_background"]
            ]
            comparisons[mode] = {
                "paired_sample_count": len(deltas),
                "median_delta_mae": float(np.median(deltas)),
                "mae_improved_count": int(sum(delta < 0 for delta in deltas)),
            }
        return comparisons

    return {
        "overall_by_background": summarize(samples),
        "by_occlusion_kind": by_kind,
        "versus_loo_median": compare_mae("loo_median") if "loo_median" in modes else {},
        "versus_hybrid_spline": (
            compare_mae("hybrid_spline") if "hybrid_spline" in modes else {}
        ),
    }


def validate_background_reconstruction(
    dark: np.ndarray,
    trusted_air: np.ndarray,
    darks: dict[int, np.ndarray],
    target_index: int,
    background_modes: tuple[str, ...],
    object_templates: list[np.ndarray],
    smoothness: float,
    seed: int,
    square_sides: tuple[int, ...] = PSEUDO_OCCLUSION_SQUARE_SIDES,
    squares_per_size: int = PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE,
    object_count: int = PSEUDO_OBJECT_OCCLUSION_COUNT,
    contamination_supports: dict[int, np.ndarray] | None = None,
    masked_fixed_pattern: np.ndarray | None = None,
    masked_fixed_pattern_diagnostics: dict[str, Any] | None = None,
    decomposition_iterations: int = FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    huber_iterations: int = FIXED_PATTERN_HUBER_ITERATIONS,
    huber_delta: float = HUBER_DELTA,
) -> dict[str, Any]:
    """Hide known air blocks and score background reconstruction inside them."""
    occlusions = generate_pseudo_occlusions(
        trusted_air=trusted_air,
        object_templates=object_templates,
        seed=seed,
        square_sides=square_sides,
        squares_per_size=squares_per_size,
        object_count=object_count,
    )
    samples = []
    for occlusion in occlusions:
        holdout = occlusion["mask"]
        train = trusted_air & ~holdout
        row = {
            "kind": occlusion["kind"],
            "size_label": occlusion["size_label"],
            "placement": occlusion["placement"],
            "holdout_blocks": int(holdout.sum()),
            "by_background": {},
        }
        for mode in background_modes:
            if mode == "loo_median":
                centered_baseline = baseline(darks, target_index, "loo_median")
                prediction = np.median(dark[train]) + centered_baseline
            elif mode == "hybrid_spline":
                prediction, _ = estimate_hybrid_dark_background(
                    dark=dark,
                    darks=darks,
                    target_index=target_index,
                    fit_mask=train,
                    smoothness=smoothness,
                    huber_delta=huber_delta,
                )
            elif mode == "masked_hybrid_spline":
                if contamination_supports is None:
                    raise ValueError(
                        "masked_hybrid_spline reconstruction validation requires "
                        "contamination_supports"
                    )
                prediction, _ = estimate_masked_hybrid_dark_background(
                    dark=dark,
                    darks=darks,
                    contamination_supports=contamination_supports,
                    target_index=target_index,
                    fit_mask=train,
                    smoothness=smoothness,
                    fixed_pattern=masked_fixed_pattern,
                    fixed_pattern_diagnostics=masked_fixed_pattern_diagnostics,
                    decomposition_iterations=decomposition_iterations,
                    huber_iterations=huber_iterations,
                    huber_delta=huber_delta,
                )
            else:
                prediction, _ = estimate_single_dark_background(
                    dark=dark,
                    fit_mask=train,
                    mode=mode,
                    smoothness=smoothness,
                )
            row["by_background"][mode] = {
                "evaluated_blocks": int(holdout.sum()),
                **_reconstruction_error_metrics(prediction[holdout], dark[holdout]),
            }
        samples.append(row)
    return {
        "trusted_air_blocks": int(trusted_air.sum()),
        "trusted_air_fraction": float(trusted_air.mean()),
        "requested_square_sides_blocks": list(square_sides),
        "requested_squares_per_size": squares_per_size,
        "requested_object_shapes": object_count,
        "generated_sample_count": len(samples),
        "summary": summarize_reconstruction_samples(samples),
        "samples": samples,
    }


def block_source_masks(
    source_mask: np.ndarray,
    block: int,
    background_max_fraction: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    fractions = block_mean(source_mask.astype(np.float32), block)
    background_fit = fractions <= background_max_fraction
    source_support = fractions > background_max_fraction
    return background_fit, source_support


def load_source_mask(
    light_index: int,
    light: np.ndarray,
    mask_archive: Any | None,
    dilation_pixels: int,
    allow_missing_archive_key: bool = False,
) -> tuple[np.ndarray, str]:
    """Load an externally generated (for example SAM) mask or use the Otsu fallback."""
    if mask_archive is None:
        mask = detect_object_mask(light)
        origin = "otsu_fallback"
    else:
        keys = (f"light_{light_index}", str(light_index))
        key = next((candidate for candidate in keys if candidate in mask_archive.files), None)
        if key is None:
            if not allow_missing_archive_key:
                raise KeyError(f"Mask archive has no key light_{light_index!s} or {light_index!s}")
            mask = detect_object_mask(light)
            origin = "otsu_fallback_missing_external"
        else:
            mask = np.asarray(mask_archive[key], dtype=bool)
            if mask.shape != light.shape:
                raise ValueError(
                    f"Mask {key!r} has shape {mask.shape}, expected light shape {light.shape}"
                )
            origin = "external_mask"
    if dilation_pixels > 0:
        mask = binary_dilation(mask, iterations=dilation_pixels)
    return mask.astype(bool, copy=False), origin


def _sam_rgb(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image[np.isfinite(image)], [0.5, 99.5])
    if high <= low:
        raise ValueError("Cannot window a constant light image for SAM")
    windowed = np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    gray = np.round(windowed * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _rough_component_boxes(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    labels, count = label(mask)
    minimum_area = max(256, int(mask.size * 0.0002))
    components = []
    for component_index, slices in enumerate(find_objects(labels), start=1):
        if slices is None:
            continue
        component = labels == component_index
        if component.sum() < minimum_area:
            continue
        row_slice, column_slice = slices
        pad = 12
        x0 = max(0, column_slice.start - pad)
        y0 = max(0, row_slice.start - pad)
        x1 = min(mask.shape[1] - 1, column_slice.stop - 1 + pad)
        y1 = min(mask.shape[0] - 1, row_slice.stop - 1 + pad)
        components.append((component, np.array([x0, y0, x1, y1], dtype=np.float32)))
    if not components:
        raise ValueError("Otsu prompt generation found no source-object component")
    return components


def build_sam_predictor(checkpoint: Path, model_type: str, device: str):
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise RuntimeError(
            "SAM mask generation requires torch and the official segment-anything package"
        ) from error
    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type {model_type!r}")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return SamPredictor(model), device


def sam_mask_from_light(light: np.ndarray, predictor: Any) -> np.ndarray:
    """Refine Otsu-derived component boxes with promptable SAM masks."""
    rough = detect_object_mask(light)
    predictor.set_image(_sam_rgb(light))
    combined = np.zeros(light.shape, dtype=bool)
    for component, box in _rough_component_boxes(rough):
        masks, predicted_iou, _ = predictor.predict(box=box, multimask_output=True)
        overlaps = []
        for mask in masks:
            union = np.logical_or(mask, component).sum()
            overlaps.append(float(np.logical_and(mask, component).sum() / max(union, 1)))
        score = 0.5 * np.asarray(predicted_iou, dtype=float) + 0.5 * np.asarray(overlaps)
        combined |= np.asarray(masks[int(np.argmax(score))], dtype=bool)
    return combined


def render_mask_contact_sheet(
    output: Path,
    previews: list[tuple[int, np.ndarray, np.ndarray]],
) -> None:
    columns = 4
    rows = int(np.ceil(len(previews) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, (index, image, mask) in zip(axes, previews):
        low, high = np.percentile(image, [0.5, 99.5])
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.contour(mask, levels=[0.5], colors=["#ff3b30"], linewidths=0.7)
        axis.set_title(f"light_{index}")
        axis.set_axis_off()
    for axis in axes[len(previews):]:
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_sam_mask_archive(
    data_dir: Path,
    pairs: tuple[tuple[int, int], ...],
    checkpoint: Path,
    model_type: str,
    device: str,
    output_npz: Path,
) -> dict[str, Any]:
    predictor, resolved_device = build_sam_predictor(checkpoint, model_type, device)
    masks = {}
    previews = []
    for light_index in sorted({light_index for light_index, _ in pairs}):
        path = data_dir / f"{light_index}-light.dcm"
        light = pydicom.dcmread(path).pixel_array.astype(np.float32)
        mask = sam_mask_from_light(light, predictor)
        masks[f"light_{light_index}"] = mask
        preview_step = max(1, int(np.ceil(max(light.shape) / 768)))
        previews.append((
            light_index,
            light[::preview_step, ::preview_step],
            mask[::preview_step, ::preview_step],
        ))
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **masks)
    contact_sheet = output_npz.with_name(f"{output_npz.stem}_contact_sheet.png")
    render_mask_contact_sheet(contact_sheet, previews)
    return {
        "mask_archive": str(output_npz.resolve()),
        "contact_sheet": str(contact_sheet.resolve()),
        "mask_count": len(masks),
        "model_type": model_type,
        "device": resolved_device,
        "checkpoint": str(checkpoint.resolve()),
    }


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


def _finite_limits(values: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, low)), float(np.percentile(finite, high))


def render_six_panel(
    output: Path,
    pair_label: str,
    x: np.ndarray,
    raw_dark: np.ndarray,
    background: np.ndarray,
    background_mode: str,
    y: np.ndarray,
    fit: dict[str, Any],
) -> None:
    prediction, residual = fit["prediction"], fit["residual"]
    common_amplitude = float(np.percentile(np.abs(y[np.isfinite(y)]), 99.5))
    common_amplitude = max(common_amplitude, 1e-6)
    x_limits, dark_limits = _finite_limits(x), _finite_limits(raw_dark)
    background_limits = _finite_limits(background)

    panels = [
        (x, "1  Source X = L - Q75(L)", "gray", x_limits),
        (raw_dark, "2  Raw dark D", "gray", dark_limits),
        (background, f"3  Background B ({background_mode})", "gray", background_limits),
        (y, "4  Observable signal Y", "coolwarm", (-common_amplitude, common_amplitude)),
        (prediction, "5  Strict OOF prediction", "coolwarm", (-common_amplitude, common_amplitude)),
        (residual, "6  Held-out residual", "coolwarm", (-common_amplitude, common_amplitude)),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), constrained_layout=True)
    for axis, (values, title, cmap, limits) in zip(axes.flat, panels):
        image = axis.imshow(values, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest")
        axis.set_title(title, fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.suptitle(
        f"Pseudo-ghost mechanism study: {pair_label}  |  {PRIMARY_BLOCK}x{PRIMARY_BLOCK} block means\n"
        f"OOF CV-R2={fit['cv_r2']:.4f}, NCC={fit['ncc']:.4f}, "
        f"residual variance ratio={fit['residual_variance_ratio']:.4f}\n"
        f"Panels 4-6 share zero-centered range [-{common_amplitude:.1f}, +{common_amplitude:.1f}] DICOM units",
        fontsize=12,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize_background_comparison(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fixed block-16 comparisons without creating per-pair artifacts."""
    by_mode: dict[str, list[dict[str, Any]]] = {}
    pair_lookup: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for pair in pair_results:
        block_row = next(
            row for row in pair["block_results"] if row["block_size"] == PRIMARY_BLOCK
        )
        key = (pair["light_index"], pair["dark_index"])
        pair_lookup[key] = {}
        for background_row in block_row["background_results"]:
            fit = next(
                row["fit"] for row in background_row["mask_results"]
                if row["mask_mode"] == "all"
            )
            null_summary = next(
                row["null_summary"] for row in background_row["mask_results"]
                if row["mask_mode"] == "all"
            )
            gate = next(
                row["detection_gate"] for row in background_row["mask_results"]
                if row["mask_mode"] == "all"
            )
            combined = {**fit, "null_summary": null_summary, "detection_gate": gate}
            mode = background_row["background_mode"]
            pair_lookup[key][mode] = combined
            by_mode.setdefault(mode, []).append(combined)

    mode_summary = {}
    for mode, rows in by_mode.items():
        mode_summary[mode] = {
            "pair_count": len(rows),
            "median_cv_r2": float(np.median([row["cv_r2"] for row in rows])),
            "median_alpha": float(np.median([row["alpha_fold_mean"] for row in rows])),
            "median_alpha_fold_cv": float(np.median([row["alpha_fold_cv"] for row in rows])),
            "median_abs_residual_neighbor_correlation": float(np.median([
                abs(row["residual_neighbor_correlation"]) for row in rows
            ])),
            "true_exceeds_null_p95_count": int(sum(
                row["null_summary"]["true_exceeds_cv_r2_p95"] for row in rows
            )),
            "detection_gate_pass_count": int(sum(
                row["detection_gate"]["passed"] for row in rows
            )),
        }

    def compare_modes(reference_mode: str) -> dict[str, Any]:
        comparisons = {}
        for mode in by_mode:
            if mode == reference_mode:
                continue
            deltas = []
            for pair, rows in pair_lookup.items():
                if reference_mode not in rows or mode not in rows:
                    continue
                reference, candidate = rows[reference_mode], rows[mode]
                deltas.append({
                    "pair": f"{pair[0]}->{pair[1]}",
                    "cv_r2": candidate["cv_r2"] - reference["cv_r2"],
                    "alpha_fold_cv": candidate["alpha_fold_cv"] - reference["alpha_fold_cv"],
                    "abs_residual_neighbor_correlation": (
                        abs(candidate["residual_neighbor_correlation"])
                        - abs(reference["residual_neighbor_correlation"])
                    ),
                })
            comparisons[mode] = {
                "pair_count": len(deltas),
                "median_delta_cv_r2": float(np.median([row["cv_r2"] for row in deltas])),
                "cv_r2_improved_count": int(sum(row["cv_r2"] > 0 for row in deltas)),
                "median_delta_alpha_fold_cv": float(np.median([
                    row["alpha_fold_cv"] for row in deltas
                ])),
                "alpha_stability_improved_count": int(sum(
                    row["alpha_fold_cv"] < 0 for row in deltas
                )),
                "median_delta_abs_residual_neighbor_correlation": float(np.median([
                    row["abs_residual_neighbor_correlation"] for row in deltas
                ])),
                "residual_neighbor_improved_count": int(sum(
                    row["abs_residual_neighbor_correlation"] < 0 for row in deltas
                )),
                "all_three_improved_count": int(sum(
                    row["cv_r2"] > 0
                    and row["alpha_fold_cv"] < 0
                    and row["abs_residual_neighbor_correlation"] < 0
                    for row in deltas
                )),
            }
        return comparisons

    return {
        "by_background": mode_summary,
        "versus_loo_median": (
            compare_modes("loo_median") if "loo_median" in by_mode else {}
        ),
        "versus_hybrid_spline": (
            compare_modes("hybrid_spline") if "hybrid_spline" in by_mode else {}
        ),
    }


def summarize_single_lag_strata(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    strata = ("robust_confirmed", "parameter_sensitive", "not_detected_control")
    return {
        stratum: summarize_background_comparison([
            pair for pair in pair_results if pair.get("cohort_stratum") == stratum
        ])
        for stratum in strata
    }


def summarize_reconstruction_validation(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            sample
            for pair in rows
            for sample in pair.get("background_reconstruction_validation", {}).get("samples", [])
        ]

    all_samples = collect(pair_results)
    strata = ("robust_confirmed", "parameter_sensitive", "not_detected_control")
    return {
        "pair_count": int(sum(
            "background_reconstruction_validation" in pair for pair in pair_results
        )),
        "sample_count": len(all_samples),
        **summarize_reconstruction_samples(all_samples),
        "by_cohort_stratum": {
            stratum: summarize_reconstruction_samples(collect([
                pair for pair in pair_results if pair.get("cohort_stratum") == stratum
            ]))
            for stratum in strata
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict OOF pseudo-ghost mechanism study")
    parser.add_argument("--data-dir", default="data/raw/AI修残影例图")
    parser.add_argument("--output-dir", default="outputs/pseudo_ghost_mechanism_v1")
    parser.add_argument("--pairs", type=parse_pairs, default=DEFAULT_PAIRS,
                        help="Comma-separated light:dark pairs, 'all', or 'single-lag'")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Use only the fixed 16x16 analysis and write no per-pair figures or NPZ files",
    )
    parser.add_argument(
        "--skip-pair-figures", action="store_true",
        help="For formal NPZ exports, omit redundant per-pair six-panel PNG files",
    )
    parser.add_argument(
        "--analysis-blocks", type=parse_block_sizes, default=BLOCK_SIZES,
        help="Comma-separated analysis block sizes; formal frozen exports use only 16",
    )
    parser.add_argument(
        "--background-modes", type=parse_background_modes, default=BACKGROUND_MODES,
        help="Comma-separated background estimators: "
             "loo_median,poly2,robust_spline,hybrid_spline,masked_hybrid_spline",
    )
    parser.add_argument(
        "--primary-background", default=DEFAULT_PRIMARY_BACKGROUND,
        choices=BACKGROUND_MODES,
        help="Background estimator saved in the existing per-pair figure and NPZ",
    )
    parser.add_argument(
        "--source-mask-npz", type=Path,
        help="Optional NPZ of external masks keyed by light_N (for example light_5); "
             "when omitted, the existing Otsu detector is used",
    )
    parser.add_argument(
        "--generate-sam-masks", type=Path,
        help="Generate one SAM mask NPZ plus one contact sheet for the selected pairs, then exit",
    )
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--sam-model-type", default="vit_b", choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--sam-device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--mask-dilation-pixels", type=int, default=24)
    parser.add_argument("--spline-smoothness", type=float, default=20.0)
    parser.add_argument("--background-huber-delta", type=float, default=HUBER_DELTA)
    parser.add_argument(
        "--fixed-pattern-iterations", type=int,
        default=FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    )
    parser.add_argument(
        "--fixed-pattern-huber-iterations", type=int,
        default=FIXED_PATTERN_HUBER_ITERATIONS,
    )
    parser.add_argument(
        "--skip-background-oof", action="store_true",
        help="Skip the block-16 spatial holdout diagnostic for faster exploratory runs",
    )
    parser.add_argument(
        "--validate-background-reconstruction", action="store_true",
        help="Hide known air blocks with square and real-object shapes and score reconstruction",
    )
    parser.add_argument("--skip-input-hashes", action="store_true",
                        help="Skip SHA-256 provenance hashes for faster exploratory runs")
    parser.add_argument(
        "--frozen-background-config", type=Path,
        help="Verify a formal modeling export against this frozen BG configuration",
    )
    args = parser.parse_args()

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    repo_root = Path(__file__).resolve().parent.parent
    pairs: tuple[tuple[int, int], ...] = args.pairs
    cohort_name = "single_lag_only" if pairs == SINGLE_LAG_PAIRS else "custom"
    analysis_blocks = (PRIMARY_BLOCK,) if args.summary_only else args.analysis_blocks
    background_modes: tuple[str, ...] = args.background_modes
    if args.primary_background not in background_modes:
        parser.error("--primary-background must also be listed in --background-modes")
    if args.mask_dilation_pixels < 0:
        parser.error("--mask-dilation-pixels must be non-negative")
    if args.spline_smoothness < 0:
        parser.error("--spline-smoothness must be non-negative")
    if args.background_huber_delta <= 0:
        parser.error("--background-huber-delta must be positive")
    if args.fixed_pattern_iterations < 1:
        parser.error("--fixed-pattern-iterations must be positive")
    if args.fixed_pattern_huber_iterations < 1:
        parser.error("--fixed-pattern-huber-iterations must be positive")
    if not args.summary_only and PRIMARY_BLOCK not in analysis_blocks:
        parser.error(f"Non-summary runs must include the primary block size {PRIMARY_BLOCK}")
    if args.generate_sam_masks and args.source_mask_npz:
        parser.error("--generate-sam-masks and --source-mask-npz are mutually exclusive")
    frozen_background = None
    if args.frozen_background_config:
        frozen_background = validate_frozen_background_config(
            config_path=args.frozen_background_config.resolve(),
            repo_root=repo_root,
            args=args,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for light_index, _ in pairs:
        path = data_dir / f"{light_index}-light.dcm"
        if not path.exists():
            raise FileNotFoundError(path)
    if args.generate_sam_masks:
        if args.sam_checkpoint is None:
            parser.error("--sam-checkpoint is required with --generate-sam-masks")
        if not args.sam_checkpoint.exists():
            raise FileNotFoundError(args.sam_checkpoint)
        generated = generate_sam_mask_archive(
            data_dir=data_dir,
            pairs=pairs,
            checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.sam_device,
            output_npz=args.generate_sam_masks,
        )
        print(json.dumps(clean_json(generated), ensure_ascii=False, indent=2))
        return
    for index in range(1, 32):
        path = data_dir / f"{index}-dark.dcm"
        if not path.exists():
            raise FileNotFoundError(path)

    dark_cache = {block: {} for block in analysis_blocks}
    for index in range(1, 32):
        image = pydicom.dcmread(data_dir / f"{index}-dark.dcm").pixel_array.astype(np.float32)
        for block in analysis_blocks:
            dark_cache[block][index] = block_mean(image, block)

    # Future-light nulls require every light from the earliest target onward.
    earliest_future = min(dark_index for _, dark_index in pairs)
    needs_masked_fixed_pattern = "masked_hybrid_spline" in background_modes
    fixed_pattern_light_indices = set(range(1, 31)) if needs_masked_fixed_pattern else set()
    needed_lights = sorted(set(
        [light for light, _ in pairs]
        + list(range(earliest_future, 32))
        + list(fixed_pattern_light_indices)
    ))
    source_light_indices = {light_index for light_index, _ in pairs}
    mask_light_indices = source_light_indices | fixed_pattern_light_indices
    light_cache = {block: {} for block in analysis_blocks}
    saturation_cache = {block: {} for block in analysis_blocks}
    source_block_masks = {block: {} for block in analysis_blocks}
    mask_archive = np.load(args.source_mask_npz) if args.source_mask_npz else None
    source_mask_origins = {}
    try:
        for index in needed_lights:
            image = pydicom.dcmread(data_dir / f"{index}-light.dcm").pixel_array.astype(np.float32)
            saturation = (image == image.max()).astype(np.float32)
            for block in analysis_blocks:
                light_cache[block][index] = block_mean(image, block)
                saturation_cache[block][index] = block_mean(saturation, block)
            if index in mask_light_indices:
                source_mask, source_mask_origins[index] = load_source_mask(
                    light_index=index,
                    light=image,
                    mask_archive=mask_archive,
                    dilation_pixels=args.mask_dilation_pixels,
                    allow_missing_archive_key=index not in source_light_indices,
                )
                for block in analysis_blocks:
                    source_block_masks[block][index] = block_source_masks(source_mask, block)
    finally:
        if mask_archive is not None:
            mask_archive.close()

    object_shape_templates = (
        extract_object_shape_templates([
            source_block_masks[PRIMARY_BLOCK][index][1]
            for index in sorted(source_light_indices)
        ])
        if args.validate_background_reconstruction else []
    )

    input_paths = [data_dir / f"{index}-dark.dcm" for index in range(1, 32)]
    input_paths.extend(data_dir / f"{index}-light.dcm" for index in needed_lights)
    if args.source_mask_npz:
        input_paths.append(args.source_mask_npz)
    provenance = {
        "hash_algorithm": None if args.skip_input_hashes else "sha256",
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                **({"sha256": sha256(path)} if not args.skip_input_hashes else {}),
            }
            for path in input_paths
        },
    }

    pair_results = []
    for light_index, dark_index in pairs:
        pair_label = f"{light_index}->{dark_index}"
        pair_dir = output_dir / f"pair_{light_index}_{dark_index}"
        if not args.summary_only:
            pair_dir.mkdir(parents=True, exist_ok=True)
        block_results = []
        primary_maps = None
        reconstruction_validation = None

        for block in analysis_blocks:
            darks, lights = dark_cache[block], light_cache[block]
            raw_dark = darks[dark_index]
            x = source(lights, light_index)
            folds = spatial_folds(raw_dark.shape)
            background_fit_mask, source_support = source_block_masks[block][light_index]
            contamination_supports = {
                index: source_block_masks[block][index][1]
                for index in fixed_pattern_light_indices
            }
            masked_fixed_pattern = None
            masked_fixed_pattern_diagnostics = None
            if needs_masked_fixed_pattern:
                masked_fixed_pattern, masked_fixed_pattern_diagnostics = (
                    estimate_masked_fixed_pattern(
                        darks=darks,
                        contamination_supports=contamination_supports,
                        target_index=dark_index,
                        smoothness=args.spline_smoothness,
                        decomposition_iterations=args.fixed_pattern_iterations,
                        huber_iterations=args.fixed_pattern_huber_iterations,
                        huber_delta=args.background_huber_delta,
                    )
                )
            future_sources = [
                (future, source(lights, future), saturation_cache[block][future])
                for future in range(dark_index, 32)
            ]

            if block == PRIMARY_BLOCK and args.validate_background_reconstruction:
                reconstruction_validation = validate_background_reconstruction(
                    dark=raw_dark,
                    trusted_air=background_fit_mask,
                    darks=darks,
                    target_index=dark_index,
                    background_modes=background_modes,
                    object_templates=object_shape_templates,
                    smoothness=args.spline_smoothness,
                    seed=RANDOM_SEED + dark_index * 1000 + 97,
                    contamination_supports=contamination_supports,
                    masked_fixed_pattern=masked_fixed_pattern,
                    masked_fixed_pattern_diagnostics=masked_fixed_pattern_diagnostics,
                    decomposition_iterations=args.fixed_pattern_iterations,
                    huber_iterations=args.fixed_pattern_huber_iterations,
                    huber_delta=args.background_huber_delta,
                )

            background_results = []
            for background_mode in background_modes:
                if background_mode == "loo_median":
                    centered_baseline = baseline(darks, dark_index, "loo_median")
                    background = np.median(raw_dark) + centered_baseline
                    fit_values = raw_dark[background_fit_mask] - background[background_fit_mask]
                    background_diagnostics = {
                        "mode": background_mode,
                        "fit_blocks": int(background_fit_mask.sum()),
                        "fit_fraction": float(background_fit_mask.mean()),
                        "fit_residual_mae": float(np.mean(np.abs(fit_values))),
                        "fit_residual_mad": float(np.median(
                            np.abs(fit_values - np.median(fit_values))
                        )),
                        "source": "other median-centered dark acquisitions",
                    }
                elif background_mode == "hybrid_spline":
                    background, background_diagnostics = estimate_hybrid_dark_background(
                        dark=raw_dark,
                        darks=darks,
                        target_index=dark_index,
                        fit_mask=background_fit_mask,
                        smoothness=args.spline_smoothness,
                        huber_delta=args.background_huber_delta,
                    )
                elif background_mode == "masked_hybrid_spline":
                    background, background_diagnostics = estimate_masked_hybrid_dark_background(
                        dark=raw_dark,
                        darks=darks,
                        contamination_supports=contamination_supports,
                        target_index=dark_index,
                        fit_mask=background_fit_mask,
                        smoothness=args.spline_smoothness,
                        fixed_pattern=masked_fixed_pattern,
                        fixed_pattern_diagnostics=masked_fixed_pattern_diagnostics,
                        decomposition_iterations=args.fixed_pattern_iterations,
                        huber_iterations=args.fixed_pattern_huber_iterations,
                        huber_delta=args.background_huber_delta,
                    )
                else:
                    background, background_diagnostics = estimate_single_dark_background(
                        dark=raw_dark,
                        fit_mask=background_fit_mask,
                        mode=background_mode,
                        smoothness=args.spline_smoothness,
                    )
                if (
                    block == PRIMARY_BLOCK
                    and background_mode != "loo_median"
                    and not args.skip_background_oof
                ):
                    background_diagnostics["spatial_oof"] = background_spatial_oof(
                        dark=raw_dark,
                        fit_mask=background_fit_mask,
                        folds=folds,
                        mode=background_mode,
                        smoothness=args.spline_smoothness,
                        darks=darks,
                        target_index=dark_index,
                        contamination_supports=contamination_supports,
                        fixed_pattern=masked_fixed_pattern,
                        fixed_pattern_diagnostics=masked_fixed_pattern_diagnostics,
                        decomposition_iterations=args.fixed_pattern_iterations,
                        huber_iterations=args.fixed_pattern_huber_iterations,
                        huber_delta=args.background_huber_delta,
                    )

                y = raw_dark - background
                mask_results = []
                for mask_mode in ("all", "exclude_saturated"):
                    fit, null_rows, null_summary = evaluate_with_nulls(
                        x=x,
                        y=y,
                        source_saturation=saturation_cache[block][light_index],
                        mask_mode=mask_mode,
                        folds=folds,
                        future_sources=future_sources,
                        seed=RANDOM_SEED + dark_index * 1000 + block,
                    )
                    mask_result = {
                        "mask_mode": mask_mode,
                        "fit": _compact_metric_view(fit) if args.summary_only else _metric_view(fit),
                        "null_summary": null_summary,
                        "detection_gate": detection_gate(fit, null_rows, null_summary),
                    }
                    if not args.summary_only:
                        mask_result["null_detail"] = null_rows
                    mask_results.append(mask_result)
                    if (
                        block == PRIMARY_BLOCK
                        and background_mode == args.primary_background
                        and mask_mode == "all"
                    ):
                        primary_maps = (
                            x, raw_dark, background, y, folds, fit,
                            background_fit_mask, source_support,
                        )
                background_results.append({
                    "background_mode": background_mode,
                    "background_diagnostics": background_diagnostics,
                    "mask_results": mask_results,
                })
            block_results.append({
                "block_size": block,
                "source_mask_origin": source_mask_origins[light_index],
                "source_support_fraction": float(source_support.mean()),
                "background_fit_fraction": float(background_fit_mask.mean()),
                "background_results": background_results,
            })

        pair_result = {
            "light_index": light_index,
            "dark_index": dark_index,
            "role": "single_lag_background_comparison",
            "cohort_stratum": single_lag_stratum((light_index, dark_index)),
            "primary_block_size": PRIMARY_BLOCK,
            "primary_background": args.primary_background,
            "source_mask_origin": source_mask_origins[light_index],
            "block_results": block_results,
        }
        if reconstruction_validation is not None:
            pair_result["background_reconstruction_validation"] = reconstruction_validation
        if not args.summary_only:
            if primary_maps is None:
                raise RuntimeError("Primary block result was not generated")
            (x, raw_dark, background, y, folds, primary_fit,
             background_fit_mask, source_support) = primary_maps
            maps_path = pair_dir / "oof_maps_block16.npz"
            figure_path = pair_dir / "six_panel_oof.png"
            if not args.skip_pair_figures:
                render_six_panel(figure_path, pair_label, x, raw_dark,
                                 background, args.primary_background, y, primary_fit)
            np.savez_compressed(
                maps_path,
                source=x.astype(np.float32),
                raw_dark=raw_dark.astype(np.float32),
                background=background.astype(np.float32),
                background_mode=np.asarray(args.primary_background),
                background_fit_mask=background_fit_mask,
                source_support=source_support,
                source_saturation_fraction=(
                    saturation_cache[PRIMARY_BLOCK][light_index].astype(np.float32)
                ),
                observable_signal=y.astype(np.float32),
                oof_prediction=primary_fit["prediction"].astype(np.float32),
                oof_residual=primary_fit["residual"].astype(np.float32),
                evaluated_mask=primary_fit["evaluated_mask"],
                spatial_folds=folds,
            )
            pair_result["primary_maps"] = str(maps_path.resolve())
            pair_result["output_artifacts"] = {
                "primary_maps": artifact_record(maps_path, output_dir),
            }
            if not args.skip_pair_figures:
                pair_result["six_panel"] = str(figure_path.resolve())
                pair_result["output_artifacts"]["six_panel"] = artifact_record(
                    figure_path, output_dir,
                )
        pair_results.append(pair_result)

    comparison_summary = summarize_background_comparison(pair_results)
    stratified_summary = (
        summarize_single_lag_strata(pair_results)
        if cohort_name == "single_lag_only" else None
    )
    reconstruction_summary = (
        summarize_reconstruction_validation(pair_results)
        if args.validate_background_reconstruction else None
    )
    result = {
        "schema_version": 5,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "study_question": "Can detectable previous-exposure structure in a later dark acquisition be explained by a low-dimensional linear spatial model?",
        "evidence_boundary": [
            "Y is an observable pseudo-ghost signal, not clean ground-truth ghost.",
            "Y may contain read noise, baseline error, drift, and older-frame memory.",
            "These selected pairs are for mechanism discovery, not prevalence or generalization claims.",
        ],
        "definitions": {
            "source": "X_(t-1) = L_(t-1) - Q75(L_(t-1)), evaluated on each block-mean grid",
            "background": "Compared estimators of B_t; single-dark surfaces are fitted only outside the expanded source mask",
            "observable_signal": "Y_t = D_t - estimated_background_t",
            "model": "Y_t = alpha * X_(t-1) + intercept + error",
            "prediction": "Four-fold strict spatial out-of-fold prediction",
        },
        "configuration": {
            "data_directory": str(data_dir.resolve()),
            "cohort": cohort_name,
            "pairs": [list(pair) for pair in pairs],
            "block_sizes": list(analysis_blocks),
            "summary_only": args.summary_only,
            "pair_figures_written": not args.skip_pair_figures,
            "primary_visualization_block_size": PRIMARY_BLOCK,
            "background_modes": list(background_modes),
            "primary_background": args.primary_background,
            "source_mask_npz": str(args.source_mask_npz.resolve()) if args.source_mask_npz else None,
            "source_mask_origins": source_mask_origins,
            "mask_dilation_pixels": args.mask_dilation_pixels,
            "background_fit_max_source_fraction": 0.01,
            "spline_smoothness": args.spline_smoothness,
            "masked_fixed_pattern": {
                "enabled": needs_masked_fixed_pattern,
                "aggregation": "contamination-masked noise-precision-weighted Huber mean",
                "uses_temporal_pixelwise_median": False,
                "previous_light_lags_excluded": [1],
                "decomposition_iterations": args.fixed_pattern_iterations,
                "huber_iterations": args.fixed_pattern_huber_iterations,
                "huber_delta": args.background_huber_delta,
                "extra_mask_fallback": "Otsu when an external archive lacks a light index",
            },
            "background_spatial_oof": not args.skip_background_oof,
            "background_reconstruction_validation": {
                "enabled": args.validate_background_reconstruction,
                "square_sides_blocks": list(PSEUDO_OCCLUSION_SQUARE_SIDES),
                "squares_per_size": PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE,
                "object_shape_count": PSEUDO_OBJECT_OCCLUSION_COUNT,
                "object_template_count": len(object_shape_templates),
                "target_values_hidden_from_single-dark_fit": True,
                "loo_level_anchor_uses_training_air_only": True,
            },
            "mask_modes": ["all", "exclude_saturated"],
            "saturation_block_fraction_threshold": 0.05,
            "training_trim_percentiles": list(TRIM_PERCENTILES),
            "held_out_target_filtering": "None; held-out Y never controls evaluation eligibility",
            "spatial_folds": 4,
            "lowpass_sigma_blocks": LOWPASS_SIGMA_BLOCKS,
            "random_seed": RANDOM_SEED,
            "nulls": ["future_light", "spatial_shift", "block_shuffle"],
            "display_common_range": "Panels 4-6 use +/- percentile_99.5(abs(Y))",
        },
        "frozen_background": frozen_background,
        "code_provenance": {
            "analysis_script": {
                "path": "scripts/analyze_pseudo_ghost_mechanism.py",
                "sha256": sha256(Path(__file__).resolve()),
            },
            "shared_spatial_analysis": {
                "path": "scripts/analyze_dark_light_pairs.py",
                "sha256": sha256(repo_root / "scripts" / "analyze_dark_light_pairs.py"),
            },
        },
        "input_provenance": provenance,
        "comparison_summary": comparison_summary,
        "stratified_comparison_summary": stratified_summary,
        "background_reconstruction_summary": reconstruction_summary,
        "pair_results": pair_results,
    }
    result_path = output_dir / "analysis_results.json"
    result_path.write_text(json.dumps(clean_json(result), ensure_ascii=False, indent=2), encoding="utf-8")

    if args.summary_only:
        compact = {
            "output": str(result_path.resolve()),
            "comparison_summary": comparison_summary,
            "stratified_comparison_summary": stratified_summary,
            "background_reconstruction_summary": reconstruction_summary,
        }
    else:
        compact = {
            "output": str(result_path.resolve()),
            "pairs": [
                {
                    "pair": f"{row['light_index']}->{row['dark_index']}",
                    "block16_all_by_background": {
                        background_row["background_mode"]: {
                            "background_diagnostics": background_row["background_diagnostics"],
                            "fit": next(
                                mask_row["fit"] for mask_row in background_row["mask_results"]
                                if mask_row["mask_mode"] == "all"
                            ),
                            "null_summary": next(
                                mask_row["null_summary"] for mask_row in background_row["mask_results"]
                                if mask_row["mask_mode"] == "all"
                            ),
                        }
                        for block_row in row["block_results"]
                        if block_row["block_size"] == PRIMARY_BLOCK
                        for background_row in block_row["background_results"]
                    },
                }
                for row in pair_results
            ],
        }
    print(json.dumps(clean_json(compact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
