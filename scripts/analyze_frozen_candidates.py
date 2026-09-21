"""Compare frozen M1 with blur and diagnostic quadratic candidates.

The background and observable Y are immutable inputs. Gaussian blur is the only
formal M2 candidate and selects one detector-wide sigma by nested spatial CV.
The quadratic model is evaluated out of fold for diagnosis only; no combined,
spatially varying, or later-stage model is fitted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_pseudo_ghost_mechanism import (
    TRIM_PERCENTILES,
    _fit_affine,
    _training_keep,
    clean_json,
)


PRIMARY_PAIRS = ("5->6", "27->28")
SIGMA_BLOCKS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
MIN_MEAN_DELTA_R2 = 0.002
MIN_EDGE_RMSE_REL_REDUCTION = 0.05
MAX_QUAD_TREND_AMPLITUDE_RATIO = 0.50


@dataclass(frozen=True)
class PairData:
    label: str
    x: np.ndarray
    y: np.ndarray
    folds: np.ndarray
    evaluated: np.ndarray
    source_support: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_pairs(dataset_dir: Path) -> tuple[list[PairData], dict[str, Any]]:
    manifest_path = dataset_dir / "analysis_results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = manifest.get("frozen_background") or {}
    if not frozen.get("verified"):
        raise ValueError("Dataset manifest is not a verified frozen-background export")
    if manifest["configuration"]["block_sizes"] != [16]:
        raise ValueError("M1 diagnostics require exactly the frozen block-16 dataset")
    if manifest["configuration"]["background_modes"] != ["masked_hybrid_spline"]:
        raise ValueError("M1 diagnostics require masked_hybrid_spline only")

    pairs = []
    for row in manifest["pair_results"]:
        label = f"{row['light_index']}->{row['dark_index']}"
        if label not in PRIMARY_PAIRS:
            continue
        artifact = row["output_artifacts"]["primary_maps"]
        path = dataset_dir / artifact["relative_path"]
        actual_hash = sha256(path)
        if actual_hash != artifact["sha256"]:
            raise ValueError(
                f"Frozen pair artifact hash mismatch for {label}: "
                f"expected {artifact['sha256']}, got {actual_hash}"
            )
        with np.load(path) as archive:
            required = {
                "source", "observable_signal", "background", "raw_dark",
                "source_support", "evaluated_mask", "oof_prediction",
                "oof_residual", "spatial_folds", "background_mode",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Frozen pair {label} is missing arrays: {sorted(missing)}")
            if str(archive["background_mode"]) != "masked_hybrid_spline":
                raise ValueError(f"Frozen pair {label} has the wrong background mode")
            x = archive["source"].astype(np.float64)
            y = archive["observable_signal"].astype(np.float64)
            raw_dark = archive["raw_dark"].astype(np.float64)
            background = archive["background"].astype(np.float64)
            folds = archive["spatial_folds"].astype(np.int8)
            evaluated = archive["evaluated_mask"].astype(bool)
            support = archive["source_support"].astype(bool)
            prediction = archive["oof_prediction"].astype(np.float64)
            residual = archive["oof_residual"].astype(np.float64)

        shapes = {
            array.shape for array in (
                x, y, raw_dark, background, folds, evaluated, support,
                prediction, residual,
            )
        }
        if len(shapes) != 1:
            raise ValueError(f"Frozen pair {label} contains inconsistent array shapes")
        if not np.allclose(y, raw_dark - background, atol=2e-4, rtol=0.0):
            raise ValueError(f"Frozen Y no longer equals raw_dark - background for {label}")
        valid = evaluated & np.isfinite(x) & np.isfinite(y) & np.isfinite(prediction)
        if set(np.unique(folds[valid]).tolist()) != {0, 1, 2, 3}:
            raise ValueError(f"Frozen pair {label} does not contain four spatial folds")
        if not np.allclose(
            residual[valid], y[valid] - prediction[valid], atol=2e-5, rtol=0.0,
        ):
            raise ValueError(f"Stored M1 residual is inconsistent for {label}")
        pairs.append(PairData(
            label=label,
            x=x,
            y=y,
            folds=folds,
            evaluated=valid,
            source_support=support,
            prediction=prediction,
            residual=residual,
        ))

    pairs.sort(key=lambda pair: PRIMARY_PAIRS.index(pair.label))
    if [pair.label for pair in pairs] != list(PRIMARY_PAIRS):
        raise ValueError(f"Frozen dataset must contain exactly {PRIMARY_PAIRS}")
    return pairs, {
        "path": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path),
        "frozen_background": frozen,
    }


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    return 1.0 - float(np.sum((y - prediction) ** 2)) / denominator


def _fit_affine_holdout(
    pair: PairData,
    predictor: np.ndarray,
    train_base: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    train, thresholds = _training_keep(
        predictor, pair.y, train_base, TRIM_PERCENTILES,
    )
    alpha, intercept = _fit_affine(predictor[train], pair.y[train])
    return alpha * predictor[test] + intercept, {
        "alpha": alpha,
        "intercept": intercept,
        "train_blocks": int(train.sum()),
        "test_blocks": int(test.sum()),
        "training_trim_thresholds": thresholds,
    }


def _inner_blur_score(
    pairs: list[PairData],
    predictors: dict[str, np.ndarray],
    outer_fold: int,
) -> tuple[float, dict[str, float]]:
    pair_scores = {}
    for pair in pairs:
        development = pair.evaluated & (pair.folds != outer_fold)
        prediction = np.full(pair.y.shape, np.nan, dtype=np.float64)
        for inner_fold in sorted(int(value) for value in np.unique(pair.folds[development])):
            test = development & (pair.folds == inner_fold)
            train_base = development & (pair.folds != inner_fold)
            prediction[test], _ = _fit_affine_holdout(
                pair, predictors[pair.label], train_base, test,
            )
        evaluated = development & np.isfinite(prediction)
        if evaluated.sum() < 20:
            raise ValueError(f"Too few inner-CV blocks for {pair.label}")
        pair_scores[pair.label] = _r2(pair.y[evaluated], prediction[evaluated])
    return float(np.mean(list(pair_scores.values()))), pair_scores


def nested_blur_oof(
    pairs: list[PairData],
    sigma_blocks: tuple[float, ...] = SIGMA_BLOCKS,
) -> dict[str, Any]:
    """Select one shared sigma inside each outer fold; fit alpha/b per pair."""
    blurred = {
        sigma: {
            pair.label: gaussian_filter(pair.x, sigma=sigma, mode="reflect")
            if sigma > 0 else pair.x
            for pair in pairs
        }
        for sigma in sigma_blocks
    }
    predictions = {
        pair.label: np.full(pair.y.shape, np.nan, dtype=np.float64) for pair in pairs
    }
    fold_parameters = {pair.label: [] for pair in pairs}
    selections = []
    for outer_fold in range(4):
        candidates = []
        for sigma in sigma_blocks:
            score, pair_scores = _inner_blur_score(
                pairs, blurred[sigma], outer_fold,
            )
            candidates.append({
                "sigma_blocks": float(sigma),
                "equal_pair_mean_inner_cv_r2": score,
                "pair_inner_cv_r2": pair_scores,
            })
        selected = max(
            candidates,
            key=lambda row: (row["equal_pair_mean_inner_cv_r2"], -row["sigma_blocks"]),
        )
        sigma = selected["sigma_blocks"]
        selections.append({
            "outer_fold": outer_fold,
            "selected_sigma_blocks": sigma,
            "candidates": candidates,
        })
        for pair in pairs:
            test = pair.evaluated & (pair.folds == outer_fold)
            train_base = pair.evaluated & (pair.folds != outer_fold)
            predictions[pair.label][test], parameters = _fit_affine_holdout(
                pair, blurred[sigma][pair.label], train_base, test,
            )
            fold_parameters[pair.label].append({
                "outer_fold": outer_fold,
                "selected_sigma_blocks": sigma,
                **parameters,
            })
    return {
        "predictions": predictions,
        "fold_parameters": fold_parameters,
        "outer_selections": selections,
        "candidate_sigma_blocks": [float(value) for value in sigma_blocks],
        "selection_scope": "One sigma shared by both pairs; alpha and intercept pair-specific",
        "selection_objective": "Equal-pair-weight mean inner spatial CV R2",
    }


def quadratic_oof(pairs: list[PairData]) -> dict[str, Any]:
    """Fit hierarchical X + X^2 models out of fold for diagnosis only."""
    predictions = {
        pair.label: np.full(pair.y.shape, np.nan, dtype=np.float64) for pair in pairs
    }
    fold_parameters = {pair.label: [] for pair in pairs}
    for outer_fold in range(4):
        for pair in pairs:
            test = pair.evaluated & (pair.folds == outer_fold)
            train_base = pair.evaluated & (pair.folds != outer_fold)
            train, thresholds = _training_keep(
                pair.x, pair.y, train_base, TRIM_PERCENTILES,
            )
            x_mean = float(np.mean(pair.x[train]))
            x_scale = max(float(np.std(pair.x[train])), 1e-12)
            train_x = (pair.x[train] - x_mean) / x_scale
            test_x = (pair.x[test] - x_mean) / x_scale
            train_design = np.column_stack([
                train_x, train_x ** 2, np.ones(train_x.size, dtype=float),
            ])
            coefficients, *_ = np.linalg.lstsq(
                train_design, pair.y[train].astype(float), rcond=None,
            )
            test_design = np.column_stack([
                test_x, test_x ** 2, np.ones(test_x.size, dtype=float),
            ])
            predictions[pair.label][test] = test_design @ coefficients
            fold_parameters[pair.label].append({
                "outer_fold": outer_fold,
                "a1_standardized": float(coefficients[0]),
                "a2_standardized_squared": float(coefficients[1]),
                "intercept": float(coefficients[2]),
                "training_x_mean": x_mean,
                "training_x_scale": x_scale,
                "train_blocks": int(train.sum()),
                "test_blocks": int(test.sum()),
                "training_trim_thresholds": thresholds,
            })
    return {
        "predictions": predictions,
        "fold_parameters": fold_parameters,
        "status": "diagnostic_only",
        "normalization": "Training-fold mean and standard deviation only",
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _neighbor_correlation(values: np.ndarray, valid: np.ndarray) -> float:
    horizontal = valid[:, :-1] & valid[:, 1:]
    vertical = valid[:-1, :] & valid[1:, :]
    first = np.concatenate([values[:, :-1][horizontal], values[:-1, :][vertical]])
    second = np.concatenate([values[:, 1:][horizontal], values[1:, :][vertical]])
    return _correlation(first, second)


def spatial_regions(pair: PairData) -> dict[str, np.ndarray]:
    dilated = binary_dilation(pair.source_support, iterations=2)
    eroded = binary_erosion(pair.source_support, iterations=2)
    return {
        "source_edge": pair.evaluated & (dilated ^ eroded),
        "source_interior": pair.evaluated & eroded,
        "outside_source": pair.evaluated & ~dilated,
    }


def model_metrics(pair: PairData, prediction: np.ndarray) -> dict[str, float]:
    valid = pair.evaluated & np.isfinite(prediction)
    y = pair.y[valid]
    predicted = prediction[valid]
    residual = y - predicted
    residual_map = np.full(pair.y.shape, np.nan, dtype=np.float64)
    residual_map[valid] = residual
    regions = spatial_regions(pair)

    def region_rmse(region: np.ndarray) -> float:
        selected = region & np.isfinite(residual_map)
        if not np.any(selected):
            return float("nan")
        return float(np.sqrt(np.mean(residual_map[selected] ** 2)))

    return {
        "evaluated_blocks": int(valid.sum()),
        "cv_r2": _r2(y, predicted),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "residual_std": float(np.std(residual)),
        "x_y_correlation": _correlation(pair.x[valid], y),
        "residual_x_correlation": _correlation(residual, pair.x[valid]),
        "residual_neighbor_correlation": _neighbor_correlation(residual_map, valid),
        "source_edge_rmse": region_rmse(regions["source_edge"]),
        "source_interior_rmse": region_rmse(regions["source_interior"]),
        "outside_source_rmse": region_rmse(regions["outside_source"]),
    }


def regression_metrics(pair: PairData) -> dict[str, float]:
    """Backward-compatible M1 metric helper used by focused tests."""
    return model_metrics(pair, pair.prediction)


def quantile_trend(
    x: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    bins: int = 24,
) -> dict[str, list[float] | list[int]]:
    xv = x[valid]
    vv = values[valid]
    edges = np.unique(np.quantile(xv, np.linspace(0.0, 1.0, bins + 1)))
    centers, means, medians, counts = [], [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (xv >= low) & (xv <= high if high == edges[-1] else xv < high)
        if not np.any(selected):
            continue
        centers.append(float(np.mean(xv[selected])))
        means.append(float(np.mean(vv[selected])))
        medians.append(float(np.median(vv[selected])))
        counts.append(int(selected.sum()))
    return {
        "x_mean": centers,
        "value_mean": means,
        "value_median": medians,
        "count": counts,
    }


def residual_trend_summary(
    pair: PairData,
    prediction: np.ndarray,
) -> dict[str, Any]:
    residual = pair.y - prediction
    trend = quantile_trend(pair.x, residual, pair.evaluated)
    means = np.asarray(trend["value_mean"], dtype=float)
    counts = np.asarray(trend["count"], dtype=float)
    return {
        "quantile_trend": trend,
        "binned_mean_peak_to_peak": float(np.ptp(means)),
        "binned_mean_weighted_rms": float(
            np.sqrt(np.sum(counts * means ** 2) / np.sum(counts))
        ),
    }


def compare_models(
    pairs: list[PairData],
    predictions: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]]]:
    results = {}
    metric_lookup = {}
    for pair in pairs:
        model_rows = {}
        metric_lookup[pair.label] = {}
        for model_name, model_predictions in predictions.items():
            metrics = model_metrics(pair, model_predictions[pair.label])
            trend = residual_trend_summary(pair, model_predictions[pair.label])
            model_rows[model_name] = {"metrics": metrics, "residual_vs_x": trend}
            metric_lookup[pair.label][model_name] = metrics
        m1_metrics = model_rows["M1"]["metrics"]
        m1_trend = model_rows["M1"]["residual_vs_x"]
        for candidate in ("Mblur", "Mquad_diagnostic"):
            metrics = model_rows[candidate]["metrics"]
            trend = model_rows[candidate]["residual_vs_x"]
            model_rows[candidate]["change_vs_M1"] = {
                "delta_cv_r2": metrics["cv_r2"] - m1_metrics["cv_r2"],
                "rmse_reduction": m1_metrics["rmse"] - metrics["rmse"],
                "source_edge_rmse_relative_reduction": (
                    1.0 - metrics["source_edge_rmse"] / m1_metrics["source_edge_rmse"]
                ),
                "residual_neighbor_correlation_reduction": (
                    m1_metrics["residual_neighbor_correlation"]
                    - metrics["residual_neighbor_correlation"]
                ),
                "binned_mean_peak_to_peak_ratio": (
                    trend["binned_mean_peak_to_peak"]
                    / max(m1_trend["binned_mean_peak_to_peak"], 1e-12)
                ),
                "binned_mean_weighted_rms_ratio": (
                    trend["binned_mean_weighted_rms"]
                    / max(m1_trend["binned_mean_weighted_rms"], 1e-12)
                ),
            }
        results[pair.label] = model_rows
    return results, metric_lookup


def interpret_candidates(
    comparisons: dict[str, Any],
    blur_selections: list[dict[str, Any]],
) -> dict[str, Any]:
    blur_deltas = [
        comparisons[label]["Mblur"]["change_vs_M1"]["delta_cv_r2"]
        for label in PRIMARY_PAIRS
    ]
    blur_checks = {
        "nonzero_sigma_selected_in_at_least_three_outer_folds": sum(
            row["selected_sigma_blocks"] > 0 for row in blur_selections
        ) >= 3,
        "all_pairs_cv_r2_improved": all(value > 0 for value in blur_deltas),
        "equal_pair_mean_delta_cv_r2_at_least_0_002": (
            float(np.mean(blur_deltas)) >= MIN_MEAN_DELTA_R2
        ),
        "all_pairs_edge_rmse_reduced_at_least_5_percent": all(
            comparisons[label]["Mblur"]["change_vs_M1"]
            ["source_edge_rmse_relative_reduction"] >= MIN_EDGE_RMSE_REL_REDUCTION
            for label in PRIMARY_PAIRS
        ),
        "all_pairs_neighbor_correlation_reduced": all(
            comparisons[label]["Mblur"]["change_vs_M1"]
            ["residual_neighbor_correlation_reduction"] > 0
            for label in PRIMARY_PAIRS
        ),
    }
    quad_change = comparisons["27->28"]["Mquad_diagnostic"]["change_vs_M1"]
    quad_checks = {
        "27_to_28_cv_r2_improved": quad_change["delta_cv_r2"] > 0,
        "27_to_28_residual_trend_amplitude_halved": (
            quad_change["binned_mean_peak_to_peak_ratio"]
            <= MAX_QUAD_TREND_AMPLITUDE_RATIO
        ),
        "27_to_28_residual_trend_weighted_rms_halved": (
            quad_change["binned_mean_weighted_rms_ratio"]
            <= MAX_QUAD_TREND_AMPLITUDE_RATIO
        ),
    }
    blur_supported = bool(all(blur_checks.values()))
    quad_signal = bool(all(quad_checks.values()))
    return {
        "spatial_diffusion_supported": blur_supported,
        "blur_checks": blur_checks,
        "quadratic_diagnostic_signal": quad_signal,
        "quadratic_checks": quad_checks,
        "combined_model_fitted": False,
        "interpretation": (
            "Both candidate mechanisms show evidence; do not infer or fit a combined model yet."
            if blur_supported and quad_signal else
            "Blur evidence only." if blur_supported else
            "Quadratic diagnostic evidence only; this is not a promoted model."
            if quad_signal else
            "Neither candidate meets the predeclared evidence checks."
        ),
        "thresholds": {
            "minimum_equal_pair_mean_delta_cv_r2": MIN_MEAN_DELTA_R2,
            "minimum_edge_rmse_relative_reduction": MIN_EDGE_RMSE_REL_REDUCTION,
            "maximum_quadratic_trend_ratio": MAX_QUAD_TREND_AMPLITUDE_RATIO,
        },
    }


def render_diagnostics(
    path: Path,
    pairs: list[PairData],
    predictions: dict[str, dict[str, np.ndarray]],
    comparisons: dict[str, Any],
    selected_sigmas: list[float],
) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(len(pairs), 4, figsize=(16, 8), constrained_layout=True)
    colors = {
        "M1": "#d95f02",
        "Mblur": "#1b9e77",
        "Mquad_diagnostic": "#7570b3",
    }
    for row_index, pair in enumerate(pairs):
        valid = pair.evaluated
        x = pair.x[valid]
        m1_residual = pair.y[valid] - predictions["M1"][pair.label][valid]
        x_limits = np.percentile(x, [0.5, 99.5])
        residual_limits = np.percentile(m1_residual, [0.5, 99.5])
        density_keep = (
            (x >= x_limits[0]) & (x <= x_limits[1])
            & (m1_residual >= residual_limits[0])
            & (m1_residual <= residual_limits[1])
        )

        axis = axes[row_index, 0]
        density = axis.hexbin(
            x[density_keep], m1_residual[density_keep], gridsize=55, mincnt=1,
            cmap="viridis", bins="log",
        )
        for model_name, label in (
            ("M1", "M1"),
            ("Mblur", "Mblur"),
            ("Mquad_diagnostic", "Mquad (diagnostic)"),
        ):
            residual = pair.y - predictions[model_name][pair.label]
            trend = quantile_trend(pair.x, residual, valid)
            axis.plot(
                trend["x_mean"], trend["value_mean"], color=colors[model_name],
                lw=2, label=label,
            )
        axis.axhline(0.0, color="#555555", lw=1)
        axis.set_xlim(x_limits)
        combined_residuals = np.concatenate([
            (pair.y - predictions[name][pair.label])[valid]
            for name in predictions
        ])
        axis.set_ylim(np.percentile(combined_residuals, [0.5, 99.5]))
        axis.set_xlabel("X (block mean intensity)")
        axis.set_ylabel("OOF residual")
        axis.set_title(f"{pair.label}  residual vs X")
        axis.legend(fontsize=8, loc="best")
        figure.colorbar(density, ax=axis, label="M1 log count")

        residual_maps = {
            name: pair.y - model_predictions[pair.label]
            for name, model_predictions in predictions.items()
        }
        display_limit = float(np.percentile(np.abs(np.concatenate([
            values[valid] for values in residual_maps.values()
        ])), 99.5))
        for column, (model_name, title) in enumerate((
            ("M1", "M1"),
            ("Mblur", "Mblur"),
            ("Mquad_diagnostic", "Mquad diagnostic"),
        ), start=1):
            axis = axes[row_index, column]
            image = axis.imshow(
                residual_maps[model_name], cmap="coolwarm",
                vmin=-display_limit, vmax=display_limit, interpolation="nearest",
            )
            if np.any(pair.source_support) and not np.all(pair.source_support):
                axis.contour(
                    pair.source_support, levels=[0.5], colors="#222222",
                    linewidths=0.45,
                )
            metrics = comparisons[pair.label][model_name]["metrics"]
            suffix = (
                f"\nouter sigma={selected_sigmas}"
                if model_name == "Mblur" else ""
            )
            axis.set_title(
                f"{title} OOF residual{suffix}\n"
                f"R2={metrics['cv_r2']:.3f}, RMSE={metrics['rmse']:.2f}, "
                f"edge={metrics['source_edge_rmse']:.2f}",
                fontsize=9,
            )
            axis.set_xlabel("Block column")
            axis.set_ylabel("Block row")
            figure.colorbar(image, ax=axis, label="residual")

    figure.suptitle(
        "Frozen BG: M1 vs blur candidate and quadratic diagnostic", fontsize=14,
    )
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen M1, blur-candidate, and quadratic diagnostic comparison",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("outputs/pseudo_ghost_mechanism_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/pseudo_ghost_mechanism_v1/model_comparison"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = (
        (repo_root / args.dataset_dir).resolve()
        if not args.dataset_dir.is_absolute() else args.dataset_dir
    )
    output_dir = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute() else args.output_dir
    )
    pairs, source_manifest = load_frozen_pairs(dataset_dir)
    blur = nested_blur_oof(pairs)
    quadratic = quadratic_oof(pairs)
    predictions = {
        "M1": {pair.label: pair.prediction for pair in pairs},
        "Mblur": blur["predictions"],
        "Mquad_diagnostic": quadratic["predictions"],
    }
    comparisons, _ = compare_models(pairs, predictions)
    interpretation = interpret_candidates(comparisons, blur["outer_selections"])
    selected_sigmas = [
        row["selected_sigma_blocks"] for row in blur["outer_selections"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xray-candidate-diagnostics-") as temporary:
        temporary_dir = Path(temporary)
        figure_path = temporary_dir / "residual_diagnostics.png"
        json_path = temporary_dir / "model_results.json"
        render_diagnostics(
            figure_path, pairs, predictions, comparisons, selected_sigmas,
        )
        result = {
            "schema_version": 3,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "analysis_scope": "Mblur formal candidate plus Mquad diagnostic",
            "study_boundary": [
                "The frozen background and Y are read-only inputs.",
                "Mblur is the only formal successor candidate in this analysis.",
                "Mquad is diagnostic only and is not promoted to M2.",
                "No combined, spatially varying, or later-stage model is fitted.",
            ],
            "source_manifest": source_manifest,
            "code_provenance": {
                "path": "scripts/analyze_frozen_candidates.py",
                "sha256": sha256(Path(__file__).resolve()),
            },
            "M1": {
                "definition": "Y = alpha * X + intercept",
                "prediction": "Stored four-fold strict spatial OOF prediction",
            },
            "Mblur": {
                "definition": "Y = alpha * GaussianBlur(X, sigma) + intercept",
                "status": "formal_candidate",
                "candidate_sigma_blocks": blur["candidate_sigma_blocks"],
                "selection_scope": blur["selection_scope"],
                "selection_objective": blur["selection_objective"],
                "outer_selections": blur["outer_selections"],
                "fold_parameters": blur["fold_parameters"],
            },
            "Mquad_diagnostic": {
                "definition": (
                    "Y = a1 * standardized(X) + a2 * standardized(X)^2 + intercept"
                ),
                "status": quadratic["status"],
                "normalization": quadratic["normalization"],
                "fold_parameters": quadratic["fold_parameters"],
            },
            "pair_comparisons": comparisons,
            "interpretation": interpretation,
            "output_artifacts": {
                "residual_diagnostics": {
                    "relative_path": "residual_diagnostics.png",
                    "size_bytes": figure_path.stat().st_size,
                    "sha256": sha256(figure_path),
                },
            },
        }
        json_path.write_text(
            json.dumps(clean_json(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        shutil.copyfile(figure_path, output_dir / figure_path.name)
        shutil.copyfile(json_path, output_dir / json_path.name)

    print(json.dumps({
        "scope": result["analysis_scope"],
        "output": str((output_dir / "model_results.json").resolve()),
        "figure": str((output_dir / "residual_diagnostics.png").resolve()),
        "selected_sigma_blocks_by_outer_fold": selected_sigmas,
        "interpretation": interpretation,
        "candidate_changes": {
            label: {
                candidate: comparisons[label][candidate]["change_vs_M1"]
                for candidate in ("Mblur", "Mquad_diagnostic")
            }
            for label in PRIMARY_PAIRS
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
