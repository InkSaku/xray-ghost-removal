"""Diagnose low-complexity intensity nonlinearity on frozen pseudo-ghost data.

This is an exploratory conditional-mean study, not a detector-physics or clean-
ground-truth validation. M1 and observable Y are read-only inputs. Msat is the
primary candidate, Mhinge is an alternative, and Mquad plus a low-degree spline
are diagnostics only. No model is promoted to M2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_frozen_candidates import (
    PairData,
    load_frozen_pairs,
    model_metrics,
    quadratic_oof,
    quantile_trend,
    residual_trend_summary,
    sha256,
)
from scripts.analyze_pseudo_ghost_mechanism import (
    TRIM_PERCENTILES,
    _training_keep,
    clean_json,
)


PRIMARY_PAIRS = ("5->6", "27->28")
SHAPE_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
HINGE_QUANTILES = (0.15, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85)
SPLINE_INTERNAL_QUANTILES = (0.20, 0.40, 0.60, 0.80)
MAX_TREND_RMS_RATIO = 0.50
MAX_LOCAL_RMSE_INCREASE = 0.05
MAX_PARAMETER_CV = 0.50
MAX_BOUNDARY_SELECTIONS = 1


def source_magnitude(x: np.ndarray) -> np.ndarray:
    """Positive source magnitude without sending positive X into an exponent."""
    return np.maximum(-np.asarray(x, dtype=np.float64), 0.0)


def _scale_from_training(
    pair: PairData,
    train: np.ndarray,
    quantile: float,
) -> float:
    s = source_magnitude(pair.x)
    eligible = train & pair.source_support & (s > 0)
    if eligible.sum() < 20:
        eligible = train & (s > 0)
    if eligible.sum() < 20:
        raise ValueError(f"Too few positive-S training blocks for {pair.label}")
    return max(float(np.quantile(s[eligible], quantile)), 1e-6)


def _sat_design(x: np.ndarray, s0: float) -> np.ndarray:
    s = source_magnitude(x)
    saturation = -np.expm1(-s / s0)
    return np.column_stack([x, x * saturation, np.ones(x.size, dtype=float)])


def _hinge_design(x: np.ndarray, tau: float) -> np.ndarray:
    s = source_magnitude(x)
    return np.column_stack([
        x,
        -np.maximum(s - tau, 0.0),
        np.ones(x.size, dtype=float),
    ])


def _fit_parametric_holdout(
    pair: PairData,
    model: str,
    quantile: float,
    train_base: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    train, thresholds = _training_keep(
        pair.x, pair.y, train_base, TRIM_PERCENTILES,
    )
    scale = _scale_from_training(pair, train, quantile)
    design: Callable[[np.ndarray, float], np.ndarray]
    if model == "Msat":
        design = _sat_design
        names = ("alpha0", "alpha1", "intercept")
        scale_name = "S0"
    elif model == "Mhinge":
        design = _hinge_design
        names = ("base_alpha", "slope_change", "intercept")
        scale_name = "tau"
    else:
        raise ValueError(f"Unknown nonlinear model {model!r}")
    coefficients, *_ = np.linalg.lstsq(
        design(pair.x[train], scale), pair.y[train], rcond=None,
    )
    prediction = design(pair.x[test], scale) @ coefficients
    parameters = {name: float(value) for name, value in zip(names, coefficients)}
    parameters.update({
        "selected_training_quantile": float(quantile),
        scale_name: scale,
        "train_blocks": int(train.sum()),
        "test_blocks": int(test.sum()),
        "training_trim_thresholds": thresholds,
    })
    if model == "Msat":
        parameters["alpha_infinity"] = float(coefficients[0] + coefficients[1])
    else:
        parameters["strong_signal_slope"] = float(coefficients[0] + coefficients[1])
    return prediction, parameters


def _inner_score(
    pair: PairData,
    model: str,
    quantile: float,
    outer_fold: int,
) -> float:
    development = pair.evaluated & (pair.folds != outer_fold)
    prediction = np.full(pair.y.shape, np.nan, dtype=np.float64)
    inner_folds = sorted(int(value) for value in np.unique(pair.folds[development]))
    for inner_fold in inner_folds:
        test = development & (pair.folds == inner_fold)
        train_base = development & (pair.folds != inner_fold)
        prediction[test], _ = _fit_parametric_holdout(
            pair, model, quantile, train_base, test,
        )
    valid = development & np.isfinite(prediction)
    if valid.sum() < 20:
        raise ValueError(f"Too few inner-CV blocks for {pair.label} {model}")
    y = pair.y[valid]
    denominator = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    return 1.0 - float(np.sum((y - prediction[valid]) ** 2)) / denominator


def nested_parametric_oof(
    pairs: list[PairData],
    model: str,
    candidate_quantiles: tuple[float, ...],
) -> dict[str, Any]:
    """Select a pair-specific shape quantile by inner spatial CV."""
    predictions = {
        pair.label: np.full(pair.y.shape, np.nan, dtype=np.float64) for pair in pairs
    }
    fold_parameters = {pair.label: [] for pair in pairs}
    outer_selections = {pair.label: [] for pair in pairs}
    for pair in pairs:
        for outer_fold in range(4):
            candidates = [
                {
                    "training_quantile": float(quantile),
                    "inner_cv_r2": _inner_score(pair, model, quantile, outer_fold),
                }
                for quantile in candidate_quantiles
            ]
            selected = max(
                candidates,
                key=lambda row: (
                    row["inner_cv_r2"],
                    -abs(row["training_quantile"] - 0.5),
                ),
            )
            quantile = selected["training_quantile"]
            test = pair.evaluated & (pair.folds == outer_fold)
            train_base = pair.evaluated & (pair.folds != outer_fold)
            predictions[pair.label][test], parameters = _fit_parametric_holdout(
                pair, model, quantile, train_base, test,
            )
            boundary_hit = quantile in (candidate_quantiles[0], candidate_quantiles[-1])
            outer_selections[pair.label].append({
                "outer_fold": outer_fold,
                "selected_training_quantile": quantile,
                "boundary_hit": boundary_hit,
                "candidates": candidates,
            })
            fold_parameters[pair.label].append({
                "outer_fold": outer_fold,
                "boundary_hit": boundary_hit,
                **parameters,
            })
    return {
        "predictions": predictions,
        "fold_parameters": fold_parameters,
        "outer_selections": outer_selections,
        "candidate_training_quantiles": [float(value) for value in candidate_quantiles],
        "selection_scope": "Pair-specific shape quantile selected inside each outer fold",
        "selection_objective": "Inner spatial CV R2",
    }


def _spline_knots(x: np.ndarray) -> tuple[np.ndarray, list[float], float, float]:
    lower, upper = (float(value) for value in np.quantile(x, [0.0, 1.0]))
    internal = np.unique(np.quantile(x, SPLINE_INTERNAL_QUANTILES)).astype(float)
    internal = internal[(internal > lower) & (internal < upper)]
    if internal.size < 2 or upper <= lower:
        raise ValueError("Training X does not support the fixed cubic spline basis")
    degree = 3
    knots = np.concatenate([
        np.repeat(lower, degree + 1),
        internal,
        np.repeat(upper, degree + 1),
    ])
    return knots, internal.tolist(), lower, upper


def spline_oof(pairs: list[PairData]) -> dict[str, Any]:
    """Fixed-low-DF cubic B-spline diagnostic fitted strictly out of fold."""
    predictions = {
        pair.label: np.full(pair.y.shape, np.nan, dtype=np.float64) for pair in pairs
    }
    fold_parameters = {pair.label: [] for pair in pairs}
    for pair in pairs:
        for outer_fold in range(4):
            test = pair.evaluated & (pair.folds == outer_fold)
            train_base = pair.evaluated & (pair.folds != outer_fold)
            train, thresholds = _training_keep(
                pair.x, pair.y, train_base, TRIM_PERCENTILES,
            )
            knots, internal, lower, upper = _spline_knots(pair.x[train])
            train_x = np.clip(pair.x[train], lower, upper)
            test_x = np.clip(pair.x[test], lower, upper)
            train_design = BSpline.design_matrix(
                train_x, knots, 3, extrapolate=False,
            ).toarray()
            test_design = BSpline.design_matrix(
                test_x, knots, 3, extrapolate=False,
            ).toarray()
            coefficients, *_ = np.linalg.lstsq(
                train_design, pair.y[train], rcond=None,
            )
            predictions[pair.label][test] = test_design @ coefficients
            fold_parameters[pair.label].append({
                "outer_fold": outer_fold,
                "internal_knots": internal,
                "training_x_min": lower,
                "training_x_max": upper,
                "basis_columns": int(train_design.shape[1]),
                "coefficients": coefficients.tolist(),
                "test_values_clipped_to_training_range": int(np.sum(
                    (pair.x[test] < lower) | (pair.x[test] > upper)
                )),
                "train_blocks": int(train.sum()),
                "test_blocks": int(test.sum()),
                "training_trim_thresholds": thresholds,
            })
    return {
        "predictions": predictions,
        "fold_parameters": fold_parameters,
        "status": "shape_diagnostic_only",
        "internal_knot_training_quantiles": list(SPLINE_INTERNAL_QUANTILES),
    }


def _relative_reduction(reference: float, candidate: float) -> float:
    return 1.0 - candidate / max(reference, 1e-12)


def compare_models(
    pairs: list[PairData],
    predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    results = {}
    for pair in pairs:
        rows = {}
        for model, model_predictions in predictions.items():
            rows[model] = {
                "metrics": model_metrics(pair, model_predictions[pair.label]),
                "residual_vs_x": residual_trend_summary(
                    pair, model_predictions[pair.label],
                ),
            }
        reference_metrics = rows["M1"]["metrics"]
        reference_trend = rows["M1"]["residual_vs_x"]
        for model in predictions:
            if model == "M1":
                continue
            metrics = rows[model]["metrics"]
            trend = rows[model]["residual_vs_x"]
            rows[model]["change_vs_M1"] = {
                "delta_cv_r2": metrics["cv_r2"] - reference_metrics["cv_r2"],
                "rmse_reduction": reference_metrics["rmse"] - metrics["rmse"],
                "source_edge_rmse_relative_reduction": _relative_reduction(
                    reference_metrics["source_edge_rmse"], metrics["source_edge_rmse"],
                ),
                "source_interior_rmse_relative_reduction": _relative_reduction(
                    reference_metrics["source_interior_rmse"],
                    metrics["source_interior_rmse"],
                ),
                "outside_source_rmse_relative_reduction": _relative_reduction(
                    reference_metrics["outside_source_rmse"],
                    metrics["outside_source_rmse"],
                ),
                "residual_neighbor_correlation_reduction": (
                    reference_metrics["residual_neighbor_correlation"]
                    - metrics["residual_neighbor_correlation"]
                ),
                "binned_mean_peak_to_peak_ratio": (
                    trend["binned_mean_peak_to_peak"]
                    / max(reference_trend["binned_mean_peak_to_peak"], 1e-12)
                ),
                "binned_mean_weighted_rms_ratio": (
                    trend["binned_mean_weighted_rms"]
                    / max(reference_trend["binned_mean_weighted_rms"], 1e-12)
                ),
            }
        results[pair.label] = rows
    return results


def parameter_stability(
    model: str,
    fold_parameters: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    scale_name = "S0" if model == "Msat" else "tau"
    result = {}
    for label, rows in fold_parameters.items():
        values = np.asarray([row[scale_name] for row in rows], dtype=float)
        cv = float(np.std(values, ddof=1) / max(abs(np.mean(values)), 1e-12))
        boundary_hits = int(sum(bool(row["boundary_hit"]) for row in rows))
        result[label] = {
            "parameter": scale_name,
            "fold_values": values.tolist(),
            "fold_cv": cv,
            "boundary_selection_count": boundary_hits,
            "stable": bool(
                cv <= MAX_PARAMETER_CV
                and boundary_hits <= MAX_BOUNDARY_SELECTIONS
            ),
        }
    return result


def interpret_candidates(
    comparisons: dict[str, Any],
    stability: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pair_checks = {}
    for model in ("Msat", "Mhinge"):
        pair_checks[model] = {}
        for label in PRIMARY_PAIRS:
            change = comparisons[label][model]["change_vs_M1"]
            checks = {
                "cv_r2_improved": change["delta_cv_r2"] > 0,
                "residual_trend_weighted_rms_halved": (
                    change["binned_mean_weighted_rms_ratio"]
                    <= MAX_TREND_RMS_RATIO
                ),
                "source_edge_rmse_not_worse_over_5_percent": (
                    change["source_edge_rmse_relative_reduction"]
                    >= -MAX_LOCAL_RMSE_INCREASE
                ),
                "outside_source_rmse_not_worse_over_5_percent": (
                    change["outside_source_rmse_relative_reduction"]
                    >= -MAX_LOCAL_RMSE_INCREASE
                ),
                "shape_parameter_stable": stability[model][label]["stable"],
            }
            pair_checks[model][label] = {
                "checks": checks,
                "passed": bool(all(checks.values())),
            }
    cross_pair = {
        model: bool(all(pair_checks[model][label]["passed"] for label in PRIMARY_PAIRS))
        for model in ("Msat", "Mhinge")
    }
    return {
        "pair_checks": pair_checks,
        "cross_pair_exploratory_evidence": cross_pair,
        "primary_candidate": "Msat",
        "alternative_candidate": "Mhinge",
        "model_promoted_to_M2": False,
        "interpretation": (
            "Exploratory conditional-mean evidence only; independent acquisition is "
            "required before any detector-physics or exposure-dependence claim."
        ),
        "thresholds": {
            "maximum_residual_trend_weighted_rms_ratio": MAX_TREND_RMS_RATIO,
            "maximum_local_rmse_relative_increase": MAX_LOCAL_RMSE_INCREASE,
            "maximum_shape_parameter_fold_cv": MAX_PARAMETER_CV,
            "maximum_boundary_selections": MAX_BOUNDARY_SELECTIONS,
        },
    }


def render_diagnostics(
    path: Path,
    pairs: list[PairData],
    predictions: dict[str, dict[str, np.ndarray]],
    comparisons: dict[str, Any],
) -> None:
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(len(pairs), 5, figsize=(20, 8), constrained_layout=True)
    colors = {
        "M1": "#d95f02",
        "Mquad_diagnostic": "#7570b3",
        "Mspline_diagnostic": "#666666",
        "Msat": "#1b9e77",
        "Mhinge": "#e7298a",
    }
    labels = {
        "M1": "M1",
        "Mquad_diagnostic": "Mquad diag",
        "Mspline_diagnostic": "spline diag",
        "Msat": "Msat primary",
        "Mhinge": "Mhinge alternative",
    }
    for row_index, pair in enumerate(pairs):
        valid = pair.evaluated
        x = pair.x[valid]
        x_limits = np.percentile(x, [0.5, 99.5])

        axis = axes[row_index, 0]
        y_limits = np.percentile(pair.y[valid], [0.5, 99.5])
        density_keep = (
            (x >= x_limits[0]) & (x <= x_limits[1])
            & (pair.y[valid] >= y_limits[0]) & (pair.y[valid] <= y_limits[1])
        )
        density = axis.hexbin(
            x[density_keep], pair.y[valid][density_keep], gridsize=55,
            mincnt=1, cmap="Greys", bins="log",
        )
        observed = quantile_trend(pair.x, pair.y, valid)
        axis.plot(
            observed["x_mean"], observed["value_mean"], color="#111111",
            lw=2.5, label="observed bin mean",
        )
        for model in predictions:
            trend = quantile_trend(pair.x, predictions[model][pair.label], valid)
            axis.plot(
                trend["x_mean"], trend["value_mean"], color=colors[model],
                lw=1.6, label=labels[model],
            )
        axis.set_xlim(x_limits)
        axis.set_xlabel("X (block mean contrast)")
        axis.set_ylabel("Y / OOF prediction")
        axis.set_title(f"{pair.label} conditional shape")
        axis.legend(fontsize=7, loc="best")
        figure.colorbar(density, ax=axis, label="log count")

        axis = axes[row_index, 1]
        combined = []
        for model in predictions:
            residual = pair.y - predictions[model][pair.label]
            combined.append(residual[valid])
            trend = quantile_trend(pair.x, residual, valid)
            axis.plot(
                trend["x_mean"], trend["value_mean"], color=colors[model],
                lw=1.8, label=labels[model],
            )
        axis.axhline(0.0, color="#555555", lw=1)
        axis.set_xlim(x_limits)
        axis.set_ylim(np.percentile(np.concatenate(combined), [0.5, 99.5]))
        axis.set_xlabel("X (block mean contrast)")
        axis.set_ylabel("OOF residual bin mean")
        axis.set_title(f"{pair.label} residual trend")
        axis.legend(fontsize=7, loc="best")

        map_models = ("M1", "Msat", "Mhinge")
        residual_maps = {
            model: pair.y - predictions[model][pair.label] for model in map_models
        }
        display_limit = float(np.percentile(np.abs(np.concatenate([
            residual_maps[model][valid] for model in map_models
        ])), 99.5))
        for column, model in enumerate(map_models, start=2):
            axis = axes[row_index, column]
            image = axis.imshow(
                residual_maps[model], cmap="coolwarm", vmin=-display_limit,
                vmax=display_limit, interpolation="nearest",
            )
            if np.any(pair.source_support) and not np.all(pair.source_support):
                axis.contour(
                    pair.source_support, levels=[0.5], colors="#222222",
                    linewidths=0.45,
                )
            metrics = comparisons[pair.label][model]["metrics"]
            axis.set_title(
                f"{labels[model]} OOF residual\n"
                f"R2={metrics['cv_r2']:.3f}, RMSE={metrics['rmse']:.2f}, "
                f"outside={metrics['outside_source_rmse']:.2f}",
                fontsize=9,
            )
            axis.set_xlabel("Block column")
            axis.set_ylabel("Block row")
            figure.colorbar(image, ax=axis, label="residual")

    figure.suptitle(
        "Frozen observable Y: low-complexity intensity nonlinearity diagnostics",
        fontsize=14,
    )
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen conditional-mean intensity nonlinearity diagnostics",
    )
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("outputs/pseudo_ghost_mechanism_v1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/pseudo_ghost_mechanism_v1/intensity_nonlinearity"),
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
    quadratic = quadratic_oof(pairs)
    spline = spline_oof(pairs)
    saturation = nested_parametric_oof(pairs, "Msat", SHAPE_QUANTILES)
    hinge = nested_parametric_oof(pairs, "Mhinge", HINGE_QUANTILES)
    predictions = {
        "M1": {pair.label: pair.prediction for pair in pairs},
        "Mquad_diagnostic": quadratic["predictions"],
        "Mspline_diagnostic": spline["predictions"],
        "Msat": saturation["predictions"],
        "Mhinge": hinge["predictions"],
    }
    comparisons = compare_models(pairs, predictions)
    stability = {
        "Msat": parameter_stability("Msat", saturation["fold_parameters"]),
        "Mhinge": parameter_stability("Mhinge", hinge["fold_parameters"]),
    }
    interpretation = interpret_candidates(comparisons, stability)

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xray-intensity-nonlinearity-") as temporary:
        temporary_dir = Path(temporary)
        figure_path = temporary_dir / "nonlinearity_diagnostics.png"
        json_path = temporary_dir / "model_results.json"
        render_diagnostics(figure_path, pairs, predictions, comparisons)
        result = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "analysis_scope": (
                "Exploratory conditional-mean intensity nonlinearity on frozen Y"
            ),
            "study_boundary": [
                "Frozen background, X, observable Y, evaluation masks, and spatial folds are read-only.",
                "X is block-mean source contrast, not incident exposure.",
                "Y is an observable pseudo-ghost signal, not clean-ground-truth lag.",
                "Msat is primary and Mhinge alternative; model class is not selected on outer results.",
                "Mquad and Mspline are diagnostics only.",
                "No candidate is promoted to M2 without independent acquisition.",
            ],
            "source_manifest": source_manifest,
            "code_provenance": {
                "path": "scripts/analyze_intensity_nonlinearity.py",
                "sha256": sha256(Path(__file__).resolve()),
                "design_path": "docs/plans/2026-09-21-intensity-nonlinearity-design.md",
                "design_sha256": sha256(
                    repo_root / "docs/plans/2026-09-21-intensity-nonlinearity-design.md"
                ),
            },
            "models": {
                "M1": {
                    "definition": "Y = alpha * X + intercept",
                    "status": "frozen_reference",
                },
                "Mquad_diagnostic": {
                    "definition": "Y = a1*z(X) + a2*z(X)^2 + intercept",
                    "status": "diagnostic_only",
                    "fold_parameters": quadratic["fold_parameters"],
                },
                "Mspline_diagnostic": {
                    "definition": "Fixed-low-DF cubic B-spline E[Y|X]",
                    "status": spline["status"],
                    "internal_knot_training_quantiles": (
                        spline["internal_knot_training_quantiles"]
                    ),
                    "fold_parameters": spline["fold_parameters"],
                },
                "Msat": {
                    "definition": (
                        "S=max(-X,0); Y=b+X*[alpha0+alpha1*(1-exp(-S/S0))]"
                    ),
                    "status": "primary_exploratory_candidate",
                    **{key: value for key, value in saturation.items()
                       if key != "predictions"},
                    "parameter_stability": stability["Msat"],
                },
                "Mhinge": {
                    "definition": "S=max(-X,0); Y=b+a*X-c*max(0,S-tau)",
                    "status": "alternative_exploratory_candidate",
                    **{key: value for key, value in hinge.items()
                       if key != "predictions"},
                    "parameter_stability": stability["Mhinge"],
                },
            },
            "pair_comparisons": comparisons,
            "interpretation": interpretation,
            "output_artifacts": {
                "nonlinearity_diagnostics": {
                    "relative_path": "nonlinearity_diagnostics.png",
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

    summary = {
        "scope": result["analysis_scope"],
        "output": str((output_dir / "model_results.json").resolve()),
        "figure": str((output_dir / "nonlinearity_diagnostics.png").resolve()),
        "interpretation": interpretation,
        "candidate_changes": {
            label: {
                model: comparisons[label][model]["change_vs_M1"]
                for model in ("Msat", "Mhinge")
            }
            for label in PRIMARY_PAIRS
        },
    }
    print(json.dumps(clean_json(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
