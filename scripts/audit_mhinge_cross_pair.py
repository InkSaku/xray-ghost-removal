"""Audit pre-frozen Mhinge reproducibility across the 13-pair single-lag cohort."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_dark_light_pairs import load_exposures
from scripts.analyze_frozen_candidates import (
    PairData,
    model_metrics,
    residual_trend_summary,
    sha256,
)
from scripts.analyze_intensity_nonlinearity import (
    HINGE_QUANTILES,
    _hinge_design,
    nested_parametric_oof,
    source_magnitude,
)
from scripts.analyze_pseudo_ghost_mechanism import (
    TRIM_PERCENTILES,
    _training_keep,
    clean_json,
)


ALL_PAIRS = (
    "1->2", "2->3", "3->4", "4->5", "5->6", "6->7", "7->8",
    "8->9", "9->10", "22->23", "23->24", "25->26", "27->28",
)
DISCOVERY_PAIRS = ("5->6", "27->28")
EXTENSION_PAIRS = ("1->2", "3->4", "7->8", "8->9", "9->10", "22->23", "23->24")
WEAK_REFERENCE_PAIRS = ("2->3", "25->26")
NEGATIVE_CONTROL_PAIRS = ("4->5", "6->7")
ELIGIBLE_PAIRS = tuple(
    label for label in ALL_PAIRS if label in DISCOVERY_PAIRS + EXTENSION_PAIRS
)

COHORT_AUDIT_SHA256 = "21b544674cb8dba66ecc0df464fd0924fe968af8b72ee3786bbe101077c487f3"
FROZEN_BACKGROUND_SHA256 = "313ad0dafc48102da57131f77534117bba366fb8893eed6cf845be1d65e4e50e"
FROZEN_MASK_SHA256 = "6a2f43410c4d92023ff7f5df72e01af31f2bb20d51feb67fabd30ad7c9658ab6"
ALPHA_ELIGIBILITY_MIN = 1e-4
MAX_TREND_RMS_RATIO = 0.50
MAX_LOCAL_RMSE_INCREASE = 0.05
MAX_PARAMETER_CV = 0.50
MAX_BOUNDARY_SELECTIONS = 1
MAX_ALPHA = 0.02
MIN_SEGMENT_BLOCKS = 50
MIN_SEGMENT_FRACTION = 0.05
MIN_DYNAMIC_RANGE_FRACTION = 0.05
MAX_STANDARDIZED_CONDITION_NUMBER = 100.0
MIN_EXTENSION_PASS_COUNT = 5
MIN_DIRECTION_COUNT = 6
BOOTSTRAP_REPETITIONS = 10_000
RANDOM_SEED = 20260921


def cohort_role(label: str) -> str:
    if label in DISCOVERY_PAIRS:
        return "discovery_reference"
    if label in EXTENSION_PAIRS:
        return "extension_primary"
    if label in WEAK_REFERENCE_PAIRS:
        return "weak_signal_reference"
    if label in NEGATIVE_CONTROL_PAIRS:
        return "negative_control"
    raise ValueError(f"Unrecognized pair {label}")


def _manifest_fit(row: dict[str, Any]) -> dict[str, Any]:
    blocks = [item for item in row["block_results"] if item["block_size"] == 16]
    if len(blocks) != 1:
        raise ValueError("Every pair must have exactly one block-16 result")
    backgrounds = [
        item for item in blocks[0]["background_results"]
        if item["background_mode"] == "masked_hybrid_spline"
    ]
    if len(backgrounds) != 1:
        raise ValueError("Every pair must have one frozen background result")
    masks = [item for item in backgrounds[0]["mask_results"] if item["mask_mode"] == "all"]
    if len(masks) != 1:
        raise ValueError("Every pair must have one all-block fit")
    return masks[0]


def load_audit_pairs(
    dataset_dir: Path,
    cohort_audit_path: Path,
) -> tuple[list[PairData], dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = dataset_dir / "analysis_results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = manifest.get("frozen_background") or {}
    if not frozen.get("verified"):
        raise ValueError("Cross-pair export is not frozen-background verified")
    if frozen.get("sha256") != FROZEN_BACKGROUND_SHA256:
        raise ValueError("Frozen background hash differs from the predeclared design")
    if frozen.get("mask_archive_sha256") != FROZEN_MASK_SHA256:
        raise ValueError("Frozen mask hash differs from the predeclared design")
    configuration = manifest["configuration"]
    manifest_pairs = tuple(f"{a}->{b}" for a, b in configuration["pairs"])
    if manifest_pairs != ALL_PAIRS:
        raise ValueError("Cross-pair export does not contain the predeclared 13 pairs")
    if configuration["block_sizes"] != [16]:
        raise ValueError("Cross-pair export must use block size 16 only")
    if configuration["background_modes"] != ["masked_hybrid_spline"]:
        raise ValueError("Cross-pair export must use the frozen background only")

    actual_audit_hash = sha256(cohort_audit_path)
    if actual_audit_hash != COHORT_AUDIT_SHA256:
        raise ValueError("Prior cohort audit hash differs from the predeclared design")
    cohort_audit = json.loads(cohort_audit_path.read_text(encoding="utf-8"))
    nominal = cohort_audit["summary"]["pair_results_by_configuration"]["nominal"]
    audit_eligible = tuple(
        label for label in ALL_PAIRS
        if nominal[label]["detection_passed"]
        and abs(float(nominal[label]["alpha"])) >= ALPHA_ELIGIBILITY_MIN
    )
    if audit_eligible != ELIGIBLE_PAIRS:
        raise ValueError(
            f"Prior audit eligible cohort changed: {audit_eligible} != {ELIGIBLE_PAIRS}"
        )

    pairs = []
    gate_snapshot = {}
    saturation_maps = {}
    for row in manifest["pair_results"]:
        label = f"{row['light_index']}->{row['dark_index']}"
        artifact = row["output_artifacts"]["primary_maps"]
        path = dataset_dir / artifact["relative_path"]
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"Frozen NPZ hash mismatch for {label}")
        with np.load(path) as archive:
            required = {
                "source", "observable_signal", "raw_dark", "background",
                "source_support", "source_saturation_fraction", "evaluated_mask",
                "oof_prediction", "oof_residual", "spatial_folds", "background_mode",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"Frozen pair {label} is missing {sorted(missing)}")
            x = archive["source"].astype(np.float64)
            y = archive["observable_signal"].astype(np.float64)
            raw_dark = archive["raw_dark"].astype(np.float64)
            background = archive["background"].astype(np.float64)
            folds = archive["spatial_folds"].astype(np.int8)
            evaluated = archive["evaluated_mask"].astype(bool)
            support = archive["source_support"].astype(bool)
            saturation = archive["source_saturation_fraction"].astype(np.float64)
            prediction = archive["oof_prediction"].astype(np.float64)
            residual = archive["oof_residual"].astype(np.float64)
            background_mode = str(archive["background_mode"])
        if background_mode != "masked_hybrid_spline":
            raise ValueError(f"Wrong background mode for {label}")
        if not np.allclose(y, raw_dark - background, atol=2e-4, rtol=0.0):
            raise ValueError(f"Frozen Y identity failed for {label}")
        valid = evaluated & np.isfinite(x) & np.isfinite(y) & np.isfinite(prediction)
        if set(np.unique(folds[valid]).tolist()) != {0, 1, 2, 3}:
            raise ValueError(f"Frozen folds are incomplete for {label}")
        if not np.allclose(residual[valid], y[valid] - prediction[valid], atol=2e-5):
            raise ValueError(f"Frozen residual identity failed for {label}")
        pair = PairData(
            label=label,
            x=x,
            y=y,
            folds=folds,
            evaluated=valid,
            source_support=support,
            prediction=prediction,
            residual=residual,
        )
        pairs.append(pair)
        saturation_maps[label] = saturation
        current_gate = _manifest_fit(row)["detection_gate"]
        gate_snapshot[label] = {
            "prior_nominal_detection_passed": bool(nominal[label]["detection_passed"]),
            "prior_nominal_alpha": float(nominal[label]["alpha"]),
            "current_export_detection_passed": bool(current_gate["passed"]),
        }
    pairs.sort(key=lambda pair: ALL_PAIRS.index(pair.label))
    if tuple(pair.label for pair in pairs) != ALL_PAIRS:
        raise ValueError("Manifest pair order/content differs from the predeclared cohort")
    return pairs, {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "frozen_background": frozen,
        "cohort_audit_path": str(cohort_audit_path.resolve()),
        "cohort_audit_sha256": actual_audit_hash,
    }, gate_snapshot, saturation_maps


def _cv(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1) / max(abs(np.mean(array)), 1e-12))


def _standardized_condition_number(design: np.ndarray) -> float:
    columns = design[:, :2].astype(float)
    scales = np.std(columns, axis=0)
    if np.any(scales <= 1e-12):
        return float("inf")
    standardized = (columns - np.mean(columns, axis=0)) / scales
    standardized = np.column_stack([standardized, np.ones(standardized.shape[0])])
    return float(np.linalg.cond(standardized))


def hinge_identifiability(
    pair: PairData,
    fold_parameters: list[dict[str, Any]],
) -> dict[str, Any]:
    fold_rows = []
    for parameters in fold_parameters:
        outer_fold = int(parameters["outer_fold"])
        tau = float(parameters["tau"])
        train_base = pair.evaluated & (pair.folds != outer_fold)
        train, _ = _training_keep(pair.x, pair.y, train_base, TRIM_PERCENTILES)
        test = pair.evaluated & (pair.folds == outer_fold)
        s = source_magnitude(pair.x)
        train_support = train & pair.source_support & (s > 0)
        test_support = test & pair.source_support & (s > 0)
        if train_support.sum() < 100:
            raise ValueError(f"Too few source-support training blocks for {pair.label}")
        support_s = s[train_support]
        p05, p50, p90, p95 = (
            float(value) for value in np.quantile(support_s, [0.05, 0.50, 0.90, 0.95])
        )
        dynamic_range = max(p95 - p05, 1e-12)
        train_low = train_support & (s <= tau)
        train_high = train_support & (s > tau)
        test_low = test_support & (s <= tau)
        test_high = test_support & (s > tau)
        train_count = int(train_support.sum())
        test_count = int(test_support.sum())
        alpha_low = float(parameters["base_alpha"])
        alpha_high = float(parameters["strong_signal_slope"])
        delta_alpha = float(parameters["slope_change"])
        fold_rows.append({
            "outer_fold": outer_fold,
            "tau_S": tau,
            "tau_X": -tau,
            "selected_training_quantile": float(parameters["selected_training_quantile"]),
            "boundary_hit": bool(parameters["boundary_hit"]),
            "tau_empirical_training_quantile": float(np.mean(support_s <= tau)),
            "tau_over_S90": tau / max(p90, 1e-12),
            "support_S_p05": p05,
            "support_S_p50": p50,
            "support_S_p90": p90,
            "support_S_p95": p95,
            "low_dynamic_range_fraction": (tau - p05) / dynamic_range,
            "high_dynamic_range_fraction": (p95 - tau) / dynamic_range,
            "train_low_blocks": int(train_low.sum()),
            "train_high_blocks": int(train_high.sum()),
            "train_low_fraction": float(train_low.sum() / train_count),
            "train_high_fraction": float(train_high.sum() / train_count),
            "test_low_blocks": int(test_low.sum()),
            "test_high_blocks": int(test_high.sum()),
            "test_low_fraction": float(test_low.sum() / max(test_count, 1)),
            "test_high_fraction": float(test_high.sum() / max(test_count, 1)),
            "standardized_design_condition_number": _standardized_condition_number(
                _hinge_design(pair.x[train], tau)
            ),
            "alpha_low": alpha_low,
            "alpha_high": alpha_high,
            "delta_alpha": delta_alpha,
            "direction": "increase" if delta_alpha > 0 else "decrease" if delta_alpha < 0 else "zero",
        })

    tau_values = [row["tau_S"] for row in fold_rows]
    alpha_low_values = [row["alpha_low"] for row in fold_rows]
    alpha_high_values = [row["alpha_high"] for row in fold_rows]
    directions = [row["direction"] for row in fold_rows]
    checks = {
        "tau_fold_cv_at_most_0_50": _cv(tau_values) <= MAX_PARAMETER_CV,
        "boundary_selection_count_at_most_1": (
            sum(bool(row["boundary_hit"]) for row in fold_rows) <= MAX_BOUNDARY_SELECTIONS
        ),
        "physical_positive_bounded_slopes": all(
            0 < row["alpha_low"] <= MAX_ALPHA and 0 < row["alpha_high"] <= MAX_ALPHA
            for row in fold_rows
        ),
        "alpha_low_fold_cv_at_most_0_50": _cv(alpha_low_values) <= MAX_PARAMETER_CV,
        "alpha_high_fold_cv_at_most_0_50": _cv(alpha_high_values) <= MAX_PARAMETER_CV,
        "slope_change_direction_consistent": len(set(directions)) == 1 and directions[0] != "zero",
        "training_segment_support_sufficient": all(
            row["train_low_blocks"] >= MIN_SEGMENT_BLOCKS
            and row["train_high_blocks"] >= MIN_SEGMENT_BLOCKS
            and row["train_low_fraction"] >= MIN_SEGMENT_FRACTION
            and row["train_high_fraction"] >= MIN_SEGMENT_FRACTION
            for row in fold_rows
        ),
        "training_dynamic_range_sufficient": all(
            row["low_dynamic_range_fraction"] >= MIN_DYNAMIC_RANGE_FRACTION
            and row["high_dynamic_range_fraction"] >= MIN_DYNAMIC_RANGE_FRACTION
            for row in fold_rows
        ),
        "standardized_condition_number_at_most_100": all(
            row["standardized_design_condition_number"]
            <= MAX_STANDARDIZED_CONDITION_NUMBER
            for row in fold_rows
        ),
    }
    return {
        "folds": fold_rows,
        "summary": {
            "tau_S_median": float(np.median(tau_values)),
            "tau_X_median": -float(np.median(tau_values)),
            "tau_fold_cv": _cv(tau_values),
            "boundary_selection_count": int(sum(row["boundary_hit"] for row in fold_rows)),
            "alpha_low_median": float(np.median(alpha_low_values)),
            "alpha_low_fold_cv": _cv(alpha_low_values),
            "alpha_high_median": float(np.median(alpha_high_values)),
            "alpha_high_fold_cv": _cv(alpha_high_values),
            "delta_alpha_median": float(np.median([
                row["delta_alpha"] for row in fold_rows
            ])),
            "direction": directions[0] if len(set(directions)) == 1 else "inconsistent",
            "tau_empirical_quantile_median": float(np.median([
                row["tau_empirical_training_quantile"] for row in fold_rows
            ])),
            "tau_over_S90_median": float(np.median([
                row["tau_over_S90"] for row in fold_rows
            ])),
            "max_standardized_condition_number": float(max(
                row["standardized_design_condition_number"] for row in fold_rows
            )),
            "min_low_dynamic_range_fraction": float(min(
                row["low_dynamic_range_fraction"] for row in fold_rows
            )),
            "min_high_dynamic_range_fraction": float(min(
                row["high_dynamic_range_fraction"] for row in fold_rows
            )),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _relative_reduction(reference: float, candidate: float) -> float:
    return 1.0 - candidate / max(reference, 1e-12)


def pair_comparison(
    pair: PairData,
    hinge_prediction: np.ndarray,
    identifiability: dict[str, Any],
) -> dict[str, Any]:
    m1_metrics = model_metrics(pair, pair.prediction)
    hinge_metrics = model_metrics(pair, hinge_prediction)
    m1_trend = residual_trend_summary(pair, pair.prediction)
    hinge_trend = residual_trend_summary(pair, hinge_prediction)
    change = {
        "delta_cv_r2": hinge_metrics["cv_r2"] - m1_metrics["cv_r2"],
        "rmse_reduction": m1_metrics["rmse"] - hinge_metrics["rmse"],
        "rmse_relative_reduction": _relative_reduction(
            m1_metrics["rmse"], hinge_metrics["rmse"],
        ),
        "residual_trend_weighted_rms_ratio": (
            hinge_trend["binned_mean_weighted_rms"]
            / max(m1_trend["binned_mean_weighted_rms"], 1e-12)
        ),
        "source_edge_rmse_relative_reduction": _relative_reduction(
            m1_metrics["source_edge_rmse"], hinge_metrics["source_edge_rmse"],
        ),
        "outside_source_rmse_relative_reduction": _relative_reduction(
            m1_metrics["outside_source_rmse"], hinge_metrics["outside_source_rmse"],
        ),
    }
    predictive_checks = {
        "cv_r2_improved": change["delta_cv_r2"] > 0,
        "rmse_improved": change["rmse_reduction"] > 0,
        "residual_trend_weighted_rms_halved": (
            change["residual_trend_weighted_rms_ratio"] <= MAX_TREND_RMS_RATIO
        ),
        "source_edge_rmse_not_worse_over_5_percent": (
            change["source_edge_rmse_relative_reduction"] >= -MAX_LOCAL_RMSE_INCREASE
        ),
        "outside_source_rmse_not_worse_over_5_percent": (
            change["outside_source_rmse_relative_reduction"] >= -MAX_LOCAL_RMSE_INCREASE
        ),
    }
    return {
        "cohort_role": cohort_role(pair.label),
        "M1": {"metrics": m1_metrics, "residual_vs_x": m1_trend},
        "Mhinge": {"metrics": hinge_metrics, "residual_vs_x": hinge_trend},
        "change_vs_M1": change,
        "identifiability": identifiability,
        "predictive_checks": predictive_checks,
        "passed_all_checks": bool(
            all(predictive_checks.values()) and identifiability["passed"]
        ),
    }


def bootstrap_median(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    samples = rng.choice(array, size=(BOOTSTRAP_REPETITIONS, array.size), replace=True)
    medians = np.median(samples, axis=1)
    return {
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": RANDOM_SEED,
        "median": float(np.median(array)),
        "percentile_2_5": float(np.percentile(medians, 2.5)),
        "percentile_97_5": float(np.percentile(medians, 97.5)),
        "interpretation": "Descriptive pair-resampling sensitivity interval; pairs are not independent acquisitions.",
    }


def aggregate_pairs(labels: tuple[str, ...], rows: dict[str, Any]) -> dict[str, Any]:
    deltas = [float(rows[label]["change_vs_M1"]["delta_cv_r2"]) for label in labels]
    directions = [rows[label]["identifiability"]["summary"]["direction"] for label in labels]
    direction_counts = {
        direction: int(sum(value == direction for value in directions))
        for direction in ("increase", "decrease", "inconsistent")
    }
    return {
        "pairs": list(labels),
        "pair_count": len(labels),
        "passed_count": int(sum(rows[label]["passed_all_checks"] for label in labels)),
        "passed_pairs": [label for label in labels if rows[label]["passed_all_checks"]],
        "delta_cv_r2": {
            "values": dict(zip(labels, deltas)),
            "median": float(np.median(deltas)),
            "bootstrap": bootstrap_median(deltas),
        },
        "slope_direction_counts": direction_counts,
    }


def extension_sensitivity(rows: dict[str, Any]) -> dict[str, Any]:
    leave_one_out = []
    for excluded in EXTENSION_PAIRS:
        included = tuple(label for label in EXTENSION_PAIRS if label != excluded)
        deltas = [rows[label]["change_vs_M1"]["delta_cv_r2"] for label in included]
        leave_one_out.append({
            "excluded_pair": excluded,
            "remaining_median_delta_cv_r2": float(np.median(deltas)),
            "remaining_passed_count": int(sum(
                rows[label]["passed_all_checks"] for label in included
            )),
            "remaining_pair_count": len(included),
        })
    best = max(
        EXTENSION_PAIRS,
        key=lambda label: rows[label]["change_vs_M1"]["delta_cv_r2"],
    )
    best_row = next(row for row in leave_one_out if row["excluded_pair"] == best)
    return {
        "leave_one_pair_out": leave_one_out,
        "largest_delta_cv_r2_pair": best,
        "remove_largest_effect": best_row,
    }


def cross_pair_decision(
    extension: dict[str, Any],
    sensitivity: dict[str, Any],
) -> dict[str, Any]:
    directions = extension["slope_direction_counts"]
    checks = {
        "at_least_5_of_7_pairs_pass": extension["passed_count"] >= MIN_EXTENSION_PASS_COUNT,
        "median_delta_cv_r2_positive": extension["delta_cv_r2"]["median"] > 0,
        "bootstrap_lower_bound_positive": (
            extension["delta_cv_r2"]["bootstrap"]["percentile_2_5"] > 0
        ),
        "all_leave_one_pair_out_medians_positive": all(
            row["remaining_median_delta_cv_r2"] > 0
            for row in sensitivity["leave_one_pair_out"]
        ),
        "at_least_6_of_7_slope_directions_match": (
            max(directions["increase"], directions["decrease"]) >= MIN_DIRECTION_COUNT
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "model_promoted_to_physical_M2": False,
        "interpretation": (
            "Same-session pre-frozen model-form reproducibility only; not independent-acquisition or detector-physics validation."
        ),
    }


def descriptive_context(
    pair: PairData,
    exposures: dict[int, dict[str, float]],
    saturation: np.ndarray,
) -> dict[str, Any]:
    light_index = int(pair.label.split("->")[0])
    s = source_magnitude(pair.x)
    selected = pair.evaluated & pair.source_support & (s > 0)
    values = s[selected]
    return {
        **exposures[light_index],
        "support_S_p05": float(np.quantile(values, 0.05)),
        "support_S_p50": float(np.quantile(values, 0.50)),
        "support_S_p90": float(np.quantile(values, 0.90)),
        "support_S_p95": float(np.quantile(values, 0.95)),
        "source_support_blocks": int(selected.sum()),
        "source_saturation_block_fraction_mean": float(np.mean(saturation[selected])),
        "source_blocks_with_any_saturation_fraction": float(np.mean(saturation[selected] > 0)),
    }


def write_csv(path: Path, rows: dict[str, Any], context: dict[str, Any]) -> None:
    fields = [
        "pair", "cohort_role", "kv", "ma", "exposure_time_ms", "mas",
        "m1_cv_r2", "mhinge_cv_r2", "delta_cv_r2", "rmse_relative_reduction",
        "trend_rms_ratio", "edge_rmse_relative_reduction",
        "outside_rmse_relative_reduction", "tau_S_median",
        "tau_empirical_quantile_median", "tau_over_S90_median", "alpha_low_median",
        "alpha_high_median", "delta_alpha_median", "direction", "tau_fold_cv",
        "alpha_low_fold_cv", "alpha_high_fold_cv", "min_low_dynamic_range_fraction",
        "min_high_dynamic_range_fraction", "max_standardized_condition_number",
        "passed_all_checks",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in ALL_PAIRS:
            row = rows[label]
            summary = row["identifiability"]["summary"]
            change = row["change_vs_M1"]
            writer.writerow({
                "pair": label,
                "cohort_role": row["cohort_role"],
                **{key: context[label][key] for key in ("kv", "ma", "exposure_time_ms", "mas")},
                "m1_cv_r2": row["M1"]["metrics"]["cv_r2"],
                "mhinge_cv_r2": row["Mhinge"]["metrics"]["cv_r2"],
                "delta_cv_r2": change["delta_cv_r2"],
                "rmse_relative_reduction": change["rmse_relative_reduction"],
                "trend_rms_ratio": change["residual_trend_weighted_rms_ratio"],
                "edge_rmse_relative_reduction": change["source_edge_rmse_relative_reduction"],
                "outside_rmse_relative_reduction": change["outside_source_rmse_relative_reduction"],
                **{key: summary[key] for key in fields if key in summary},
                "passed_all_checks": row["passed_all_checks"],
            })


def render_summary(path: Path, rows: dict[str, Any]) -> None:
    labels = list(ALL_PAIRS)
    y = np.arange(len(labels))
    role_colors = {
        "extension_primary": "#1b9e77",
        "discovery_reference": "#7570b3",
        "weak_signal_reference": "#e6ab02",
        "negative_control": "#999999",
    }
    colors = [role_colors[rows[label]["cohort_role"]] for label in labels]
    figure, axes = plt.subplots(1, 4, figsize=(18, 7), constrained_layout=True)

    delta = [rows[label]["change_vs_M1"]["delta_cv_r2"] for label in labels]
    axes[0].barh(y, delta, color=colors)
    axes[0].axvline(0, color="#333333", lw=1)
    axes[0].set_title("OOF delta CV-R2")

    trend = [rows[label]["change_vs_M1"]["residual_trend_weighted_rms_ratio"] for label in labels]
    axes[1].barh(y, trend, color=colors)
    axes[1].axvline(MAX_TREND_RMS_RATIO, color="#b2182b", ls="--", lw=1)
    axes[1].set_title("Residual trend RMS ratio")

    outside_increase = [
        -rows[label]["change_vs_M1"]["outside_source_rmse_relative_reduction"]
        for label in labels
    ]
    axes[2].barh(y, outside_increase, color=colors)
    axes[2].axvline(MAX_LOCAL_RMSE_INCREASE, color="#b2182b", ls="--", lw=1)
    axes[2].axvline(0, color="#333333", lw=1)
    axes[2].set_title("Outside-source RMSE increase")

    for index, label in enumerate(labels):
        summary = rows[label]["identifiability"]["summary"]
        low = summary["alpha_low_median"]
        high = summary["alpha_high_median"]
        axes[3].plot([low, high], [index, index], color=colors[index], lw=2)
        axes[3].scatter([low], [index], color="#2166ac", s=24, zorder=3)
        axes[3].scatter([high], [index], color="#b2182b", s=24, zorder=3)
    axes[3].axvline(MAX_ALPHA, color="#b2182b", ls="--", lw=1)
    axes[3].set_title("Median alpha: low (blue) -> high (red)")

    for axis in axes:
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Mhinge cross-pair pre-frozen reproducibility audit\n"
        "green=extension, purple=discovery, yellow=weak, gray=negative",
        fontsize=14,
    )
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Mhinge across frozen single-lag pairs")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/mhinge_cross_pair_v1"))
    parser.add_argument(
        "--cohort-audit", type=Path,
        default=Path("outputs/background_freeze_audit_v1/background_freeze_audit.json"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/AI修残影例图"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/mhinge_cross_pair_v1/model_audit"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (repo_root / path).resolve()

    dataset_dir = resolve(args.dataset_dir)
    cohort_audit_path = resolve(args.cohort_audit)
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)
    pairs, provenance, gate_snapshot, saturation_maps = load_audit_pairs(
        dataset_dir, cohort_audit_path,
    )
    exposures_path = data_dir / "拍摄参数记录.xlsx"
    exposures = load_exposures(exposures_path)
    hinge = nested_parametric_oof(pairs, "Mhinge", HINGE_QUANTILES)

    comparisons = {}
    context = {}
    for pair in pairs:
        identifiability = hinge_identifiability(
            pair, hinge["fold_parameters"][pair.label],
        )
        comparisons[pair.label] = pair_comparison(
            pair, hinge["predictions"][pair.label], identifiability,
        )
        context[pair.label] = descriptive_context(
            pair, exposures, saturation_maps[pair.label],
        )

    extension = aggregate_pairs(EXTENSION_PAIRS, comparisons)
    discovery = aggregate_pairs(DISCOVERY_PAIRS, comparisons)
    eligible = aggregate_pairs(ELIGIBLE_PAIRS, comparisons)
    sensitivity = extension_sensitivity(comparisons)
    decision = cross_pair_decision(extension, sensitivity)

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xray-mhinge-cross-pair-") as temporary:
        temporary_dir = Path(temporary)
        figure_path = temporary_dir / "cross_pair_summary.png"
        csv_path = temporary_dir / "pair_summary.csv"
        json_path = temporary_dir / "audit_results.json"
        render_summary(figure_path, comparisons)
        write_csv(csv_path, comparisons, context)
        result = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "analysis_scope": "Mhinge cross-pair pre-frozen reproducibility audit",
            "evidence_boundary": [
                "The seven extension pairs are primary because they did not select the Mhinge model class.",
                "Discovery pairs 5->6 and 27->28 are reference-only in the primary decision.",
                "Pairs are equal-weight summary units but are not independent acquisitions.",
                "Bootstrap intervals are descriptive sensitivity intervals.",
                "No result is clean-ground-truth or detector-physics validation.",
            ],
            "predeclared_cohorts": {
                "all_single_lag": list(ALL_PAIRS),
                "discovery_reference": list(DISCOVERY_PAIRS),
                "extension_primary": list(EXTENSION_PAIRS),
                "weak_signal_reference": list(WEAK_REFERENCE_PAIRS),
                "negative_control": list(NEGATIVE_CONTROL_PAIRS),
                "eligible_secondary": list(ELIGIBLE_PAIRS),
            },
            "source_provenance": {
                **provenance,
                "exposure_workbook": str(exposures_path.resolve()),
                "exposure_workbook_sha256": sha256(exposures_path),
                "gate_snapshot": gate_snapshot,
            },
            "code_provenance": {
                "path": "scripts/audit_mhinge_cross_pair.py",
                "sha256": sha256(Path(__file__).resolve()),
                "design_path": "docs/plans/2026-09-21-mhinge-cross-pair-audit.md",
                "design_sha256": sha256(
                    repo_root / "docs/plans/2026-09-21-mhinge-cross-pair-audit.md"
                ),
            },
            "model": {
                "definition": "S=max(-X,0); Y=b+alpha_low*X-delta_alpha*max(0,S-tau_S)",
                "alpha_high_definition": "alpha_high=alpha_low+delta_alpha",
                "candidate_training_quantiles": list(HINGE_QUANTILES),
                "selection": "Pair-specific inner spatial CV inside every outer fold",
                "fold_parameters": hinge["fold_parameters"],
            },
            "thresholds": {
                "alpha_eligibility_minimum": ALPHA_ELIGIBILITY_MIN,
                "maximum_trend_rms_ratio": MAX_TREND_RMS_RATIO,
                "maximum_local_rmse_increase": MAX_LOCAL_RMSE_INCREASE,
                "maximum_parameter_cv": MAX_PARAMETER_CV,
                "maximum_boundary_selections": MAX_BOUNDARY_SELECTIONS,
                "maximum_alpha": MAX_ALPHA,
                "minimum_segment_blocks": MIN_SEGMENT_BLOCKS,
                "minimum_segment_fraction": MIN_SEGMENT_FRACTION,
                "minimum_dynamic_range_fraction": MIN_DYNAMIC_RANGE_FRACTION,
                "maximum_standardized_condition_number": MAX_STANDARDIZED_CONDITION_NUMBER,
                "minimum_extension_pass_count": MIN_EXTENSION_PASS_COUNT,
                "minimum_matching_direction_count": MIN_DIRECTION_COUNT,
            },
            "descriptive_context": context,
            "pair_results": comparisons,
            "aggregate": {
                "extension_primary": extension,
                "discovery_reference": discovery,
                "eligible_secondary": eligible,
            },
            "extension_sensitivity": sensitivity,
            "decision": decision,
            "output_artifacts": {
                "cross_pair_summary": {
                    "relative_path": "cross_pair_summary.png",
                    "size_bytes": figure_path.stat().st_size,
                    "sha256": sha256(figure_path),
                },
                "pair_summary": {
                    "relative_path": "pair_summary.csv",
                    "size_bytes": csv_path.stat().st_size,
                    "sha256": sha256(csv_path),
                },
            },
        }
        json_path.write_text(
            json.dumps(clean_json(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for path in (figure_path, csv_path, json_path):
            shutil.copyfile(path, output_dir / path.name)

    print(json.dumps(clean_json({
        "output": str((output_dir / "audit_results.json").resolve()),
        "extension_primary": extension,
        "decision": decision,
        "remove_largest_effect": sensitivity["remove_largest_effect"],
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
