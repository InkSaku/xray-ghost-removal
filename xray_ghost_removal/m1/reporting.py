"""M1 reporting; extracted without changing numerical definitions."""
from __future__ import annotations
from typing import Any
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xray_ghost_removal.m1.config import PRIMARY_BLOCK
from xray_ghost_removal.m1.validation import summarize_reconstruction_samples

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
