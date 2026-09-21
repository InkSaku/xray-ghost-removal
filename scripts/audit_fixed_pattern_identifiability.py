"""Audit whether the frozen fixed-pattern estimate has spatial data support.

This is deliberately not another background-model search.  It reuses the
frozen masked-hybrid configuration and measures raw coverage, final-weight
effective coverage, and sensitivity of the affine ghost conclusion to
excluding weakly supported detector blocks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

try:
    from analyze_pseudo_ghost_mechanism import (
        PRIMARY_BLOCK,
        SINGLE_LAG_PAIRS,
        block_mean,
        block_source_masks,
        clean_json,
        detection_gate,
        estimate_masked_fixed_pattern,
        estimate_masked_hybrid_dark_background,
        evaluate_with_nulls,
        load_source_mask,
        source,
        spatial_folds,
    )
except ModuleNotFoundError:  # pragma: no cover - module execution fallback
    from scripts.analyze_pseudo_ghost_mechanism import (  # type: ignore
        PRIMARY_BLOCK,
        SINGLE_LAG_PAIRS,
        block_mean,
        block_source_masks,
        clean_json,
        detection_gate,
        estimate_masked_fixed_pattern,
        estimate_masked_hybrid_dark_background,
        evaluate_with_nulls,
        load_source_mask,
        source,
        spatial_folds,
    )


FOCUS_PAIRS = {(5, 6), (27, 28), (4, 5), (6, 7)}
CORE_PAIRS = {(5, 6), (27, 28)}
SCENARIOS = {
    "all": lambda coverage, effective: np.ones_like(coverage, dtype=bool),
    "coverage_ge_5": lambda coverage, effective: coverage >= 5,
    "effective_coverage_ge_3": lambda coverage, effective: effective >= 3,
}
THRESHOLDS = {
    "source_coverage_median_min": 10.0,
    "source_coverage_below_5_fraction_max": 0.01,
    "source_effective_coverage_median_min": 5.0,
    "source_effective_coverage_below_3_fraction_max": 0.01,
    "core_alpha_relative_change_max": 0.20,
}
RANDOM_SEED = 20260921


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        raise ValueError("Cannot summarize an empty spatial region")
    return {
        "block_count": int(selected.size),
        "min": float(np.min(selected)),
        "p01": float(np.percentile(selected, 1)),
        "median": float(np.median(selected)),
        "max": float(np.max(selected)),
        "equal_0_fraction": float(np.mean(selected == 0)),
        "less_equal_2_fraction": float(np.mean(selected <= 2)),
        "below_3_fraction": float(np.mean(selected < 3)),
        "below_5_fraction": float(np.mean(selected < 5)),
        "below_10_fraction": float(np.mean(selected < 10)),
    }


def classify_decision(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair = {row["pair"]: row for row in pair_rows}
    core_labels = ("5->6", "27->28")
    focus_labels = ("5->6", "27->28", "4->5", "6->7")
    coverage_checks = {
        label: bool(
            by_pair[label]["source_region"]["coverage"]["median"]
            >= THRESHOLDS["source_coverage_median_min"]
            and by_pair[label]["source_region"]["coverage"]["below_5_fraction"]
            < THRESHOLDS["source_coverage_below_5_fraction_max"]
            and by_pair[label]["source_region"]["coverage"]["equal_0_fraction"] == 0
        )
        for label in core_labels
    }
    effective_checks = {
        label: bool(
            by_pair[label]["source_region"]["effective_coverage"]["median"]
            >= THRESHOLDS["source_effective_coverage_median_min"]
            and by_pair[label]["source_region"]["effective_coverage"]["below_3_fraction"]
            < THRESHOLDS["source_effective_coverage_below_3_fraction_max"]
        )
        for label in core_labels
    }
    detection_stability = {
        label: all(
            scenario["detection_passed"]
            == by_pair[label]["sensitivity"]["all"]["detection_passed"]
            for scenario in by_pair[label]["sensitivity"].values()
        )
        for label in focus_labels
    }
    alpha_stability = {
        label: all(
            scenario["alpha_relative_change_vs_all"] is None
            or abs(scenario["alpha_relative_change_vs_all"])
            <= THRESHOLDS["core_alpha_relative_change_max"]
            for scenario in by_pair[label]["sensitivity"].values()
        )
        for label in core_labels
    }
    support_passed = all(coverage_checks.values()) and all(effective_checks.values())
    conclusion_passed = all(detection_stability.values()) and all(alpha_stability.values())
    if support_passed and conclusion_passed:
        grade = "pass"
        recommendation = "freeze_background_and_continue_to_new_ghost_models"
    elif conclusion_passed and all(
        by_pair[label]["source_region"]["coverage"]["equal_0_fraction"] == 0
        for label in core_labels
    ):
        grade = "conditional_pass"
        recommendation = "freeze_background_but_flag_low_support_blocks"
    else:
        grade = "fail"
        recommendation = "do_not_tune_background;_consider_joint_F_and_ghost_estimation"
    return {
        "grade": grade,
        "recommendation": recommendation,
        "support_passed": support_passed,
        "conclusion_stability_passed": conclusion_passed,
        "checks": {
            "core_raw_coverage": coverage_checks,
            "core_effective_coverage": effective_checks,
            "focus_detection_stability": detection_stability,
            "core_alpha_stability": alpha_stability,
        },
        "thresholds": THRESHOLDS,
    }


def plot_map(axis: plt.Axes, values: np.ndarray, title: str, cmap: str,
             vmin: float | None = None, vmax: float | None = None) -> None:
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_axis_off()
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def render_global(output: Path, frequency: np.ndarray,
                  coverages: list[np.ndarray]) -> None:
    median_coverage = np.median(np.stack(coverages), axis=0)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_map(axes[0], frequency, "Object occurrence frequency", "magma", 0, 1)
    plot_map(axes[1], median_coverage, "Median leave-target-out coverage", "viridis")
    axes[2].hist(median_coverage.ravel(), bins=np.arange(0.5, 31.5, 1), color="#376795")
    axes[2].axvline(5, color="#c43c39", linestyle="--", label="C = 5")
    axes[2].set_title("Detector-block coverage distribution")
    axes[2].set_xlabel("Available dark acquisitions")
    axes[2].set_ylabel("16x16 blocks")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def render_pair(output: Path, label: str, source_support: np.ndarray,
                coverage: np.ndarray, effective: np.ndarray, y: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    plot_map(axes[0], source_support.astype(float), "Source support", "gray", 0, 1)
    plot_map(axes[1], coverage, "Raw coverage C", "viridis", 0, 30)
    plot_map(axes[2], effective, "Effective coverage Neff", "viridis", 0, 30)
    limit = float(np.percentile(np.abs(y[np.isfinite(y)]), 99.5))
    plot_map(axes[3], y, "Observable residual Y", "coolwarm", -limit, limit)
    for axis in axes[1:]:
        axis.contour(source_support, levels=[0.5], colors=["white"], linewidths=0.45)
    figure.suptitle(f"Fixed-pattern identifiability audit: {label} (16x16 blocks)")
    figure.tight_layout()
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "pair", "role", "full_coverage_median", "source_coverage_median",
        "source_coverage_equal_0_fraction", "source_coverage_le_2_fraction",
        "source_coverage_below_5_fraction", "source_effective_coverage_median",
        "source_effective_coverage_below_3_fraction",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "pair": row["pair"],
                "role": row["role"],
                "full_coverage_median": row["full_image"]["coverage"]["median"],
                "source_coverage_median": row["source_region"]["coverage"]["median"],
                "source_coverage_equal_0_fraction": row["source_region"]["coverage"]["equal_0_fraction"],
                "source_coverage_le_2_fraction": row["source_region"]["coverage"]["less_equal_2_fraction"],
                "source_coverage_below_5_fraction": row["source_region"]["coverage"]["below_5_fraction"],
                "source_effective_coverage_median": row["source_region"]["effective_coverage"]["median"],
                "source_effective_coverage_below_3_fraction": row["source_region"]["effective_coverage"]["below_3_fraction"],
            })


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    decision = result["decision"]
    lines = [
        "# Fixed-pattern spatial identifiability audit",
        "",
        f"- Decision: **{decision['grade']}**",
        f"- Recommendation: `{decision['recommendation']}`",
        f"- Analysis unit: {PRIMARY_BLOCK}x{PRIMARY_BLOCK} detector blocks",
        "- Background algorithm and affine ghost model were not changed.",
        "",
        "| Pair | Role | Source C median | Source C=0 | Source C<=2 | Source C<5 | Source Neff median | Neff<3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["pairs"]:
        coverage = row["source_region"]["coverage"]
        effective = row["source_region"]["effective_coverage"]
        lines.append(
            f"| {row['pair']} | {row['role']} | {coverage['median']:.2f} | "
            f"{coverage['equal_0_fraction']:.3%} | {coverage['less_equal_2_fraction']:.3%} | "
            f"{coverage['below_5_fraction']:.3%} | {effective['median']:.2f} | "
            f"{effective['below_3_fraction']:.3%} |"
        )
    lines.extend(["", "## Low-support exclusion sensitivity", ""])
    for label in ("5->6", "27->28", "4->5", "6->7"):
        row = next(item for item in result["pairs"] if item["pair"] == label)
        parts = []
        for name, scenario in row["sensitivity"].items():
            relative = scenario["alpha_relative_change_vs_all"]
            relative_text = "baseline" if relative is None else f"alpha change {relative:+.1%}"
            parts.append(
                f"`{name}`: detection={scenario['detection_passed']}, {relative_text}"
            )
        lines.append(f"- **{label}** — " + "; ".join(parts))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/AI修残影例图"))
    parser.add_argument("--frozen-config", type=Path,
                        default=Path("configs/background_frozen_v1.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/fixed_pattern_identifiability_v1"))
    args = parser.parse_args()

    config = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    mask_path = Path(config["mask"]["archive_relative_path"])
    if sha256(mask_path) != config["mask"]["sha256"]:
        raise ValueError("Frozen source-mask archive hash does not match the config")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    darks: dict[int, np.ndarray] = {}
    lights: dict[int, np.ndarray] = {}
    saturations: dict[int, np.ndarray] = {}
    for index in range(1, 32):
        dark = pydicom.dcmread(args.data_dir / f"{index}-dark.dcm").pixel_array.astype(np.float32)
        light = pydicom.dcmread(args.data_dir / f"{index}-light.dcm").pixel_array.astype(np.float32)
        darks[index] = block_mean(dark, PRIMARY_BLOCK)
        lights[index] = block_mean(light, PRIMARY_BLOCK)
        saturations[index] = block_mean((light == light.max()).astype(np.float32), PRIMARY_BLOCK)

    supports: dict[int, np.ndarray] = {}
    mask_origins: dict[int, str] = {}
    with np.load(mask_path) as archive:
        for light_index in range(1, 31):
            raw_light = pydicom.dcmread(
                args.data_dir / f"{light_index}-light.dcm"
            ).pixel_array.astype(np.float32)
            mask, origin = load_source_mask(
                light_index=light_index,
                light=raw_light,
                mask_archive=archive,
                dilation_pixels=int(config["mask"]["dilation_pixels"]),
                allow_missing_archive_key=True,
            )
            _, supports[light_index] = block_source_masks(
                mask,
                PRIMARY_BLOCK,
                float(config["mask"]["background_fit_max_source_fraction"]),
            )
            mask_origins[light_index] = origin

    object_frequency = np.mean(np.stack(list(supports.values())), axis=0)
    folds = spatial_folds(darks[1].shape)
    pair_rows = []
    coverage_maps = []
    figure_payloads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for light_index, dark_index in SINGLE_LAG_PAIRS:
        fixed, diagnostics = estimate_masked_fixed_pattern(
            darks=darks,
            contamination_supports=supports,
            target_index=dark_index,
            smoothness=float(config["spline_smoothness"]),
            decomposition_iterations=int(config["fixed_pattern"]["decomposition_iterations"]),
            huber_iterations=int(config["fixed_pattern"]["huber_iterations"]),
            huber_delta=float(config["huber_delta"]),
            include_support_maps=True,
        )
        coverage = diagnostics.pop("coverage_map")
        effective = diagnostics.pop("effective_coverage_map")
        coverage_maps.append(coverage)
        source_support = supports[light_index]
        fit_mask = ~source_support
        background, _ = estimate_masked_hybrid_dark_background(
            dark=darks[dark_index],
            darks=darks,
            contamination_supports=supports,
            target_index=dark_index,
            fit_mask=fit_mask,
            smoothness=float(config["spline_smoothness"]),
            fixed_pattern=fixed,
            fixed_pattern_diagnostics=diagnostics,
            decomposition_iterations=int(config["fixed_pattern"]["decomposition_iterations"]),
            huber_iterations=int(config["fixed_pattern"]["huber_iterations"]),
            huber_delta=float(config["huber_delta"]),
        )
        y = darks[dark_index] - background
        x = source(lights, light_index)
        label = f"{light_index}->{dark_index}"
        role = "core" if (light_index, dark_index) in CORE_PAIRS else (
            "control" if (light_index, dark_index) in FOCUS_PAIRS else "cohort"
        )
        row: dict[str, Any] = {
            "pair": label,
            "role": role,
            "target_dark_excluded": True,
            "full_image": {
                "coverage": distribution(coverage, np.ones_like(coverage, dtype=bool)),
                "effective_coverage": distribution(effective, np.ones_like(coverage, dtype=bool)),
            },
            "source_region": {
                "source_block_count": int(source_support.sum()),
                "coverage": distribution(coverage, source_support),
                "effective_coverage": distribution(effective, source_support),
            },
            "fixed_pattern_diagnostics": diagnostics,
        }
        if (light_index, dark_index) in FOCUS_PAIRS:
            future_sources = [
                (future, source(lights, future), saturations[future])
                for future in range(dark_index, 32)
            ]
            sensitivity = {}
            for scenario_name, selector in SCENARIOS.items():
                base_mask = selector(coverage, effective)
                fit, null_rows, null_summary = evaluate_with_nulls(
                    x=x,
                    y=y,
                    source_saturation=saturations[light_index],
                    mask_mode="all",
                    folds=folds,
                    future_sources=future_sources,
                    seed=RANDOM_SEED + 100 * light_index,
                    base_mask=base_mask,
                )
                sensitivity[scenario_name] = {
                    "eligible_block_count": int(base_mask.sum()),
                    "alpha": float(fit["alpha_fold_mean"]),
                    "alpha_fold_cv": float(fit["alpha_fold_cv"]),
                    "cv_r2": float(fit["cv_r2"]),
                    "null_cv_r2_p95": float(null_summary["cv_r2_p95"]),
                    "detection_passed": bool(
                        detection_gate(fit, null_rows, null_summary)["passed"]
                    ),
                }
            reference_alpha = sensitivity["all"]["alpha"]
            for scenario_name, scenario in sensitivity.items():
                scenario["alpha_relative_change_vs_all"] = (
                    None if scenario_name == "all" else float(
                        (scenario["alpha"] - reference_alpha)
                        / max(abs(reference_alpha), 1e-12)
                    )
                )
            row["sensitivity"] = sensitivity
        pair_rows.append(row)
        if (light_index, dark_index) in CORE_PAIRS:
            figure_payloads[label] = (source_support, coverage, effective, y)

    result = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "study_question": "Is frozen F spatially supported where ghost is analyzed?",
        "background_changed": False,
        "ghost_model_changed": False,
        "analysis_block_size": PRIMARY_BLOCK,
        "frozen_config": config,
        "mask_archive_sha256_verified": True,
        "mask_origin_counts": {
            origin: sum(value == origin for value in mask_origins.values())
            for origin in sorted(set(mask_origins.values()))
        },
        "pairs": pair_rows,
    }
    result["decision"] = classify_decision(pair_rows)

    render_global(args.output_dir / "global_support.png", object_frequency, coverage_maps)
    for label, payload in figure_payloads.items():
        render_pair(args.output_dir / f"pair_{label.replace('->', '_')}_support.png", label, *payload)
    write_csv(args.output_dir / "coverage_summary.csv", pair_rows)
    write_markdown(args.output_dir / "identifiability_audit.md", result)
    (args.output_dir / "identifiability_audit.json").write_text(
        json.dumps(clean_json(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(clean_json(result["decision"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
