"""Audit realistic BG holdouts and fold-refitted BG effects on ghost models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_frozen_candidates import PairData, model_metrics, sha256
from scripts.analyze_intensity_nonlinearity import HINGE_QUANTILES, nested_parametric_oof
from scripts.analyze_pseudo_ghost_mechanism import clean_json, strict_affine_oof


MODEL_ORDER = ("M0", "M1", "Mhinge")
CORE_PAIRS = ("5->6", "27->28")
ALPHA_ELIGIBILITY_MINIMUM = 1e-4


def intercept_oof(pair: PairData) -> np.ndarray:
    prediction = np.full(pair.y.shape, np.nan, dtype=np.float64)
    for fold in sorted(int(value) for value in np.unique(pair.folds[pair.evaluated])):
        train = pair.evaluated & (pair.folds != fold)
        test = pair.evaluated & (pair.folds == fold)
        prediction[test] = float(np.mean(pair.y[train]))
    return prediction


def load_pair_variants(dataset_dir: Path) -> tuple[list[PairData], list[PairData]]:
    nominal_pairs = []
    crossfit_pairs = []
    for pair_dir in sorted(dataset_dir.glob("pair_*_*")):
        fields = pair_dir.name.split("_")
        label = f"{int(fields[1])}->{int(fields[2])}"
        with np.load(pair_dir / "oof_maps_block16.npz") as archive:
            required = {
                "source", "observable_signal", "oof_prediction", "oof_residual",
                "evaluated_mask", "source_support", "spatial_folds",
                "crossfit_observable_signal", "crossfit_oof_prediction",
                "crossfit_oof_residual", "crossfit_evaluated_mask",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{label} is missing cross-fit arrays: {sorted(missing)}")
            common = {
                "label": label,
                "x": archive["source"].astype(np.float64),
                "folds": archive["spatial_folds"].astype(np.int8),
                "source_support": archive["source_support"].astype(bool),
            }
            nominal_pairs.append(PairData(
                **common,
                y=archive["observable_signal"].astype(np.float64),
                evaluated=archive["evaluated_mask"].astype(bool),
                prediction=archive["oof_prediction"].astype(np.float64),
                residual=archive["oof_residual"].astype(np.float64),
            ))
            crossfit_pairs.append(PairData(
                **common,
                y=archive["crossfit_observable_signal"].astype(np.float64),
                evaluated=archive["crossfit_evaluated_mask"].astype(bool),
                prediction=archive["crossfit_oof_prediction"].astype(np.float64),
                residual=archive["crossfit_oof_residual"].astype(np.float64),
            ))
    if len(nominal_pairs) != 13:
        raise ValueError(f"Expected 13 single-lag pairs, found {len(nominal_pairs)}")
    return nominal_pairs, crossfit_pairs


def evaluate_models(pairs: list[PairData]) -> dict[str, Any]:
    hinge = nested_parametric_oof(pairs, "Mhinge", HINGE_QUANTILES)
    rows = {}
    for pair in pairs:
        affine = strict_affine_oof(pair.x, pair.y, pair.evaluated, pair.folds)
        pair = replace(
            pair,
            prediction=affine["prediction"],
            residual=affine["residual"],
            evaluated=affine["evaluated_mask"],
        )
        predictions = {
            "M0": intercept_oof(pair),
            "M1": affine["prediction"],
            "Mhinge": hinge["predictions"][pair.label],
        }
        metrics = {
            model: model_metrics(pair, prediction)
            for model, prediction in predictions.items()
        }
        ranking = sorted(
            MODEL_ORDER,
            key=lambda model: (-metrics[model]["cv_r2"], MODEL_ORDER.index(model)),
        )
        rows[pair.label] = {
            "alpha": float(affine["alpha_fold_mean"]),
            "alpha_fold_cv": float(affine["alpha_fold_cv"]),
            "model_metrics": metrics,
            "cv_r2_ranking": ranking,
            "mhinge_delta_cv_r2_vs_m1": float(
                metrics["Mhinge"]["cv_r2"] - metrics["M1"]["cv_r2"]
            ),
        }
    return rows


def summarize_pair_changes(
    nominal: dict[str, Any],
    crossfit: dict[str, Any],
) -> dict[str, Any]:
    detail = {}
    for label in nominal:
        reference = nominal[label]
        updated = crossfit[label]
        alpha_change = (
            updated["alpha"] - reference["alpha"]
        ) / max(abs(reference["alpha"]), 1e-12)
        detail[label] = {
            "nominal_alpha": reference["alpha"],
            "crossfit_alpha": updated["alpha"],
            "alpha_relative_change": float(alpha_change),
            "nominal_ranking": reference["cv_r2_ranking"],
            "crossfit_ranking": updated["cv_r2_ranking"],
            "ranking_changed": reference["cv_r2_ranking"] != updated["cv_r2_ranking"],
            "nominal_mhinge_delta_cv_r2_vs_m1": reference[
                "mhinge_delta_cv_r2_vs_m1"
            ],
            "crossfit_mhinge_delta_cv_r2_vs_m1": updated[
                "mhinge_delta_cv_r2_vs_m1"
            ],
            "mhinge_improvement_direction_changed": bool(
                np.sign(reference["mhinge_delta_cv_r2_vs_m1"])
                != np.sign(updated["mhinge_delta_cv_r2_vs_m1"])
            ),
        }
    changes = np.asarray([row["alpha_relative_change"] for row in detail.values()])
    eligible = [
        row for row in detail.values()
        if abs(row["nominal_alpha"]) >= ALPHA_ELIGIBILITY_MINIMUM
    ]
    core = [detail[label] for label in CORE_PAIRS]
    return {
        "pair_count": len(detail),
        "median_absolute_alpha_relative_change": float(np.median(np.abs(changes))),
        "maximum_absolute_alpha_relative_change": float(np.max(np.abs(changes))),
        "eligible_alpha_minimum": ALPHA_ELIGIBILITY_MINIMUM,
        "eligible_pair_count": len(eligible),
        "eligible_maximum_absolute_alpha_relative_change": float(max(
            abs(row["alpha_relative_change"]) for row in eligible
        )),
        "core_maximum_absolute_alpha_relative_change": float(max(
            abs(row["alpha_relative_change"]) for row in core
        )),
        "ranking_changed_count": int(sum(row["ranking_changed"] for row in detail.values())),
        "mhinge_improvement_direction_changed_count": int(sum(
            row["mhinge_improvement_direction_changed"] for row in detail.values()
        )),
        "detail": detail,
    }


def summarize_realistic_holdouts(analysis: dict[str, Any]) -> dict[str, Any]:
    attempts = []
    samples = []
    for pair in analysis["pair_results"]:
        label = f"{pair['light_index']}->{pair['dark_index']}"
        validation = pair["background_reconstruction_validation"]
        attempts.extend({"target_pair": label, **row} for row in validation[
            "realistic_occlusion_attempts"
        ])
        samples.extend(
            {"target_pair": label, **row}
            for row in validation["samples"]
            if row["kind"] == "real_source_full_shape"
        )
    verified = [row for row in attempts if row["status"] == "validated"]
    unverified = [row for row in attempts if row["status"] == "unverified"]
    errors = [row["by_background"]["masked_hybrid_spline"] for row in samples]
    candidate_labels = sorted({row["label"] for row in attempts})
    return {
        "attempt_count": len(attempts),
        "validated_count": len(verified),
        "unverified_count": len(unverified),
        "original_position_count": int(sum(
            row.get("placement") == "original" for row in verified
        )),
        "translated_count": int(sum(
            row.get("placement") == "nearest_valid_translation" for row in verified
        )),
        "validated_requested_fraction_range": (
            [float(min(row["requested_fraction"] for row in verified)),
             float(max(row["requested_fraction"] for row in verified))]
            if verified else None
        ),
        "attempted_requested_fraction_range": (
            [float(min(row["requested_fraction"] for row in attempts)),
             float(max(row["requested_fraction"] for row in attempts))]
            if attempts else None
        ),
        "unverified_requested_fraction_range": (
            [float(min(row["requested_fraction"] for row in unverified)),
             float(max(row["requested_fraction"] for row in unverified))]
            if unverified else None
        ),
        "median_displacement_blocks": (
            float(np.median([row["displacement_blocks"] for row in verified]))
            if verified else None
        ),
        "median_mae": (
            float(np.median([row["mae"] for row in errors])) if errors else None
        ),
        "median_absolute_bias": (
            float(np.median([abs(row["bias"]) for row in errors])) if errors else None
        ),
        "unverified_by_reason": {
            reason: int(sum(row.get("reason") == reason for row in unverified))
            for reason in sorted({row.get("reason") for row in unverified})
        },
        "by_source_candidate": {
            label: {
                "attempted": int(sum(row["label"] == label for row in attempts)),
                "validated": int(sum(row["label"] == label for row in verified)),
                "unverified": int(sum(row["label"] == label for row in unverified)),
                "requested_fraction": next(
                    row["requested_fraction"] for row in attempts if row["label"] == label
                ),
            }
            for label in candidate_labels
        },
        "attempts": attempts,
        "samples": samples,
    }


def render_markdown(result: dict[str, Any]) -> str:
    holdout = result["realistic_holdout_summary"]
    change = result["crossfit_model_summary"]
    lines = [
        "# Background realistic-holdout and end-to-end cross-fit audit",
        "",
        "This audit keeps the frozen single-frame BG definition and refits it after "
        "excluding each spatial test fold from the current dark image.",
        "",
        "## Real-shape holdouts",
        "",
        f"- Attempts: {holdout['attempt_count']}",
        f"- Validated: {holdout['validated_count']}",
        f"- Unverified: {holdout['unverified_count']}",
        f"- Original-position validations: {holdout['original_position_count']}",
        f"- Nearest translated validations: {holdout['translated_count']}",
        "- Validated source-area range: "
        f"{100 * holdout['validated_requested_fraction_range'][0]:.2f}% to "
        f"{100 * holdout['validated_requested_fraction_range'][1]:.2f}% of the image",
        "- Attempted source-area range: "
        f"{100 * holdout['attempted_requested_fraction_range'][0]:.2f}% to "
        f"{100 * holdout['attempted_requested_fraction_range'][1]:.2f}% of the image",
        f"- Median translation: {holdout['median_displacement_blocks']:.2f} blocks",
        f"- Median reconstruction MAE: {holdout['median_mae']:.6f}",
        f"- Median absolute bias: {holdout['median_absolute_bias']:.6f}",
        "",
        "Unverified attempts retain their requested area, shape, and position in the JSON; "
        "they are not silently dropped or resized.",
        "No full shape could be validated at its original detector position. The validated "
        "cases therefore test area and shape after the smallest feasible translation, while "
        "position-specific reconstruction remains unverified.",
        "",
        "## Fold-refitted BG impact",
        "",
        f"- Median absolute alpha change: "
        f"{100 * change['median_absolute_alpha_relative_change']:.2f}%",
        f"- Maximum absolute alpha change: "
        f"{100 * change['maximum_absolute_alpha_relative_change']:.2f}%",
        f"- Maximum change for |alpha| >= {change['eligible_alpha_minimum']:.0e}: "
        f"{100 * change['eligible_maximum_absolute_alpha_relative_change']:.2f}%",
        f"- Maximum change for core pairs 5->6 and 27->28: "
        f"{100 * change['core_maximum_absolute_alpha_relative_change']:.2f}%",
        f"- Full M0/M1/Mhinge ranking changes: {change['ranking_changed_count']}"
        f"/{change['pair_count']}",
        f"- Mhinge-vs-M1 improvement direction changes: "
        f"{change['mhinge_improvement_direction_changed_count']}/{change['pair_count']}",
        "",
        "| Pair | alpha nominal | alpha cross-fit | relative change | nominal rank | cross-fit rank |",
        "|---|---:|---:|---:|---|---|",
    ]
    for label, row in change["detail"].items():
        lines.append(
            f"| {label} | {row['nominal_alpha']:.6g} | {row['crossfit_alpha']:.6g} | "
            f"{100 * row['alpha_relative_change']:+.2f}% | "
            f"{' > '.join(row['nominal_ranking'])} | "
            f"{' > '.join(row['crossfit_ranking'])} |"
        )
    lines.extend([
        "",
        "The test remains an internal known-air and same-session sensitivity audit. It "
        "does not provide the true BG under an actually exposed object.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/AI修残影例图"))
    parser.add_argument(
        "--source-mask-npz", type=Path,
        default=Path("outputs/pseudo_ghost_mechanism_v1/sam_masks_single_lag.npz"),
    )
    parser.add_argument(
        "--frozen-config", type=Path, default=Path("configs/background_frozen_v1.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/background_freeze_audit_v1"),
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else (repo_root / path).resolve()

    data_dir = resolve(args.data_dir)
    mask_path = resolve(args.source_mask_npz)
    config_path = resolve(args.frozen_config)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="xray-bg-crossfit-") as temporary:
        dataset_dir = Path(temporary) / "dataset"
        command = [
            sys.executable,
            str(repo_root / "scripts" / "analyze_pseudo_ghost_mechanism.py"),
            "--data-dir", str(data_dir),
            "--output-dir", str(dataset_dir),
            "--pairs", "single-lag",
            "--analysis-blocks", "16",
            "--background-modes", "masked_hybrid_spline",
            "--primary-background", "masked_hybrid_spline",
            "--source-mask-npz", str(mask_path),
            "--frozen-background-config", str(config_path),
            "--skip-input-hashes",
            "--skip-background-oof",
            "--skip-pair-figures",
            "--validate-background-reconstruction",
            "--validate-realistic-occlusions",
            "--end-to-end-background-oof",
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Cross-fit dataset generation failed:\n" + completed.stdout[-8000:]
            )
        analysis_path = dataset_dir / "analysis_results.json"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        nominal_pairs, crossfit_pairs = load_pair_variants(dataset_dir)
        nominal_models = evaluate_models(nominal_pairs)
        crossfit_models = evaluate_models(crossfit_pairs)
        result = {
            "schema_version": 2,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "scope": "Frozen-BG realistic holdouts and spatial-fold BG cross-fitting",
            "evidence_boundary": [
                "Only the immediately previous light is used as the ghost predictor.",
                "Fixed detector structure may use other target-excluded dark acquisitions.",
                "Real-shape holdouts are evaluated only where target dark values are known.",
                "This is not clean-ground-truth validation under an actually exposed object.",
            ],
            "provenance": {
                "source_mask_sha256": sha256(mask_path),
                "frozen_config_sha256": sha256(config_path),
                "analysis_script_sha256": sha256(
                    repo_root / "scripts" / "analyze_pseudo_ghost_mechanism.py"
                ),
                "audit_script_sha256": sha256(Path(__file__).resolve()),
            },
            "realistic_holdout_summary": summarize_realistic_holdouts(analysis),
            "crossfit_model_summary": summarize_pair_changes(
                nominal_models, crossfit_models,
            ),
            "nominal_models": nominal_models,
            "crossfit_models": crossfit_models,
        }

    json_path = output_dir / "background_crossfit_audit.json"
    report_path = output_dir / "background_crossfit_audit.md"
    json_path.write_text(
        json.dumps(clean_json(result), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "report": str(report_path),
        "realistic_holdouts": {
            "validated": result["realistic_holdout_summary"]["validated_count"],
            "unverified": result["realistic_holdout_summary"]["unverified_count"],
        },
        "crossfit": {
            key: result["crossfit_model_summary"][key]
            for key in (
                "median_absolute_alpha_relative_change",
                "maximum_absolute_alpha_relative_change",
                "ranking_changed_count",
                "mhinge_improvement_direction_changed_count",
            )
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
