"""Run the finite sensitivity audit used to freeze the background estimator.

The audit deliberately keeps the affine ghost model unchanged.  It reruns the
predeclared 13-pair cohort under one-factor-at-a-time background perturbations,
then evaluates detection stability, alpha stability, pseudo-occlusion
reconstruction, and held-out source-aligned residual attenuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


CORE_PAIRS = ("5->6", "27->28")
CONTROL_PAIRS = ("4->5", "6->7")
NOMINAL_CONFIG = {
    "mask_dilation_pixels": 24,
    "spline_smoothness": 20.0,
    "huber_delta": 1.5,
    "fixed_pattern_iterations": 2,
    "fixed_pattern_huber_iterations": 4,
    "mask_source": "sam_with_otsu_fallback",
}
SENSITIVITY_CONFIGS = (
    {"name": "nominal", "gate": True},
    {"name": "dilation_12", "gate": True, "mask_dilation_pixels": 12},
    {"name": "dilation_48", "gate": True, "mask_dilation_pixels": 48},
    {"name": "smoothness_10", "gate": True, "spline_smoothness": 10.0},
    {"name": "smoothness_40", "gate": True, "spline_smoothness": 40.0},
    {"name": "huber_delta_1", "gate": True, "huber_delta": 1.0},
    {"name": "huber_delta_2", "gate": True, "huber_delta": 2.0},
    {"name": "fixed_pattern_iter_1", "gate": True, "fixed_pattern_iterations": 1},
    {"name": "fixed_pattern_iter_3", "gate": True, "fixed_pattern_iterations": 3},
    {"name": "otsu_only_stress", "gate": False, "mask_source": "otsu_only"},
)
THRESHOLDS = {
    "core_detection_min_rate": 0.90,
    "control_nondetection_min_rate": 0.90,
    "alpha_relative_tolerance": 0.20,
    "core_alpha_within_tolerance_min_rate": 0.80,
    "eligible_alpha_within_tolerance_min_rate": 0.75,
    "eligible_nominal_alpha_minimum": 1e-4,
    "pseudo_improved_pair_min_rate": 0.70,
    "source_alignment_ratio_maximum": 0.25,
    "core_residual_variance_ratio_maximum": 0.20,
}
BOOTSTRAP_SAMPLES = 10000
RANDOM_SEED = 20260921


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def resolved_config(row: dict[str, Any]) -> dict[str, Any]:
    return {**NOMINAL_CONFIG, **row}


def extract_pair_rows(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output = {}
    for pair in analysis["pair_results"]:
        label = f"{pair['light_index']}->{pair['dark_index']}"
        block = next(row for row in pair["block_results"] if row["block_size"] == 16)
        backgrounds = {row["background_mode"]: row for row in block["background_results"]}

        def mask_row(mode: str) -> dict[str, Any]:
            return next(
                row for row in backgrounds[mode]["mask_results"]
                if row["mask_mode"] == "all"
            )

        masked = mask_row("masked_hybrid_spline")
        old = mask_row("hybrid_spline")
        samples = pair["background_reconstruction_validation"]["samples"]
        reconstruction_deltas = [
            sample["by_background"]["masked_hybrid_spline"]["mae"]
            - sample["by_background"]["hybrid_spline"]["mae"]
            for sample in samples
        ]
        y_source = abs(float(masked["fit"]["low_frequency_y_source_ncc"]))
        residual_source = abs(float(masked["fit"]["low_frequency_residual_source_ncc"]))
        output[label] = {
            "cohort_stratum": pair.get("cohort_stratum"),
            "detection_passed": bool(masked["detection_gate"]["passed"]),
            "alpha": float(masked["fit"]["alpha_fold_mean"]),
            "alpha_fold_cv": float(masked["fit"]["alpha_fold_cv"]),
            "cv_r2": float(masked["fit"]["cv_r2"]),
            "null_cv_r2_p95": float(masked["null_summary"]["cv_r2_p95"]),
            "residual_variance_ratio": float(masked["fit"]["residual_variance_ratio"]),
            "low_frequency_y_source_ncc_abs": y_source,
            "low_frequency_residual_source_ncc_abs": residual_source,
            "source_alignment_ratio": residual_source / max(y_source, 1e-12),
            "reconstruction_sample_count": len(reconstruction_deltas),
            "reconstruction_delta_mae_median": float(np.median(reconstruction_deltas)),
            "reconstruction_improved_samples": int(sum(value < 0 for value in reconstruction_deltas)),
            "old_cv_r2": float(old["fit"]["cv_r2"]),
            "old_alpha": float(old["fit"]["alpha_fold_mean"]),
        }
    return output


def bootstrap_median(values: list[float], seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_SAMPLES, array.size))
    medians = np.median(array[indices], axis=1)
    return {
        "median": float(np.median(array)),
        "ci025": float(np.percentile(medians, 2.5)),
        "ci975": float(np.percentile(medians, 97.5)),
    }


def summarize_audit(
    analyses: dict[str, dict[str, Any]],
    configurations: list[dict[str, Any]],
) -> dict[str, Any]:
    rows_by_config = {
        name: extract_pair_rows(analysis) for name, analysis in analyses.items()
    }
    gated_names = [row["name"] for row in configurations if row["gate"]]
    nominal = rows_by_config["nominal"]

    detection = {"core": {}, "controls": {}}
    for pair in CORE_PAIRS:
        values = [rows_by_config[name][pair]["detection_passed"] for name in gated_names]
        detection["core"][pair] = {
            "pass_count": int(sum(values)),
            "configuration_count": len(values),
            "pass_rate": float(np.mean(values)),
        }
    for pair in CONTROL_PAIRS:
        values = [not rows_by_config[name][pair]["detection_passed"] for name in gated_names]
        detection["controls"][pair] = {
            "nondetection_count": int(sum(values)),
            "configuration_count": len(values),
            "nondetection_rate": float(np.mean(values)),
        }

    alpha_rows = []
    for name in gated_names:
        for pair, row in rows_by_config[name].items():
            reference = nominal[pair]["alpha"]
            relative_change = (row["alpha"] - reference) / max(abs(reference), 1e-12)
            alpha_rows.append({
                "configuration": name,
                "pair": pair,
                "relative_change": float(relative_change),
                "within_tolerance": bool(
                    abs(relative_change) <= THRESHOLDS["alpha_relative_tolerance"]
                ),
            })
    core_alpha = [row for row in alpha_rows if row["pair"] in CORE_PAIRS]
    eligible_pairs = [
        pair for pair, row in nominal.items()
        if row["detection_passed"]
        and abs(row["alpha"]) >= THRESHOLDS["eligible_nominal_alpha_minimum"]
    ]
    eligible_alpha = [row for row in alpha_rows if row["pair"] in eligible_pairs]
    alpha_summary = {
        "eligible_pairs": eligible_pairs,
        "core_within_tolerance_rate": float(np.mean([
            row["within_tolerance"] for row in core_alpha
        ])),
        "eligible_within_tolerance_rate": float(np.mean([
            row["within_tolerance"] for row in eligible_alpha
        ])),
        "core_max_absolute_relative_change": float(max(
            abs(row["relative_change"]) for row in core_alpha
        )),
        "eligible_max_absolute_relative_change": float(max(
            abs(row["relative_change"]) for row in eligible_alpha
        )),
        "detail": alpha_rows,
    }

    reconstruction = {}
    for config_index, name in enumerate(rows_by_config):
        deltas = [
            row["reconstruction_delta_mae_median"]
            for row in rows_by_config[name].values()
        ]
        reconstruction[name] = {
            "pair_count": len(deltas),
            "improved_pair_count": int(sum(value < 0 for value in deltas)),
            "improved_pair_rate": float(np.mean(np.asarray(deltas) < 0)),
            "pair_median_delta_mae": bootstrap_median(
                deltas, RANDOM_SEED + config_index,
            ),
        }

    source_alignment = {}
    for name in gated_names:
        source_alignment[name] = {
            pair: {
                "ratio": rows_by_config[name][pair]["source_alignment_ratio"],
                "residual_variance_ratio": rows_by_config[name][pair][
                    "residual_variance_ratio"
                ],
            }
            for pair in CORE_PAIRS
        }

    checks = {
        "core_detection_stable": all(
            row["pass_rate"] >= THRESHOLDS["core_detection_min_rate"]
            for row in detection["core"].values()
        ),
        "controls_remain_nondetected": all(
            row["nondetection_rate"] >= THRESHOLDS["control_nondetection_min_rate"]
            for row in detection["controls"].values()
        ),
        "core_alpha_stable": (
            alpha_summary["core_within_tolerance_rate"]
            >= THRESHOLDS["core_alpha_within_tolerance_min_rate"]
        ),
        "eligible_alpha_stable": (
            alpha_summary["eligible_within_tolerance_rate"]
            >= THRESHOLDS["eligible_alpha_within_tolerance_min_rate"]
        ),
        "pseudo_occlusion_consistently_better": all(
            reconstruction[name]["improved_pair_rate"]
            >= THRESHOLDS["pseudo_improved_pair_min_rate"]
            and reconstruction[name]["pair_median_delta_mae"]["median"] < 0
            for name in gated_names
        ),
        "nominal_pseudo_occlusion_ci_below_zero": (
            reconstruction["nominal"]["pair_median_delta_mae"]["ci975"] < 0
        ),
        "core_source_alignment_attenuated": all(
            metrics["ratio"] <= THRESHOLDS["source_alignment_ratio_maximum"]
            for rows in source_alignment.values() for metrics in rows.values()
        ),
        "core_residual_variance_reduced": all(
            metrics["residual_variance_ratio"]
            <= THRESHOLDS["core_residual_variance_ratio_maximum"]
            for rows in source_alignment.values() for metrics in rows.values()
        ),
    }
    return {
        "freeze_recommended": bool(all(checks.values())),
        "checks": checks,
        "thresholds": THRESHOLDS,
        "gated_configurations": gated_names,
        "stress_test_configurations": [
            row["name"] for row in configurations if not row["gate"]
        ],
        "detection_stability": detection,
        "alpha_stability": alpha_summary,
        "pseudo_occlusion": reconstruction,
        "source_alignment": source_alignment,
        "pair_results_by_configuration": rows_by_config,
    }


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    decision = "FREEZE" if summary["freeze_recommended"] else "DO NOT FREEZE"
    lines = [
        "# Background freeze audit",
        "",
        f"Decision: **{decision}**",
        "",
        "The affine model remains `Y = alpha * X + intercept + error`; only background "
        "parameters were perturbed.",
        "",
        "## Gate checks",
        "",
        "| Check | Pass |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {'yes' if passed else 'no'} |"
        for name, passed in summary["checks"].items()
    )
    lines.extend(["", "## Detection stability", "", "| Pair | Role | Stable rate |", "|---|---|---:|"])
    for pair, row in summary["detection_stability"]["core"].items():
        lines.append(f"| {pair} | core positive | {row['pass_rate']:.3f} |")
    for pair, row in summary["detection_stability"]["controls"].items():
        lines.append(f"| {pair} | nondetected control | {row['nondetection_rate']:.3f} |")
    alpha = summary["alpha_stability"]
    nominal_reconstruction = summary["pseudo_occlusion"]["nominal"]
    stress = summary["pair_results_by_configuration"].get("otsu_only_stress", {})
    lines.extend([
        "",
        "## Compact metrics",
        "",
        f"- Core alpha within +/-20%: {alpha['core_within_tolerance_rate']:.3f}",
        f"- Eligible alpha within +/-20%: {alpha['eligible_within_tolerance_rate']:.3f}",
        "- Nominal pseudo-occlusion pair median delta MAE: "
        f"{nominal_reconstruction['pair_median_delta_mae']['median']:.6f} "
        f"(pair-bootstrap 95% CI "
        f"{nominal_reconstruction['pair_median_delta_mae']['ci025']:.6f} to "
        f"{nominal_reconstruction['pair_median_delta_mae']['ci975']:.6f})",
        f"- Nominal improved pairs: {nominal_reconstruction['improved_pair_count']}"
        f"/{nominal_reconstruction['pair_count']}",
        "",
        "## Configuration scope",
        "",
        f"Gated: {', '.join(summary['gated_configurations'])}",
        "",
        f"Stress only: {', '.join(summary['stress_test_configurations'])}",
        "",
        "The Otsu-only run is a segmentation-method stress test and is reported but not "
        "used to tune or select the frozen SAM-based configuration.",
        "",
    ])
    if stress:
        lines.extend([
            "## Otsu-only stress observation",
            "",
            "| Pair | Expected role | Detection passed |",
            "|---|---|---:|",
        ])
        for pair in (*CORE_PAIRS, *CONTROL_PAIRS):
            role = "core positive" if pair in CORE_PAIRS else "nondetected control"
            passed = "yes" if stress[pair]["detection_passed"] else "no"
            lines.append(f"| {pair} | {role} | {passed} |")
        lines.extend([
            "",
            "Otsu-only changes the 4->5 control to detected. The freeze decision therefore "
            "applies to the hashed SAM mask archive and does not declare SAM and Otsu "
            "interchangeable.",
            "",
        ])
    return "\n".join(lines)


def run_configuration(
    repo_root: Path,
    output_root: Path,
    data_dir: Path,
    mask_archive: Path,
    config: dict[str, Any],
    reuse_existing: bool,
) -> dict[str, Any]:
    name = config["name"]
    config_dir = output_root / "runs" / name
    result_path = config_dir / "analysis_results.json"
    if reuse_existing and result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    command = [
        sys.executable,
        str(repo_root / "scripts" / "analyze_pseudo_ghost_mechanism.py"),
        "--data-dir", str(data_dir),
        "--output-dir", str(config_dir),
        "--pairs", "single-lag",
        "--summary-only",
        "--background-modes", "hybrid_spline,masked_hybrid_spline",
        "--primary-background", "masked_hybrid_spline",
        "--mask-dilation-pixels", str(config["mask_dilation_pixels"]),
        "--spline-smoothness", str(config["spline_smoothness"]),
        "--background-huber-delta", str(config["huber_delta"]),
        "--fixed-pattern-iterations", str(config["fixed_pattern_iterations"]),
        "--fixed-pattern-huber-iterations", str(config["fixed_pattern_huber_iterations"]),
        "--skip-background-oof",
        "--validate-background-reconstruction",
    ]
    if config["mask_source"] != "otsu_only":
        command.extend(["--source-mask-npz", str(mask_archive)])
    if name != "nominal":
        command.append("--skip-input-hashes")

    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Configuration {name} failed; see {config_dir / 'run.log'}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite background freeze sensitivity audit")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/AI修残影例图"))
    parser.add_argument(
        "--source-mask-npz", type=Path,
        default=Path("outputs/pseudo_ghost_mechanism_v1/sam_masks_single_lag.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/background_freeze_audit_v1"),
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--keep-runs", action="store_true",
        help="Keep per-configuration intermediate runs; default uses a temporary directory",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = (repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir
    mask_archive = (
        (repo_root / args.source_mask_npz).resolve()
        if not args.source_mask_npz.is_absolute() else args.source_mask_npz
    )
    output_root = (
        (repo_root / args.output_dir).resolve()
        if not args.output_dir.is_absolute() else args.output_dir
    )
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)
    if not mask_archive.exists():
        raise FileNotFoundError(mask_archive)
    output_root.mkdir(parents=True, exist_ok=True)

    configurations = [resolved_config(row) for row in SENSITIVITY_CONFIGS]

    def run_all(run_root: Path) -> dict[str, dict[str, Any]]:
        analyses = {}
        for index, config in enumerate(configurations, start=1):
            print(f"[{index}/{len(configurations)}] {config['name']}", flush=True)
            analyses[config["name"]] = run_configuration(
                repo_root=repo_root,
                output_root=run_root,
                data_dir=data_dir,
                mask_archive=mask_archive,
                config=config,
                reuse_existing=args.reuse_existing,
            )
        return analyses

    if args.reuse_existing or args.keep_runs:
        analyses = run_all(output_root)
    else:
        with tempfile.TemporaryDirectory(prefix="xray-bg-freeze-") as temporary:
            analyses = run_all(Path(temporary))

    summary = summarize_audit(analyses, configurations)
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    result = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision_scope": (
            "Freeze the nominal hashed-SAM background configuration only; "
            "affine ghost model unchanged"
        ),
        "git_commit_before_audit_changes": git_commit,
        "source_mask_archive": str(mask_archive),
        "source_mask_sha256": _sha256(mask_archive),
        "configurations": configurations,
        "summary": summary,
    }
    result_path = output_root / "background_freeze_audit.json"
    report_path = output_root / "background_freeze_audit.md"
    result_path.write_text(
        json.dumps(_clean_json(result), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    report_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({
        "decision": "freeze" if summary["freeze_recommended"] else "do_not_freeze",
        "checks": summary["checks"],
        "json": str(result_path),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
