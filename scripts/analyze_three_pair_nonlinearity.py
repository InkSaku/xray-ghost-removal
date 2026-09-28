"""Retrospective three-pair view of the frozen intensity-response diagnostics.

The original two-pair candidate decision and its hashed figure remain intact.
This script adds 3->4 as a descriptive comparison using the same frozen arrays,
model definitions, and spatial folds; it does not make a new M2 decision.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.analyze_frozen_candidates import (
    PairData,
    load_frozen_pairs,
    quadratic_oof,
    sha256,
)
from scripts.analyze_intensity_nonlinearity import (
    HINGE_QUANTILES,
    SHAPE_QUANTILES,
    compare_models,
    nested_parametric_oof,
    parameter_stability,
    render_diagnostics,
    spline_oof,
)
from scripts.analyze_pseudo_ghost_mechanism import clean_json


PAIRS = ("3->4", "5->6", "27->28")
DEFAULT_OUTPUT = Path(
    "outputs/pseudo_ghost_mechanism_v1/intensity_nonlinearity_three_pair_v1"
)


def load_three_pairs(repo_root: Path) -> tuple[list[PairData], dict]:
    original_dir = repo_root / "outputs/pseudo_ghost_mechanism_v1"
    original_pairs, original_provenance = load_frozen_pairs(original_dir)
    cross_manifest_path = repo_root / "outputs/mhinge_cross_pair_v1/analysis_results.json"
    cross_manifest = json.loads(cross_manifest_path.read_text(encoding="utf-8"))
    if cross_manifest.get("frozen_background") != original_provenance["frozen_background"]:
        raise ValueError("The three-pair export uses a different frozen background")
    if cross_manifest["configuration"]["block_sizes"] != [16]:
        raise ValueError("The three-pair export must use block-16 maps")
    if cross_manifest["configuration"]["background_modes"] != ["masked_hybrid_spline"]:
        raise ValueError("The three-pair export uses a different background method")
    rows = [
        row for row in cross_manifest["pair_results"]
        if (row["light_index"], row["dark_index"]) == (3, 4)
    ]
    if len(rows) != 1:
        raise ValueError("The frozen 13-pair manifest must contain one 3->4 pair")
    artifact = rows[0]["output_artifacts"]["primary_maps"]
    # The retained original archive is byte-identical to the 13-pair export.
    archive_path = original_dir / "pair_3_4/oof_maps_block16.npz"
    if sha256(archive_path) != artifact["sha256"]:
        raise ValueError("The retained 3->4 map differs from the frozen manifest")
    with np.load(archive_path) as archive:
        required = {
            "source", "observable_signal", "raw_dark", "background",
            "source_support", "evaluated_mask", "oof_prediction",
            "oof_residual", "spatial_folds", "background_mode",
        }
        if required - set(archive.files):
            raise ValueError("The retained 3->4 archive is incomplete")
        if str(archive["background_mode"]) != "masked_hybrid_spline":
            raise ValueError("The retained 3->4 background method differs")
        x = archive["source"].astype(np.float64)
        y = archive["observable_signal"].astype(np.float64)
        raw = archive["raw_dark"].astype(np.float64)
        background = archive["background"].astype(np.float64)
        support = archive["source_support"].astype(bool)
        evaluated = archive["evaluated_mask"].astype(bool)
        prediction = archive["oof_prediction"].astype(np.float64)
        residual = archive["oof_residual"].astype(np.float64)
        folds = archive["spatial_folds"].astype(np.int8)
    if len({a.shape for a in (x, y, raw, background, support, evaluated,
                               prediction, residual, folds)}) != 1:
        raise ValueError("The 3->4 arrays have inconsistent shapes")
    if not np.allclose(y, raw - background, atol=2e-4, rtol=0):
        raise ValueError("The 3->4 observable signal differs from raw minus BG")
    valid = evaluated & np.isfinite(x) & np.isfinite(y) & np.isfinite(prediction)
    if set(np.unique(folds[valid]).tolist()) != {0, 1, 2, 3}:
        raise ValueError("The 3->4 spatial folds are incomplete")
    if not np.allclose(residual[valid], y[valid] - prediction[valid], atol=2e-5, rtol=0):
        raise ValueError("The 3->4 M1 residual is inconsistent")
    added_pair = PairData(
        label="3->4", x=x, y=y, folds=folds, evaluated=valid,
        source_support=support, prediction=prediction, residual=residual,
    )
    pairs = [added_pair, *original_pairs]
    if tuple(pair.label for pair in pairs) != PAIRS:
        raise ValueError("Three-pair order differs from the declared cohort")
    provenance = {
        "original_two_pair_manifest": original_provenance,
        "cross_pair_manifest": {
            "path": str(cross_manifest_path.resolve()),
            "sha256": sha256(cross_manifest_path),
        },
        "retained_3_4_archive": {
            "path": str(archive_path.resolve()),
            "sha256": artifact["sha256"],
        },
    }
    return pairs, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    pairs, provenance = load_three_pairs(repo_root)
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
    original_result_path = (
        repo_root / "outputs/pseudo_ghost_mechanism_v1/intensity_nonlinearity/model_results.json"
    )
    original_result = json.loads(original_result_path.read_text(encoding="utf-8"))
    for label in ("5->6", "27->28"):
        for model in predictions:
            old = original_result["pair_comparisons"][label][model]["metrics"]
            new = comparisons[label][model]["metrics"]
            for key in ("cv_r2", "rmse", "source_edge_rmse", "outside_source_rmse"):
                if not np.isclose(old[key], new[key], atol=1e-9, rtol=0):
                    raise ValueError(f"Original frozen result changed: {label} {model} {key}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="xray-three-pair-nonlinearity-") as temporary:
        temporary_dir = Path(temporary)
        figure_path = temporary_dir / "nonlinearity_diagnostics.png"
        result_path = temporary_dir / "model_results.json"
        render_diagnostics(
            figure_path, pairs, predictions, comparisons,
            title="Frozen observable Y: retrospective three-pair intensity comparison",
        )
        result = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "cohort": list(PAIRS),
            "scope": "Retrospective three-pair descriptive comparison on frozen block-16 data",
            "evidence_boundary": [
                "3->4 was added after the original two-pair model decision.",
                "The original hashed two-pair result and figure remain retained as audit evidence.",
                "All models reuse the original functions and spatial folds; no model is promoted to M2.",
                "X is source contrast, not measured incident dose; Y is not clean ground truth.",
                "Residual structure may include background error and older-frame memory.",
            ],
            "source_provenance": provenance,
            "original_two_pair_decision": {
                "path": str(original_result_path.resolve()),
                "sha256": sha256(original_result_path),
            },
            "code_provenance": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "models": {
                "M1": {"status": "frozen_reference"},
                "Mquad_diagnostic": {"status": "diagnostic_only",
                                     "fold_parameters": quadratic["fold_parameters"]},
                "Mspline_diagnostic": {"status": "diagnostic_only",
                                       "fold_parameters": spline["fold_parameters"]},
                "Msat": {"status": "original_primary_candidate_retrospectively_applied_to_3_4",
                         "candidate_quantiles": list(SHAPE_QUANTILES),
                         "fold_parameters": saturation["fold_parameters"],
                         "parameter_stability": stability["Msat"]},
                "Mhinge": {"status": "original_alternative_candidate_retrospectively_applied_to_3_4",
                           "candidate_quantiles": list(HINGE_QUANTILES),
                           "fold_parameters": hinge["fold_parameters"],
                           "parameter_stability": stability["Mhinge"]},
            },
            "pair_comparisons": comparisons,
            "output_artifacts": {
                "nonlinearity_diagnostics": {
                    "relative_path": figure_path.name,
                    "sha256": sha256(figure_path),
                    "size_bytes": figure_path.stat().st_size,
                }
            },
        }
        result_path.write_text(
            json.dumps(clean_json(result), ensure_ascii=False, indent=2), encoding="utf-8",
        )
        shutil.copyfile(figure_path, output_dir / figure_path.name)
        shutil.copyfile(result_path, output_dir / result_path.name)
    print(json.dumps({
        "figure": str((output_dir / "nonlinearity_diagnostics.png").resolve()),
        "result": str((output_dir / "model_results.json").resolve()),
        "summary": {
            label: {
                model: {
                    "r2": comparisons[label][model]["metrics"]["cv_r2"],
                    "rmse": comparisons[label][model]["metrics"]["rmse"],
                }
                for model in ("M1", "Msat", "Mhinge")
            }
            for label in PAIRS
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
