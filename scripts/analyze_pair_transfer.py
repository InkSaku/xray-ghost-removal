"""Exploratory two-way M1 transfer on frozen 3->4 and 5->6 block maps.

The target is observable Y = dark - estimated background, not true ghost or
clean-image ground truth. All model choices and calibration regions are fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_frozen_candidates import PairData, model_metrics, sha256
from scripts.analyze_pseudo_ghost_mechanism import (
    TRIM_PERCENTILES,
    _fit_affine,
    _training_keep,
    clean_json,
)


PAIRS = ("3->4", "5->6")
TILE_SIDE = 24  # Frozen folds consist of 24 x 24 block tiles.
TILE_COUNTS = (4, 8, 16)  # 6.25%, 12.5%, 25% of the full image.
BUFFER_BLOCKS = 2


def load_pairs(dataset_dir: Path, fallback_dir: Path) -> tuple[dict[str, PairData], dict]:
    manifest_path = dataset_dir / "analysis_results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("frozen_background", {}).get("verified"):
        raise ValueError("The 13-pair manifest does not verify its frozen background")
    config = manifest["configuration"]
    if config["block_sizes"] != [16] or config["background_modes"] != ["masked_hybrid_spline"]:
        raise ValueError("Expected the frozen block-16 masked hybrid background")

    pairs: dict[str, PairData] = {}
    paths: dict[str, dict] = {}
    for row in manifest["pair_results"]:
        label = f"{row['light_index']}->{row['dark_index']}"
        if label not in PAIRS:
            continue
        artifact = row["output_artifacts"]["primary_maps"]
        path = dataset_dir / artifact["relative_path"]
        fallback_used = False
        if not path.exists() and label == "3->4":
            path = fallback_dir / artifact["relative_path"]
            fallback_used = True
        if not path.exists() or sha256(path) != artifact["sha256"]:
            raise ValueError(f"Missing or hash-mismatched frozen NPZ for {label}: {path}")
        with np.load(path) as archive:
            required = {
                "source", "observable_signal", "raw_dark", "background",
                "source_support", "evaluated_mask", "oof_prediction",
                "oof_residual", "spatial_folds", "background_mode",
            }
            if required - set(archive.files):
                raise ValueError(f"Incomplete frozen NPZ for {label}")
            if str(archive["background_mode"]) != "masked_hybrid_spline":
                raise ValueError(f"Wrong frozen background mode for {label}")
            x = archive["source"].astype(np.float64)
            y = archive["observable_signal"].astype(np.float64)
            raw = archive["raw_dark"].astype(np.float64)
            background = archive["background"].astype(np.float64)
            evaluated = archive["evaluated_mask"].astype(bool)
            folds = archive["spatial_folds"].astype(np.int8)
            support = archive["source_support"].astype(bool)
            oof = archive["oof_prediction"].astype(np.float64)
            residual = archive["oof_residual"].astype(np.float64)
        if {a.shape for a in (x, y, raw, background, evaluated, folds, support, oof, residual)} != {(192, 192)}:
            raise ValueError(f"Unexpected block-map shapes for {label}")
        if not np.allclose(y, raw - background, atol=2e-4, rtol=0):
            raise ValueError(f"Observable Y identity failed for {label}")
        valid = evaluated & np.isfinite(x) & np.isfinite(y) & np.isfinite(oof)
        if not np.allclose(residual[valid], y[valid] - oof[valid], atol=2e-5, rtol=0):
            raise ValueError(f"Stored OOF residual identity failed for {label}")
        if set(np.unique(folds[valid]).tolist()) != {0, 1, 2, 3}:
            raise ValueError(f"Incomplete spatial folds for {label}")
        pairs[label] = PairData(label, x, y, folds, valid, support, oof, residual)
        paths[label] = {
            "path": str(path), "sha256": artifact["sha256"],
            "fallback_copy_used": fallback_used,
        }
    if set(pairs) != set(PAIRS):
        raise ValueError("The frozen manifest must contain both selected pairs")
    return pairs, {
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "frozen_background": manifest["frozen_background"], "pairs": paths,
    }


def calibration_tiles(pair: PairData, fold: int) -> tuple[np.ndarray, list[int]]:
    """Choose source-rich and air tiles using X/support only; never inspect Y."""
    rr, cc = np.indices(pair.x.shape)
    tile = (rr // TILE_SIDE) * (pair.x.shape[1] // TILE_SIDE) + cc // TILE_SIDE
    ids = np.unique(tile[pair.folds == fold])
    if ids.size != 16 or any(np.sum((tile == t) & (pair.folds == fold)) != TILE_SIDE**2 for t in ids):
        raise ValueError("Frozen spatial fold is not a set of 16 complete tiles")
    source = sorted(
        (int(t) for t in ids if np.any(pair.source_support[tile == t])),
        key=lambda t: (-int(np.sum(pair.source_support[tile == t])), t),
    )
    air = sorted(int(t) for t in ids if not np.any(pair.source_support[tile == t]))
    if len(source) < 4 or len(air) < 4:
        raise ValueError("Four source and air calibration tiles are required")
    # Two source-bearing and two air tiles at the first budget; four of each
    # at the second. Remaining tiles complete the entire calibration fold.
    order = source[:2] + air[:2] + source[2:4] + air[2:4]
    order += [int(t) for t in ids if int(t) not in order]
    return tile, order


def masks_for_budget(pair: PairData, fold: int, tile_count: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    tile, order = calibration_tiles(pair, fold)
    calibration = pair.evaluated & (pair.folds == fold) & np.isin(tile, order[:tile_count])
    test = pair.evaluated & (pair.folds != fold)
    # Keep the test region identical at every calibration budget. Otherwise a
    # changing score could be caused by dropping more difficult border blocks.
    whole_calibration_fold = pair.evaluated & (pair.folds == fold)
    test &= ~binary_dilation(whole_calibration_fold, iterations=BUFFER_BLOCKS)
    if np.any(calibration & test) or int(test.sum()) < 1000:
        raise ValueError("Calibration/test partition is invalid")
    return calibration, test, order[:tile_count]


def fit_on(pair: PairData, base: np.ndarray) -> tuple[float, float, dict]:
    kept, thresholds = _training_keep(pair.x, pair.y, base, TRIM_PERCENTILES)
    alpha, intercept = _fit_affine(pair.x[kept], pair.y[kept])
    return alpha, intercept, {"fit_blocks": int(kept.sum()), "trim_thresholds": thresholds}


def scored(pair: PairData, prediction: np.ndarray, test: np.ndarray) -> dict:
    return model_metrics(replace(pair, evaluated=test), prediction)


def analyze(pairs: dict[str, PairData]) -> tuple[dict, dict]:
    results = {}
    display = {}
    for source_label, target_label in ((PAIRS[0], PAIRS[1]), (PAIRS[1], PAIRS[0])):
        source, target = pairs[source_label], pairs[target_label]
        alpha, intercept, fit_details = fit_on(source, source.evaluated)
        direct = alpha * target.x + intercept
        direction = f"{source_label} -> {target_label}"
        folds = []
        for fold in range(4):
            budgets = []
            for count in TILE_COUNTS:
                calibration, test, selected = masks_for_budget(target, fold, count)
                local_alpha, local_intercept, local_fit = fit_on(target, calibration)
                calibrated = local_alpha * target.x + local_intercept
                budgets.append({
                    "tile_count": count,
                    "calibration_fraction_full_image": float(calibration.sum() / target.evaluated.sum()),
                    "calibration_blocks": int(calibration.sum()),
                    "calibration_source_blocks": int((calibration & target.source_support).sum()),
                    "test_blocks": int(test.sum()),
                    "selected_tile_ids": selected,
                    "calibrated_alpha": local_alpha,
                    "calibrated_intercept": local_intercept,
                    "calibrated_fit": local_fit,
                    "direct": scored(target, direct, test),
                    "calibrated": scored(target, calibrated, test),
                    "within_pair_oof": scored(target, target.prediction, test),
                })
                if fold == 0 and count == 16:
                    display[direction] = (target, test, direct, calibrated)
            folds.append({"calibration_fold": fold, "budgets": budgets})
        results[direction] = {
            "source_pair": source_label, "target_pair": target_label,
            "source_fit": {"alpha": alpha, "intercept": intercept, **fit_details},
            "folds": folds,
        }
    return results, display


def plot(results: dict, display: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for row, (direction, result) in enumerate(results.items()):
        ax = axes[row, 0]
        for name, marker in (("direct", "o"), ("calibrated", "s"), ("within_pair_oof", "^")):
            means = [
                np.mean([f["budgets"][i][name]["rmse"] for f in result["folds"]])
                for i in range(len(TILE_COUNTS))
            ]
            ax.plot([100*c/64 for c in TILE_COUNTS], means, marker=marker, label=name)
        ax.set(title=direction, xlabel="Target image used for calibration (%)", ylabel="RMSE of observable Y")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        target, test, direct, calibrated = display[direction]
        predictions = (direct, calibrated, target.prediction)
        labels = ("Direct transfer", "25% calibration", "Within-pair OOF")
        residuals = [np.where(test, target.y - prediction, np.nan) for prediction in predictions]
        lim = max(float(np.nanpercentile(np.abs(r), 99)) for r in residuals)
        for col, (label, residual) in enumerate(zip(labels, residuals), start=1):
            image = axes[row, col].imshow(residual, cmap="RdBu_r", vmin=-lim, vmax=lim)
            axes[row, col].set(title=f"{label} residual\nfold 0 held-out blocks", xticks=[], yticks=[])
            fig.colorbar(image, ax=axes[row, col], fraction=.046, pad=.04)
    fig.suptitle("Exploratory two-way affine transfer; 16×16 block Y = dark − estimated background", fontsize=14)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/mhinge_cross_pair_v1"))
    parser.add_argument("--fallback-dir", type=Path, default=Path("outputs/pseudo_ghost_mechanism_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pair_transfer_v1"))
    args = parser.parse_args()
    pairs, provenance = load_pairs(args.dataset_dir, args.fallback_dir)
    results, display = analyze(pairs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = args.output_dir / "transfer_diagnostics.png"
    plot(results, display, figure)
    report = {
        "schema_version": "pair_transfer_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_question": "Does a frozen per-pair M1 transfer between 3->4 and 5->6, and how much target calibration helps?",
        "evidence_boundary": "Exploratory same-session pseudo-ghost prediction; both pairs were previously inspected. No clean ground truth or independent acquisition test.",
        "definition": {
            "Y": "raw_dark - frozen estimated background",
            "X": "frozen previous-light source map",
            "model": "alpha * X + intercept, fit with training-only 0.5/99.5 percentile trimming",
            "direct": "fit on all eligible source-pair blocks; never fit target Y",
            "calibration": "fit on 4, 8, or 16 target-fold tiles chosen from X/support only; four folds rotated",
            "evaluation": "remaining three folds excluding two-block dilation around calibration; methods compared on identical blocks within each budget and fold",
            "within_pair_baseline": "stored four-fold spatial OOF M1 prediction; for calibration comparisons, scored on the same remaining target blocks",
            "tile_side_blocks": TILE_SIDE, "buffer_blocks": BUFFER_BLOCKS,
            "tile_counts": list(TILE_COUNTS), "trim_percentiles": list(TRIM_PERCENTILES),
        },
        "provenance": {**provenance, "script_sha256": sha256(Path(__file__))},
        "directions": results,
        "figure": {"path": str(figure), "sha256": sha256(figure)},
    }
    (args.output_dir / "transfer_results.json").write_text(
        json.dumps(clean_json(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        direction: {
            "source_alpha": row["source_fit"]["alpha"],
            "mean_rmse_25pct": {
                name: float(np.mean([f["budgets"][-1][name]["rmse"] for f in row["folds"]]))
                for name in ("direct", "calibrated", "within_pair_oof")
            },
        }
        for direction, row in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
