"""Mechanism study for observable CR afterglow in dark acquisitions.

This script deliberately does not treat a dark image as clean ground truth.  It
constructs an observable signal

    Y_t = D_t - median(D_t) - B_{-t}

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
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pairs(value: str) -> tuple[tuple[int, int], ...]:
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
        "low_frequency_y_prediction_ncc": _correlation(y_low, p_low, evaluated),
        "low_frequency_residual_source_ncc": _correlation(r_low, x_low, evaluated),
        "residual_neighbor_correlation": _neighbor_correlation(residual, evaluated),
    }


def _metric_view(fit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fit.items()
            if key not in {"prediction", "residual", "evaluated_mask"}}


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


def evaluate_with_nulls(
    x: np.ndarray,
    y: np.ndarray,
    source_saturation: np.ndarray,
    mask_mode: str,
    folds: np.ndarray,
    future_sources: list[tuple[int, np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    mask = np.ones_like(y, dtype=bool)
    if mask_mode == "exclude_saturated":
        mask &= source_saturation < 0.05
    true_fit = strict_affine_oof(x, y, mask, folds)
    null_rows = []

    for light_index, future_x, future_saturation in future_sources:
        future_mask = np.ones_like(y, dtype=bool)
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
        null_mask = np.ones_like(y, dtype=bool)
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
    loo_baseline: np.ndarray,
    y: np.ndarray,
    fit: dict[str, Any],
) -> None:
    prediction, residual = fit["prediction"], fit["residual"]
    common_amplitude = float(np.percentile(np.abs(y[np.isfinite(y)]), 99.5))
    common_amplitude = max(common_amplitude, 1e-6)
    baseline_amplitude = float(np.percentile(np.abs(loo_baseline[np.isfinite(loo_baseline)]), 99.5))
    baseline_amplitude = max(baseline_amplitude, 1e-6)
    x_limits, dark_limits = _finite_limits(x), _finite_limits(raw_dark)

    panels = [
        (x, "1  Source X = L - Q75(L)", "gray", x_limits),
        (raw_dark, "2  Raw dark D", "gray", dark_limits),
        (loo_baseline, "3  LOO baseline B", "coolwarm", (-baseline_amplitude, baseline_amplitude)),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict OOF pseudo-ghost mechanism study")
    parser.add_argument("--data-dir", default="data/raw/AI修残影例图")
    parser.add_argument("--output-dir", default="outputs/pseudo_ghost_mechanism_v1")
    parser.add_argument("--pairs", type=parse_pairs, default=DEFAULT_PAIRS,
                        help="Comma-separated light:dark pairs (default: 5:6,27:28)")
    parser.add_argument("--skip-input-hashes", action="store_true",
                        help="Skip SHA-256 provenance hashes for faster exploratory runs")
    args = parser.parse_args()

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    pairs: tuple[tuple[int, int], ...] = args.pairs
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 32):
        path = data_dir / f"{index}-dark.dcm"
        if not path.exists():
            raise FileNotFoundError(path)
    for light_index, _ in pairs:
        path = data_dir / f"{light_index}-light.dcm"
        if not path.exists():
            raise FileNotFoundError(path)

    dark_cache = {block: {} for block in BLOCK_SIZES}
    for index in range(1, 32):
        image = pydicom.dcmread(data_dir / f"{index}-dark.dcm").pixel_array.astype(np.float32)
        for block in BLOCK_SIZES:
            dark_cache[block][index] = block_mean(image, block)

    # Future-light nulls require every light from the earliest target onward.
    earliest_future = min(dark_index for _, dark_index in pairs)
    needed_lights = sorted(set([light for light, _ in pairs] + list(range(earliest_future, 32))))
    light_cache = {block: {} for block in BLOCK_SIZES}
    saturation_cache = {block: {} for block in BLOCK_SIZES}
    for index in needed_lights:
        image = pydicom.dcmread(data_dir / f"{index}-light.dcm").pixel_array.astype(np.float32)
        saturation = (image == image.max()).astype(np.float32)
        for block in BLOCK_SIZES:
            light_cache[block][index] = block_mean(image, block)
            saturation_cache[block][index] = block_mean(saturation, block)

    input_paths = [data_dir / f"{index}-dark.dcm" for index in range(1, 32)]
    input_paths.extend(data_dir / f"{index}-light.dcm" for index in needed_lights)
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
        pair_dir.mkdir(parents=True, exist_ok=True)
        block_results = []
        primary_maps = None

        for block in BLOCK_SIZES:
            darks, lights = dark_cache[block], light_cache[block]
            loo_baseline = baseline(darks, dark_index, "loo_median")
            raw_dark = darks[dark_index]
            y = raw_dark - np.median(raw_dark) - loo_baseline
            x = source(lights, light_index)
            folds = spatial_folds(y.shape)
            future_sources = [
                (future, source(lights, future), saturation_cache[block][future])
                for future in range(dark_index, 32)
            ]

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
                mask_results.append({
                    "mask_mode": mask_mode,
                    "fit": _metric_view(fit),
                    "null_summary": null_summary,
                    "null_detail": null_rows,
                })
                if block == PRIMARY_BLOCK and mask_mode == "all":
                    primary_maps = (x, raw_dark, loo_baseline, y, folds, fit)
            block_results.append({"block_size": block, "mask_results": mask_results})

        if primary_maps is None:
            raise RuntimeError("Primary block result was not generated")
        x, raw_dark, loo_baseline, y, folds, primary_fit = primary_maps
        render_six_panel(pair_dir / "six_panel_oof.png", pair_label, x, raw_dark,
                         loo_baseline, y, primary_fit)
        np.savez_compressed(
            pair_dir / "oof_maps_block16.npz",
            source=x.astype(np.float32),
            raw_dark=raw_dark.astype(np.float32),
            loo_baseline=loo_baseline.astype(np.float32),
            observable_signal=y.astype(np.float32),
            oof_prediction=primary_fit["prediction"].astype(np.float32),
            oof_residual=primary_fit["residual"].astype(np.float32),
            evaluated_mask=primary_fit["evaluated_mask"],
            spatial_folds=folds,
        )
        pair_results.append({
            "light_index": light_index,
            "dark_index": dark_index,
            "role": "single_frame_mechanism_discovery",
            "primary_block_size": PRIMARY_BLOCK,
            "six_panel": str((pair_dir / "six_panel_oof.png").resolve()),
            "primary_maps": str((pair_dir / "oof_maps_block16.npz").resolve()),
            "block_results": block_results,
        })

    result = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "study_question": "Can detectable previous-exposure structure in a later dark acquisition be explained by a low-dimensional linear spatial model?",
        "evidence_boundary": [
            "Y is an observable pseudo-ghost signal, not clean ground-truth ghost.",
            "Y may contain read noise, baseline error, drift, and older-frame memory.",
            "These selected pairs are for mechanism discovery, not prevalence or generalization claims.",
        ],
        "definitions": {
            "source": "X_(t-1) = L_(t-1) - Q75(L_(t-1)), evaluated on each block-mean grid",
            "baseline": "B_(-t) = pixelwise median of all other median-centered dark block maps",
            "observable_signal": "Y_t = D_t - median(D_t) - B_(-t)",
            "model": "Y_t = alpha * X_(t-1) + intercept + error",
            "prediction": "Four-fold strict spatial out-of-fold prediction",
        },
        "configuration": {
            "data_directory": str(data_dir.resolve()),
            "pairs": [list(pair) for pair in pairs],
            "block_sizes": list(BLOCK_SIZES),
            "primary_visualization_block_size": PRIMARY_BLOCK,
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
        "input_provenance": provenance,
        "pair_results": pair_results,
    }
    result_path = output_dir / "analysis_results.json"
    result_path.write_text(json.dumps(clean_json(result), ensure_ascii=False, indent=2), encoding="utf-8")

    compact = {
        "output": str(result_path.resolve()),
        "pairs": [
            {
                "pair": f"{row['light_index']}->{row['dark_index']}",
                "block16_all": {
                    "fit": next(
                        mask_row["fit"] for block_row in row["block_results"]
                        if block_row["block_size"] == PRIMARY_BLOCK
                        for mask_row in block_row["mask_results"]
                        if mask_row["mask_mode"] == "all"
                    ),
                    "null_summary": next(
                        mask_row["null_summary"] for block_row in row["block_results"]
                        if block_row["block_size"] == PRIMARY_BLOCK
                        for mask_row in block_row["mask_results"]
                        if mask_row["mask_mode"] == "all"
                    ),
                },
            }
            for row in pair_results
        ],
    }
    print(json.dumps(clean_json(compact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
