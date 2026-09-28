"""Relate frozen single-frame ghost estimates to recorded exposure settings.

This analysis reuses the 13-pair frozen-background export.  It does not fit
multi-frame terms, rebuild the background, or reinterpret dark images as
clean ground truth.  Re-running replaces the two stable output artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_dark_light_pairs import load_exposures
from scripts.analyze_frozen_candidates import sha256
from scripts.analyze_pseudo_ghost_mechanism import clean_json, strict_affine_oof


ALL_PAIRS = (
    "1->2", "2->3", "3->4", "4->5", "5->6", "6->7", "7->8",
    "8->9", "9->10", "22->23", "23->24", "25->26", "27->28",
)
LOW_SATURATION_THRESHOLD = 0.01
PERMUTATIONS = 10_000
RANDOM_SEED = 20260922
SPATIAL_ZONE_GRID = 3
MIN_ZONE_SOURCE_BLOCKS = 50
MIN_ZONE_SOURCE_FRACTION = 0.02
TECHNICAL_HEADER_FIELDS = (
    "ImageType", "Modality", "SamplesPerPixel", "PhotometricInterpretation",
    "Rows", "Columns", "BitsAllocated", "BitsStored", "HighBit",
    "PixelRepresentation", "NumberOfFrames", "PlanarConfiguration",
    "RescaleIntercept", "RescaleSlope", "RescaleType", "PresentationLUTShape",
    "PixelIntensityRelationship", "PixelIntensityRelationshipSign", "Sensitivity",
    "ExposureIndex", "TargetExposureIndex", "DeviationIndex", "DetectorType",
    "DetectorConfiguration", "AcquisitionDeviceProcessingDescription",
)


def _manifest_fit(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    return blocks[0], masks[0]["fit"], masks[0]["detection_gate"]


def spearman_summary(
    x: list[float],
    y: list[float],
    *,
    seed: int,
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    valid = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[valid], yv[valid]
    if xv.size < 4 or np.unique(xv).size < 2 or np.unique(yv).size < 2:
        return {"n": int(xv.size), "rho": None, "permutation_p_two_sided": None,
                "leave_one_out_rho_min": None, "leave_one_out_rho_max": None}
    rho = float(spearmanr(xv, yv).statistic)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = float(spearmanr(xv, rng.permutation(yv)).statistic)
        exceed += abs(permuted) >= abs(rho) - 1e-15
    loo = [
        float(spearmanr(np.delete(xv, index), np.delete(yv, index)).statistic)
        for index in range(xv.size)
    ]
    return {
        "n": int(xv.size),
        "rho": rho,
        "permutation_p_two_sided": float((exceed + 1) / (permutations + 1)),
        "permutations": permutations,
        "leave_one_out_rho_min": float(np.nanmin(loo)),
        "leave_one_out_rho_max": float(np.nanmax(loo)),
    }


def _ghost_contrast_rms(
    prediction: np.ndarray,
    evaluated: np.ndarray,
    support: np.ndarray,
) -> float:
    source = evaluated & support & np.isfinite(prediction)
    outside = evaluated & ~support & np.isfinite(prediction)
    if source.sum() == 0 or outside.sum() == 0:
        return float("nan")
    baseline = float(np.median(prediction[outside]))
    return float(np.sqrt(np.mean((prediction[source] - baseline) ** 2)))


def source_morphology(
    source: np.ndarray,
    evaluated: np.ndarray,
    support: np.ndarray,
) -> dict[str, Any]:
    valid = evaluated & support & np.isfinite(source)
    if valid.sum() == 0:
        raise ValueError("Source support is empty")
    rows, columns = np.indices(source.shape)
    magnitude = np.maximum(-source[valid], 0.0)
    boundary = valid & ~ndimage.binary_erosion(valid)
    labels, _ = ndimage.label(valid, structure=np.ones((3, 3), dtype=np.int8))
    component_sizes = np.bincount(labels.ravel())[1:]
    return {
        "centroid_row_fraction": float(np.mean(rows[valid]) / max(source.shape[0] - 1, 1)),
        "centroid_column_fraction": float(
            np.mean(columns[valid]) / max(source.shape[1] - 1, 1)
        ),
        "boundary_to_area_ratio": float(boundary.sum() / valid.sum()),
        "component_count_min_4_blocks": int(np.sum(component_sizes >= 4)),
        "magnitude_mean": float(np.mean(magnitude)),
        "magnitude_cv": float(np.std(magnitude) / max(np.mean(magnitude), 1e-12)),
        "magnitude_p10": float(np.quantile(magnitude, 0.10)),
        "magnitude_p50": float(np.quantile(magnitude, 0.50)),
        "magnitude_p90": float(np.quantile(magnitude, 0.90)),
    }


def spatial_zone_alphas(
    source: np.ndarray,
    signal: np.ndarray,
    evaluated: np.ndarray,
    support: np.ndarray,
    folds: np.ndarray,
) -> list[dict[str, Any]]:
    rows, columns = np.indices(source.shape)
    results = []
    for zone_row in range(SPATIAL_ZONE_GRID):
        for zone_column in range(SPATIAL_ZONE_GRID):
            zone = (
                (rows >= zone_row * source.shape[0] // SPATIAL_ZONE_GRID)
                & (rows < (zone_row + 1) * source.shape[0] // SPATIAL_ZONE_GRID)
                & (columns >= zone_column * source.shape[1] // SPATIAL_ZONE_GRID)
                & (columns < (zone_column + 1) * source.shape[1] // SPATIAL_ZONE_GRID)
            )
            mask = evaluated & zone
            source_blocks = int(np.sum(mask & support))
            source_fraction = float(source_blocks / max(mask.sum(), 1))
            eligible = bool(
                source_blocks >= MIN_ZONE_SOURCE_BLOCKS
                and source_fraction >= MIN_ZONE_SOURCE_FRACTION
            )
            row = {
                "zone": f"r{zone_row + 1}c{zone_column + 1}",
                "zone_row": zone_row,
                "zone_column": zone_column,
                "evaluated_blocks": int(mask.sum()),
                "source_blocks": source_blocks,
                "source_fraction": source_fraction,
                "eligible": eligible,
                "predictively_reliable": False,
            }
            if eligible:
                fit = strict_affine_oof(source, signal, mask, folds)
                row.update({
                    "alpha": float(fit["alpha_fold_mean"]),
                    "alpha_fold_cv": float(fit["alpha_fold_cv"]),
                    "cv_r2": float(fit["cv_r2"]),
                    "predictively_reliable": bool(
                        fit["alpha_fold_mean"] > 0
                        and fit["cv_r2"] > 0
                        and fit["alpha_fold_cv"] <= 0.5
                    ),
                })
            results.append(row)
    return results


def load_pair_rows(
    dataset_dir: Path,
    workbook_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = dataset_dir / "analysis_results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen = manifest.get("frozen_background") or {}
    if not frozen.get("verified"):
        raise ValueError("Input export is not frozen-background verified")
    configuration = manifest["configuration"]
    labels = tuple(f"{a}->{b}" for a, b in configuration["pairs"])
    if labels != ALL_PAIRS:
        raise ValueError("Input export does not contain the frozen 13-pair cohort")
    if configuration["block_sizes"] != [16]:
        raise ValueError("Exposure analysis requires the frozen block-16 export")
    if configuration["background_modes"] != ["masked_hybrid_spline"]:
        raise ValueError("Exposure analysis requires the frozen background only")

    exposures = load_exposures(workbook_path)
    rows = []
    for manifest_row in manifest["pair_results"]:
        light_index = int(manifest_row["light_index"])
        dark_index = int(manifest_row["dark_index"])
        label = f"{light_index}->{dark_index}"
        block, fit, gate = _manifest_fit(manifest_row)
        artifact = manifest_row["output_artifacts"]["primary_maps"]
        artifact_path = dataset_dir / artifact["relative_path"]
        if sha256(artifact_path) != artifact["sha256"]:
            raise ValueError(f"Frozen NPZ hash mismatch for {label}")
        with np.load(artifact_path) as archive:
            prediction = archive["oof_prediction"].astype(np.float64)
            evaluated = archive["evaluated_mask"].astype(bool)
            support = archive["source_support"].astype(bool)
            saturation = archive["source_saturation_fraction"].astype(np.float64)
            source = archive["source"].astype(np.float64)
            signal = archive["observable_signal"].astype(np.float64)
            folds = archive["spatial_folds"].astype(np.int8)
        source_valid = evaluated & support & np.isfinite(source)
        saturation_mean = float(np.mean(saturation[source_valid]))
        morphology = source_morphology(source, evaluated, support)
        record = exposures[light_index]
        rows.append({
            "pair": label,
            "light_index": light_index,
            "dark_index": dark_index,
            "cohort_stratum": manifest_row["cohort_stratum"],
            "detected": bool(gate["passed"]),
            "alpha": float(fit["alpha_fold_mean"]),
            "alpha_fold_cv": float(fit["alpha_fold_cv"]),
            "cv_r2": float(fit["cv_r2"]),
            "ghost_contrast_rms": _ghost_contrast_rms(prediction, evaluated, support),
            "kv": float(record["kv"]),
            "ma": float(record["ma"]),
            "exposure_time_ms": float(record["exposure_time_ms"]),
            "mas": float(record["mas"]),
            "source_support_fraction": float(block["source_support_fraction"]),
            "source_saturation_fraction_mean": saturation_mean,
            "source_morphology": morphology,
            "source_magnitude_p50": morphology["magnitude_p50"],
            "source_magnitude_p90": morphology["magnitude_p90"],
            "spatial_zone_alphas": spatial_zone_alphas(
                source, signal, evaluated, support, folds
            ),
        })
    rows.sort(key=lambda row: ALL_PAIRS.index(row["pair"]))
    return rows, {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "workbook_path": str(workbook_path.resolve()),
        "workbook_sha256": sha256(workbook_path),
        "frozen_background": frozen,
    }


def repeated_setting_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["kv"], row["ma"], row["exposure_time_ms"])].append(row)
    results = []
    for setting, members in groups.items():
        if len(members) < 2:
            continue
        alpha = np.asarray([row["alpha"] for row in members], dtype=float)
        ghost = np.asarray([row["ghost_contrast_rms"] for row in members], dtype=float)
        stronger = max(members, key=lambda row: row["alpha"])
        weaker = min(members, key=lambda row: row["alpha"])
        stronger_zones = {
            row["zone"]: row for row in stronger["spatial_zone_alphas"] if row["eligible"]
        }
        weaker_zones = {
            row["zone"]: row for row in weaker["spatial_zone_alphas"] if row["eligible"]
        }
        common_zones = sorted(set(stronger_zones) & set(weaker_zones))
        common_rows = [
            {
                "zone": zone,
                "stronger_alpha": stronger_zones[zone]["alpha"],
                "weaker_alpha": weaker_zones[zone]["alpha"],
                "same_order_as_global": bool(
                    stronger_zones[zone]["alpha"] > weaker_zones[zone]["alpha"]
                ),
            }
            for zone in common_zones
        ]
        stronger_reliable = {
            row["zone"]: row
            for row in stronger["spatial_zone_alphas"]
            if row["predictively_reliable"]
        }
        weaker_reliable = {
            row["zone"]: row
            for row in weaker["spatial_zone_alphas"]
            if row["predictively_reliable"]
        }
        common_reliable_zones = sorted(set(stronger_reliable) & set(weaker_reliable))
        common_reliable_rows = [
            {
                "zone": zone,
                "stronger_alpha": stronger_reliable[zone]["alpha"],
                "weaker_alpha": weaker_reliable[zone]["alpha"],
                "alpha_ratio": (
                    stronger_reliable[zone]["alpha"] / weaker_reliable[zone]["alpha"]
                ),
                "same_order_as_global": bool(
                    stronger_reliable[zone]["alpha"] > weaker_reliable[zone]["alpha"]
                ),
            }
            for zone in common_reliable_zones
        ]
        positive_ratios = [
            row["stronger_alpha"] / row["weaker_alpha"]
            for row in common_rows
            if row["stronger_alpha"] > 0 and row["weaker_alpha"] > 0
        ]
        stronger_morphology = stronger["source_morphology"]
        weaker_morphology = weaker["source_morphology"]
        centroid_distance = float(np.hypot(
            stronger_morphology["centroid_row_fraction"]
            - weaker_morphology["centroid_row_fraction"],
            stronger_morphology["centroid_column_fraction"]
            - weaker_morphology["centroid_column_fraction"],
        ))
        results.append({
            "setting": {
                "kv": setting[0], "ma": setting[1], "exposure_time_ms": setting[2],
                "mas": float(members[0]["mas"]),
            },
            "pairs": [row["pair"] for row in members],
            "detected": [row["detected"] for row in members],
            "alpha_min": float(np.min(alpha)),
            "alpha_max": float(np.max(alpha)),
            "alpha_max_to_min_ratio": float(np.max(alpha) / max(np.min(alpha), 1e-12)),
            "ghost_contrast_rms_min": float(np.min(ghost)),
            "ghost_contrast_rms_max": float(np.max(ghost)),
            "ghost_contrast_rms_max_to_min_ratio": float(
                np.max(ghost) / max(np.min(ghost), 1e-12)
            ),
            "source_morphology_contrast": {
                "stronger_pair": stronger["pair"],
                "weaker_pair": weaker["pair"],
                "centroid_distance_fraction": centroid_distance,
                "support_fraction_ratio": float(
                    stronger["source_support_fraction"]
                    / max(weaker["source_support_fraction"], 1e-12)
                ),
                "stronger_magnitude_p50": stronger_morphology["magnitude_p50"],
                "weaker_magnitude_p50": weaker_morphology["magnitude_p50"],
                "magnitude_p50_ratio": (
                    float(
                        stronger_morphology["magnitude_p50"]
                        / weaker_morphology["magnitude_p50"]
                    )
                    if weaker_morphology["magnitude_p50"] > 0 else None
                ),
                "boundary_to_area_ratio_ratio": float(
                    stronger_morphology["boundary_to_area_ratio"]
                    / max(weaker_morphology["boundary_to_area_ratio"], 1e-12)
                ),
            },
            "spatial_consistency": {
                "grid": [SPATIAL_ZONE_GRID, SPATIAL_ZONE_GRID],
                "stronger_pair": stronger["pair"],
                "weaker_pair": weaker["pair"],
                "common_eligible_zone_count": len(common_rows),
                "same_order_as_global_count": int(sum(
                    row["same_order_as_global"] for row in common_rows
                )),
                "same_order_as_global_fraction": (
                    float(np.mean([row["same_order_as_global"] for row in common_rows]))
                    if common_rows else None
                ),
                "positive_common_zone_ratio_median": (
                    float(np.median(positive_ratios)) if positive_ratios else None
                ),
                "common_zones": common_rows,
                "predictively_reliable_definition": (
                    "Both pair-zone fits have positive alpha, CV-R2 > 0, and alpha fold CV <= 0.5."
                ),
                "common_reliable_zone_count": len(common_reliable_rows),
                "reliable_same_order_count": int(sum(
                    row["same_order_as_global"] for row in common_reliable_rows
                )),
                "reliable_same_order_fraction": (
                    float(np.mean([
                        row["same_order_as_global"] for row in common_reliable_rows
                    ])) if common_reliable_rows else None
                ),
                "reliable_zone_ratio_median": (
                    float(np.median([
                        row["alpha_ratio"] for row in common_reliable_rows
                    ])) if common_reliable_rows else None
                ),
                "common_reliable_zones": common_reliable_rows,
            },
        })
    return sorted(results, key=lambda row: (row["setting"]["kv"], row["setting"]["mas"]))


def _header_value(dataset: pydicom.Dataset, keyword: str) -> Any:
    if keyword not in dataset:
        return None
    value = dataset.data_element(keyword).value
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def technical_header_audit(
    rows: list[dict[str, Any]],
    dicom_dir: Path,
) -> dict[str, Any]:
    repeated_labels = {
        label
        for group in repeated_setting_summary(rows)
        for label in group["pairs"]
    }
    selected = [row for row in rows if row["pair"] in repeated_labels]
    datasets: dict[str, dict[str, pydicom.Dataset]] = {}
    for row in selected:
        datasets[row["pair"]] = {
            "source_light": pydicom.dcmread(
                dicom_dir / f"{row['light_index']}-light.dcm", stop_before_pixels=True
            ),
            "target_dark": pydicom.dcmread(
                dicom_dir / f"{row['dark_index']}-dark.dcm", stop_before_pixels=True
            ),
        }

    field_summary = {}
    for keyword in TECHNICAL_HEADER_FIELDS:
        by_role = {}
        for role in ("source_light", "target_dark"):
            values = [_header_value(pair[role], keyword) for pair in datasets.values()]
            present = [value for value in values if value is not None]
            canonical = [json.dumps(value, sort_keys=True) for value in present]
            by_role[role] = {
                "present_count": len(present),
                "pair_count": len(values),
                "distinct_value_count": len(set(canonical)),
                "identical_when_present": bool(present and len(set(canonical)) == 1),
                "common_value": present[0] if present and len(set(canonical)) == 1 else None,
            }
        field_summary[keyword] = by_role

    explanatory = (
        "RescaleIntercept", "RescaleSlope", "RescaleType", "PresentationLUTShape",
        "PixelIntensityRelationship", "PixelIntensityRelationshipSign", "Sensitivity",
        "ExposureIndex", "TargetExposureIndex", "DeviationIndex", "DetectorType",
        "DetectorConfiguration", "AcquisitionDeviceProcessingDescription",
    )
    missing_explanatory = [
        keyword for keyword in explanatory
        if all(
            field_summary[keyword][role]["present_count"] == 0
            for role in ("source_light", "target_dark")
        )
    ]
    available_fields = [
        keyword for keyword in TECHNICAL_HEADER_FIELDS
        if any(
            field_summary[keyword][role]["present_count"] > 0
            for role in ("source_light", "target_dark")
        )
    ]
    varying_available_fields = [
        keyword for keyword in available_fields
        if any(
            field_summary[keyword][role]["distinct_value_count"] > 1
            for role in ("source_light", "target_dark")
        )
    ]
    return {
        "repeated_pair_count": len(selected),
        "technical_fields_checked": list(TECHNICAL_HEADER_FIELDS),
        "available_fields": available_fields,
        "missing_explanatory_fields": missing_explanatory,
        "varying_available_fields": varying_available_fields,
        "technical_variation_found": bool(varying_available_fields),
        "field_summary": field_summary,
        "interpretation": (
            "Public DICOM technical fields do not expose a reader/processing difference "
            "among the exact-repeat pairs; the fields that could test this explanation "
            "are absent rather than confirmed equal."
        ),
    }


def association_set(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    relations = (
        ("mas", "alpha"),
        ("mas", "ghost_contrast_rms"),
        ("source_magnitude_p50", "alpha"),
        ("source_magnitude_p50", "ghost_contrast_rms"),
        ("source_support_fraction", "alpha"),
        ("source_saturation_fraction_mean", "alpha"),
    )
    return {
        f"{x}_vs_{y}": spearman_summary(
            [row[x] for row in rows], [row[y] for row in rows], seed=seed + index
        )
        for index, (x, y) in enumerate(relations)
    }


def render_summary(rows: list[dict[str, Any]], repeated: list[dict[str, Any]], path: Path) -> None:
    colors = {70.0: "#2878b5", 100.0: "#f39c34", 110.0: "#c33c54"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row in rows:
        marker = "o" if row["detected"] else "X"
        color = colors.get(row["kv"], "#555555")
        axes[0, 0].scatter(row["mas"], row["alpha"], color=color, marker=marker, s=55)
        axes[0, 1].scatter(
            row["mas"], row["ghost_contrast_rms"], color=color, marker=marker, s=55
        )
        axes[0, 2].scatter(
            row["source_magnitude_p50"], row["alpha"], color=color, marker=marker, s=55
        )
        axes[0, 0].annotate(row["pair"], (row["mas"], row["alpha"]), fontsize=7)
        axes[0, 1].annotate(
            row["pair"], (row["mas"], row["ghost_contrast_rms"]), fontsize=7
        )
        axes[0, 2].annotate(
            row["pair"], (row["source_magnitude_p50"], row["alpha"]), fontsize=7
        )
    axes[0, 0].set(xlabel="mAs", ylabel="Frozen M1 alpha", xscale="log", yscale="log")
    axes[0, 1].set(
        xlabel="mAs", ylabel="OOF ghost contrast RMS", xscale="log", yscale="log"
    )
    axes[0, 2].set(
        xlabel="Source magnitude median (DICOM value)", ylabel="Frozen M1 alpha",
        yscale="log",
    )

    labels = [
        f"{item['setting']['kv']:.0f}kV/{item['setting']['ma']:.0f}mA/"
        f"{item['setting']['exposure_time_ms']:.0f}ms"
        for item in repeated
    ]
    positions = np.arange(len(repeated))
    for position, item in zip(positions, repeated):
        member_rows = [row for row in rows if row["pair"] in item["pairs"]]
        values = [row["alpha"] for row in member_rows]
        axes[1, 0].plot([position] * len(values), values, "o", color="#555555")
        axes[1, 0].plot([position, position], [min(values), max(values)], color="#999999")
        for value, member in zip(values, member_rows):
            axes[1, 0].annotate(member["pair"], (position, value), fontsize=7)
    axes[1, 0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1, 0].set(ylabel="Frozen M1 alpha", yscale="log", title="Exact repeated settings")

    global_ratios = [item["alpha_max_to_min_ratio"] for item in repeated]
    axes[1, 1].bar(positions, global_ratios, color="#777777")
    axes[1, 1].axhline(1.0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1, 1].set(
        ylabel="Global alpha max/min", yscale="log", title="Repeat-pair heterogeneity"
    )

    order_fractions = [
        item["spatial_consistency"]["same_order_as_global_fraction"] or 0.0
        for item in repeated
    ]
    axes[1, 2].bar(positions, order_fractions, color="#4c956c")
    for position, item, fraction in zip(positions, repeated, order_fractions):
        spatial = item["spatial_consistency"]
        zone_ratio = spatial["positive_common_zone_ratio_median"]
        ratio_text = "NA" if zone_ratio is None else f"{zone_ratio:.1f}x"
        reliable_text = (
            f"rel {spatial['reliable_same_order_count']}/"
            f"{spatial['common_reliable_zone_count']}"
            if spatial["common_reliable_zone_count"] else "rel NA"
        )
        axes[1, 2].text(
            position, min(fraction + 0.035, 1.08),
            f"{spatial['same_order_as_global_count']}/"
            f"{spatial['common_eligible_zone_count']}\n{ratio_text}\n{reliable_text}",
            ha="center", va="bottom", fontsize=8,
        )
    axes[1, 2].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1, 2].set(
        ylabel="Zones matching global ordering", ylim=(0, 1.30),
        title="Fixed 3x3 spatial diagnostic",
    )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=f"{kv:.0f} kV")
        for kv, color in colors.items()
    ] + [
        Line2D([0], [0], marker="o", linestyle="", color="#555555", label="detected"),
        Line2D([0], [0], marker="X", linestyle="", color="#555555", label="not detected"),
    ]
    axes[0, 0].legend(handles=legend_handles, fontsize=8, loc="lower right")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle("Single-frame exposure and exact-repeat attribution (frozen 13-pair evidence)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/mhinge_cross_pair_v1"))
    parser.add_argument(
        "--workbook", type=Path,
        default=Path("data/raw/AI修残影例图/拍摄参数记录.xlsx"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/mhinge_cross_pair_v1/exposure_association"),
    )
    args = parser.parse_args()

    rows, provenance = load_pair_rows(args.dataset_dir, args.workbook)
    script_path = Path(__file__).resolve()
    provenance["analysis_script"] = str(script_path)
    provenance["analysis_script_sha256"] = sha256(script_path)
    repeated = repeated_setting_summary(rows)
    header_audit = technical_header_audit(rows, args.workbook.parent)
    detected = [row for row in rows if row["detected"]]
    low_saturation = [
        row for row in rows
        if row["source_saturation_fraction_mean"] <= LOW_SATURATION_THRESHOLD
    ]
    associations = {
        "all_pairs": association_set(rows, RANDOM_SEED),
        "detected_pairs_sensitivity": association_set(detected, RANDOM_SEED + 100),
        "low_saturation_sensitivity": association_set(low_saturation, RANDOM_SEED + 200),
    }
    primary_alpha = associations["all_pairs"]["mas_vs_alpha"]
    primary_ghost = associations["all_pairs"]["mas_vs_ghost_contrast_rms"]
    high_heterogeneity = [
        item for item in repeated if item["alpha_max_to_min_ratio"] >= 10.0
    ]
    high_heterogeneity_common_zones = sum(
        item["spatial_consistency"]["common_eligible_zone_count"]
        for item in high_heterogeneity
    )
    high_heterogeneity_order_matches = sum(
        item["spatial_consistency"]["same_order_as_global_count"]
        for item in high_heterogeneity
    )
    high_heterogeneity_reliable_zones = sum(
        item["spatial_consistency"]["common_reliable_zone_count"]
        for item in high_heterogeneity
    )
    high_heterogeneity_reliable_matches = sum(
        item["spatial_consistency"]["reliable_same_order_count"]
        for item in high_heterogeneity
    )
    high_heterogeneity_groups_with_reliable_zones = sum(
        item["spatial_consistency"]["common_reliable_zone_count"] > 0
        for item in high_heterogeneity
    )
    result = {
        "schema_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(),
        "study_question": (
            "Within the frozen single-frame cohort, do recorded exposure settings alone "
            "describe between-pair variation in alpha or predicted ghost contrast?"
        ),
        "scope": {
            "single_frame_only": True,
            "previous_frame_lag": 1,
            "multi_frame_terms_fitted": False,
            "background_reestimated": False,
            "clean_ground_truth_available": False,
            "pair_count": len(rows),
            "detected_pair_count": len(detected),
            "low_saturation_threshold": LOW_SATURATION_THRESHOLD,
            "low_saturation_pair_count": len(low_saturation),
            "exact_repeat_group_count": len(repeated),
        },
        "definitions": {
            "alpha": "Mean of four frozen spatial-OOF M1 fold slopes.",
            "ghost_contrast_rms": (
                "RMS of the frozen OOF M1 prediction on source-support blocks after "
                "subtracting its median outside source support."
            ),
            "permutation_p": (
                "Descriptive within-cohort label permutation; pairs share one acquisition "
                "session and are not treated as independent biological samples."
            ),
            "spatial_zone_alpha": (
                "Frozen M1 refit independently in a fixed 3x3 detector grid. A zone is "
                "eligible with at least 50 source blocks and 2% source coverage."
            ),
        },
        "source_provenance": provenance,
        "analysis_configuration": {
            "associations": "Spearman rank correlations",
            "permutations": PERMUTATIONS,
            "random_seed": RANDOM_SEED,
            "exact_repeat_key": ["kv", "ma", "exposure_time_ms"],
            "spatial_zone_grid": [SPATIAL_ZONE_GRID, SPATIAL_ZONE_GRID],
            "minimum_zone_source_blocks": MIN_ZONE_SOURCE_BLOCKS,
            "minimum_zone_source_fraction": MIN_ZONE_SOURCE_FRACTION,
        },
        "pair_results": rows,
        "associations": associations,
        "exact_repeated_settings": repeated,
        "technical_header_audit": header_audit,
        "interpretation": {
            "exposure_parameters_alone_supported": False,
            "reason": (
                "The all-pair mAs associations are weak and leave-one-out unstable, while "
                "exactly repeated exposure settings show substantial between-pair alpha "
                "and ghost-contrast variation."
            ),
            "all_pair_mas_alpha_rho": primary_alpha["rho"],
            "all_pair_mas_ghost_contrast_rho": primary_ghost["rho"],
            "reader_processing_difference_testable_from_public_dicom": False,
            "reader_processing_reason": header_audit["interpretation"],
            "simple_source_morphology_sufficient": False,
            "simple_source_morphology_reason": (
                "The 70 kV / 100 mA / 100 ms repeat has nearly identical source median, "
                "support area, and centroid but a 33.7-fold global alpha difference."
            ),
            "difference_confined_to_one_detector_region": False,
            "high_heterogeneity_common_zone_count": high_heterogeneity_common_zones,
            "high_heterogeneity_same_order_zone_count": high_heterogeneity_order_matches,
            "high_heterogeneity_reliable_zone_count": high_heterogeneity_reliable_zones,
            "high_heterogeneity_reliable_same_order_zone_count": (
                high_heterogeneity_reliable_matches
            ),
            "high_heterogeneity_groups_with_reliable_zones": (
                high_heterogeneity_groups_with_reliable_zones
            ),
            "spatial_reason": (
                "Across the three repeat groups with at least 10-fold global alpha "
                "heterogeneity, all 10 support-eligible common zones preserve the global "
                "ordering. Only four zones across two groups pass the additional local "
                "predictive-reliability screen, and all four preserve the ordering; the "
                "third group has no reliable common zone. The clearest evidence is the "
                "fully detected 70 kV / 100 mA / 100 ms repeat, whose two reliable common "
                "zones have 28.5-fold and 33.7-fold alpha ratios. This favors a global-state "
                "difference over one localized source-shape effect, but remains diagnostic."
            ),
            "boundary": (
                "This is same-session descriptive evidence using processed DICOM signal; "
                "the global-state interpretation is an inference because reader/plate "
                "state metadata are absent. It does not establish a dose law or removal accuracy."
            ),
        },
        "output_artifacts": {
            "json": "exposure_association.json",
            "figure": "exposure_association.png",
            "retention": "Stable paths; reruns overwrite these two artifacts.",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="single_frame_exposure_") as temporary:
        temporary_dir = Path(temporary)
        json_path = temporary_dir / "exposure_association.json"
        figure_path = temporary_dir / "exposure_association.png"
        json_path.write_text(
            json.dumps(clean_json(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        render_summary(rows, repeated, figure_path)
        shutil.move(str(json_path), args.output_dir / json_path.name)
        shutil.move(str(figure_path), args.output_dir / figure_path.name)
    print(json.dumps(clean_json(result["interpretation"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
