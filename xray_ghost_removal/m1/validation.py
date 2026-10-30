"""M1 validation; extracted without changing numerical definitions."""
from __future__ import annotations
from typing import Any
from scipy.ndimage import find_objects
from scipy.ndimage import label
import numpy as np
from xray_ghost_removal.m1.background import (
    baseline,
    estimate_hybrid_dark_background,
    estimate_masked_hybrid_dark_background,
    estimate_single_dark_background,
)
from xray_ghost_removal.m1.config import (
    FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    FIXED_PATTERN_HUBER_ITERATIONS,
    HUBER_DELTA,
    PSEUDO_OBJECT_OCCLUSION_COUNT,
    PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE,
    PSEUDO_OCCLUSION_SQUARE_SIDES,
)

def extract_object_shape_templates(
    source_supports: list[np.ndarray],
    min_blocks: int = 20,
    max_fraction: float = 0.12,
) -> list[np.ndarray]:
    """Extract reusable connected-component shapes from real source masks."""
    templates = []
    for support in source_supports:
        labels, _ = label(support)
        maximum = int(support.size * max_fraction)
        for component_index, slices in enumerate(find_objects(labels), start=1):
            if slices is None:
                continue
            component = labels[slices] == component_index
            area = int(component.sum())
            if min_blocks <= area <= maximum:
                templates.append(component)
    return sorted(templates, key=lambda mask: int(mask.sum()))


def realistic_occlusion_candidates(
    source_supports: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    """Keep each real source mask at its full area, shape, and detector position."""
    candidates = []
    for light_index, support in sorted(source_supports.items()):
        support = np.asarray(support, dtype=bool)
        locations = np.argwhere(support)
        if locations.size == 0:
            candidates.append({
                "label": f"light_{light_index}",
                "status": "unverified",
                "reason": "empty_source_support",
                "requested_blocks": 0,
                "requested_fraction": 0.0,
            })
            continue
        row_min, column_min = locations.min(axis=0)
        row_max, column_max = locations.max(axis=0) + 1
        candidates.append({
            "label": f"light_{light_index}",
            "status": "candidate",
            "shape": support[row_min:row_max, column_min:column_max].copy(),
            "requested_top_left": [int(row_min), int(column_min)],
            "requested_blocks": int(support.sum()),
            "requested_fraction": float(support.mean()),
            "bounding_box_blocks": [
                int(row_max - row_min), int(column_max - column_min),
            ],
        })
    return candidates


def _place_realistic_occlusion(
    candidate: dict[str, Any],
    trusted_air: np.ndarray,
) -> dict[str, Any]:
    """Place a full real mask in trusted air, minimizing displacement from its origin."""
    if candidate.get("status") == "unverified":
        return {key: value for key, value in candidate.items() if key != "shape"}
    shape = np.asarray(candidate["shape"], dtype=bool)
    height, width = shape.shape
    requested_row, requested_column = candidate["requested_top_left"]
    base = {key: value for key, value in candidate.items() if key != "shape"}
    if height > trusted_air.shape[0] or width > trusted_air.shape[1]:
        return {
            **base,
            "status": "unverified",
            "reason": "bounding_box_larger_than_analysis_grid",
        }

    positions = [
        (row, column)
        for row in range(trusted_air.shape[0] - height + 1)
        for column in range(trusted_air.shape[1] - width + 1)
    ]
    positions.sort(key=lambda position: (
        (position[0] - requested_row) ** 2 + (position[1] - requested_column) ** 2,
        abs(position[0] - requested_row) + abs(position[1] - requested_column),
        position,
    ))
    for row, column in positions:
        window = trusted_air[row:row + height, column:column + width]
        if not np.all(window[shape]):
            continue
        mask = np.zeros_like(trusted_air, dtype=bool)
        mask[row:row + height, column:column + width] = shape
        displacement = float(np.hypot(row - requested_row, column - requested_column))
        return {
            **base,
            "status": "validated",
            "placement": "original" if displacement == 0 else "nearest_valid_translation",
            "placed_top_left": [int(row), int(column)],
            "displacement_blocks": displacement,
            "mask": mask,
        }
    return {
        **base,
        "status": "unverified",
        "reason": "full_shape_does_not_fit_inside_trusted_air",
    }


def _place_pseudo_occlusion(
    shape: np.ndarray,
    trusted_air: np.ndarray,
    occupied: np.ndarray,
    rng: np.random.Generator,
    attempts: int = 1000,
) -> np.ndarray | None:
    height, width = shape.shape
    if height > trusted_air.shape[0] or width > trusted_air.shape[1]:
        return None
    area = int(shape.sum())
    for _ in range(attempts):
        row = int(rng.integers(0, trusted_air.shape[0] - height + 1))
        column = int(rng.integers(0, trusted_air.shape[1] - width + 1))
        air_window = trusted_air[row:row + height, column:column + width]
        if not np.all(air_window[shape]):
            continue
        occupied_window = occupied[row:row + height, column:column + width]
        if np.logical_and(occupied_window, shape).sum() > 0.25 * area:
            continue
        placed = np.zeros_like(trusted_air, dtype=bool)
        placed[row:row + height, column:column + width] = shape
        return placed
    return None


def generate_pseudo_occlusions(
    trusted_air: np.ndarray,
    object_templates: list[np.ndarray],
    seed: int,
    square_sides: tuple[int, ...] = PSEUDO_OCCLUSION_SQUARE_SIDES,
    squares_per_size: int = PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE,
    object_count: int = PSEUDO_OBJECT_OCCLUSION_COUNT,
) -> list[dict[str, Any]]:
    """Place deterministic square and real-object-shaped holdouts in trusted air."""
    rng = np.random.default_rng(seed)
    occupied = np.zeros_like(trusted_air, dtype=bool)
    occlusions = []

    for side in square_sides:
        shape = np.ones((side, side), dtype=bool)
        for placement in range(squares_per_size):
            mask = _place_pseudo_occlusion(shape, trusted_air, occupied, rng)
            if mask is None:
                continue
            occupied |= mask
            occlusions.append({
                "kind": "square",
                "size_label": f"{side}x{side}_blocks",
                "placement": placement + 1,
                "mask": mask,
            })

    if object_templates and object_count > 0:
        candidate_indices = np.unique(np.round(
            np.linspace(0, len(object_templates) - 1, min(len(object_templates), 9))
        ).astype(int))
        candidates = [object_templates[index] for index in candidate_indices]
        target_indices = np.unique(np.round(
            np.linspace(0, len(candidates) - 1, min(object_count, len(candidates)))
        ).astype(int))
        ordered = [candidates[index] for index in target_indices]
        ordered.extend(mask for index, mask in enumerate(candidates) if index not in target_indices)
        for shape in ordered:
            if sum(row["kind"] == "object_shape" for row in occlusions) >= object_count:
                break
            mask = _place_pseudo_occlusion(shape, trusted_air, occupied, rng)
            if mask is None:
                continue
            occupied |= mask
            occlusions.append({
                "kind": "object_shape",
                "size_label": f"{int(shape.sum())}_blocks",
                "placement": 1,
                "mask": mask,
            })
    return occlusions


def _reconstruction_error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction.astype(np.float64) - target.astype(np.float64)
    absolute = np.abs(error)
    return {
        "mae": float(np.mean(absolute)),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
    }


def summarize_reconstruction_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({mode for sample in samples for mode in sample["by_background"]})

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        output = {}
        for mode in modes:
            values = [row["by_background"][mode] for row in rows if mode in row["by_background"]]
            if not values:
                continue
            output[mode] = {
                "sample_count": len(values),
                "evaluated_blocks": int(sum(row["evaluated_blocks"] for row in values)),
                "median_sample_mae": float(np.median([row["mae"] for row in values])),
                "median_absolute_bias": float(np.median([abs(row["bias"]) for row in values])),
                "median_sample_rmse": float(np.median([row["rmse"] for row in values])),
                "median_sample_p95_absolute_error": float(np.median([
                    row["p95_absolute_error"] for row in values
                ])),
            }
        return output

    by_kind = {
        kind: summarize([sample for sample in samples if sample["kind"] == kind])
        for kind in sorted({sample["kind"] for sample in samples})
    }
    def compare_mae(reference_mode: str) -> dict[str, Any]:
        comparisons = {}
        for mode in modes:
            if mode == reference_mode:
                continue
            deltas = [
                sample["by_background"][mode]["mae"]
                - sample["by_background"][reference_mode]["mae"]
                for sample in samples
                if mode in sample["by_background"]
                and reference_mode in sample["by_background"]
            ]
            comparisons[mode] = {
                "paired_sample_count": len(deltas),
                "median_delta_mae": float(np.median(deltas)),
                "mae_improved_count": int(sum(delta < 0 for delta in deltas)),
            }
        return comparisons

    return {
        "overall_by_background": summarize(samples),
        "by_occlusion_kind": by_kind,
        "versus_loo_median": compare_mae("loo_median") if "loo_median" in modes else {},
        "versus_hybrid_spline": (
            compare_mae("hybrid_spline") if "hybrid_spline" in modes else {}
        ),
    }


def validate_background_reconstruction(
    dark: np.ndarray,
    trusted_air: np.ndarray,
    darks: dict[int, np.ndarray],
    target_index: int,
    background_modes: tuple[str, ...],
    object_templates: list[np.ndarray],
    smoothness: float,
    seed: int,
    square_sides: tuple[int, ...] = PSEUDO_OCCLUSION_SQUARE_SIDES,
    squares_per_size: int = PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE,
    object_count: int = PSEUDO_OBJECT_OCCLUSION_COUNT,
    realistic_candidates: list[dict[str, Any]] | None = None,
    contamination_supports: dict[int, np.ndarray] | None = None,
    masked_fixed_pattern: np.ndarray | None = None,
    masked_fixed_pattern_diagnostics: dict[str, Any] | None = None,
    decomposition_iterations: int = FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    huber_iterations: int = FIXED_PATTERN_HUBER_ITERATIONS,
    huber_delta: float = HUBER_DELTA,
) -> dict[str, Any]:
    """Hide known air blocks and score background reconstruction inside them."""
    occlusions = generate_pseudo_occlusions(
        trusted_air=trusted_air,
        object_templates=object_templates,
        seed=seed,
        square_sides=square_sides,
        squares_per_size=squares_per_size,
        object_count=object_count,
    )
    realistic_attempts = [
        _place_realistic_occlusion(candidate, trusted_air)
        for candidate in (realistic_candidates or [])
    ]
    occlusions.extend(
        {
            "kind": "real_source_full_shape",
            "size_label": attempt["label"],
            "placement": attempt["placement"],
            "mask": attempt["mask"],
            "requested_blocks": attempt["requested_blocks"],
            "requested_fraction": attempt["requested_fraction"],
            "requested_top_left": attempt["requested_top_left"],
            "placed_top_left": attempt["placed_top_left"],
            "displacement_blocks": attempt["displacement_blocks"],
        }
        for attempt in realistic_attempts
        if attempt["status"] == "validated"
    )
    samples = []
    for occlusion in occlusions:
        holdout = occlusion["mask"]
        train = trusted_air & ~holdout
        row = {
            "kind": occlusion["kind"],
            "size_label": occlusion["size_label"],
            "placement": occlusion["placement"],
            "holdout_blocks": int(holdout.sum()),
            "by_background": {},
        }
        for key in (
            "requested_blocks", "requested_fraction", "requested_top_left",
            "placed_top_left", "displacement_blocks",
        ):
            if key in occlusion:
                row[key] = occlusion[key]
        for mode in background_modes:
            if mode == "loo_median":
                centered_baseline = baseline(darks, target_index, "loo_median")
                prediction = np.median(dark[train]) + centered_baseline
            elif mode == "hybrid_spline":
                prediction, _ = estimate_hybrid_dark_background(
                    dark=dark,
                    darks=darks,
                    target_index=target_index,
                    fit_mask=train,
                    smoothness=smoothness,
                    huber_delta=huber_delta,
                )
            elif mode == "masked_hybrid_spline":
                if contamination_supports is None:
                    raise ValueError(
                        "masked_hybrid_spline reconstruction validation requires "
                        "contamination_supports"
                    )
                prediction, _ = estimate_masked_hybrid_dark_background(
                    dark=dark,
                    darks=darks,
                    contamination_supports=contamination_supports,
                    target_index=target_index,
                    fit_mask=train,
                    smoothness=smoothness,
                    fixed_pattern=masked_fixed_pattern,
                    fixed_pattern_diagnostics=masked_fixed_pattern_diagnostics,
                    decomposition_iterations=decomposition_iterations,
                    huber_iterations=huber_iterations,
                    huber_delta=huber_delta,
                )
            else:
                prediction, _ = estimate_single_dark_background(
                    dark=dark,
                    fit_mask=train,
                    mode=mode,
                    smoothness=smoothness,
                )
            row["by_background"][mode] = {
                "evaluated_blocks": int(holdout.sum()),
                **_reconstruction_error_metrics(prediction[holdout], dark[holdout]),
            }
        samples.append(row)
    return {
        "trusted_air_blocks": int(trusted_air.sum()),
        "trusted_air_fraction": float(trusted_air.mean()),
        "requested_square_sides_blocks": list(square_sides),
        "requested_squares_per_size": squares_per_size,
        "requested_object_shapes": object_count,
        "generated_sample_count": len(samples),
        "realistic_occlusion_attempts": [
            {key: value for key, value in attempt.items() if key != "mask"}
            for attempt in realistic_attempts
        ],
        "realistic_validated_count": int(sum(
            attempt["status"] == "validated" for attempt in realistic_attempts
        )),
        "realistic_unverified_count": int(sum(
            attempt["status"] == "unverified" for attempt in realistic_attempts
        )),
        "summary": summarize_reconstruction_samples(samples),
        "samples": samples,
    }
