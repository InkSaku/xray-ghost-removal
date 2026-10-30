"""M1 config; extracted without changing numerical definitions."""
from __future__ import annotations
from typing import Any
from pathlib import Path
import argparse
import json
from xray_ghost_removal.common.provenance import sha256
from xray_ghost_removal.common.spatial import BLOCK_SIZES

DEFAULT_PAIRS = ((5, 6), (27, 28))

PRIMARY_BLOCK = 16

TRIM_PERCENTILES = (0.5, 99.5)

LOWPASS_SIGMA_BLOCKS = 2.0

BACKGROUND_MODES = (
    "loo_median",
    "poly2",
    "robust_spline",
    "hybrid_spline",
    "masked_hybrid_spline",
)

DEFAULT_PRIMARY_BACKGROUND = "masked_hybrid_spline"

HUBER_DELTA = 1.5

ROBUST_ITERATIONS = 5

FIXED_PATTERN_DECOMPOSITION_ITERATIONS = 2

FIXED_PATTERN_HUBER_ITERATIONS = 4

SINGLE_LAG_PAIRS = (
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (22, 23), (23, 24), (25, 26), (27, 28),
)

SINGLE_LAG_ROBUST = frozenset({(5, 6), (27, 28)})

SINGLE_LAG_NOT_DETECTED = frozenset({(4, 5), (6, 7)})

DETECTION_MIN_CV_R2 = 0.05

DETECTION_MIN_NULL_MARGIN = 0.05

DETECTION_MAX_ALPHA_CV = 0.25

DETECTION_MAX_ALPHA = 0.02

PSEUDO_OCCLUSION_SQUARE_SIDES = (12, 24, 40)

PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE = 2

PSEUDO_OBJECT_OCCLUSION_COUNT = 3

def parse_pairs(value: str) -> tuple[tuple[int, int], ...]:
    normalized = value.strip().lower().replace("_", "-")
    if normalized == "all":
        return tuple((light_index, light_index + 1) for light_index in range(1, 31))
    if normalized in {"single-lag", "single-lag-only"}:
        return SINGLE_LAG_PAIRS
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


def single_lag_stratum(pair: tuple[int, int]) -> str | None:
    if pair not in SINGLE_LAG_PAIRS:
        return None
    if pair in SINGLE_LAG_ROBUST:
        return "robust_confirmed"
    if pair in SINGLE_LAG_NOT_DETECTED:
        return "not_detected_control"
    return "parameter_sensitive"


def parse_background_modes(value: str) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    unknown = sorted(set(modes) - set(BACKGROUND_MODES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown background mode(s): {', '.join(unknown)}; "
            f"choose from {', '.join(BACKGROUND_MODES)}"
        )
    if not modes:
        raise argparse.ArgumentTypeError("At least one background mode is required")
    return modes


def parse_block_sizes(value: str) -> tuple[int, ...]:
    """Parse a non-empty subset of the supported block sizes."""
    try:
        blocks = tuple(dict.fromkeys(
            int(item.strip()) for item in value.split(",") if item.strip()
        ))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Block sizes must be comma-separated integers"
        ) from error
    unknown = sorted(set(blocks) - set(BLOCK_SIZES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported block size(s): {unknown}; choose from {BLOCK_SIZES}"
        )
    if not blocks:
        raise argparse.ArgumentTypeError("At least one block size is required")
    return blocks


def validate_frozen_background_config(
    config_path: Path,
    repo_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fail closed if a formal modeling export differs from the frozen BG."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise ValueError(f"Background configuration is not frozen: {config_path}")
    if config["mask"]["dilation_pixels"] != 0:
        raise ValueError(
            "Frozen background requires mask dilation, which is prohibited. "
            "Preserve the historical config/export; a no-dilation analysis "
            "has a different scientific definition and cannot overwrite it."
        )

    expected = {
        "primary_background": config["background_mode"],
        "mask_dilation_pixels": config["mask"]["dilation_pixels"],
        "spline_smoothness": config["spline_smoothness"],
        "background_huber_delta": config["huber_delta"],
        "fixed_pattern_iterations": config["fixed_pattern"]["decomposition_iterations"],
        "fixed_pattern_huber_iterations": config["fixed_pattern"]["huber_iterations"],
    }
    mismatches = {
        name: {"expected": expected_value, "actual": getattr(args, name)}
        for name, expected_value in expected.items()
        if getattr(args, name) != expected_value
    }
    if config["background_mode"] not in args.background_modes:
        mismatches["background_modes"] = {
            "expected_to_include": config["background_mode"],
            "actual": list(args.background_modes),
        }
    if tuple(args.analysis_blocks) != (int(config["block_size"]),):
        mismatches["analysis_blocks"] = {
            "expected": [int(config["block_size"])],
            "actual": list(args.analysis_blocks),
        }
    if args.summary_only:
        mismatches["summary_only"] = {"expected": False, "actual": True}
    if mismatches:
        raise ValueError(f"Formal export differs from frozen BG: {mismatches}")

    archive_path = repo_root / config["mask"]["archive_relative_path"]
    if args.source_mask_npz is None:
        raise ValueError("Formal frozen export requires --source-mask-npz")
    if args.source_mask_npz.resolve() != archive_path.resolve():
        raise ValueError(
            f"Frozen mask path mismatch: expected {archive_path}, got {args.source_mask_npz}"
        )
    actual_mask_hash = sha256(archive_path)
    if actual_mask_hash != config["mask"]["sha256"]:
        raise ValueError(
            f"Frozen mask SHA-256 mismatch: expected {config['mask']['sha256']}, "
            f"got {actual_mask_hash}"
        )
    return {
        "path": str(config_path.resolve()),
        "sha256": sha256(config_path),
        "mask_archive_sha256": actual_mask_hash,
        "verified": True,
    }


def artifact_record(path: Path, output_dir: Path) -> dict[str, Any]:
    """Return portable provenance for a generated artifact."""
    return {
        "relative_path": str(path.resolve().relative_to(output_dir.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
