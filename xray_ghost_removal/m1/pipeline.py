"""M1 pipeline; extracted without changing numerical definitions."""
from __future__ import annotations
from pathlib import Path
from xray_ghost_removal.common.paths import PROJECT_ROOT, require_current_m1_output
import argparse
from datetime import datetime
import json
import numpy as np
import pydicom
from xray_ghost_removal.common.provenance import clean_json, sha256, package_sources
from xray_ghost_removal.common.spatial import BLOCK_SIZES, RANDOM_SEED, block_mean, source, spatial_folds
from xray_ghost_removal.m1.background import (
    background_spatial_oof,
    baseline,
    crossfit_observable_signal,
    estimate_hybrid_dark_background,
    estimate_masked_fixed_pattern,
    estimate_masked_hybrid_dark_background,
    estimate_single_dark_background,
)
from xray_ghost_removal.m1.config import (
    BACKGROUND_MODES,
    DEFAULT_PAIRS,
    DEFAULT_PRIMARY_BACKGROUND,
    FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    FIXED_PATTERN_HUBER_ITERATIONS,
    HUBER_DELTA,
    LOWPASS_SIGMA_BLOCKS,
    PRIMARY_BLOCK,
    PSEUDO_OBJECT_OCCLUSION_COUNT,
    PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE,
    PSEUDO_OCCLUSION_SQUARE_SIDES,
    SINGLE_LAG_PAIRS,
    TRIM_PERCENTILES,
    artifact_record,
    parse_background_modes,
    parse_block_sizes,
    parse_pairs,
    single_lag_stratum,
    validate_frozen_background_config,
)
from xray_ghost_removal.m1.fitting import (
    _compact_metric_view,
    _metric_view,
    detection_gate,
    evaluate_with_nulls,
)
from xray_ghost_removal.m1.reporting import (
    render_six_panel,
    summarize_background_comparison,
    summarize_reconstruction_validation,
    summarize_single_lag_strata,
)
from xray_ghost_removal.m1.segmentation import block_source_masks, generate_sam_mask_archive, load_source_mask
from xray_ghost_removal.m1.validation import (
    extract_object_shape_templates,
    realistic_occlusion_candidates,
    validate_background_reconstruction,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Strict OOF pseudo-ghost mechanism study")
    parser.add_argument("--data-dir", default="data/raw/AI修残影例图")
    parser.add_argument("--output-dir", default="outputs/m1")
    parser.add_argument("--pairs", type=parse_pairs, default=DEFAULT_PAIRS,
                        help="Comma-separated light:dark pairs, 'all', or 'single-lag'")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Use only the fixed 16x16 analysis and write no per-pair figures or NPZ files",
    )
    parser.add_argument(
        "--skip-pair-figures", action="store_true",
        help="For formal NPZ exports, omit redundant per-pair six-panel PNG files",
    )
    parser.add_argument(
        "--analysis-blocks", type=parse_block_sizes, default=BLOCK_SIZES,
        help="Comma-separated analysis block sizes; formal frozen exports use only 16",
    )
    parser.add_argument(
        "--background-modes", type=parse_background_modes, default=BACKGROUND_MODES,
        help="Comma-separated background estimators: "
             "loo_median,poly2,robust_spline,hybrid_spline,masked_hybrid_spline",
    )
    parser.add_argument(
        "--primary-background", default=DEFAULT_PRIMARY_BACKGROUND,
        choices=BACKGROUND_MODES,
        help="Background estimator saved in the existing per-pair figure and NPZ",
    )
    parser.add_argument(
        "--source-mask-npz", type=Path,
        help="Optional NPZ of external masks keyed by light_N (for example light_5); "
             "when omitted, the existing Otsu detector is used",
    )
    parser.add_argument(
        "--generate-sam-masks", type=Path,
        help="Generate one SAM mask NPZ plus one contact sheet for the selected pairs, then exit",
    )
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--sam-model-type", default="vit_b", choices=("vit_b", "vit_l", "vit_h"))
    parser.add_argument("--sam-device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument(
        "--mask-dilation-pixels", type=int, default=0, choices=(0,),
        help="Compatibility option; only 0 is allowed because mask dilation is prohibited",
    )
    parser.add_argument("--spline-smoothness", type=float, default=20.0)
    parser.add_argument("--background-huber-delta", type=float, default=HUBER_DELTA)
    parser.add_argument(
        "--fixed-pattern-iterations", type=int,
        default=FIXED_PATTERN_DECOMPOSITION_ITERATIONS,
    )
    parser.add_argument(
        "--fixed-pattern-huber-iterations", type=int,
        default=FIXED_PATTERN_HUBER_ITERATIONS,
    )
    parser.add_argument(
        "--skip-background-oof", action="store_true",
        help="Skip the block-16 spatial holdout diagnostic for faster exploratory runs",
    )
    parser.add_argument(
        "--validate-background-reconstruction", action="store_true",
        help="Hide known air blocks with square and real-object shapes and score reconstruction",
    )
    parser.add_argument(
        "--validate-realistic-occlusions", action="store_true",
        help=(
            "Also attempt every full real source mask at its original or nearest valid "
            "position; requires --validate-background-reconstruction"
        ),
    )
    parser.add_argument(
        "--end-to-end-background-oof", action="store_true",
        help=(
            "Refit BG after excluding each spatial test fold, then rerun the ghost fit "
            "on the resulting cross-fitted observable signal"
        ),
    )
    parser.add_argument("--skip-input-hashes", action="store_true",
                        help="Skip SHA-256 provenance hashes for faster exploratory runs")
    parser.add_argument(
        "--frozen-background-config", type=Path,
        help="Verify a formal modeling export against this frozen BG configuration",
    )
    args = parser.parse_args()

    data_dir, output_dir = Path(args.data_dir), Path(args.output_dir)
    require_current_m1_output(output_dir)
    if args.generate_sam_masks:
        require_current_m1_output(args.generate_sam_masks)
    repo_root = PROJECT_ROOT
    pairs: tuple[tuple[int, int], ...] = args.pairs
    cohort_name = "single_lag_only" if pairs == SINGLE_LAG_PAIRS else "custom"
    analysis_blocks = (PRIMARY_BLOCK,) if args.summary_only else args.analysis_blocks
    background_modes: tuple[str, ...] = args.background_modes
    if args.primary_background not in background_modes:
        parser.error("--primary-background must also be listed in --background-modes")
    if args.mask_dilation_pixels != 0:
        parser.error("Mask dilation is prohibited; --mask-dilation-pixels must be 0")
    if args.spline_smoothness < 0:
        parser.error("--spline-smoothness must be non-negative")
    if args.background_huber_delta <= 0:
        parser.error("--background-huber-delta must be positive")
    if args.fixed_pattern_iterations < 1:
        parser.error("--fixed-pattern-iterations must be positive")
    if args.fixed_pattern_huber_iterations < 1:
        parser.error("--fixed-pattern-huber-iterations must be positive")
    if not args.summary_only and PRIMARY_BLOCK not in analysis_blocks:
        parser.error(f"Non-summary runs must include the primary block size {PRIMARY_BLOCK}")
    if args.generate_sam_masks and args.source_mask_npz:
        parser.error("--generate-sam-masks and --source-mask-npz are mutually exclusive")
    if args.validate_realistic_occlusions and not args.validate_background_reconstruction:
        parser.error(
            "--validate-realistic-occlusions requires --validate-background-reconstruction"
        )
    frozen_background = None
    if args.frozen_background_config:
        frozen_background = validate_frozen_background_config(
            config_path=args.frozen_background_config.resolve(),
            repo_root=repo_root,
            args=args,
        )
    existing_manifest = output_dir / "analysis_results.json"
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if ((previous.get("frozen_background") or {}).get("verified")
                and previous.get("configuration", {}).get("mask_dilation_pixels", 0) != 0):
            parser.error(
                "Cannot overwrite a frozen dilated-mask export with a no-dilation "
                "analysis. Preserve the historical evidence and use a separate "
                "output for the changed scientific definition."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for light_index, _ in pairs:
        path = data_dir / f"{light_index}-light.dcm"
        if not path.exists():
            raise FileNotFoundError(path)
    if args.generate_sam_masks:
        if args.sam_checkpoint is None:
            parser.error("--sam-checkpoint is required with --generate-sam-masks")
        if not args.sam_checkpoint.exists():
            raise FileNotFoundError(args.sam_checkpoint)
        generated = generate_sam_mask_archive(
            data_dir=data_dir,
            pairs=pairs,
            checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.sam_device,
            output_npz=args.generate_sam_masks,
        )
        print(json.dumps(clean_json(generated), ensure_ascii=False, indent=2))
        return
    for index in range(1, 32):
        path = data_dir / f"{index}-dark.dcm"
        if not path.exists():
            raise FileNotFoundError(path)

    dark_cache = {block: {} for block in analysis_blocks}
    for index in range(1, 32):
        image = pydicom.dcmread(data_dir / f"{index}-dark.dcm").pixel_array.astype(np.float32)
        for block in analysis_blocks:
            dark_cache[block][index] = block_mean(image, block)

    # Future-light nulls require every light from the earliest target onward.
    earliest_future = min(dark_index for _, dark_index in pairs)
    needs_masked_fixed_pattern = "masked_hybrid_spline" in background_modes
    fixed_pattern_light_indices = set(range(1, 31)) if needs_masked_fixed_pattern else set()
    needed_lights = sorted(set(
        [light for light, _ in pairs]
        + list(range(earliest_future, 32))
        + list(fixed_pattern_light_indices)
    ))
    source_light_indices = {light_index for light_index, _ in pairs}
    mask_light_indices = source_light_indices | fixed_pattern_light_indices
    light_cache = {block: {} for block in analysis_blocks}
    saturation_cache = {block: {} for block in analysis_blocks}
    source_block_masks = {block: {} for block in analysis_blocks}
    mask_archive = np.load(args.source_mask_npz) if args.source_mask_npz else None
    source_mask_origins = {}
    try:
        for index in needed_lights:
            image = pydicom.dcmread(data_dir / f"{index}-light.dcm").pixel_array.astype(np.float32)
            saturation = (image == image.max()).astype(np.float32)
            for block in analysis_blocks:
                light_cache[block][index] = block_mean(image, block)
                saturation_cache[block][index] = block_mean(saturation, block)
            if index in mask_light_indices:
                source_mask, source_mask_origins[index] = load_source_mask(
                    light_index=index,
                    light=image,
                    mask_archive=mask_archive,
                    dilation_pixels=args.mask_dilation_pixels,
                    allow_missing_archive_key=index not in source_light_indices,
                )
                for block in analysis_blocks:
                    source_block_masks[block][index] = block_source_masks(source_mask, block)
    finally:
        if mask_archive is not None:
            mask_archive.close()

    object_shape_templates = (
        extract_object_shape_templates([
            source_block_masks[PRIMARY_BLOCK][index][1]
            for index in sorted(source_light_indices)
        ])
        if args.validate_background_reconstruction else []
    )
    full_shape_occlusion_candidates = (
        realistic_occlusion_candidates({
            index: source_block_masks[PRIMARY_BLOCK][index][1]
            for index in sorted(source_light_indices)
        })
        if args.validate_realistic_occlusions else []
    )

    input_paths = [data_dir / f"{index}-dark.dcm" for index in range(1, 32)]
    input_paths.extend(data_dir / f"{index}-light.dcm" for index in needed_lights)
    if args.source_mask_npz:
        input_paths.append(args.source_mask_npz)
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
        if not args.summary_only:
            pair_dir.mkdir(parents=True, exist_ok=True)
        block_results = []
        primary_maps = None
        primary_crossfit_maps = None
        reconstruction_validation = None

        for block in analysis_blocks:
            darks, lights = dark_cache[block], light_cache[block]
            raw_dark = darks[dark_index]
            x = source(lights, light_index)
            folds = spatial_folds(raw_dark.shape)
            background_fit_mask, source_support = source_block_masks[block][light_index]
            contamination_supports = {
                index: source_block_masks[block][index][1]
                for index in fixed_pattern_light_indices
            }
            masked_fixed_pattern = None
            masked_fixed_pattern_diagnostics = None
            if needs_masked_fixed_pattern:
                masked_fixed_pattern, masked_fixed_pattern_diagnostics = (
                    estimate_masked_fixed_pattern(
                        darks=darks,
                        contamination_supports=contamination_supports,
                        target_index=dark_index,
                        smoothness=args.spline_smoothness,
                        decomposition_iterations=args.fixed_pattern_iterations,
                        huber_iterations=args.fixed_pattern_huber_iterations,
                        huber_delta=args.background_huber_delta,
                    )
                )
            future_sources = [
                (future, source(lights, future), saturation_cache[block][future])
                for future in range(dark_index, 32)
            ]

            if block == PRIMARY_BLOCK and args.validate_background_reconstruction:
                reconstruction_validation = validate_background_reconstruction(
                    dark=raw_dark,
                    trusted_air=background_fit_mask,
                    darks=darks,
                    target_index=dark_index,
                    background_modes=background_modes,
                    object_templates=object_shape_templates,
                    realistic_candidates=full_shape_occlusion_candidates,
                    smoothness=args.spline_smoothness,
                    seed=RANDOM_SEED + dark_index * 1000 + 97,
                    contamination_supports=contamination_supports,
                    masked_fixed_pattern=masked_fixed_pattern,
                    masked_fixed_pattern_diagnostics=masked_fixed_pattern_diagnostics,
                    decomposition_iterations=args.fixed_pattern_iterations,
                    huber_iterations=args.fixed_pattern_huber_iterations,
                    huber_delta=args.background_huber_delta,
                )

            background_results = []
            for background_mode in background_modes:
                end_to_end_oof = None
                if background_mode == "loo_median":
                    centered_baseline = baseline(darks, dark_index, "loo_median")
                    background = np.median(raw_dark) + centered_baseline
                    fit_values = raw_dark[background_fit_mask] - background[background_fit_mask]
                    background_diagnostics = {
                        "mode": background_mode,
                        "fit_blocks": int(background_fit_mask.sum()),
                        "fit_fraction": float(background_fit_mask.mean()),
                        "fit_residual_mae": float(np.mean(np.abs(fit_values))),
                        "fit_residual_mad": float(np.median(
                            np.abs(fit_values - np.median(fit_values))
                        )),
                        "source": "other median-centered dark acquisitions",
                    }
                elif background_mode == "hybrid_spline":
                    background, background_diagnostics = estimate_hybrid_dark_background(
                        dark=raw_dark,
                        darks=darks,
                        target_index=dark_index,
                        fit_mask=background_fit_mask,
                        smoothness=args.spline_smoothness,
                        huber_delta=args.background_huber_delta,
                    )
                elif background_mode == "masked_hybrid_spline":
                    background, background_diagnostics = estimate_masked_hybrid_dark_background(
                        dark=raw_dark,
                        darks=darks,
                        contamination_supports=contamination_supports,
                        target_index=dark_index,
                        fit_mask=background_fit_mask,
                        smoothness=args.spline_smoothness,
                        fixed_pattern=masked_fixed_pattern,
                        fixed_pattern_diagnostics=masked_fixed_pattern_diagnostics,
                        decomposition_iterations=args.fixed_pattern_iterations,
                        huber_iterations=args.fixed_pattern_huber_iterations,
                        huber_delta=args.background_huber_delta,
                    )
                else:
                    background, background_diagnostics = estimate_single_dark_background(
                        dark=raw_dark,
                        fit_mask=background_fit_mask,
                        mode=background_mode,
                        smoothness=args.spline_smoothness,
                    )
                if (
                    block == PRIMARY_BLOCK
                    and background_mode != "loo_median"
                    and not args.skip_background_oof
                ):
                    background_diagnostics["spatial_oof"] = background_spatial_oof(
                        dark=raw_dark,
                        fit_mask=background_fit_mask,
                        folds=folds,
                        mode=background_mode,
                        smoothness=args.spline_smoothness,
                        darks=darks,
                        target_index=dark_index,
                        contamination_supports=contamination_supports,
                        fixed_pattern=masked_fixed_pattern,
                        fixed_pattern_diagnostics=masked_fixed_pattern_diagnostics,
                        decomposition_iterations=args.fixed_pattern_iterations,
                        huber_iterations=args.fixed_pattern_huber_iterations,
                        huber_delta=args.background_huber_delta,
                    )

                y = raw_dark - background
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
                    mask_result = {
                        "mask_mode": mask_mode,
                        "fit": _compact_metric_view(fit) if args.summary_only else _metric_view(fit),
                        "null_summary": null_summary,
                        "detection_gate": detection_gate(fit, null_rows, null_summary),
                    }
                    if not args.summary_only:
                        mask_result["null_detail"] = null_rows
                    mask_results.append(mask_result)
                    if (
                        block == PRIMARY_BLOCK
                        and background_mode == args.primary_background
                        and mask_mode == "all"
                    ):
                        primary_maps = (
                            x, raw_dark, background, y, folds, fit,
                            background_fit_mask, source_support,
                        )
                if block == PRIMARY_BLOCK and args.end_to_end_background_oof:
                    crossfit_y, crossfit_background, crossfit_background_diagnostics = (
                        crossfit_observable_signal(
                            dark=raw_dark,
                            fit_mask=background_fit_mask,
                            folds=folds,
                            mode=background_mode,
                            smoothness=args.spline_smoothness,
                            darks=darks,
                            target_index=dark_index,
                            contamination_supports=contamination_supports,
                            fixed_pattern=masked_fixed_pattern,
                            fixed_pattern_diagnostics=masked_fixed_pattern_diagnostics,
                            decomposition_iterations=args.fixed_pattern_iterations,
                            huber_iterations=args.fixed_pattern_huber_iterations,
                            huber_delta=args.background_huber_delta,
                        )
                    )
                    crossfit_fit, crossfit_null_rows, crossfit_null_summary = evaluate_with_nulls(
                        x=x,
                        y=crossfit_y,
                        source_saturation=saturation_cache[block][light_index],
                        mask_mode="all",
                        folds=folds,
                        future_sources=future_sources,
                        seed=RANDOM_SEED + dark_index * 1000 + block,
                    )
                    end_to_end_oof = {
                        "background": crossfit_background_diagnostics,
                        "fit": (
                            _compact_metric_view(crossfit_fit)
                            if args.summary_only else _metric_view(crossfit_fit)
                        ),
                        "null_summary": crossfit_null_summary,
                        "detection_gate": detection_gate(
                            crossfit_fit, crossfit_null_rows, crossfit_null_summary,
                        ),
                    }
                    if not args.summary_only:
                        end_to_end_oof["null_detail"] = crossfit_null_rows
                    if background_mode == args.primary_background:
                        primary_crossfit_maps = (
                            crossfit_y, crossfit_background, crossfit_fit,
                        )
                background_results.append({
                    "background_mode": background_mode,
                    "background_diagnostics": background_diagnostics,
                    "mask_results": mask_results,
                    **({"end_to_end_background_oof": end_to_end_oof}
                       if end_to_end_oof is not None else {}),
                })
            block_results.append({
                "block_size": block,
                "source_mask_origin": source_mask_origins[light_index],
                "source_support_fraction": float(source_support.mean()),
                "background_fit_fraction": float(background_fit_mask.mean()),
                "background_results": background_results,
            })

        pair_result = {
            "light_index": light_index,
            "dark_index": dark_index,
            "role": "single_lag_background_comparison",
            "cohort_stratum": single_lag_stratum((light_index, dark_index)),
            "primary_block_size": PRIMARY_BLOCK,
            "primary_background": args.primary_background,
            "source_mask_origin": source_mask_origins[light_index],
            "block_results": block_results,
        }
        if reconstruction_validation is not None:
            pair_result["background_reconstruction_validation"] = reconstruction_validation
        if not args.summary_only:
            if primary_maps is None:
                raise RuntimeError("Primary block result was not generated")
            (x, raw_dark, background, y, folds, primary_fit,
             background_fit_mask, source_support) = primary_maps
            maps_path = pair_dir / "oof_maps_block16.npz"
            figure_path = pair_dir / "six_panel_oof.png"
            if not args.skip_pair_figures:
                render_six_panel(figure_path, pair_label, x, raw_dark,
                                 background, args.primary_background, y, primary_fit)
            map_arrays = {
                "source": x.astype(np.float32),
                "raw_dark": raw_dark.astype(np.float32),
                "background": background.astype(np.float32),
                "background_mode": np.asarray(args.primary_background),
                "background_fit_mask": background_fit_mask,
                "source_support": source_support,
                "source_saturation_fraction": (
                    saturation_cache[PRIMARY_BLOCK][light_index].astype(np.float32)
                ),
                "observable_signal": y.astype(np.float32),
                "oof_prediction": primary_fit["prediction"].astype(np.float32),
                "oof_residual": primary_fit["residual"].astype(np.float32),
                "evaluated_mask": primary_fit["evaluated_mask"],
                "spatial_folds": folds,
            }
            if primary_crossfit_maps is not None:
                crossfit_y, crossfit_background, crossfit_fit = primary_crossfit_maps
                map_arrays.update({
                    "crossfit_background": crossfit_background.astype(np.float32),
                    "crossfit_observable_signal": crossfit_y.astype(np.float32),
                    "crossfit_oof_prediction": crossfit_fit["prediction"].astype(np.float32),
                    "crossfit_oof_residual": crossfit_fit["residual"].astype(np.float32),
                    "crossfit_evaluated_mask": crossfit_fit["evaluated_mask"],
                })
            np.savez_compressed(
                maps_path,
                **map_arrays,
            )
            pair_result["primary_maps"] = str(maps_path.resolve())
            pair_result["output_artifacts"] = {
                "primary_maps": artifact_record(maps_path, output_dir),
            }
            if not args.skip_pair_figures:
                pair_result["six_panel"] = str(figure_path.resolve())
                pair_result["output_artifacts"]["six_panel"] = artifact_record(
                    figure_path, output_dir,
                )
        pair_results.append(pair_result)

    comparison_summary = summarize_background_comparison(pair_results)
    stratified_summary = (
        summarize_single_lag_strata(pair_results)
        if cohort_name == "single_lag_only" else None
    )
    reconstruction_summary = (
        summarize_reconstruction_validation(pair_results)
        if args.validate_background_reconstruction else None
    )
    result = {
        "schema_version": 6,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "study_question": "Can detectable previous-exposure structure in a later dark acquisition be explained by a low-dimensional linear spatial model?",
        "evidence_boundary": [
            "Y is an observable pseudo-ghost signal, not clean ground-truth ghost.",
            "Y may contain read noise, baseline error, drift, and older-frame memory.",
            "These selected pairs are for mechanism discovery, not prevalence or generalization claims.",
        ],
        "definitions": {
            "source": "X_(t-1) = L_(t-1) - Q75(L_(t-1)), evaluated on each block-mean grid",
            "background": "Compared estimators of B_t; single-dark surfaces are fitted only outside the expanded source mask",
            "observable_signal": "Y_t = D_t - estimated_background_t",
            "model": "Y_t = alpha * X_(t-1) + intercept + error",
            "prediction": "Four-fold strict spatial out-of-fold prediction",
        },
        "configuration": {
            "data_directory": str(data_dir.resolve()),
            "cohort": cohort_name,
            "pairs": [list(pair) for pair in pairs],
            "block_sizes": list(analysis_blocks),
            "summary_only": args.summary_only,
            "pair_figures_written": not args.skip_pair_figures,
            "primary_visualization_block_size": PRIMARY_BLOCK,
            "background_modes": list(background_modes),
            "primary_background": args.primary_background,
            "source_mask_npz": str(args.source_mask_npz.resolve()) if args.source_mask_npz else None,
            "source_mask_origins": source_mask_origins,
            "mask_dilation_pixels": args.mask_dilation_pixels,
            "background_fit_max_source_fraction": 0.01,
            "spline_smoothness": args.spline_smoothness,
            "masked_fixed_pattern": {
                "enabled": needs_masked_fixed_pattern,
                "aggregation": "contamination-masked noise-precision-weighted Huber mean",
                "uses_temporal_pixelwise_median": False,
                "previous_light_lags_excluded": [1],
                "decomposition_iterations": args.fixed_pattern_iterations,
                "huber_iterations": args.fixed_pattern_huber_iterations,
                "huber_delta": args.background_huber_delta,
                "extra_mask_fallback": "Otsu when an external archive lacks a light index",
            },
            "background_spatial_oof": not args.skip_background_oof,
            "end_to_end_background_oof": args.end_to_end_background_oof,
            "background_reconstruction_validation": {
                "enabled": args.validate_background_reconstruction,
                "square_sides_blocks": list(PSEUDO_OCCLUSION_SQUARE_SIDES),
                "squares_per_size": PSEUDO_OCCLUSIONS_PER_SQUARE_SIZE,
                "object_shape_count": PSEUDO_OBJECT_OCCLUSION_COUNT,
                "object_template_count": len(object_shape_templates),
                "full_shape_candidate_count": len(full_shape_occlusion_candidates),
                "full_shape_validation_enabled": args.validate_realistic_occlusions,
                "target_values_hidden_from_single-dark_fit": True,
                "loo_level_anchor_uses_training_air_only": True,
            },
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
        "frozen_background": frozen_background,
        "code_provenance": {
            "package_sources": package_sources(),
            "analysis_script": {
                "path": "xray_ghost_removal/m1/pipeline.py",
                "sha256": sha256(Path(__file__).resolve()),
            },
            "shared_spatial_analysis": {
                "path": "xray_ghost_removal/common/spatial.py",
                "sha256": sha256(repo_root / "xray_ghost_removal/common/spatial.py"),
            },
        },
        "input_provenance": provenance,
        "comparison_summary": comparison_summary,
        "stratified_comparison_summary": stratified_summary,
        "background_reconstruction_summary": reconstruction_summary,
        "pair_results": pair_results,
    }
    result_path = output_dir / "analysis_results.json"
    result_path.write_text(json.dumps(clean_json(result), ensure_ascii=False, indent=2), encoding="utf-8")

    if args.summary_only:
        compact = {
            "output": str(result_path.resolve()),
            "comparison_summary": comparison_summary,
            "stratified_comparison_summary": stratified_summary,
            "background_reconstruction_summary": reconstruction_summary,
        }
    else:
        compact = {
            "output": str(result_path.resolve()),
            "pairs": [
                {
                    "pair": f"{row['light_index']}->{row['dark_index']}",
                    "block16_all_by_background": {
                        background_row["background_mode"]: {
                            "background_diagnostics": background_row["background_diagnostics"],
                            "fit": next(
                                mask_row["fit"] for mask_row in background_row["mask_results"]
                                if mask_row["mask_mode"] == "all"
                            ),
                            "null_summary": next(
                                mask_row["null_summary"] for mask_row in background_row["mask_results"]
                                if mask_row["mask_mode"] == "all"
                            ),
                        }
                        for block_row in row["block_results"]
                        if block_row["block_size"] == PRIMARY_BLOCK
                        for background_row in block_row["background_results"]
                    },
                }
                for row in pair_results
            ],
        }
    print(json.dumps(clean_json(compact), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
