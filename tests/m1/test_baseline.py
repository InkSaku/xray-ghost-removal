"""Focused tests for strict spatial OOF pseudo-ghost fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from xray_ghost_removal.m1.background import background_spatial_oof, crossfit_observable_signal, estimate_hybrid_dark_background, estimate_masked_hybrid_dark_background, estimate_single_dark_background
from xray_ghost_removal.m1.config import parse_block_sizes, parse_pairs, single_lag_stratum, validate_frozen_background_config
from xray_ghost_removal.m1.fitting import detection_gate, strict_affine_oof
from xray_ghost_removal.m1.segmentation import load_source_mask
from xray_ghost_removal.m1.validation import generate_pseudo_occlusions, realistic_occlusion_candidates, validate_background_reconstruction
from xray_ghost_removal.common.spatial import spatial_folds


def test_parse_all_single_lag_pairs():
    pairs = parse_pairs("all")
    assert len(pairs) == 30
    assert pairs[0] == (1, 2)
    assert pairs[-1] == (30, 31)


def test_parse_block_sizes_rejects_unknown_values():
    assert parse_block_sizes("16,32,16") == (16, 32)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_block_sizes("16,12")


def test_frozen_background_validation_checks_mask_hash_and_parameters(tmp_path):
    mask_path = tmp_path / "mask.npz"
    mask_path.write_bytes(b"frozen mask")
    mask_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    config = {
        "status": "frozen",
        "background_mode": "masked_hybrid_spline",
        "block_size": 16,
        "mask": {
            "archive_relative_path": "mask.npz",
            "sha256": mask_hash,
            "dilation_pixels": 0,
        },
        "spline_smoothness": 20.0,
        "huber_delta": 1.5,
        "fixed_pattern": {"decomposition_iterations": 2, "huber_iterations": 4},
    }
    config_path = tmp_path / "frozen.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args = argparse.Namespace(
        primary_background="masked_hybrid_spline",
        background_modes=("masked_hybrid_spline",),
        mask_dilation_pixels=0,
        spline_smoothness=20.0,
        background_huber_delta=1.5,
        fixed_pattern_iterations=2,
        fixed_pattern_huber_iterations=4,
        analysis_blocks=(16,),
        summary_only=False,
        source_mask_npz=mask_path,
    )

    result = validate_frozen_background_config(config_path, tmp_path, args)
    assert result["verified"]
    assert result["mask_archive_sha256"] == mask_hash

    args.spline_smoothness = 40.0
    with pytest.raises(ValueError, match="differs from frozen BG"):
        validate_frozen_background_config(config_path, tmp_path, args)

    config["mask"]["dilation_pixels"] = 24
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="dilation, which is prohibited"):
        validate_frozen_background_config(config_path, tmp_path, args)


def test_source_masks_cannot_expand_external_or_fallback_masks(tmp_path):
    expected = np.zeros((64, 64), dtype=bool)
    expected[12:40, 14:38] = True
    expected[22:30, 20:28] = False
    expected[50, 4:55] = True
    image = np.where(expected, 1000.0, 10000.0)
    path = tmp_path / "masks.npz"
    np.savez(path, light_1=expected)
    with np.load(path) as archive:
        for masks in (archive, None):
            actual, _ = load_source_mask(1, image, masks)
            assert np.array_equal(actual, expected)
            for amount in (-1, 1, 24):
                with pytest.raises(ValueError, match="dilation is prohibited"):
                    load_source_mask(1, image, masks, dilation_pixels=amount)


def test_single_lag_cohort_is_prestratified():
    pairs = parse_pairs("single-lag")
    strata = [single_lag_stratum(pair) for pair in pairs]
    assert len(pairs) == 13
    assert strata.count("robust_confirmed") == 2
    assert strata.count("parameter_sensitive") == 9
    assert strata.count("not_detected_control") == 2
    assert single_lag_stratum((11, 12)) is None


def test_detection_gate_rejects_null_significant_but_tiny_effect():
    fit = {"alpha_fold_mean": 0.0001, "alpha_fold_cv": 0.05, "cv_r2": 0.02}
    null_rows = [{"kind": "future_light", "status": "ok", "cv_r2": 0.001}]
    null_summary = {"cv_r2_p95": 0.002}
    result = detection_gate(fit, null_rows, null_summary)
    assert not result["passed"]
    assert not result["checks"]["minimum_cv_r2"]
    assert not result["checks"]["minimum_null_margin"]


def test_detection_gate_accepts_stable_practical_effect():
    fit = {"alpha_fold_mean": 0.001, "alpha_fold_cv": 0.05, "cv_r2": 0.5}
    null_rows = [{"kind": "future_light", "status": "ok", "cv_r2": 0.1}]
    null_summary = {"cv_r2_p95": 0.15}
    assert detection_gate(fit, null_rows, null_summary)["passed"]


def test_strict_affine_oof_recovers_known_model():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 5000, (64, 64))
    y = 0.0008 * x + 2.5 + rng.normal(0, 0.4, x.shape)
    mask = np.ones_like(x, dtype=bool)
    result = strict_affine_oof(x, y, mask, spatial_folds(x.shape))

    assert abs(result["alpha_fold_mean"] - 0.0008) < 0.00003
    assert result["cv_r2"] > 0.98
    assert result["ncc"] > 0.99
    assert result["low_frequency_y_source_ncc"] > 0.99
    assert np.isfinite(result["prediction"]).all()
    assert np.isfinite(result["residual"]).all()


def test_held_out_targets_do_not_change_their_fold_parameters():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(64, 64))
    y = 0.4 * x + rng.normal(0, 0.1, x.shape)
    folds = spatial_folds(x.shape)
    mask = np.ones_like(x, dtype=bool)
    original = strict_affine_oof(x, y, mask, folds)

    changed_y = y.copy()
    changed_y[folds == 0] += 10000.0
    changed = strict_affine_oof(x, changed_y, mask, folds)

    fold0_original = next(row for row in original["fold_parameters"] if row["fold"] == 0)
    fold0_changed = next(row for row in changed["fold_parameters"] if row["fold"] == 0)
    assert fold0_changed["alpha"] == fold0_original["alpha"]
    assert fold0_changed["intercept"] == fold0_original["intercept"]


def test_single_dark_surfaces_extrapolate_into_excluded_source_region():
    height, width = 48, 52
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    true_background = 1000.0 + 35.0 * xx - 20.0 * yy + 12.0 * xx * yy + 8.0 * xx ** 2
    source_support = np.zeros((height, width), dtype=bool)
    source_support[14:34, 15:39] = True
    contaminated_dark = true_background.copy()
    contaminated_dark[source_support] += 50.0

    polynomial, _ = estimate_single_dark_background(
        contaminated_dark, ~source_support, "poly2",
    )
    spline, spline_diagnostics = estimate_single_dark_background(
        contaminated_dark, ~source_support, "robust_spline", huber_delta=1.0,
    )

    assert np.mean(np.abs(polynomial[source_support] - true_background[source_support])) < 1e-6
    assert np.mean(np.abs(spline[source_support] - true_background[source_support])) < 3.0
    assert spline_diagnostics["huber_delta"] == 1.0


def test_hybrid_background_preserves_fixed_pattern_and_fits_current_drift():
    height, width = 48, 52
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    fixed_pattern = 7.0 * np.where((np.indices((height, width)).sum(axis=0) % 2) == 0, 1, -1)
    smooth_drift = 1000.0 + 28.0 * xx - 16.0 * yy + 7.0 * xx * yy
    true_background = fixed_pattern + smooth_drift
    source_support = np.zeros((height, width), dtype=bool)
    source_support[14:34, 15:39] = True
    contaminated = true_background.copy()
    contaminated[source_support] += 80.0
    darks = {
        index: fixed_pattern + 900.0 + index
        for index in range(1, 32)
    }
    darks[6] = contaminated

    estimate, diagnostics = estimate_hybrid_dark_background(
        contaminated, darks, 6, ~source_support, smoothness=20.0,
    )

    assert diagnostics["target_dark_in_fixed_pattern"] is False
    assert np.mean(np.abs(estimate[source_support] - true_background[source_support])) < 3.0


def test_masked_hybrid_fixed_pattern_rejects_repeated_temporal_contamination():
    height, width = 48, 52
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    fixed_pattern = 7.0 * np.where(
        (np.indices((height, width)).sum(axis=0) % 2) == 0, 1, -1,
    )
    source_support = np.zeros((height, width), dtype=bool)
    source_support[14:34, 15:39] = True
    empty_support = np.zeros_like(source_support)
    darks = {}
    contamination_supports = {}
    for index in range(1, 32):
        smooth_drift = 1000.0 + 3.0 * index + 28.0 * xx - 16.0 * yy + 7.0 * xx * yy
        darks[index] = fixed_pattern + smooth_drift
        if 2 <= index <= 20:
            darks[index] = darks[index] + 80.0 * source_support
        if index <= 30:
            contamination_supports[index] = (
                source_support.copy() if index <= 19 or index == 30 else empty_support.copy()
            )

    target_index = 31
    darks[target_index] = darks[target_index] + 50.0 * source_support
    true_background = (
        fixed_pattern + 1000.0 + 3.0 * target_index
        + 28.0 * xx - 16.0 * yy + 7.0 * xx * yy
    )
    masked, diagnostics = estimate_masked_hybrid_dark_background(
        dark=darks[target_index],
        darks=darks,
        contamination_supports=contamination_supports,
        target_index=target_index,
        fit_mask=~source_support,
        smoothness=20.0,
    )
    median_hybrid, _ = estimate_hybrid_dark_background(
        darks[target_index], darks, target_index, ~source_support, smoothness=20.0,
    )

    masked_error = np.mean(np.abs(masked[source_support] - true_background[source_support]))
    median_error = np.mean(
        np.abs(median_hybrid[source_support] - true_background[source_support])
    )
    assert diagnostics["fixed_pattern"]["uses_temporal_pixelwise_median"] is False
    assert diagnostics["fixed_pattern"]["target_dark_in_fixed_pattern"] is False
    assert diagnostics["fixed_pattern"]["coverage_min"] > 0
    assert masked_error < 1.0
    assert masked_error < 0.05 * median_error


def test_background_spatial_oof_never_needs_excluded_source_values():
    height, width = 48, 52
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    background = 800.0 + 18.0 * xx + 9.0 * yy + 4.0 * xx * yy
    source_support = np.zeros_like(background, dtype=bool)
    source_support[12:36, 16:38] = True
    dark = background.copy()
    dark[source_support] += 10000.0

    metrics = background_spatial_oof(
        dark, ~source_support, spatial_folds(dark.shape), "poly2",
    )

    assert metrics["evaluated_blocks"] == int((~source_support).sum())
    assert metrics["mae"] < 1e-6


def test_crossfit_background_does_not_use_heldout_fold_dark_values():
    height, width = 48, 52
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    dark = 800.0 + 18.0 * xx + 9.0 * yy + 4.0 * xx * yy
    folds = spatial_folds(dark.shape)
    fit_mask = np.ones_like(dark, dtype=bool)

    _, original_background, diagnostics = crossfit_observable_signal(
        dark, fit_mask, folds, "poly2",
    )
    changed = dark.copy()
    changed[folds == 0] += 10000.0
    _, changed_background, _ = crossfit_observable_signal(
        changed, fit_mask, folds, "poly2",
    )

    assert diagnostics["target_fold_dark_values_used_in_its_background_fit"] is False
    assert np.allclose(
        original_background[folds == 0], changed_background[folds == 0], atol=1e-6,
    )


def test_pseudo_occlusions_stay_inside_trusted_air():
    trusted_air = np.ones((64, 64), dtype=bool)
    trusted_air[20:35, 24:40] = False
    object_shape = np.zeros((14, 18), dtype=bool)
    object_shape[2:12, 3:7] = True
    object_shape[8:12, 3:16] = True

    samples = generate_pseudo_occlusions(
        trusted_air,
        [object_shape],
        seed=23,
        square_sides=(8, 12),
        squares_per_size=1,
        object_count=1,
    )

    assert {row["kind"] for row in samples} == {"square", "object_shape"}
    assert all(np.all(trusted_air[row["mask"]]) for row in samples)


def test_full_real_shape_is_kept_or_reported_unverified():
    support = np.zeros((32, 32), dtype=bool)
    support[8:24, 6:26] = True
    candidates = realistic_occlusion_candidates({7: support})
    trusted_air = np.ones_like(support)
    trusted_air[8:24, 6:26] = False

    result = validate_background_reconstruction(
        dark=np.ones((32, 32), dtype=float),
        trusted_air=trusted_air,
        darks={2: np.ones((32, 32), dtype=float)},
        target_index=2,
        background_modes=("poly2",),
        object_templates=[],
        realistic_candidates=candidates,
        smoothness=20.0,
        seed=31,
        square_sides=(),
        object_count=0,
    )

    attempts = result["realistic_occlusion_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["requested_blocks"] == int(support.sum())
    if attempts[0]["status"] == "validated":
        sample = result["samples"][0]
        assert sample["kind"] == "real_source_full_shape"
        assert sample["holdout_blocks"] == int(support.sum())
    else:
        assert attempts[0]["status"] == "unverified"
        assert result["generated_sample_count"] == 0


def test_pseudo_occlusion_validation_recovers_known_quadratic_background():
    height, width = 64, 68
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    background = 900.0 + 22.0 * xx - 17.0 * yy + 6.0 * xx * yy + 5.0 * xx ** 2
    trusted_air = np.ones_like(background, dtype=bool)
    object_shape = np.zeros((16, 20), dtype=bool)
    object_shape[2:14, 3:8] = True
    object_shape[10:14, 3:18] = True

    result = validate_background_reconstruction(
        dark=background,
        trusted_air=trusted_air,
        darks={2: background},
        target_index=2,
        background_modes=("poly2", "robust_spline"),
        object_templates=[object_shape],
        smoothness=20.0,
        seed=31,
        square_sides=(8, 12),
        squares_per_size=1,
        object_count=1,
    )

    assert result["generated_sample_count"] == 3
    polynomial = result["summary"]["overall_by_background"]["poly2"]
    spline = result["summary"]["overall_by_background"]["robust_spline"]
    assert polynomial["median_sample_mae"] < 1e-6
    assert spline["median_sample_mae"] < 3.0


def test_public_m1_pipeline_writes_complete_provenance_without_touching_frozen_data(tmp_path):
    """Run the migrated orchestration on a temporary complete synthetic sequence."""
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    from xray_ghost_removal.cli import main
    from xray_ghost_removal.common.provenance import sha256

    raw, output = tmp_path/'raw', tmp_path/'output'
    raw.mkdir()
    rng = np.random.default_rng(27)
    yy, xx = np.mgrid[:256, :256]
    previous = np.zeros((256, 256))
    for index in range(1, 32):
        light = np.full((256, 256), 30000.)
        y, x = 16 + index*5 % 144, 16 + index*7 % 144
        light[y:y+64, x:x+64] = 8000
        light += rng.normal(0, 10, light.shape)
        dark = 1200 + .03*xx + .02*yy + .003*previous + rng.normal(0, 1, xx.shape)
        for role, array in [('dark', dark), ('light', light)]:
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
            ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
            ds.Rows, ds.Columns = array.shape
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
            ds.PixelRepresentation, ds.SamplesPerPixel = 0, 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.PixelData = np.rint(array).astype('<u2').tobytes()
            ds.save_as(raw/f'{index}-{role}.dcm', enforce_file_format=True)
        previous = light - 30000
    main(['m1', 'fit', '--data-dir', str(raw), '--output-dir', str(output),
          '--pairs', '5:6', '--analysis-blocks', '16', '--background-modes', 'poly2',
          '--primary-background', 'poly2', '--skip-background-oof', '--skip-pair-figures'])
    result = json.loads((output/'analysis_results.json').read_text())
    assert result['configuration']['mask_dilation_pixels'] == 0
    assert len(result['pair_results']) == 1
    inputs = result['input_provenance']['files']
    assert {'5-light.dcm', '6-dark.dcm'} <= inputs.keys()
    assert all(sha256(raw/name) == info['sha256'] for name, info in inputs.items())
    blocks = result['pair_results'][0]['block_results'][0]
    fit = blocks['background_results'][0]['mask_results'][0]['fit']
    assert fit['alpha_fold_mean'] == pytest.approx(.003, abs=5e-5)
    sources = result['code_provenance']['package_sources']['files']
    assert 'xray_ghost_removal/m1/background.py' in sources
    assert 'xray_ghost_removal/m1/fitting.py' in sources
    assert all(sha256(Path(path)) == digest for path, digest in sources.items())
