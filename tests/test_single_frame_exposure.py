"""Focused tests for the frozen single-frame exposure association."""

import numpy as np
import pytest

from scripts.analyze_single_frame_exposure import (
    repeated_setting_summary,
    source_morphology,
    spearman_summary,
)


def test_spearman_summary_reports_stable_monotonic_relation():
    result = spearman_summary(
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        seed=17,
        permutations=99,
    )
    assert result["rho"] == pytest.approx(1.0)
    assert result["leave_one_out_rho_min"] == pytest.approx(1.0)
    assert result["leave_one_out_rho_max"] == pytest.approx(1.0)


def test_repeated_settings_require_exact_kv_ma_and_duration_match():
    morphology = {
        "centroid_row_fraction": 0.5,
        "centroid_column_fraction": 0.5,
        "boundary_to_area_ratio": 0.2,
        "magnitude_p50": 100.0,
    }
    zones = [
        {"zone": "r2c2", "eligible": True, "predictively_reliable": True,
         "alpha": 0.001},
        {"zone": "r1c1", "eligible": False, "predictively_reliable": False},
    ]
    rows = [
        {
            "pair": "1->2", "kv": 70.0, "ma": 100.0, "exposure_time_ms": 100.0,
            "mas": 10.0, "alpha": 0.001, "ghost_contrast_rms": 10.0,
            "detected": True, "source_support_fraction": 0.2,
            "source_morphology": morphology, "spatial_zone_alphas": zones,
        },
        {
            "pair": "2->3", "kv": 70.0, "ma": 100.0, "exposure_time_ms": 100.0,
            "mas": 10.0, "alpha": 0.002, "ghost_contrast_rms": 20.0,
            "detected": True, "source_support_fraction": 0.2,
            "source_morphology": morphology,
            "spatial_zone_alphas": [
                {"zone": "r2c2", "eligible": True, "predictively_reliable": True,
                 "alpha": 0.002},
                {"zone": "r1c1", "eligible": False, "predictively_reliable": False},
            ],
        },
        {
            "pair": "3->4", "kv": 70.0, "ma": 80.0, "exposure_time_ms": 125.0,
            "mas": 10.0, "alpha": 0.003, "ghost_contrast_rms": 30.0,
            "detected": True, "source_support_fraction": 0.2,
            "source_morphology": morphology, "spatial_zone_alphas": zones,
        },
    ]
    result = repeated_setting_summary(rows)
    assert len(result) == 1
    assert result[0]["pairs"] == ["1->2", "2->3"]
    assert result[0]["alpha_max_to_min_ratio"] == 2.0
    assert result[0]["spatial_consistency"]["same_order_as_global_fraction"] == 1.0


def test_source_morphology_uses_evaluated_support_only():
    source = np.zeros((5, 5), dtype=float)
    source[1:3, 2:4] = -10.0
    support = source < 0
    evaluated = np.ones_like(support)
    result = source_morphology(source, evaluated, support)
    assert result["magnitude_p50"] == 10.0
    assert result["component_count_min_4_blocks"] == 1
    assert result["centroid_row_fraction"] == pytest.approx(0.375)
    assert result["centroid_column_fraction"] == pytest.approx(0.625)
