"""Focused tests for the finite background-freeze audit."""

from pathlib import Path

import pytest

from scripts.audit_background_freeze import (
    SENSITIVITY_CONFIGS,
    bootstrap_median,
    extract_loo_comparison,
    resolved_config,
)


def test_audit_source_defaults_to_temporary_intermediate_runs():
    source = Path("scripts/audit_background_freeze.py").read_text(encoding="utf-8")
    assert "tempfile.TemporaryDirectory" in source
    assert '"--keep-runs"' in source


def test_freeze_grid_has_one_nominal_and_separate_stress_test():
    configurations = [resolved_config(row) for row in SENSITIVITY_CONFIGS]
    assert [row["name"] for row in configurations].count("nominal") == 1
    gated = [row for row in configurations if row["gate"]]
    stress = [row for row in configurations if not row["gate"]]
    assert len(gated) == 9
    assert [row["name"] for row in stress] == ["otsu_only_stress"]
    assert all(row["mask_source"] == "sam_with_otsu_fallback" for row in gated)


def test_pair_bootstrap_preserves_strictly_negative_improvement():
    summary = bootstrap_median([-0.8, -0.5, -0.2, -0.1], seed=7)
    assert summary["median"] < 0
    assert summary["ci025"] < 0
    assert summary["ci975"] < 0


def test_loo_comparison_preserves_pairing_and_nested_sample_counts():
    def sample(kind: str, loo: float, masked: float) -> dict:
        return {
            "kind": kind,
            "by_background": {
                "loo_median": {"mae": loo},
                "masked_hybrid_spline": {"mae": masked},
            },
        }

    analysis = {
        "pair_results": [
            {
                "light_index": 1,
                "dark_index": 2,
                "background_reconstruction_validation": {
                    "samples": [sample("square", 1.0, 0.6), sample("object_shape", 1.2, 0.8)],
                },
            },
            {
                "light_index": 2,
                "dark_index": 3,
                "background_reconstruction_validation": {
                    "samples": [sample("square", 0.7, 0.5), sample("object_shape", 0.9, 1.0)],
                },
            },
        ],
    }

    comparison = extract_loo_comparison(analysis)

    assert comparison["sample_count"] == 4
    assert comparison["improved_sample_count"] == 3
    assert comparison["improved_pair_count"] == 2
    assert comparison["pairs"][0]["median_delta_mae"] == pytest.approx(-0.4)
