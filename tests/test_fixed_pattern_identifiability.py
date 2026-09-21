"""Tests for the fixed-pattern spatial identifiability audit."""

import numpy as np

from scripts.analyze_pseudo_ghost_mechanism import estimate_masked_fixed_pattern
from scripts.audit_fixed_pattern_identifiability import distribution


def test_fixed_pattern_support_maps_use_final_weights():
    shape = (12, 14)
    darks = {
        index: np.full(shape, 1000.0 + index, dtype=float)
        for index in range(1, 32)
    }
    supports = {index: np.zeros(shape, dtype=bool) for index in range(1, 31)}
    supports[1][:, :3] = True

    _, diagnostics = estimate_masked_fixed_pattern(
        darks=darks,
        contamination_supports=supports,
        target_index=31,
        decomposition_iterations=1,
        huber_iterations=1,
        include_support_maps=True,
    )

    coverage = diagnostics["coverage_map"]
    effective = diagnostics["effective_coverage_map"]
    assert np.all(coverage[:, :3] == 29)
    assert np.all(coverage[:, 3:] == 30)
    assert np.all(effective > 0)
    assert np.all(effective <= coverage + 1e-9)


def test_distribution_reports_low_support_fractions():
    values = np.array([[0, 1, 2], [3, 4, 10]], dtype=float)
    summary = distribution(values, np.ones_like(values, dtype=bool))
    assert summary["equal_0_fraction"] == 1 / 6
    assert summary["less_equal_2_fraction"] == 3 / 6
    assert summary["below_5_fraction"] == 5 / 6
    assert summary["median"] == 2.5
