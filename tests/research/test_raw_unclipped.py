"""Unclipped laws: per-pair offsets never enter a law, and fits recover planted structure."""
import numpy as np

from xray_ghost_removal.research.raw_unclipped import (
    SIGMAS, Unclipped, fit_centered, fit_model, position_basis, residual,
)


def _unit(rng, offset, slope_fn, spill=0.0, n=600, sigma=SIGMAS[3]):
    rows, cols = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    xu = rng.uniform(500, 30000, n)
    blur = {s: rng.uniform(0, 1, n) for s in SIGMAS}
    y = offset + slope_fn(rows, cols) * xu + spill * blur[sigma] + rng.normal(0, 0.5, n)
    return Unclipped('p', 'd', 2.3, [], rows, cols, xu, y, np.ones(n), blur)


def test_per_pair_offsets_do_not_change_the_fitted_law():
    rng = np.random.default_rng(1)
    units = [_unit(rng, 0.0, lambda r, c: 0.003 + 0 * r) for _ in range(4)]
    shifted = [Unclipped(**{**u.__dict__, 'y': u.y + 1000.0 * k}) for k, u in enumerate(units)]
    a = fit_centered('mconst', units)['beta']
    b = fit_centered('mconst', shifted)['beta']
    assert np.allclose(a, b) and abs(a[0] - 0.003) < 1e-4
    assert np.allclose(residual(fit_centered('mconst', units), units[0]),
                       residual(fit_centered('mconst', shifted), shifted[0]))


def test_position_map_recovers_a_vertical_slope_gradient():
    rng = np.random.default_rng(2)
    law = lambda r, c: 0.003 - 0.001 * r            # stronger at the top (row = -1)
    units = [_unit(rng, 50.0, law) for _ in range(5)]
    beta = fit_centered('mmap', units)['beta']
    assert np.allclose(position_basis(np.array([-1.0, 1.0]), np.array([0.0, 0.0])) @ beta, [0.004, 0.002], atol=1e-4)


def test_spill_fit_selects_the_planted_sigma_with_positive_gamma():
    rng = np.random.default_rng(3)
    units = [_unit(rng, 0.0, lambda r, c: 0.002 + 0 * r, spill=40.0) for _ in range(5)]
    fit = fit_model('mspill', units)
    assert fit['sigma'] == SIGMAS[3] and not fit['at_grid_edge'] and abs(fit['beta'][1] - 40.0) < 1.0
