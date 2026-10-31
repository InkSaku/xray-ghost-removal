"""Response shapes: grid fits recover known curves and flag non-identifiable edge choices."""
import numpy as np

from xray_ghost_removal.research.raw_response import GRIDS, basis, fit_shape, predict_shape
from xray_ghost_removal.research.raw_split import PairData


def test_saturation_and_power_fits_recover_known_shapes():
    x = np.linspace(100.0, 54000.0, 4000)
    tau = GRIDS['msat'][12]
    sat = fit_shape('msat', x, 300.0 * basis('msat', x, tau) + 50.0)
    assert sat['theta'] == tau and not sat['at_grid_edge']
    assert abs(sat['c'] - 300.0) < 1e-6 and abs(sat['b'] - 50.0) < 1e-6
    p = GRIDS['mpow'][30]
    power = fit_shape('mpow', x, 800.0 * basis('mpow', x, p) + 20.0)
    assert power['theta'] == p and not power['at_grid_edge']


def test_linear_truth_is_flagged_at_the_saturation_grid_edge():
    # A straight line is the tau -> infinity limit, so the best tau is the largest grid value.
    x = np.linspace(100.0, 54000.0, 4000)
    fit = fit_shape('msat', x, 0.003 * x + 60.0)
    assert fit['index'] == len(GRIDS['msat']) - 1 and fit['at_grid_edge']


def test_prediction_mixes_map_and_curve_by_plateau_fraction():
    d = PairData(x=np.zeros((1, 3)), xu=np.array([[np.nan, 20000.0, 20000.0]]),
                 y=np.zeros((1, 3)), s=np.array([[1.0, 0.5, 0.0]]))
    split = {'G': np.full((1, 3), 3000.0)}
    fit = {'theta': 1.0, 'c': 100.0, 'b': 10.0}
    curve = 100.0 * (20000.0 / 55000.0) + 10.0
    assert np.allclose(predict_shape('mpow', fit, split, d), [[3000.0, 1500.0 + 0.5 * curve, curve]])
