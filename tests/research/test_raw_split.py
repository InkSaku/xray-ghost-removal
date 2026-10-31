"""Msplit: the map sees only full-plateau training blocks, and boundaries mix unclipped pixels."""
import numpy as np
import pytest

from xray_ghost_removal.research.raw_split import MIN_MAP_SAMPLES, PairData, learn_map, predict_split


def test_map_averages_only_full_plateau_blocks_and_fill_never_alters_observed_values():
    shape = (12, 12)
    ss, ys = [], []
    for k in range(4):
        s = np.ones(shape)
        s[0:3, 0:3] = 0.0                       # always covered by an object: needs filling
        s[5, 5] = 0.99                          # boundary block: must not enter the map
        ss.append(s)
        y = np.full(shape, 100.0 * (k + 1))     # plateau ghost 100, 200, 300, 400
        y[0:3, 0:3] = 1e6                       # values under the object must never leak in
        y[5, 5] = -1e6
        ys.append(y)
    g, count, filled = learn_map(ys, ss)
    assert count[5, 5] == 0 and count[6, 6] == 4
    observed = count >= MIN_MAP_SAMPLES
    assert np.allclose(g[observed], 250.0)
    assert filled[0:3, 0:3].all() and filled[5, 5] and not filled[observed].any()
    assert np.allclose(g[filled], 250.0)        # filled from observed blocks only


def test_map_requires_minimum_samples():
    s = np.ones((6, 6))
    y = np.ones((6, 6))
    with pytest.raises(ValueError):
        learn_map([y] * (MIN_MAP_SAMPLES - 1), [s] * (MIN_MAP_SAMPLES - 1))
    g, count, filled = learn_map([y] * MIN_MAP_SAMPLES, [s] * MIN_MAP_SAMPLES)
    assert count.min() == MIN_MAP_SAMPLES and not filled.any() and np.allclose(g, 1.0)


def test_boundary_block_mixes_map_and_unclipped_pixel_term():
    s = np.array([[1.0, 0.5, 0.0]])
    d = PairData(x=np.array([[54000.0, 37000.0, 20000.0]]),
                 xu=np.array([[np.nan, 20000.0, 20000.0]]),
                 y=np.zeros((1, 3)), s=s)
    params = {'G': np.full((1, 3), 3000.0), 'a': 0.002, 'b': 72.0, 'c': 50.0}
    pixel = 0.002 * 20000 + 72
    assert np.allclose(predict_split(params, d), [[3000.0, 0.5 * 3000 + 0.5 * pixel, pixel]])
    assert np.allclose(predict_split(params, d, use_pixels=False), [[3000.0, 1525.0, 50.0]])
