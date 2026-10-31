"""Medge: edge features count plateau pixels only, and the alternating fit recovers planted attenuation."""
import numpy as np

from xray_ghost_removal.research.raw_cohort import PLATEAU
from xray_ghost_removal.research.raw_edge import LAMBDAS, edge_features, fit_edge, predict_edge
from xray_ghost_removal.research.raw_split import PairData, predict_split


def test_edge_feature_averages_only_plateau_pixels():
    pixels = np.full((32, 32), PLATEAU, dtype='<u2')
    pixels[:, :16] = 1000                       # left half is an object
    e, dist = edge_features(pixels)
    assert np.all(e[LAMBDAS[0]][:, 0] == 0) and np.isnan(dist[:, 0]).all()
    # Right-hand blocks: plateau pixels at distance 1..16 from the object column 15.
    expected = np.mean(np.exp(-np.arange(1, 17) / LAMBDAS[0]))
    assert np.allclose(e[LAMBDAS[0]][:, 1], expected)
    assert np.allclose(dist[:, 1], np.mean(np.arange(1, 17)))


def _pairs(rng, kappa, lam, n=8):
    data, es = [], []
    g = 2000.0 + 800.0 * np.linspace(1, 0, 24)[:, None] * np.ones((24, 24))
    for _ in range(n):
        s = (rng.uniform(size=(24, 24)) > 0.25).astype(float)
        s[rng.uniform(size=(24, 24)) < 0.1] = 0.5
        e = {l: np.where(s > 0, rng.uniform(0, 1, (24, 24)) ** (8.0 / l), 0.0) for l in LAMBDAS}
        xu = np.where(s < 1, rng.uniform(500, 30000, (24, 24)), np.nan)
        y = s * g * (1 - kappa * e[lam]) + (1 - s) * (0.003 * np.nan_to_num(xu) + 60.0)
        data.append(PairData(x=np.zeros((24, 24)), xu=xu, y=y + rng.normal(0, 1, y.shape), s=s))
        es.append(e)
    return data, es


def test_alternating_fit_recovers_planted_kappa_and_lambda():
    rng = np.random.default_rng(4)
    data, es = _pairs(rng, kappa=0.4, lam=LAMBDAS[3])
    fit = fit_edge(data, es)['edge']
    assert fit['lambda'] == LAMBDAS[3] and abs(fit['kappa'] - 0.4) < 0.02 and not fit['at_grid_edge']


def test_zero_kappa_reduces_to_msplit():
    rng = np.random.default_rng(5)
    data, es = _pairs(rng, kappa=0.0, lam=LAMBDAS[2])
    split = {'G': np.full((24, 24), 2500.0), 'a': 0.003, 'b': 60.0}
    edge = {**split, 'kappa': 0.0}
    assert np.allclose(predict_edge(edge, data[0], es[0][LAMBDAS[2]]), predict_split(split, data[0]))


def test_neighbor_feature_weights_decay_by_nearest_unclipped_value():
    pixels = np.full((32, 32), PLATEAU, dtype='<u2')
    pixels[:, :16] = 11000                      # object at 20% of the plateau value
    e, f, _ = edge_features(pixels, neighbor=True)
    assert np.allclose(f[LAMBDAS[0]][:, 1], 0.2 * e[LAMBDAS[0]][:, 1])
    assert np.all(f[LAMBDAS[0]][:, 0] == 0)


def test_two_feature_attenuation_recovers_planted_kappas():
    from xray_ghost_removal.research.raw_edge import fit_attenuation
    from xray_ghost_removal.research.raw_split import fit_split
    rng = np.random.default_rng(6)
    g = 2000.0 + 800.0 * np.linspace(1, 0, 24)[:, None] * np.ones((24, 24))
    data, feats = [], []
    for _ in range(8):
        s = (rng.uniform(size=(24, 24)) > 0.25).astype(float)
        e = np.where(s > 0, rng.uniform(0, 1, (24, 24)), 0.0)
        f = e * rng.uniform(0, 1, (24, 24))
        xu = np.where(s < 1, rng.uniform(500, 30000, (24, 24)), np.nan)
        y = s * g * (1 - 0.2 * e - 0.5 * f) + (1 - s) * (0.003 * np.nan_to_num(xu) + 60.0)
        data.append(PairData(x=np.zeros((24, 24)), xu=xu, y=y + rng.normal(0, 1, y.shape), s=s))
        feats.append([e, f])
    fit = fit_attenuation(data, feats, fit_split(data))
    assert fit['converged'] and np.allclose(fit['kappas'], [0.2, 0.5], atol=0.02)
