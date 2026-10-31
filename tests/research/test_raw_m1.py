"""RAW M1: held-out pairs never contribute to their own parameters."""
import numpy as np

from xray_ghost_removal.research.raw_m1 import leave_one_out


def test_leave_one_out_parameters_ignore_the_held_out_target():
    rng = np.random.default_rng(0)
    xs = [rng.normal(size=(24, 24)) for _ in range(6)]
    ys = [0.04 * x + 2.0 + 0.01 * rng.normal(size=x.shape) for x in xs]
    groups = ['a', 'a', 'a', 'b', 'b', 'b']
    for by in ('pair', 'group'):
        before = leave_one_out(xs, ys, groups, by)
        for i in range(len(xs)):
            changed = list(ys)
            changed[i] = rng.normal(scale=1e4, size=ys[i].shape)
            assert leave_one_out(xs, changed, groups, by)[i] == before[i]
        assert all(abs(alpha - 0.04) < 1e-3 for alpha, _ in before)
