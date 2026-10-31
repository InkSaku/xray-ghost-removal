"""RAW cohort: pairing stays inside directories and plateau statistics count exact pixels."""
from datetime import datetime, timedelta

import numpy as np

from xray_ghost_removal.research.raw_cohort import (
    CLEAN_DARK_COUNTERS, PLATEAU, Frame, load_frames, pair_frames, training_indices,
)


def _frame(directory, counter, high, seconds):
    fraction = np.ones((4, 4)) if high else np.zeros((4, 4))
    stamp = datetime(2026, 10, 29, 14) + timedelta(seconds=seconds)
    return Frame(f'{directory}/{stamp:%Y%m%d-%H%M%S-%f}-{counter:08X}.raw', directory,
                 stamp, counter, '', np.zeros((4, 4)), fraction)


def test_pairs_are_adjacent_within_directory_and_flags_use_counters_and_time():
    first = CLEAN_DARK_COUNTERS[0]
    frames = [
        _frame('a', first, False, 0),          # clean dark, never a target
        _frame('a', first + 2, True, 10),      # counter gap before the high frame
        _frame('a', first + 3, False, 12),
        _frame('a', first + 4, True, 40),
        _frame('a', first + 5, False, 70),     # long filename gap
        _frame('b', 0x30, True, 100),          # high at the end of another directory
        _frame('c', 0x31, False, 101),         # low in a different directory
    ]
    pairs = pair_frames(frames)
    assert [(p.high.counter, p.low.counter) for p in pairs] == [(first + 2, first + 3), (first + 4, first + 5)]
    assert pairs[0].flags == ['preceded_by_counter_gap']
    assert pairs[1].flags == ['long_filename_gap']


def test_plateau_fraction_counts_exact_pixels_without_growth(tmp_path):
    image = np.full((32, 32), 1000, dtype='<u2')
    image[0, 0] = PLATEAU                      # one pixel in block (0, 0)
    image[16:32, 16:32] = PLATEAU              # one full block
    folder = tmp_path / 'd'
    folder.mkdir()
    image.tofile(folder / '20261029-140000-000-20000001.raw')
    frame, = load_frames(tmp_path, (32, 32), '<u2')
    assert frame.plateau_fraction.tolist() == [[1 / 256, 0.0], [0.0, 1.0]]
    assert frame.counter == 0x20000001 and frame.directory == 'd'
    # Unclipped mean ignores plateau pixels, so a partly clipped block is not pulled toward 55000.
    assert frame.unclipped_mean[0, 0] == 1000.0 and frame.unclipped_mean[0, 1] == 1000.0
    assert np.isnan(frame.unclipped_mean[1, 1])


def test_training_indices_never_include_the_held_out_pair():
    groups = ['a', 'a', 'a', 'b', 'b']
    for i in range(len(groups)):
        within = training_indices(groups, i, 'pair')
        across = training_indices(groups, i, 'group')
        assert i not in within and i not in across
        assert all(groups[j] == groups[i] for j in within)
        assert all(groups[j] != groups[i] for j in across)
