"""Check that target calibration and scored blocks remain spatially separate."""

from dataclasses import replace

import numpy as np
from scipy.ndimage import binary_dilation

from scripts.analyze_frozen_candidates import PairData
from scripts.analyze_pair_transfer import (
    BUFFER_BLOCKS,
    TILE_COUNTS,
    calibration_tiles,
    masks_for_budget,
)


def test_calibration_selection_uses_source_only_and_buffers_test():
    rr, cc = np.indices((192, 192))
    tile_row, tile_col = rr // 24, cc // 24
    folds = (2 * (tile_row % 2) + (tile_col // 2) % 2).astype(np.int8)
    support = (tile_col >= 2) & (tile_col <= 5)
    pair = PairData(
        "synthetic", -support.astype(float), np.zeros((192, 192)), folds,
        np.ones((192, 192), dtype=bool), support,
        np.zeros((192, 192)), np.zeros((192, 192)),
    )
    changed_target_y = replace(pair, y=np.full((192, 192), 999.0))
    for fold in range(4):
        _, order = calibration_tiles(pair, fold)
        _, changed_order = calibration_tiles(changed_target_y, fold)
        assert order == changed_order
        previous_test = None
        for count in TILE_COUNTS:
            calibration, test, selected = masks_for_budget(pair, fold, count)
            assert len(selected) == count
            assert calibration.sum() == count * 24 * 24
            assert not np.any(calibration & test)
            if previous_test is not None:
                assert np.array_equal(test, previous_test)
            previous_test = test
            for distance in range(1, BUFFER_BLOCKS + 1):
                assert not np.any(binary_dilation(calibration, iterations=distance) & test)
