"""Shared block means, source centering, folds, and spatial nulls."""
from __future__ import annotations
import numpy as np

BLOCK_SIZES = (8, 16, 32, 64)


N_SPATIAL_NULL = 8


RANDOM_SEED = 20260914


def block_mean(image: np.ndarray, block: int = 16) -> np.ndarray:
    h = image.shape[0] // block * block
    w = image.shape[1] // block * block
    return image[:h, :w].reshape(h // block, block, w // block, block).mean((1, 3))


def spatial_folds(shape: tuple[int, int]) -> np.ndarray:
    """Four folds distributed over an 8x8 macro-tile grid."""
    h, w = shape
    rr = np.minimum(np.arange(h) * 8 // h, 7)[:, None]
    cc = np.minimum(np.arange(w) * 8 // w, 7)[None, :]
    return ((rr + 2 * cc) % 4).astype(np.int8)


def source(light: dict[int, np.ndarray], index: int) -> np.ndarray:
    image = light[index]
    return image - np.percentile(image, 75)


def spatial_nulls(x: np.ndarray, sat: np.ndarray, seed: int):
    h, w = x.shape
    shifts = [(h // 3, 0), (0, w // 3), (h // 2, w // 2), (h // 4, w // 2)]
    for i, shift in enumerate(shifts):
        yield np.roll(x, shift, (0, 1)), np.roll(sat, shift, (0, 1)), f"roll_{i+1}"
    rng = np.random.default_rng(seed)
    for i in range(N_SPATIAL_NULL - len(shifts)):
        order = rng.permutation(x.size)
        yield x.ravel()[order].reshape(x.shape), sat.ravel()[order].reshape(x.shape), f"shuffle_{i+1}"
