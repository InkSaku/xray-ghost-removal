"""Synthetic ghost pair generation for training.

Creates (ghosted, clean, previous, ghost_contribution) samples.
The ghost_contribution is the KNOWN signal to subtract, providing
direct supervision for the ghost predictor.
"""

import random
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter


def load_all_images(data_dir: str | Path) -> list[np.ndarray]:
    """Load all DICOM images as float32 arrays."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.dcm"), key=lambda p: int(p.stem))
    return [pydicom.dcmread(str(f)).pixel_array.astype(np.float32) for f in files]


def extract_random_patch(
    image: np.ndarray, patch_size: int = 256,
) -> tuple[np.ndarray, int, int]:
    h, w = image.shape
    y = random.randint(0, h - patch_size)
    x = random.randint(0, w - patch_size)
    return image[y:y + patch_size, x:x + patch_size], y, x


def synthesize_ghost_pair(
    clean: np.ndarray,
    ghost_source: np.ndarray,
    alpha_range: tuple[float, float] = (0.005, 0.15),
    blur_sigma_range: tuple[float, float] = (0.0, 3.0),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a ghosted image and return the exact ghost contribution.

    Returns (ghosted, ghost_contribution, alpha).
    ghost_contribution is the exact signal that was added, so the
    network can be trained to predict it directly.
    """
    alpha = random.uniform(*alpha_range)
    blur_sigma = random.uniform(*blur_sigma_range)

    ghost_pattern = ghost_source.copy()
    if blur_sigma > 0.1:
        ghost_pattern = gaussian_filter(ghost_pattern, sigma=blur_sigma)

    bg_level = np.percentile(ghost_pattern, 75)
    ghost_contribution = alpha * (ghost_pattern - bg_level)

    ghosted = clean + ghost_contribution
    ghosted = np.clip(ghosted, 0, 65535)

    # Actual contribution after clipping
    actual_contribution = ghosted - clean

    return ghosted, actual_contribution, alpha


def _augment_patch(patch: np.ndarray) -> np.ndarray:
    if random.random() > 0.5:
        patch = np.flip(patch, axis=0).copy()
    if random.random() > 0.5:
        patch = np.flip(patch, axis=1).copy()
    k = random.randint(0, 3)
    if k > 0:
        patch = np.rot90(patch, k).copy()
    return patch


class GhostPatchDataset:
    """Generates synthetic ghost patches with direct ghost supervision.

    Each sample: (input_2ch, ghost_target_1ch)
    - input_2ch: [ghosted_patch, previous_patch] normalized to [0,1]
    - ghost_target_1ch: the ghost contribution (scaled to match normalized space)

    The model learns to predict the ghost pattern DIRECTLY,
    not to reconstruct the full clean image. This gives a much
    stronger learning signal since the ghost is only ~0.5-1% of
    the image range.
    """

    def __init__(
        self,
        images: list[np.ndarray],
        patch_size: int = 256,
        samples_per_epoch: int = 4000,
        alpha_range: tuple[float, float] = (0.005, 0.15),
        augment: bool = True,
    ):
        self.images = images
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.alpha_range = alpha_range
        self.augment = augment
        self._compute_normalization()

    def _compute_normalization(self):
        all_vals = np.concatenate([img.ravel()[::100] for img in self.images])
        self.global_min = float(np.percentile(all_vals, 0.5))
        self.global_max = float(np.percentile(all_vals, 99.5))
        self.value_range = max(self.global_max - self.global_min, 1.0)

    def normalize(self, patch: np.ndarray) -> np.ndarray:
        return (patch - self.global_min) / self.value_range

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        import torch

        n = len(self.images)
        clean_idx = random.randint(0, n - 1)
        ghost_idx = random.randint(0, n - 1)
        while ghost_idx == clean_idx:
            ghost_idx = random.randint(0, n - 1)

        clean_img = self.images[clean_idx]
        ghost_src = self.images[ghost_idx]

        patch_clean, y, x = extract_random_patch(clean_img, self.patch_size)
        patch_ghost_src = ghost_src[y:y + self.patch_size, x:x + self.patch_size]

        if self.augment:
            seed = random.randint(0, 2**31)
            random.seed(seed)
            patch_clean = _augment_patch(patch_clean)
            random.seed(seed)
            patch_ghost_src = _augment_patch(patch_ghost_src)

        patch_ghosted, ghost_contrib, _ = synthesize_ghost_pair(
            patch_clean, patch_ghost_src, self.alpha_range,
        )

        inp = np.stack([
            self.normalize(patch_ghosted),
            self.normalize(patch_ghost_src),
        ], axis=0).astype(np.float32)

        # Ghost target in normalized space (same scale as input)
        ghost_target = (ghost_contrib / self.value_range)[np.newaxis].astype(np.float32)

        return torch.from_numpy(inp), torch.from_numpy(ghost_target)
