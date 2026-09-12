"""Physics-based ghost artifact removal for CR X-ray images.

Uses diffusion inpainting: identifies ghost-contaminated air regions,
then fills them by propagating values from clean (ghost-free) air neighbors.
This avoids the halo artifacts of template subtraction methods.

For CR (Computed Radiography) MONOCHROME2 images:
- Background (air) = HIGH pixel values (high X-ray exposure)
- Objects = LOW pixel values (X-rays absorbed)
- Ghost artifacts appear as faint structure in air regions
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, zoom


@dataclass
class GhostResult:
    cleaned: np.ndarray
    correction: np.ndarray
    ghost_zone_fraction: float
    clean_air_fraction: float


def _otsu_threshold(image: np.ndarray, n_bins: int = 256) -> float:
    """Compute Otsu's threshold for bimodal separation."""
    img_min, img_max = image.min(), image.max()
    if img_max == img_min:
        return float(img_min)

    hist, bin_edges = np.histogram(image.ravel(), bins=n_bins, range=(img_min, img_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    w0 = np.cumsum(hist).astype(np.float64)
    w1 = total - w0
    mu0 = np.cumsum(hist * bin_centers) / np.maximum(w0, 1)
    mu1 = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / np.maximum(w1, 1)

    variance = w0 * w1 * (mu0 - mu1) ** 2
    idx = np.argmax(variance)
    return float(bin_centers[idx])


def detect_object_mask(image: np.ndarray) -> np.ndarray:
    """Detect object regions in a CR image (low pixel values = objects)."""
    threshold = _otsu_threshold(image)
    mask = image < threshold

    struct = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_closing(mask, struct, iterations=5)
    mask = ndimage.binary_opening(mask, struct, iterations=3)
    # Dilate slightly to include object penumbra
    mask = ndimage.binary_dilation(mask, struct, iterations=8)

    return mask


def detect_air_mask(image: np.ndarray) -> np.ndarray:
    """Detect air/background regions in a CR image (high pixel values = air)."""
    threshold = _otsu_threshold(image)
    mask = image > threshold

    struct = ndimage.generate_binary_structure(2, 1)
    mask = ndimage.binary_erosion(mask, struct, iterations=5)

    return mask


def remove_ghost_inpainting(
    current: np.ndarray,
    previous_images: list[np.ndarray],
    ds_factor: int = 4,
    diffusion_iterations: int = 400,
    diffusion_sigma: float = 3.0,
    correction_sigma: float = 8.0,
    taper_sigma: float = 5.0,
) -> GhostResult:
    """Remove ghost artifacts by inpainting contaminated air regions.

    Identifies air regions contaminated by previous image objects,
    then fills them by iterative diffusion from clean air neighbors.

    Args:
        current: Current image with ghost artifacts
        previous_images: List of preceding images (nearest first)
        ds_factor: Downsampling factor for diffusion (speed vs quality)
        diffusion_iterations: Number of diffusion passes
        diffusion_sigma: Gaussian sigma for each diffusion step
        correction_sigma: Smoothing applied to final correction map
        taper_sigma: Sigma for soft mask at air/object boundary
    """
    h, w = current.shape
    air_mask = detect_air_mask(current)

    ghost_mask = np.zeros((h, w), dtype=bool)
    for prev in previous_images:
        ghost_mask |= detect_object_mask(prev)

    ghost_zone = air_mask & ghost_mask
    clean_air = air_mask & ~ghost_mask

    ghost_frac = ghost_zone.sum() / current.size
    clean_frac = clean_air.sum() / current.size

    if ghost_frac < 0.001 or clean_frac < 0.01:
        return GhostResult(
            cleaned=current.copy(),
            correction=np.zeros_like(current),
            ghost_zone_fraction=ghost_frac,
            clean_air_fraction=clean_frac,
        )

    curr_ds = current[::ds_factor, ::ds_factor]
    ghost_ds = ghost_zone[::ds_factor, ::ds_factor]
    clean_ds = clean_air[::ds_factor, ::ds_factor]
    air_ds = air_mask[::ds_factor, ::ds_factor]

    if clean_ds.sum() > 0:
        seed_value = float(np.median(curr_ds[clean_ds]))
    else:
        seed_value = float(np.median(curr_ds[air_ds])) if air_ds.sum() > 0 else float(np.median(curr_ds))

    # Build a "source" image: clean air keeps original values,
    # ghost zone + object region get the seed value.
    # This prevents dark object pixels from leaking into ghost zone.
    source = np.full_like(curr_ds, seed_value)
    source[clean_ds] = curr_ds[clean_ds]

    # Normalized convolution: diffuse from clean air into ghost zone,
    # ignoring non-air pixels. weight=1 for clean air, 0 elsewhere.
    weight = clean_ds.astype(np.float64)

    filled = source.copy()
    for _ in range(diffusion_iterations):
        num = gaussian_filter(filled * weight, sigma=diffusion_sigma)
        den = gaussian_filter(weight, sigma=diffusion_sigma)
        den = np.maximum(den, 1e-10)
        interpolated = num / den
        # Update ghost zone from interpolated; keep clean air original
        filled[ghost_ds] = interpolated[ghost_ds]
        # Mark newly filled ghost pixels as contributing to future iterations
        weight[ghost_ds] = np.clip(
            gaussian_filter(weight, sigma=diffusion_sigma)[ghost_ds], 0, 1,
        )

    expected = zoom(
        filled,
        (h / filled.shape[0], w / filled.shape[1]),
        order=3,
    )[:h, :w]

    raw_correction = expected - current
    correction = gaussian_filter(raw_correction, sigma=correction_sigma)

    taper = gaussian_filter(air_mask.astype(np.float64), sigma=taper_sigma)
    taper = np.clip(taper, 0, 1)

    masked_correction = correction * taper
    cleaned = current + masked_correction
    cleaned = np.clip(cleaned, 0, np.iinfo(np.uint16).max)

    return GhostResult(
        cleaned=cleaned,
        correction=masked_correction,
        ghost_zone_fraction=ghost_frac,
        clean_air_fraction=clean_frac,
    )


def process_sequence(
    images: list[np.ndarray],
    n_previous: int = 3,
    **kwargs,
) -> list[GhostResult]:
    """Process a full sequence of images, removing ghosts from each."""
    results = []
    for i in range(len(images)):
        prev_start = max(0, i - n_previous)
        previous = images[prev_start:i]
        result = remove_ghost_inpainting(images[i], previous, **kwargs)
        results.append(result)
    return results
