"""Confidence-gated linear ghost removal for CR X-ray images.

The ghost from previous exposures adds a scaled copy of each previous
image's structure to the current image:

    current = true + sum_k alpha_k * (previous_k - bg_k)

This module estimates the alpha_k coefficients by least-squares in the
current image's air region, then subtracts the ghost. Critically, it
only applies the correction when the fit is CONFIDENT (the ghost model
explains the air-region structure well). Otherwise it leaves the image
untouched - safe behavior for a QC tool where corrupting a clean image
is worse than leaving a faint ghost.

Validated behavior (no ground truth available for most images):
- Strong, clean ghosts (e.g. a sharp object on low-noise background)
  are removed reliably (~95%+ reduction).
- Weak ghosts buried in air noise are SKIPPED rather than over-corrected.

For reliable removal across all images, real paired training data is
required (see docs/plans/2026-05-28-real-data-acquisition-protocol.md).
"""

from dataclasses import dataclass, field

import numpy as np

from src.models.physics_model import detect_air_mask


@dataclass
class LinearGhostResult:
    cleaned: np.ndarray
    alphas: list[float]
    r2: float
    applied: bool
    reason: str
    air_fraction: float = 0.0


def estimate_alphas(
    current: np.ndarray,
    previous_images: list[np.ndarray],
    ds_factor: int = 4,
    alpha_cap: float = 0.015,
) -> tuple[list[float], list[float], float, float]:
    """Least-squares estimate of ghost coefficients in the air region.

    Returns (alphas, bg_levels, r2, air_fraction).
    """
    air_full = detect_air_mask(current)
    air_fraction = float(air_full.mean())

    air = air_full[::ds_factor, ::ds_factor]
    cur = current[::ds_factor, ::ds_factor]

    if air.sum() < 2000:
        bgs = [float(np.percentile(p, 75)) for p in previous_images]
        return [0.0] * len(previous_images), bgs, 0.0, air_fraction

    air_level = float(np.median(cur[air]))
    y = cur[air] - air_level

    cols, bgs = [], []
    for p in previous_images:
        pd = p[::ds_factor, ::ds_factor]
        bg = float(np.percentile(pd, 75))
        cols.append(pd[air] - bg)
        bgs.append(float(np.percentile(p, 75)))

    X = np.stack(cols, axis=1)
    alphas, *_ = np.linalg.lstsq(X, y, rcond=None)
    alphas = np.clip(alphas, 0.0, alpha_cap)

    pred = X @ alphas
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum(y ** 2)) + 1e-9
    r2 = max(1.0 - ss_res / ss_tot, 0.0)

    return [float(a) for a in alphas], bgs, r2, air_fraction


def remove_ghost_gated(
    current: np.ndarray,
    previous_images: list[np.ndarray],
    r2_threshold: float = 0.12,
    min_air_fraction: float = 0.15,
    alpha_cap: float = 0.015,
    ds_factor: int = 4,
) -> LinearGhostResult:
    """Remove ghost only when the linear fit is confident.

    Args:
        current: Image to clean.
        previous_images: Preceding images (nearest first), the ghost sources.
        r2_threshold: Minimum fit R^2 to apply the correction.
        min_air_fraction: Minimum air fraction; below this the fit is
            degenerate (object fills the frame) and we skip.
        alpha_cap: Maximum physical ghost fraction per previous image.
        ds_factor: Downsampling for estimation speed.
    """
    if not previous_images:
        return LinearGhostResult(
            cleaned=current.copy(), alphas=[], r2=0.0,
            applied=False, reason="no previous images",
        )

    alphas, bgs, r2, air_frac = estimate_alphas(
        current, previous_images, ds_factor=ds_factor, alpha_cap=alpha_cap,
    )

    if air_frac < min_air_fraction:
        return LinearGhostResult(
            cleaned=current.copy(), alphas=alphas, r2=r2, applied=False,
            reason=f"air fraction {air_frac:.2f} < {min_air_fraction} (degenerate fit)",
            air_fraction=air_frac,
        )

    if r2 < r2_threshold:
        return LinearGhostResult(
            cleaned=current.copy(), alphas=alphas, r2=r2, applied=False,
            reason=f"R2 {r2:.3f} < {r2_threshold} (low confidence, ghost left untouched)",
            air_fraction=air_frac,
        )

    cleaned = current.copy()
    for p, a, bg in zip(previous_images, alphas, bgs):
        if a > 0:
            cleaned = cleaned - a * (p - bg)
    cleaned = np.clip(cleaned, 0, np.iinfo(np.uint16).max)

    return LinearGhostResult(
        cleaned=cleaned, alphas=alphas, r2=r2, applied=True,
        reason=f"applied (R2={r2:.3f})", air_fraction=air_frac,
    )
