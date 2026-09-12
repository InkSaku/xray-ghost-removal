"""Seamless ghost removal for CR X-ray images.

Goal: make the de-ghosted region undetectable to a human - it must match
the surrounding background in BOTH level and noise texture, with no halo.

Method (per nearest previous image):
  1. Detect the ghost zone = air(current) AND object(previous).
  2. Diffusion-inpaint the clean-air level into the ghost zone, using only
     clean air (air not contaminated by this previous) as the source.
  3. Seamless replace: keep the current image's HIGH-frequency content
     (noise texture) and swap only the LOW-frequency component for the
     clean-air target:
         cleaned = current + lowpass(expected) - lowpass(current)
     Low-pass uses normalized convolution over air only, so dark object
     pixels never bleed across edges (this is what eliminates halos).

Only the nearest previous image is used by default: with multiple large
previous objects their union leaves no clean air to inpaint from.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from src.models.physics_model import detect_air_mask, detect_object_mask


@dataclass
class SeamlessResult:
    cleaned: np.ndarray
    applied: bool
    reason: str
    ghost_zone_fraction: float = 0.0
    clean_air_fraction: float = 0.0


def remove_ghost_seamless(
    current: np.ndarray,
    previous: np.ndarray,
    ds_factor: int = 4,
    diffusion_iterations: int = 300,
    diffusion_sigma: float = 3.0,
    lowfreq_sigma: float = 16.0,
    apply_taper_sigma: float = 4.0,
    min_clean_air: float = 0.02,
    min_ghost_zone: float = 0.002,
) -> SeamlessResult:
    """Remove the ghost of `previous` from `current` seamlessly.

    Returns the cleaned image and whether a correction was applied.
    Saturated or object-filled images (no usable air) are returned unchanged.
    """
    h, w = current.shape
    air = detect_air_mask(current)

    # Saturated background (no variation) -> no ghost can live in the air.
    if air.sum() > 1000 and float(np.std(current[air])) < 1.0:
        return SeamlessResult(current.copy(), False, "saturated air (no ghost signal)")

    obj_prev = detect_object_mask(previous)
    ghost_zone = air & obj_prev
    clean_air = air & ~obj_prev

    # Calibrate alpha from the pure-background ghost zone. alpha is a property
    # of the plate, so the same value applies over the object - letting us
    # SUBTRACT the known ghost where it overlaps the object (inpainting can't).
    bg_level = float(np.percentile(previous, 75))
    alpha = 0.0
    if ghost_zone.sum() > 500 and clean_air.sum() > 500:
        air_lvl = float(np.median(current[clean_air]))
        xb = (previous[ghost_zone] - bg_level)
        yb = (current[ghost_zone] - air_lvl)
        denom = float(np.dot(xb, xb))
        if denom > 1e-6:
            alpha = float(np.clip(np.dot(xb, yb) / denom, 0.0, 0.05))

    gz_frac = ghost_zone.sum() / current.size
    ca_frac = clean_air.sum() / current.size

    if ca_frac < min_clean_air:
        return SeamlessResult(current.copy(), False,
                              f"clean-air fraction {ca_frac:.3f} too low", gz_frac, ca_frac)
    if gz_frac < min_ghost_zone:
        return SeamlessResult(current.copy(), False,
                              f"ghost-zone fraction {gz_frac:.4f} too low", gz_frac, ca_frac)

    # --- 1. Diffusion-inpaint clean-air level into the ghost zone ---
    cur_ds = current[::ds_factor, ::ds_factor]
    gz = ghost_zone[::ds_factor, ::ds_factor]
    ca = clean_air[::ds_factor, ::ds_factor]

    seed = float(np.median(cur_ds[ca]))
    source = np.full_like(cur_ds, seed)
    source[ca] = cur_ds[ca]
    weight = ca.astype(np.float64)
    filled = source.copy()
    for _ in range(diffusion_iterations):
        num = gaussian_filter(filled * weight, diffusion_sigma)
        den = np.maximum(gaussian_filter(weight, diffusion_sigma), 1e-10)
        filled[gz] = (num / den)[gz]
        weight[gz] = np.clip(gaussian_filter(weight, diffusion_sigma)[gz], 0, 1)

    expected = zoom(filled, (h / filled.shape[0], w / filled.shape[1]), order=3)[:h, :w]

    # --- 2. Seamless low-frequency swap (normalized convolution over air) ---
    valid = air.astype(np.float64)
    den = np.maximum(gaussian_filter(valid, lowfreq_sigma), 1e-10)
    cur_lf = gaussian_filter(current * valid, lowfreq_sigma) / den
    exp_lf = gaussian_filter(expected * valid, lowfreq_sigma) / den

    correction = exp_lf - cur_lf

    # SAFETY: only correct FLAT background. Protect any real structure/texture
    # in the current image (the object, its edges, fine detail). A ghost that
    # overlaps the real object cannot be removed without corrupting the object,
    # so we leave those regions untouched.
    local_mean = gaussian_filter(current, 6.0)
    local_var = gaussian_filter(current ** 2, 6.0) - local_mean ** 2
    local_std = np.sqrt(np.maximum(local_var, 0))
    # background noise level = median local_std within clean air
    bg_std = float(np.median(local_std[clean_air])) if clean_air.sum() > 0 else float(np.median(local_std))
    flat = local_std < (bg_std * 3.0 + 1.0)   # structured regions excluded

    apply_region = air & flat
    apply_mask = np.clip(gaussian_filter(apply_region.astype(np.float64), apply_taper_sigma), 0, 1)

    # Flat background: seamless inpainting (preserves noise, no halo).
    # Structured/object regions: subtract the calibrated ghost so overlaps
    # are removed without corrupting the object.
    ghost_est = alpha * (previous - bg_level)
    cleaned = current + correction * apply_mask - ghost_est * (1.0 - apply_mask)
    cleaned = np.clip(cleaned, 0, np.iinfo(np.uint16).max)

    return SeamlessResult(cleaned, True, f"applied (alpha={alpha:.4f})", gz_frac, ca_frac)


def remove_ghost_iterative(
    current: np.ndarray,
    previous_images: list[np.ndarray],
    **kwargs,
) -> SeamlessResult:
    """Remove ghosts from multiple previous exposures, one layer at a time.

    A single inpainting pass can only use one previous image (the union of
    several previous objects leaves no clean air to inpaint from). Applying
    the seamless remover iteratively - nearest previous first, then earlier
    ones on the running result - peels off each ghost layer in turn.

    `previous_images` must be ordered nearest-first.
    """
    work = current.copy()
    any_applied = False
    reasons = []
    last = None
    for k, prev in enumerate(previous_images):
        res = remove_ghost_seamless(work, prev, **kwargs)
        work = res.cleaned
        any_applied = any_applied or res.applied
        reasons.append(f"prev[{k}]:{'ok' if res.applied else res.reason}")
        last = res

    return SeamlessResult(
        cleaned=work,
        applied=any_applied,
        reason="; ".join(reasons),
        ghost_zone_fraction=last.ghost_zone_fraction if last else 0.0,
        clean_air_fraction=last.clean_air_fraction if last else 0.0,
    )
