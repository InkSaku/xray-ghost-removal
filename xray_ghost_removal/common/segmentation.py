"""Threshold segmentation with unchanged mask boundaries."""
import numpy as np

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
    """Threshold low-valued objects without expanding or filling their mask.

    Keep the threshold boundary at the supplied image's resolution. Fixed-step
    morphology on a block-mean image otherwise expands it by whole blocks.
    """
    threshold = _otsu_threshold(image)
    return image < threshold


def detect_air_mask(image: np.ndarray) -> np.ndarray:
    """Threshold high-valued air without eroding it to grow the exclusion zone."""
    threshold = _otsu_threshold(image)
    return image > threshold
