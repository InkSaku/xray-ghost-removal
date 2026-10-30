"""Smoke tests. Verify the install and the core algorithms without any DICOM data.

These build a tiny synthetic scene with a known ghost and check that each remover
runs, returns the right shapes, and reduces the ghost it was given. They are not
accuracy tests. Real validation needs paired ground-truth data, see DATA.md.

Run: python -m pytest tests/ -v
"""


import numpy as np
import pytest


from xray_ghost_removal.legacy.linear import estimate_alphas, remove_ghost_gated
from xray_ghost_removal.common.segmentation import detect_air_mask, detect_object_mask
from xray_ghost_removal.legacy.physics import remove_ghost_inpainting
from xray_ghost_removal.legacy.seamless import remove_ghost_iterative, remove_ghost_seamless

AIR_LEVEL = 30000.0     # CR air is bright, MONOCHROME2
OBJECT_LEVEL = 8000.0
NOISE_SIGMA = 200.0
ALPHA_TRUE = 0.01
SIZE = 512


def _scene(object_box, seed=0):
    """Bright air with one dark rectangle, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    img = np.full((SIZE, SIZE), AIR_LEVEL, dtype=np.float32)
    r0, r1, c0, c1 = object_box
    img[r0:r1, c0:c1] = OBJECT_LEVEL
    return img + rng.normal(0, NOISE_SIGMA, img.shape).astype(np.float32)


@pytest.fixture
def ghost_pair():
    """(current_with_ghost, previous, clean_current). Ghost sources do not overlap."""
    previous = _scene((60, 200, 60, 200), seed=1)
    clean = _scene((300, 440, 300, 440), seed=2)
    bg_prev = float(np.percentile(previous, 75))
    ghosted = clean + ALPHA_TRUE * (previous - bg_prev)
    return ghosted.astype(np.float32), previous, clean


def test_air_and_object_masks():
    """Neither object growth nor air erosion changes the threshold boundary."""
    box = (100, 300, 100, 300)
    img = _scene(box)
    air = detect_air_mask(img)
    obj = detect_object_mask(img)
    assert air.shape == img.shape and air.dtype == bool
    assert air.mean() > 0.5, "most of the frame is air"
    assert obj.mean() > 0.05, "the object must be detected"

    r0, r1, c0, c1 = box
    assert obj[r0 + 20:r1 - 20, c0 + 20:c1 - 20].all(), "the object core must be object"
    assert not air[r0 + 20:r1 - 20, c0 + 20:c1 - 20].any(), "the object core is not air"
    assert not (air & obj).any(), "air and object must stay disjoint"


def test_object_mask_preserves_thin_features_and_holes_without_expansion():
    expected = np.zeros((64, 64), dtype=bool)
    expected[10:40, 10:40] = True
    expected[20:30, 20:30] = False
    expected[45, 5:55] = True
    expected[0, 60] = True
    image = np.where(expected, OBJECT_LEVEL, AIR_LEVEL)
    assert np.array_equal(detect_object_mask(image), expected)
    assert np.array_equal(detect_air_mask(image), ~expected)


def test_estimate_alphas_recovers_the_planted_coefficient(ghost_pair):
    ghosted, previous, _ = ghost_pair
    alphas, _, r2, air_fraction = estimate_alphas(ghosted, [previous], ds_factor=1)
    assert len(alphas) == 1
    assert abs(alphas[0] - ALPHA_TRUE) < 0.004, f"alpha={alphas[0]:.4f}, planted {ALPHA_TRUE}"
    assert 0.0 <= r2 <= 1.0
    assert air_fraction > 0.5


def test_gated_removal_reduces_the_ghost(ghost_pair):
    ghosted, previous, clean = ghost_pair
    res = remove_ghost_gated(ghosted, [previous], r2_threshold=0.0, ds_factor=1)
    assert res.cleaned.shape == ghosted.shape
    assert res.applied, f"correction was skipped, reason {res.reason}"
    ghost_zone = detect_air_mask(clean) & detect_object_mask(previous)
    before = float(np.mean(np.abs(ghosted[ghost_zone] - clean[ghost_zone])))
    after = float(np.mean(np.abs(res.cleaned[ghost_zone] - clean[ghost_zone])))
    assert after < before, f"error grew, {before:.1f} to {after:.1f}"


def test_gate_skips_when_there_is_no_air():
    """A frame filled by the object gives a degenerate fit and must be left alone."""
    solid = np.full((SIZE, SIZE), OBJECT_LEVEL, dtype=np.float32)
    previous = _scene((60, 200, 60, 200))
    res = remove_ghost_gated(solid, [previous], ds_factor=1)
    assert not res.applied
    assert np.array_equal(res.cleaned, solid), "a skipped image must be returned untouched"


def test_seamless_removal_preserves_noise_texture(ghost_pair):
    ghosted, previous, clean = ghost_pair
    res = remove_ghost_seamless(ghosted, previous, ds_factor=2, diffusion_iterations=50)
    assert res.cleaned.shape == ghosted.shape
    if res.applied:
        air = detect_air_mask(clean)
        assert abs(float(np.std(res.cleaned[air])) - NOISE_SIGMA) < NOISE_SIGMA, \
            "the correction must not flatten the noise"


def test_iterative_removal_runs_over_several_previous_images(ghost_pair):
    ghosted, previous, _ = ghost_pair
    second = _scene((60, 200, 300, 440), seed=3)
    res = remove_ghost_iterative(ghosted, [previous, second], ds_factor=2,
                                 diffusion_iterations=50)
    assert res.cleaned.shape == ghosted.shape
    assert isinstance(res.reason, str) and res.reason


def test_inpainting_baseline_runs(ghost_pair):
    ghosted, previous, _ = ghost_pair
    res = remove_ghost_inpainting(ghosted, [previous])
    assert res.cleaned.shape == ghosted.shape
    assert np.isfinite(res.cleaned).all()


def test_dicom_utils_import():
    from xray_ghost_removal.common import dicom as dicom_utils
    for name in ("load_dicom", "save_dicom", "load_sequence", "normalize_to_float"):
        assert hasattr(dicom_utils, name)


def test_unet_forward_shape():
    torch = pytest.importorskip("torch")
    from xray_ghost_removal.legacy.unet import GhostUNet
    model = GhostUNet(in_channels=2, base_filters=8)
    cleaned, ghost = model(torch.zeros(1, 2, 128, 128))
    assert cleaned.shape == (1, 1, 128, 128)
    assert ghost.shape == (1, 1, 128, 128)
