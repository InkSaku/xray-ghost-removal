"""M1 segmentation; extracted without changing numerical definitions."""
from __future__ import annotations
from typing import Any
from pathlib import Path
from xray_ghost_removal.common.segmentation import detect_object_mask
from scipy.ndimage import find_objects
from scipy.ndimage import label
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pydicom
from xray_ghost_removal.common.spatial import block_mean

def block_source_masks(
    source_mask: np.ndarray,
    block: int,
    background_max_fraction: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    fractions = block_mean(source_mask.astype(np.float32), block)
    background_fit = fractions <= background_max_fraction
    source_support = fractions > background_max_fraction
    return background_fit, source_support


def load_source_mask(
    light_index: int,
    light: np.ndarray,
    mask_archive: Any | None,
    dilation_pixels: int = 0,
    allow_missing_archive_key: bool = False,
) -> tuple[np.ndarray, str]:
    """Load an external or threshold mask unchanged; expansion is prohibited."""
    if dilation_pixels != 0:
        raise ValueError("Mask dilation is prohibited; dilation_pixels must be 0")
    if mask_archive is None:
        mask = detect_object_mask(light)
        origin = "otsu_fallback"
    else:
        keys = (f"light_{light_index}", str(light_index))
        key = next((candidate for candidate in keys if candidate in mask_archive.files), None)
        if key is None:
            if not allow_missing_archive_key:
                raise KeyError(f"Mask archive has no key light_{light_index!s} or {light_index!s}")
            mask = detect_object_mask(light)
            origin = "otsu_fallback_missing_external"
        else:
            mask = np.asarray(mask_archive[key], dtype=bool)
            if mask.shape != light.shape:
                raise ValueError(
                    f"Mask {key!r} has shape {mask.shape}, expected light shape {light.shape}"
                )
            origin = "external_mask"
    return mask.astype(bool, copy=False), origin


def _sam_rgb(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image[np.isfinite(image)], [0.5, 99.5])
    if high <= low:
        raise ValueError("Cannot window a constant light image for SAM")
    windowed = np.clip((image.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    gray = np.round(windowed * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _rough_component_boxes(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    labels, count = label(mask)
    minimum_area = max(256, int(mask.size * 0.0002))
    components = []
    for component_index, slices in enumerate(find_objects(labels), start=1):
        if slices is None:
            continue
        component = labels == component_index
        if component.sum() < minimum_area:
            continue
        row_slice, column_slice = slices
        pad = 12
        x0 = max(0, column_slice.start - pad)
        y0 = max(0, row_slice.start - pad)
        x1 = min(mask.shape[1] - 1, column_slice.stop - 1 + pad)
        y1 = min(mask.shape[0] - 1, row_slice.stop - 1 + pad)
        components.append((component, np.array([x0, y0, x1, y1], dtype=np.float32)))
    if not components:
        raise ValueError("Otsu prompt generation found no source-object component")
    return components


def build_sam_predictor(checkpoint: Path, model_type: str, device: str):
    try:
        import torch
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise RuntimeError(
            "SAM mask generation requires torch and the official segment-anything package"
        ) from error
    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type {model_type!r}")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return SamPredictor(model), device


def sam_mask_from_light(light: np.ndarray, predictor: Any) -> np.ndarray:
    """Refine Otsu-derived component boxes with promptable SAM masks."""
    rough = detect_object_mask(light)
    predictor.set_image(_sam_rgb(light))
    combined = np.zeros(light.shape, dtype=bool)
    for component, box in _rough_component_boxes(rough):
        masks, predicted_iou, _ = predictor.predict(box=box, multimask_output=True)
        overlaps = []
        for mask in masks:
            union = np.logical_or(mask, component).sum()
            overlaps.append(float(np.logical_and(mask, component).sum() / max(union, 1)))
        score = 0.5 * np.asarray(predicted_iou, dtype=float) + 0.5 * np.asarray(overlaps)
        combined |= np.asarray(masks[int(np.argmax(score))], dtype=bool)
    return combined


def render_mask_contact_sheet(
    output: Path,
    previews: list[tuple[int, np.ndarray, np.ndarray]],
) -> None:
    columns = 4
    rows = int(np.ceil(len(previews) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis, (index, image, mask) in zip(axes, previews):
        low, high = np.percentile(image, [0.5, 99.5])
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.contour(mask, levels=[0.5], colors=["#ff3b30"], linewidths=0.7)
        axis.set_title(f"light_{index}")
        axis.set_axis_off()
    for axis in axes[len(previews):]:
        axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_sam_mask_archive(
    data_dir: Path,
    pairs: tuple[tuple[int, int], ...],
    checkpoint: Path,
    model_type: str,
    device: str,
    output_npz: Path,
) -> dict[str, Any]:
    predictor, resolved_device = build_sam_predictor(checkpoint, model_type, device)
    masks = {}
    previews = []
    for light_index in sorted({light_index for light_index, _ in pairs}):
        path = data_dir / f"{light_index}-light.dcm"
        light = pydicom.dcmread(path).pixel_array.astype(np.float32)
        mask = sam_mask_from_light(light, predictor)
        masks[f"light_{light_index}"] = mask
        preview_step = max(1, int(np.ceil(max(light.shape) / 768)))
        previews.append((
            light_index,
            light[::preview_step, ::preview_step],
            mask[::preview_step, ::preview_step],
        ))
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **masks)
    contact_sheet = output_npz.with_name(f"{output_npz.stem}_contact_sheet.png")
    render_mask_contact_sheet(contact_sheet, previews)
    return {
        "mask_archive": str(output_npz.resolve()),
        "contact_sheet": str(contact_sheet.resolve()),
        "mask_count": len(masks),
        "model_type": model_type,
        "device": resolved_device,
        "checkpoint": str(checkpoint.resolve()),
    }
