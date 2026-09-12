"""DICOM I/O utilities for CR X-ray images."""

from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian


def load_dicom(path: str | Path) -> tuple[np.ndarray, pydicom.Dataset]:
    """Load a DICOM file, returning pixel array as float32 and the dataset."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    return arr, ds


def save_dicom(
    arr: np.ndarray,
    reference_ds: pydicom.Dataset,
    output_path: str | Path,
) -> None:
    """Save array as DICOM, preserving headers from reference dataset."""
    ds = reference_ds.copy()

    arr_clipped = np.clip(arr, 0, np.iinfo(np.uint16).max)
    ds.PixelData = arr_clipped.astype(np.uint16).tobytes()
    ds.Rows, ds.Columns = arr_clipped.shape

    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(output_path))


def load_sequence(
    data_dir: str | Path,
    max_images: Optional[int] = None,
) -> list[tuple[np.ndarray, pydicom.Dataset, str]]:
    """Load all DICOM files in directory, sorted by numeric filename.

    Returns list of (pixel_array, dataset, filename) tuples.
    """
    data_dir = Path(data_dir)
    dcm_files = sorted(
        data_dir.glob("*.dcm"),
        key=lambda p: int(p.stem),
    )
    if max_images is not None:
        dcm_files = dcm_files[:max_images]

    results = []
    for f in dcm_files:
        arr, ds = load_dicom(f)
        results.append((arr, ds, f.name))
    return results


def normalize_to_float(arr: np.ndarray) -> np.ndarray:
    """Normalize uint16 range to [0, 1] float32."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - arr_min) / (arr_max - arr_min)).astype(np.float32)


def denormalize_from_float(
    arr: np.ndarray,
    original_min: float,
    original_max: float,
) -> np.ndarray:
    """Convert [0, 1] float32 back to original value range."""
    return arr * (original_max - original_min) + original_min
