"""DICOM I/O utilities for CR X-ray images."""

from pathlib import Path
from typing import Optional
from copy import deepcopy
from datetime import datetime

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def load_dicom(path: str | Path) -> tuple[np.ndarray, pydicom.Dataset]:
    """Load a DICOM file, returning pixel array as float32 and the dataset."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    return arr, ds


def save_dicom(
    arr: np.ndarray,
    reference_ds: pydicom.Dataset,
    output_path: str | Path,
    *,
    study_uid: str | None = None,
    series_uid: str | None = None,
    series_description: str = "Ghost removal - research review",
    derivation_description: str = "Processed stored pixels; research review only",
    source_images: list[pydicom.Dataset] | None = None,
) -> dict:
    """Write a single-frame monochrome Secondary Capture review derivative.

    Input is in STORED pixel units, not rescaled or windowed units. Geometry,
    grayscale transforms and acquisition information are copied; the source
    dataset is never mutated. Unsupported pixel layouts fail explicitly.
    Returns pre-quantization clipping counts for audit. This does not certify
    the source IOD or restore absent acquisition metadata.
    """
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape != (int(reference_ds.Rows), int(reference_ds.Columns)):
        raise ValueError("Expected unchanged two-dimensional source geometry")
    if not np.isfinite(arr).all():
        raise ValueError("Cannot export non-finite pixels")
    allocated, stored = int(reference_ds.BitsAllocated), int(reference_ds.BitsStored)
    if (allocated not in (8, 16) or not 1 <= stored <= allocated
            or int(reference_ds.PixelRepresentation) != 0
            or int(reference_ds.SamplesPerPixel) != 1
            or int(reference_ds.get("NumberOfFrames", 1)) != 1
            or reference_ds.PhotometricInterpretation not in ("MONOCHROME1", "MONOCHROME2")):
        raise ValueError("Only unsigned 8/16-bit single-frame monochrome review export is supported")
    output_path = Path(output_path)
    source_filename = getattr(reference_ds, "filename", None)
    if source_filename and output_path.resolve() == Path(source_filename).resolve():
        raise ValueError("Refusing to overwrite source DICOM")
    upper = 2**stored - 1
    quantized = np.rint(np.clip(arr, 0, upper)).astype("u1" if allocated == 8 else "<u2")
    ds = deepcopy(reference_ds)
    ds.file_meta = FileMetaDataset()
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = study_uid or ds.get("StudyInstanceUID") or generate_uid()
    ds.SeriesInstanceUID = series_uid or generate_uid()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.ImageType = ["DERIVED", "SECONDARY"]
    ds.ConversionType = "WSD"
    ds.SeriesDescription = series_description
    ds.DerivationDescription = derivation_description
    stamp = datetime.now()
    ds.InstanceCreationDate = stamp.strftime("%Y%m%d")
    ds.InstanceCreationTime = stamp.strftime("%H%M%S.%f")
    ds.DateOfSecondaryCapture = ds.InstanceCreationDate
    ds.TimeOfSecondaryCapture = ds.InstanceCreationTime
    for key in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                "StudyDate", "StudyTime", "ReferringPhysicianName", "StudyID",
                "AccessionNumber", "SeriesNumber", "InstanceNumber", "PatientOrientation"):
        if key not in ds:
            setattr(ds, key, "")
    ds.Modality = ds.get("Modality", "OT")
    # Remove stale storage-layout, signatures and cached extrema information.
    for key in ("NumberOfFrames", "PlanarConfiguration", "SmallestImagePixelValue",
                "LargestImagePixelValue", "DigitalSignaturesSequence", "MACParametersSequence",
                "DataSetTrailingPadding", "NumberOfStudyRelatedSeries",
                "NumberOfStudyRelatedInstances", "NumberOfSeriesRelatedInstances"):
        if key in ds:
            del ds[key]
    if "SourceImageSequence" in ds:
        del ds.SourceImageSequence
    if source_images is not None:
        refs = []
        for source in source_images:
            ref = Dataset()
            ref.ReferencedSOPClassUID = source.SOPClassUID
            ref.ReferencedSOPInstanceUID = source.SOPInstanceUID
            ref.SpatialLocationsPreserved = "YES"
            refs.append(ref)
        if refs:
            ds.SourceImageSequence = refs
    ds.HighBit = stored - 1
    ds.PixelData = quantized.tobytes()
    if len(ds.PixelData) % 2:
        ds.PixelData += b"\0"
    ds["PixelData"].VR = "OB" if allocated == 8 else "OW"
    ds["PixelData"].is_undefined_length = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if int(pydicom.__version__.split(".")[0]) >= 3:
        pydicom.dcmwrite(str(output_path), ds, implicit_vr=False, little_endian=True,
                        enforce_file_format=True)
    else:
        ds.is_implicit_VR, ds.is_little_endian = False, True
        pydicom.dcmwrite(str(output_path), ds, write_like_original=False)
    reread = pydicom.dcmread(str(output_path))
    if not np.array_equal(reread.pixel_array, quantized):
        raise RuntimeError("DICOM pixel round-trip verification failed")
    return {"clipped_low": int(np.count_nonzero(arr < 0)),
            "clipped_high": int(np.count_nonzero(arr > upper)),
            "stored_range": [0, upper], "round_trip_equal": True,
            "sop_instance_uid": str(ds.SOPInstanceUID),
            "series_instance_uid": str(ds.SeriesInstanceUID)}


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


def display_window(a, center, width):
    return np.clip((a-(center-0.5))/(width-1)+0.5, 0, 1)
