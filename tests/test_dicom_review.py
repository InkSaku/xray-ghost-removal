"""Regression checks for research DICOM export and unclipped diagnostics."""
from copy import deepcopy

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from src.utils.dicom_utils import save_dicom
from scripts.prepare_dicom_review import choose_roi, display_window, select_cohort


def reference(bits=12, allocated=16):
    d = Dataset()
    d.file_meta = FileMetaDataset()
    d.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    d.SOPClassUID = SecondaryCaptureImageStorage
    d.SOPInstanceUID = generate_uid()
    d.StudyInstanceUID = generate_uid()
    d.SeriesInstanceUID = generate_uid()
    d.Rows, d.Columns = 2, 3
    d.BitsAllocated, d.BitsStored, d.HighBit = allocated, bits, bits-1
    d.PixelRepresentation, d.SamplesPerPixel = 0, 1
    d.PhotometricInterpretation = 'MONOCHROME2'
    d.PixelSpacing = [0.2, 0.2]
    d.WindowCenter, d.WindowWidth = 1000, 500
    d.RescaleSlope, d.RescaleIntercept = 2, -10
    d.PixelData = np.zeros((2,3),dtype='uint16').tobytes()
    return d


def test_roundtrip_bits_clipping_provenance_and_no_mutation(tmp_path):
    source = reference()
    original = deepcopy(source)
    a = np.array([[-0.6, 0.4, 1000.7], [4094.7, 4095.2, 6000]])
    path = tmp_path/'derived.dcm'
    stats = save_dicom(a, source, path, source_images=[source])
    d = pydicom.dcmread(path)
    np.testing.assert_array_equal(d.pixel_array, [[0,0,1001],[4095,4095,4095]])
    assert stats['clipped_low'] == 1 and stats['clipped_high'] == 2
    assert source == original
    assert d.SOPInstanceUID != source.SOPInstanceUID
    assert d.SeriesInstanceUID != source.SeriesInstanceUID
    assert d.StudyInstanceUID == source.StudyInstanceUID
    assert d.file_meta.MediaStorageSOPInstanceUID == d.SOPInstanceUID
    assert d.SourceImageSequence[0].ReferencedSOPInstanceUID == source.SOPInstanceUID
    assert d.WindowWidth == source.WindowWidth and d.RescaleSlope == 2
    assert d.PixelSpacing == source.PixelSpacing
    assert d.ImageType == ['DERIVED', 'SECONDARY']


@pytest.mark.parametrize('problem', ['nan','shape','signed','color','multiframe'])
def test_reject_unsupported_pixels(tmp_path, problem):
    d = reference()
    a = np.zeros((2,3))
    if problem == 'nan': a[0,0] = np.nan
    if problem == 'shape': a = a.T
    if problem == 'signed': d.PixelRepresentation = 1
    if problem == 'color': d.SamplesPerPixel = 3
    if problem == 'multiframe': d.NumberOfFrames = 2
    with pytest.raises(ValueError):
        save_dicom(a, d, tmp_path/'bad.dcm')
    assert not (tmp_path/'bad.dcm').exists()


def test_no_source_overwrite(tmp_path):
    d = reference()
    d.filename = str(tmp_path/'raw.dcm')
    with pytest.raises(ValueError, match='overwrite'):
        save_dicom(np.zeros((2,3)), d, d.filename)


def test_odd_length_8bit_pixels(tmp_path):
    d = reference(bits=8, allocated=8)
    d.Rows, d.Columns = 1,3
    save_dicom(np.array([[1,2,255]]), d, tmp_path/'odd.dcm')
    np.testing.assert_array_equal(pydicom.dcmread(tmp_path/'odd.dcm').pixel_array, [[1,2,255]])


def test_roi_excludes_endpoints_and_uses_same_input_tile():
    a = np.ones((128,128))*100
    a[:64,:64] = 65535
    mask = np.ones_like(a, dtype=bool)
    assert choose_roi(a,mask,65535) == [0,64,64]
    assert choose_roi(a,np.zeros_like(mask),65535) is None


def test_window_standard_boundaries():
    np.testing.assert_allclose(display_window(np.array([0,1,2,3]),2,4), [0,1/3,2/3,1])


def test_detect_pileup_below_storage_limit_in_selection():
    rows = [dict(index=i, observed_max_fraction=0, high_endpoint_fraction=0,
                 source_air_correlation_block16=0.5) for i in range(1,31)]
    rows[16]['observed_max_fraction'] = 0.94  # e.g. 55000, not 65535
    rows[8]['source_air_correlation_block16'] = 0.001
    selected = select_cohort(rows)
    assert 17 in selected and 18 in selected and 9 in selected
    assert any('pile-up' in why for why in selected[17])


def test_roi_excludes_empirical_plateau():
    a = np.ones((128,128))*100
    a[:64,:64] = 55000
    assert choose_roi(a,np.ones_like(a,dtype=bool),65535) == [0,64,64]
