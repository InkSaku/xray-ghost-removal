"""Regression checks for research DICOM export and unclipped diagnostics."""
from copy import deepcopy

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from xray_ghost_removal.common.dicom import save_dicom
from xray_ghost_removal.common.dicom import display_window


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


def test_window_standard_boundaries():
    np.testing.assert_allclose(display_window(np.array([0,1,2,3]),2,4), [0,1/3,2/3,1])


def test_m1_lift_preserves_texture_and_unevaluated_blocks():
    from xray_ghost_removal.m1.review import apply_frozen_m1
    dark = np.arange(24, dtype=float).reshape(4, 6) + 100
    prediction = np.array([[3., np.nan, -2.], [1., 4., 5.]])
    valid = np.array([[True, False, True], [True, True, True]])
    after = apply_frozen_m1(dark, prediction, valid, block_size=2)
    np.testing.assert_array_equal(after[:2, 2:4], dark[:2, 2:4])
    np.testing.assert_allclose(after[:2, :2], dark[:2, :2]-3)
    np.testing.assert_allclose(after[:2, 4:], dark[:2, 4:]+2)
    for y in (0,2):
        for x in (0,2,4):
            np.testing.assert_allclose(np.diff(after[y:y+2,x:x+2],axis=1),
                                       np.diff(dark[y:y+2,x:x+2],axis=1))


def test_m1_lift_rejects_geometry_and_valid_nan():
    from xray_ghost_removal.m1.review import apply_frozen_m1
    with pytest.raises(ValueError, match='geometry'):
        apply_frozen_m1(np.zeros((3,4)), np.zeros((2,2)), np.ones((2,2),bool), 2)
    with pytest.raises(ValueError, match='Non-finite'):
        apply_frozen_m1(np.zeros((4,4)), np.full((2,2),np.nan), np.ones((2,2),bool), 2)
