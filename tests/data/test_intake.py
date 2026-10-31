"""Exercise packaged RAW intake through the public CLI using temporary files."""
import json
from pathlib import Path
import numpy as np
import pydicom
import pytest

from xray_ghost_removal.cli import main
from xray_ghost_removal.common.exposure import workbook_record
from xray_ghost_removal.common.provenance import sha256


def test_raw_intake_cli_and_malformed_size(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    np.full((16, 16), 55000, dtype='<u2').tofile(raw/'20261029-140000-000-00000001.raw')
    np.full((16, 16), 1200, dtype='<u2').tofile(raw/'20261029-140002-000-00000002.raw')
    output = tmp_path / 'output'
    args = ['data', 'inspect-raw', '--data-dir', str(raw), '--output-dir', str(output),
            '--shape', '16', '16', '--dtype', '<u2']
    main(args)
    report = json.loads((output/'inventory.json').read_text())
    assert report['total_files'] == report['unique_sha256'] == 2
    assert report['assumed_layout']['confirmed'] is False
    assert report['groups'][0]['candidate_pairs'][0]['filename_gap_seconds'] == 2
    assert (output/'overview.png').is_file()
    (raw/'20261029-140004-000-00000003.raw').write_bytes(b'bad')
    before = (output/'inventory.json').read_bytes()
    with pytest.raises(ValueError, match='byte count'):
        main(args)
    assert (output/'inventory.json').read_bytes() == before


def test_workbook_record_uses_project_dependencies(tmp_path):
    from openpyxl import Workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet['A1'] = 'exposure record'
    for index in range(1, 32):
        for col, value in enumerate([index, 70, 250, 200], 1):
            sheet.cell(index+3, col, value)
    path = tmp_path / 'exposures.xlsx'
    workbook.save(path)
    record = workbook_record(path)
    assert len(record['rows']) == 31
    assert record['rows'][0] == dict(index=1, kv=70, ma=250, ms=200, mas=50, row=4)
    assert record['note'] == 'exposure record'
    assert len(record['sha256']) == 64


@pytest.mark.parametrize('dtype', ['<u2', '>u2'])
def test_raw_dicom_batch_preserves_pixels_order_and_unknown_metadata(tmp_path, dtype):
    raw = tmp_path / 'raw'
    expected = {}
    for group, values in [('第一组', [1100, 55000, 1600]), ('第二组', [55000, 1700])]:
        directory = raw / group
        directory.mkdir(parents=True)
        for index, value in enumerate(values, 1):
            pixels = np.full((16, 16), value, dtype=dtype)
            # Exercise unsigned endpoints and byte order, including values above 32767.
            pixels[0, :3] = [0, 32768, 65535]
            name = f'20261029-14000{index}-000-{index:08X}.raw'
            path = directory / name
            pixels.tofile(path)
            expected[path.relative_to(raw).as_posix()] = (pixels.copy(), sha256(path))
    intake = tmp_path / 'intake'
    main(['data', 'inspect-raw', '--data-dir', str(raw), '--output-dir', str(intake),
          '--shape', '16', '16', '--dtype', dtype])
    inventory_bytes = (intake / 'inventory.json').read_bytes()
    output = intake / 'dicom'
    args = ['data', 'export-raw-dicom', '--inventory', str(intake / 'inventory.json'),
            '--output-dir', str(output)]
    main(args)
    report_bytes = (output / 'manifest.json').read_bytes()
    report = json.loads(report_bytes)
    assert (report['total_files'], report['total_studies'], report['total_series']) == (5, 2, 4)
    assert report['pixel_round_trip_equal'] is True
    assert report['assumed_layout']['confirmed'] is False
    studies, series, instances = set(), set(), set()
    windows = {}
    for row in report['files']:
        pixels, digest = expected[row['source_file']]
        path = output / row['dicom_file']
        ds = pydicom.dcmread(path)
        np.testing.assert_array_equal(ds.pixel_array, pixels)
        assert sha256(raw / row['source_file']) == digest == row['source_sha256']
        assert sha256(path) == row['dicom_sha256']
        assert ds.PixelRepresentation == 0 and ds.BitsStored == 16
        assert row['source_file'] in ds.ImageComments
        assert int(ds.InstanceNumber) == int(Path(row['source_file']).stem.rsplit('-', 1)[1], 16)
        assert ds.PatientID == 'XRAY-RAW-REVIEW'
        assert ds.StudyDate == ds.StudyTime == ''
        for key in ('AcquisitionDate', 'AcquisitionTime', 'PixelSpacing', 'KVP', 'ExposureTime'):
            assert key not in ds
        studies.add(str(ds.StudyInstanceUID))
        series.add(str(ds.SeriesInstanceUID))
        instances.add(str(ds.SOPInstanceUID))
        window = (float(ds.WindowCenter), float(ds.WindowWidth))
        assert windows.setdefault(str(ds.SeriesInstanceUID), window) == window
    assert (len(studies), len(series), len(instances)) == (2, 4, 5)
    assert (intake / 'inventory.json').read_bytes() == inventory_bytes
    # A changed input must not replace an already verified review package.
    next(raw.rglob('*.raw')).write_bytes(np.zeros((16, 16), dtype=dtype).tobytes())
    with pytest.raises(ValueError, match='hash differs'):
        main(args)
    assert (output / 'manifest.json').read_bytes() == report_bytes


def test_raw_dicom_export_protects_raw_and_unmanaged_output(tmp_path):
    from xray_ghost_removal.data.raw_dicom import export_batch
    raw = tmp_path / 'raw'
    raw.mkdir()
    np.full((16, 16), 1200, dtype='<u2').tofile(raw / '20261029-140000-000-00000001.raw')
    intake = tmp_path / 'intake'
    main(['data', 'inspect-raw', '--data-dir', str(raw), '--output-dir', str(intake),
          '--shape', '16', '16', '--dtype', '<u2'])
    inventory = intake / 'inventory.json'
    for destination in (raw, raw / 'dicom', tmp_path):
        with pytest.raises(ValueError, match='must be separate'):
            export_batch(inventory, destination)
    unmanaged = tmp_path / 'unmanaged'
    unmanaged.mkdir()
    (unmanaged / 'keep.txt').write_text('keep')
    with pytest.raises(ValueError, match='not a managed'):
        export_batch(inventory, unmanaged)
    assert (unmanaged / 'keep.txt').read_text() == 'keep'
