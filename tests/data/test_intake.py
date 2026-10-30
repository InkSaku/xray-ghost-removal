"""Exercise packaged RAW intake through the public CLI using temporary files."""
import json
import numpy as np
import pytest

from xray_ghost_removal.cli import main
from xray_ghost_removal.common.exposure import workbook_record


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
