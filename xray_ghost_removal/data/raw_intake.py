"""Read-only RAW intake; pixel layout is an explicit, unverified assumption.

No masks, corrections, exposure labels, or physical time estimates are produced.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ghost-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def inspect_batch(data_dir, output_dir, shape, dtype):
    files = sorted(data_dir.rglob('*.raw'))
    if not files:
        raise ValueError('No .raw files found')
    dtype = np.dtype(dtype)
    expected = int(np.prod(shape)) * dtype.itemsize
    for path in files:
        if path.stat().st_size != expected:
            raise ValueError(f'Unexpected byte count: {path}; expected {expected}')
        datetime.strptime(path.stem[:19], '%Y%m%d-%H%M%S-%f')
        int(path.stem.rsplit('-', 1)[1], 16)
    groups = []
    previews = []
    for parent in sorted({p.parent for p in files}):
        rows, tiles = [], []
        for path in (p for p in files if p.parent == parent):
            a = np.memmap(path, mode='r', dtype=dtype, shape=shape)
            sample = a[::8, ::8]
            high = float(np.mean(a == 55000))
            rows.append(dict(
                file=str(path.relative_to(data_dir)), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                filename_timestamp=datetime.strptime(path.stem[:19], '%Y%m%d-%H%M%S-%f').isoformat(),
                filename_counter=int(path.stem.rsplit('-', 1)[1], 16),
                minimum=int(a.min()), maximum=int(a.max()), mean=float(a.mean()),
                sample_quantiles=dict(zip(('p0', 'p1', 'p50', 'p99', 'p100'),
                                          np.percentile(sample, [0, 1, 50, 99, 100]).tolist())),
                fraction_equal_55000=high, fraction_zero=float(np.mean(a == 0)),
                fraction_equal_65535=float(np.mean(a == 65535)),
                observed_class='majority_55000' if high > .5 else 'other',
            ))
            # Means for display only; no segmentation or quantitative mask.
            h, w = a.shape
            tiles.append(np.asarray(a[:h//16*16, :w//16*16]).reshape(h//16, 16, w//16, 16).mean((1, 3)))
        pairs, gaps = [], []
        for i, (a, b) in enumerate(zip(rows, rows[1:])):
            step = b['filename_counter'] - a['filename_counter']
            if step != 1:
                gaps.append(dict(before=a['file'], after=b['file'], counter_step=step))
            if a['observed_class'] == 'majority_55000' and b['observed_class'] == 'other':
                pairs.append(dict(
                    high_frame=a['file'], following_low_frame=b['file'], counter_step=step,
                    filename_gap_seconds=(datetime.fromisoformat(b['filename_timestamp']) -
                                          datetime.fromisoformat(a['filename_timestamp'])).total_seconds(),
                ))
        first = next((i for i, r in enumerate(rows[:-1]) if
                      r['observed_class'] == 'majority_55000' and rows[i+1]['observed_class'] == 'other'), None)
        previews.append([(tiles[i], rows[i]) for i in range(first, min(first+4, len(rows)))] if first is not None else [])
        groups.append(dict(directory=str(parent.relative_to(data_dir)), count=len(rows),
                           observed_classes=dict(Counter(r['observed_class'] for r in rows)),
                           counter_gaps=gaps, candidate_pairs=pairs, files=rows))
    result = dict(
        generated_at=datetime.now(timezone.utc).isoformat(), input_root=str(data_dir.resolve()),
        assumed_layout=dict(shape=list(shape), dtype=dtype.str, header_bytes=0, confirmed=False),
        interpretation_notes=[
            'Layout is caller supplied; consistent sizes and readable structure do not confirm vendor specification.',
            'majority_55000 means >50% of pixels equal 55000; it is not a verified exposure label.',
            '55000 is an observed plateau, not the uint16 maximum; clipping/processing cause is unknown.',
            'Candidate pairs are adjacent exported high/low files, not verified first post-exposure dark frames.',
            'Filename times are not validated acquisition/readout times or physical delay measurements.',
            'Counter gaps do not establish whether unexported detector readouts occurred.',
            'No masks, background models, ghost correction, or physical coefficients were computed.',
        ],
        total_files=len(files), total_bytes=sum(r['bytes'] for g in groups for r in g['files']),
        unique_sha256=len({r['sha256'] for g in groups for r in g['files']}), groups=groups,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(groups), 4, figsize=(12, 3.2*len(groups)), squeeze=False)
    for gi, (axrow, examples) in enumerate(zip(axes, previews)):
        for ax in axrow:
            ax.axis('off')
        for ax, (tile, row) in zip(axrow, examples):
            lo, hi = np.percentile(tile, [1, 99])
            ax.imshow(tile, cmap='gray', vmin=lo, vmax=max(lo+1, hi))
            ax.set_title(f"Group {gi+1} | {row['filename_counter']:08X}\n{row['observed_class']}\nwindow {lo:.0f}:{hi:.0f}", fontsize=9)
    fig.suptitle('RAW intake | unconfirmed layout | individual display windows | no correction')
    fig.tight_layout()
    fig.savefig(output_dir / 'overview.png', dpi=130)
    plt.close(fig)
    (output_dir / 'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/raw_intake'))
    parser.add_argument('--shape', type=int, nargs=2, required=True, metavar=('ROWS', 'COLS'))
    parser.add_argument('--dtype', required=True, choices=('<u2', '>u2'))
    args = parser.parse_args()
    if min(args.shape) < 16:
        parser.error('Both dimensions must be at least 16')
    result = inspect_batch(args.data_dir, args.output_dir, tuple(args.shape), args.dtype)
    print(json.dumps(dict(total_files=result['total_files'], unique_sha256=result['unique_sha256'],
                          groups=[dict(directory=g['directory'], count=g['count'],
                                       observed_classes=g['observed_classes'], candidate_pairs=len(g['candidate_pairs']),
                                       counter_gaps=len(g['counter_gaps'])) for g in result['groups']]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
