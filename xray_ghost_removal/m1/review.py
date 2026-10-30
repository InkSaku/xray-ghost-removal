"""Export frozen M1 OOF corrections as full-size research DICOMs.

Each frozen block prediction is subtracted from its corresponding 16x16 pixels.
No refitting, interpolation, background subtraction, or seamless inpainting.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from xray_ghost_removal.common.paths import PROJECT_ROOT
import tempfile
from copy import deepcopy

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ghost-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.uid import generate_uid

ROOT = PROJECT_ROOT
from xray_ghost_removal.common.dicom import save_dicom, display_window
from xray_ghost_removal.common.spatial import block_mean as blocks
from xray_ghost_removal.m1.frozen import load_m1_pair

RAW = ROOT / 'data/raw/AI修残影例图'
OUT = ROOT / 'outputs/dicom_review'


from xray_ghost_removal.common.provenance import sha256 as sha


def apply_frozen_m1(dark, prediction, evaluated, block_size=16):
    """Lift the frozen OOF correction without altering within-block differences."""
    dark = np.asarray(dark, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    evaluated = np.asarray(evaluated, dtype=bool)
    if prediction.shape != evaluated.shape or dark.shape != tuple(n*block_size for n in prediction.shape):
        raise ValueError('Full-resolution geometry must exactly match the frozen block grid')
    if not np.isfinite(prediction[evaluated]).all():
        raise ValueError('Non-finite prediction in evaluated blocks')
    correction = np.repeat(np.repeat(np.where(evaluated, prediction, 0), block_size, axis=0), block_size, axis=1)
    return dark-correction


def m1_preview(light, dark, before, after, maps, output, window):
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    wc, ww = window
    for ax, a, title in zip(axes[0], (before, after, after-before),
                            ('Original dark', 'M1 corrected dark', 'Correction: after - before')):
        if title.startswith('Correction'):
            limit = max(float(np.percentile(np.abs(a), 99.5)), 1)
            ax.imshow(a[::4, ::4], cmap='coolwarm', vmin=-limit, vmax=limit)
        else:
            ax.imshow(display_window(a, wc, ww)[::4, ::4], cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
    valid = maps['evaluated_mask'].astype(bool)
    limit = max(float(np.percentile(np.abs(maps['observable_signal'][valid]), 99.5)), 1)
    for ax, key, title in zip(axes[1], ('observable_signal', 'oof_prediction', 'oof_residual'),
                             ('Frozen Y = dark - background', 'Frozen M1 OOF prediction', 'Frozen residual Y - prediction')):
        im = ax.imshow(np.ma.array(maps[key], mask=~valid), cmap='coolwarm', vmin=-limit, vmax=limit)
        ax.set_title(title)
    fig.colorbar(im, ax=list(axes[1]), fraction=0.025, pad=0.02)
    for ax in axes.ravel():
        ax.axis('off')
    fig.suptitle(f'{light}-light -> {dark}-dark | M1 frozen OOF | shared WC={wc:.2f}, WW={ww:.2f}')
    fig.subplots_adjust(top=0.9, bottom=0.03, left=0.02, right=0.91, hspace=0.12)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    import shutil
    from datetime import datetime, timezone
    manifest_path = ROOT/'outputs/mhinge_cross_pair_v1/analysis_results.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('frozen_background', {}).get('verified'):
        raise ValueError('Unverified frozen background')
    config = manifest['configuration']
    if config['block_sizes'] != [16] or config['background_modes'] != ['masked_hybrid_spline']:
        raise ValueError('Expected frozen block16 masked-hybrid data')
    pairs = ((3,4), (5,6), (27,28))
    study = generate_uid()
    # Build and verify in OS temp space before replacing superseded review files.
    with tempfile.TemporaryDirectory(prefix='m1-dicom-review-') as staging:
        stage = Path(staging)
        records = []
        for light, dark in pairs:
            maps, fit, provenance = load_m1_pair(manifest, light, dark)
            sources = {}
            for index, kind in ((light, 'light'), (dark, 'dark')):
                name = f'{index}-{kind}.dcm'
                path = RAW/name
                expected = manifest['input_provenance']['files'][name]['sha256']
                if sha(path) != expected:
                    raise ValueError(f'Raw DICOM hash mismatch: {name}')
                ds = pydicom.dcmread(path)
                if (ds.PhotometricInterpretation != 'MONOCHROME2' or ds.PixelRepresentation != 0
                    or float(ds.get('RescaleSlope', 1)) != 1 or float(ds.get('RescaleIntercept', 0)) != 0
                    or 'ModalityLUTSequence' in ds or 'VOILUTSequence' in ds or 'PixelPaddingValue' in ds):
                    raise ValueError(f'Unsupported stored pixel semantics: {name}')
                sources[kind] = ds
            before = sources['dark'].pixel_array.astype(np.float64)
            if not np.allclose(blocks(before), maps['raw_dark'], atol=0.002, rtol=0):
                raise ValueError('Raw dark does not match frozen block data')
            source_blocks = blocks(sources['light'].pixel_array.astype(np.float64))
            source_blocks -= np.percentile(source_blocks, 75)
            if not np.allclose(source_blocks, maps['source'], atol=0.002, rtol=0):
                raise ValueError('Raw light does not match frozen centered source')
            proposed = apply_frozen_m1(before, maps['oof_prediction'], maps['evaluated_mask'])
            valid = maps['evaluated_mask'].astype(bool)
            if not np.allclose(blocks(proposed)[valid], (maps['raw_dark']-maps['oof_prediction'])[valid], atol=0.002, rtol=0):
                raise ValueError('Full-size correction does not reproduce frozen block subtraction')
            # Input-only window highlights the low-amplitude ghost in dark images.
            lo, hi = np.percentile(before, [0.5, 99.5])
            window = [float((lo+hi+1)/2), float(max(hi-lo+1, 2))]
            copies = {}
            for kind in ('light', 'dark'):
                ds = deepcopy(sources[kind])
                if kind == 'dark':
                    ds.WindowCenter, ds.WindowWidth = window
                else:
                    a,b = np.percentile(ds.pixel_array, [0.5,99.5])
                    ds.WindowCenter, ds.WindowWidth = float((a+b+1)/2), float(max(b-a+1,2))
                ds.SeriesNumber, ds.InstanceNumber = dark*10+(kind=='dark'), 1
                filename = f'pair_{light}_{dark}_{kind}_original.dcm'
                audit = save_dicom(ds.pixel_array, ds, stage/'dicom'/filename, study_uid=study,
                    series_description=f'{light}->{dark} ORIGINAL {kind.upper()} - research',
                    derivation_description='Pixel-identical copy; source file SHA256 in review_results.json')
                copies[kind] = pydicom.dcmread(stage/'dicom'/filename)
                assert np.array_equal(copies[kind].pixel_array, sources[kind].pixel_array)
            ds = deepcopy(copies['dark'])
            ds.SeriesNumber = dark*10+2
            filename = f'pair_{light}_{dark}_dark_m1.dcm'
            audit = save_dicom(proposed, ds, stage/'dicom'/filename, study_uid=study,
                series_description=f'{light}->{dark} M1 OOF CORRECTED DARK - research',
                derivation_description='Dark minus frozen M1 OOF prediction; constant correction per 16x16 block; BG retained; unevaluated blocks unchanged',
                source_images=[copies['dark'], copies['light']])
            after_ds = pydicom.dcmread(stage/'dicom'/filename)
            after = after_ds.pixel_array.astype(np.float64)
            assert (float(after_ds.WindowCenter), float(after_ds.WindowWidth)) == tuple(window)
            quantization_target = np.clip(proposed, 0, 2**int(ds.BitsStored)-1)
            assert np.max(np.abs(after-quantization_target)) <= 0.5+1e-8
            m1_preview(light, dark, before, after, maps, stage/f'pair_{light}_{dark}_comparison.png', window)
            records.append(dict(pair=f'{light}->{dark}', frozen_maps=provenance,
                inputs={f'{i}-{k}.dcm':sha(RAW/f'{i}-{k}.dcm') for i,k in ((light,'light'),(dark,'dark'))},
                fit=fit, window=window, export=audit, processed_sha256=sha(stage/'dicom'/filename),
                evaluated_blocks=int(valid.sum()), unchanged_blocks=int((~valid).sum()),
                max_quantization_error=float(np.max(np.abs(after-quantization_target))),
                max_storage_clipping_change=float(np.max(np.abs(proposed-quantization_target))),
                correction_range=[float((before-proposed).min()),float((before-proposed).max())]))
            print(f"{light}->{dark}: R2={fit['cv_r2']:.6f}; clipping={audit['clipped_low']}/{audit['clipped_high']}; DICOM verified", flush=True)
        uids = [str(pydicom.dcmread(p,stop_before_pixels=True).SOPInstanceUID) for p in (stage/'dicom').glob('*.dcm')]
        assert len(uids) == len(set(uids)) == 9
        report = dict(generated_at=datetime.now(timezone.utc).isoformat(), method='frozen M1 strict spatial OOF',
            correction='D - nearest-block lift of frozen OOF prediction; background retained; invalid blocks unchanged',
            evidence_scope='Block16 model; full-size export preserves within-block texture and may expose block boundaries; no clean ground truth',
            manifest_sha256=sha(manifest_path), code_sha256=sha(Path(__file__)), pairs=records)
        (stage/'review_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        lines = ['# M1 DICOM 阅片', '', '使用冻结四折空间 OOF 预测。输出为 dark−预测残影，保留 BG。',
                 '每个 16×16 block 使用常数修正，不重新拟合、不插值；无效 block 保持原样。',
                 '原 dark 与处理后 dark 的窗宽窗位相同；全尺寸导出未新增像素级模型验证。', '',
                 '每对包含前序 light 原图、目标 dark 原图、M1 修正 dark，共 9 个 DICOM。', '',
                 '| Pair | OOF R² | 对比图 |', '|---|---:|---|']
        lines += [f'| {r["pair"]} | {r["fit"]["cv_r2"]:.6f} | pair_{r["pair"].replace("->","_")}_comparison.png |' for r in records]
        (stage/'review.md').write_text('\n'.join(lines)+'\n')
        OUT.mkdir(parents=True, exist_ok=True)
        # This stable directory owns only review derivatives, not frozen evidence.
        for path in OUT.iterdir():
            if path.name == 'dicom' and path.is_dir():
                shutil.rmtree(path)
            elif path.is_file() and (path.suffix in ('.png', '.json') or path.name == 'review.md'):
                path.unlink()
        for path in stage.iterdir():
            shutil.move(str(path), OUT/path.name)
    print(f'Results: {OUT}')


if __name__ == '__main__':
    main()
