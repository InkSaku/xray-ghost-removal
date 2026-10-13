"""Export frozen M1 OOF corrections as full-size research DICOMs.

Each frozen block prediction is subtracted from its corresponding 16x16 pixels.
No refitting, interpolation, background subtraction, or seamless inpainting.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from copy import deepcopy

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ghost-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import binary_erosion, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.physics_model import detect_air_mask, detect_object_mask
from src.utils.dicom_utils import save_dicom

RAW = ROOT / 'data/raw/AI修残影例图'
OUT = ROOT / 'outputs/dicom_review'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocks(a, size=16):
    h, w = a.shape
    return a[:h//size*size, :w//size*size].reshape(h//size, size, w//size, size).mean((1, 3))


def correlation(a, b):
    if len(a) < 20 or min(np.std(a), np.std(b)) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def screen_inputs():
    rows = []
    previous = None
    for idx in range(1, 31):
        path = RAW / f'{idx}.dcm'
        ds = pydicom.dcmread(path)
        a = ds.pixel_array.astype(np.float64)
        # This review method assumes stored unsigned MONOCHROME2 pixels.
        if (ds.PhotometricInterpretation != 'MONOCHROME2' or ds.PixelRepresentation != 0
                or float(ds.get('RescaleSlope', 1)) != 1 or float(ds.get('RescaleIntercept', 0)) != 0
                or 'ModalityLUTSequence' in ds or 'VOILUTSequence' in ds
                or 'PixelPaddingValue' in ds):
            raise ValueError(f'Unsupported review grayscale semantics: {path.name}')
        upper = 2**int(ds.BitsStored)-1
        b = blocks(a)
        air = detect_air_mask(b)
        corr = correlation(b[air], previous[air]) if previous is not None else None
        rows.append(dict(index=idx, sha256=sha(path), minimum=float(a.min()), maximum=float(a.max()),
                         low_endpoint_fraction=float(np.mean(a == 0)),
                         high_endpoint_fraction=float(np.mean(a == upper)),
                         observed_max_fraction=float(np.mean(a == a.max())),
                         source_air_correlation_block16=corr))
        previous = b
    return rows


def select_cohort(rows):
    reasons = {1: ['unchanged export control; not a clean-ground-truth control'],
               4: ['historical strong-ghost example'], 6: ['historical difficult example'],
               18: ['historical adverse-metric example']}
    high = max(rows[1:], key=lambda r: r['observed_max_fraction'])
    if high['observed_max_fraction'] > 0.001:
        reasons.setdefault(high['index'], []).append('largest target observed-maximum pile-up (>0.1%); suspected clipping')
    source = max(rows[:-1], key=lambda r: r['observed_max_fraction'])
    if source['observed_max_fraction'] > 0.001:
        reasons.setdefault(source['index']+1, []).append('preceded by largest source observed-maximum pile-up')
    eligible = [r for r in rows[1:] if r['source_air_correlation_block16'] is not None
                and r['observed_max_fraction'] < 0.001]
    if eligible:
        weak = min(eligible, key=lambda r: abs(r['source_air_correlation_block16']))
        reasons.setdefault(weak['index'], []).append('weakest absolute input block-air association; NOT confirmed ghost-free')
    return reasons


def choose_roi(a, clean_air, upper, size=64):
    """Input-only low-variation, unsaturated full clean-air tile; fixed for comparison."""
    best = None
    ceiling = min(upper, float(a.max())) if np.mean(a == a.max()) > 0.001 else upper
    for y in range(0, a.shape[0]-size+1, size):
        for x in range(0, a.shape[1]-size+1, size):
            tile = a[y:y+size, x:x+size]
            if (clean_air[y:y+size, x:x+size].all() and tile.min() > 0 and tile.max() < ceiling):
                score = float(np.std(tile))
                if best is None or score < best[0]:
                    best = (score, y, x)
    return None if best is None else [best[1], best[2], size]


def noise_stats(a):
    dx, dy = np.diff(a, axis=1), np.diff(a, axis=0)
    def mad(v):
        return float(np.median(np.abs(v-np.median(v))) / (0.67448975*np.sqrt(2)))
    hp = a-gaussian_filter(a, 2)
    return dict(horizontal_difference_sigma=mad(dx), vertical_difference_sigma=mad(dy),
                highpass_std=float(np.std(hp)),
                highpass_neighbor_correlation=correlation(hp[:, :-1].ravel(), hp[:, 1:].ravel()))


def region_stats(before, after, mask):
    n = int(mask.sum())
    if n < 100:
        return dict(pixels=n, status='unverified: insufficient region')
    delta = after[mask]-before[mask]
    return dict(pixels=n, mean_change=float(delta.mean()), rms_change=float(np.sqrt(np.mean(delta**2))),
                p99_absolute_change=float(np.percentile(np.abs(delta), 99)))


def display_window(a, center, width):
    return np.clip((a-(center-0.5))/(width-1)+0.5, 0, 1)


def preview(idx, before, after, roi, windows, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    delta = after-before
    limit = max(float(np.percentile(np.abs(delta), 99.5)), 1)
    for row, (wc, ww) in enumerate(windows):
        for col, (a, title) in enumerate(((before, 'Original pixels'), (after, 'Single-lag seamless'))):
            axes[row, col].imshow(display_window(a, wc, ww)[::4, ::4], cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title(f'{title}\nWC={wc:.1f}, WW={ww:.1f}')
            if roi:
                y, x, size = roi
                axes[row, col].add_patch(plt.Rectangle((x/4,y/4),size/4,size/4,fill=False,edgecolor='red'))
        axes[row, 2].imshow(delta[::4, ::4], cmap='coolwarm', vmin=-limit, vmax=limit)
        axes[row, 2].set_title(f'After - before (stored units)\ncolor limits +/- {limit:.1f}')
    for ax in axes.ravel():
        ax.axis('off')
    fig.suptitle(f'Frame {idx:02d} | research review, no clean reference | red: fixed noise ROI')
    fig.tight_layout()
    fig.savefig(output, dpi=110)
    plt.close(fig)



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


def load_m1_pair(manifest, light, dark):
    row = next(r for r in manifest['pair_results'] if (r['light_index'], r['dark_index']) == (light, dark))
    artifact = row['output_artifacts']['primary_maps']
    path = ROOT/'outputs/mhinge_cross_pair_v1'/artifact['relative_path']
    if not path.exists():
        path = ROOT/'outputs/pseudo_ghost_mechanism_v1'/artifact['relative_path']
    if sha(path) != artifact['sha256']:
        raise ValueError(f'Frozen map hash mismatch: {path}')
    with np.load(path) as z:
        maps = {k: z[k].copy() for k in z.files}
    if str(maps['background_mode']) != 'masked_hybrid_spline':
        raise ValueError('Wrong frozen background mode')
    bg = next(r for r in row['block_results'] if r['block_size'] == 16)
    bg = next(r for r in bg['background_results'] if r['background_mode'] == 'masked_hybrid_spline')
    result = next(r for r in bg['mask_results'] if r['mask_mode'] == 'exclude_saturated')
    valid = maps['evaluated_mask'].astype(bool)
    pred = maps['oof_prediction']
    if not result['detection_gate']['passed'] or abs(result['fit']['alpha_fold_mean']) < 1e-4:
        raise ValueError('Pair does not pass the frozen practical-effect gate')
    if not np.allclose(maps['observable_signal'], maps['raw_dark']-maps['background'], atol=2e-4, rtol=0):
        raise ValueError('Frozen Y identity failed')
    if not np.allclose(maps['oof_residual'][valid], (maps['observable_signal']-pred)[valid], atol=2e-5, rtol=0):
        raise ValueError('Frozen residual identity failed')
    # Verify the stored prediction against the recorded fold-specific model.
    for params in result['fit']['fold_parameters']:
        mask = valid & (maps['spatial_folds'] == params['fold'])
        expected = params['alpha']*maps['source'][mask]+params['intercept']
        if not np.allclose(pred[mask], expected, atol=2e-5, rtol=1e-6):
            raise ValueError('Frozen OOF prediction does not match model parameters')
    return maps, result['fit'], dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


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
