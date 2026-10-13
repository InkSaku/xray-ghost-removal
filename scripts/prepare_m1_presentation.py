"""Screen display candidates and export honest x1 semisynthetic demonstrations.

Reuses frozen measured Y and descriptive full-fit M1, as run_m1_semisynthetic.
No refit, gain, registration or changes to frozen scientific evidence.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_dicom_review import blocks, load_m1_pair, sha, display_window
from scripts.run_m1_semisynthetic import lift, synthesize, error_metrics, review_template
from src.utils.dicom_utils import save_dicom
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import binary_dilation

RAW = ROOT / 'data/raw/AI修残影例图'
OUT = ROOT / 'outputs/dicom_review/presentation'
DONORS = ((3, 4), (5, 6))
# Curated after input-only screening and visual review; not a generalization cohort.
CASES = ((5, 6, 8), (3, 4, 30), (5, 6, 4))


def read_raw(manifest, index, role):
    path = RAW / f'{index}-{role}.dcm'
    if sha(path) != manifest['input_provenance']['files'][path.name]['sha256']:
        raise ValueError(f'Input hash mismatch: {path.name}')
    ds = pydicom.dcmread(path)
    if (ds.PhotometricInterpretation != 'MONOCHROME2' or ds.PixelRepresentation != 0
            or float(ds.get('RescaleSlope', 1)) != 1 or float(ds.get('RescaleIntercept', 0)) != 0
            or any(k in ds for k in ('ModalityLUTSequence', 'VOILUTSequence', 'PixelPaddingValue'))):
        raise ValueError(f'Unsupported grayscale: {path.name}')
    return ds


def screening(manifest, models):
    rows = []
    for target in range(1, 32):
        a = read_raw(manifest, target, 'light').pixel_array
        b = blocks(a)
        # Coarse display-only exclusion, NOT a scientific object segmentation.
        obj = binary_dilation(b < np.percentile(b, 75)-1000, iterations=2)
        for pair, (maps, _, _) in models.items():
            support = maps['source_support'].astype(bool) & maps['evaluated_mask'].astype(bool)
            visible = support & ~obj
            vals = b[visible]
            rows.append(dict(donor=list(pair), baseline_light=target,
                exposed_support_fraction=float(visible.sum()/max(support.sum(), 1)),
                baseline_air_p95_p05=float(np.ptp(np.percentile(vals, [5, 95]))) if vals.size else None,
                baseline_maximum=int(a.max()), baseline_maximum_fraction=float(np.mean(a == a.max())),
                input_sha256=sha(RAW/f'{target}-light.dcm')))
    return rows


def structure_window(a):
    """Full object display, computed independently for each stored image."""
    lo, hi = np.percentile(a, [.5, 99.5])
    return float((lo+hi+1)/2), float(max(hi-lo+1, 2))


def draw_case(case, arrays, valid, windows, metrics, pileup, path):
    plt.rcParams.update({'font.family': 'Arial Unicode MS', 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(15, 9), facecolor='#f6f7f9')
    grid = fig.add_gridspec(2, 3, left=.035, right=.965, top=.85, bottom=.13,
                          wspace=.10, hspace=.22, height_ratios=[1, .8])
    fig.text(.035, .95, '实测残影叠加与 M1 校正', fontsize=23, weight='bold', color='#18283e')
    fig.text(.035, .903, f'{case}  ·  实测信号 ×1  ·  校正前后同窗比较  ·  半合成演示', fontsize=13)
    small = [blocks(a) for a in arrays]
    for i, (a, title, window) in enumerate(zip(small,
            ['原 light：结构窗', '加入残影：残影细节窗', '校正后：相同残影细节窗'], windows.values())):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(display_window(a, *window), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{title}\nWC={window[0]:.1f}  WW={window[1]:.1f}', fontsize=12, pad=9); ax.axis('off')
    e0, e1 = small[1]-small[0], small[2]-small[0]
    limit = max(float(np.percentile(np.abs(e0[valid]), 99)), 1)
    for i, (a, title) in enumerate([(e0, '加入后 − 底图'), (e1, '校正后 − 底图')]):
        ax = fig.add_subplot(grid[1, i])
        im = ax.imshow(np.ma.array(a, mask=~valid), cmap='RdBu_r', vmin=-limit,
                       vmax=limit, interpolation='nearest')
        ax.set_title(title + '（同一色标）', fontsize=12); ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=.75, pad=.025)
    ax = fig.add_subplot(grid[1, 2]); ax.axis('off')
    before, after = metrics['before']['rmse'], metrics['after']['rmse']
    ax.text(0, .95, '相对底图的附加误差', fontsize=15, weight='bold', va='top')
    ax.text(0, .78, f'RMSE   {before:.2f} → {after:.2f}\n评价区域：全部冻结有效像素\n单位：存储像素值\n\n原 light 用结构窗\n加入残影与校正后使用完全相同 WC/WW\n展示采用 16×16 均值；DICOM 保留全尺寸',
            fontsize=11, va='top', linespacing=1.65)
    caution = (f'演示底图有最大值堆积（{pileup:.1%} 像素）；平坦背景有利于看清残影，不代表恢复了被截断的细节。'
               if pileup > .001 else '本底图未见明显最大值堆积；原图噪声与背景起伏仍会遮盖弱残影。')
    fig.text(.035, .078, caution, fontsize=11, color='#994000')
    fig.text(.035, .035, '边界：底图不是真实 clean；供体参与过拟合；此处验证已加入信号的扣除，不是独立真实残影去除验证。',
             fontsize=10, color='#555555')
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    manifest_path = ROOT/'outputs/mhinge_cross_pair_v1/analysis_results.json'
    manifest = json.loads(manifest_path.read_text())
    if (not manifest['frozen_background']['verified']
            or manifest['configuration']['block_sizes'] != [16]
            or manifest['configuration']['background_modes'] != ['masked_hybrid_spline']):
        raise ValueError('Unexpected frozen definition')
    if sha(ROOT/'configs/background_frozen_v1.json') != manifest['frozen_background']['sha256']:
        raise ValueError('Frozen configuration changed')
    models = {pair: load_m1_pair(manifest, *pair) for pair in DONORS}
    for (light, dark), (maps, _, _) in models.items():
        x = blocks(read_raw(manifest, light, 'light').pixel_array)
        x -= np.percentile(x, 75)
        d = blocks(read_raw(manifest, dark, 'dark').pixel_array)
        assert np.allclose(x, maps['source'], atol=.002, rtol=0)
        assert np.allclose(d, maps['raw_dark'], atol=.002, rtol=0)
    candidates = screening(manifest, models)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai'))
    records = []
    with tempfile.TemporaryDirectory(prefix='m1-presentation-') as tmp:
        stage = Path(tmp)
        for light, dark, target in CASES:
            case = f'G{light:02d}{dark:02d}_B{target:02d}'
            maps, fit, provenance = models[(light, dark)]
            ds = read_raw(manifest, target, 'light'); base = ds.pixel_array.astype(float)
            valid = maps['evaluated_mask'].astype(bool)
            signal = lift(maps['observable_signal'], valid, base.shape)
            prediction = lift(fit['alpha_full_descriptive']*maps['source'].astype(float)
                              + fit['intercept_full_descriptive'], valid, base.shape)
            ghosted, corrected = synthesize(base, signal, prediction)
            assert np.allclose(corrected-base, signal-prediction, atol=1e-10, rtol=0)
            row = next(r for r in candidates if r['donor'] == [light, dark] and r['baseline_light'] == target)
            # Baseline shows structure; before/after share the ghost-detail window.
            # Windowing only: pixel data and quantitative errors are unaffected.
            wc = float(np.percentile(base, 75))
            detail_window = (wc, 180.0 if row['baseline_maximum_fraction'] > .001 else 250.0)
            windows = dict(baseline=structure_window(base),
                ghosted=detail_window, corrected=detail_window)
            ds = review_template(ds, generate_uid(), target, stamp)
            ds.StudyDescription = f'M1 DEMO {case} - SEMISYNTHETIC'
            ds.VOILUTFunction = 'LINEAR'
            decoded = []; artifacts = {}; baseline_ds = None
            for index, (kind, a) in enumerate(zip(('baseline', 'ghosted', 'corrected'), (base, ghosted, corrected)), 1):
                ref = deepcopy(ds); ref.SeriesNumber = index; ref.InstanceNumber = 1
                ref.WindowCenter, ref.WindowWidth = windows[kind]
                ref.WindowCenterWidthExplanation = 'OBJECT STRUCTURE' if kind == 'baseline' else 'SHARED GHOST DETAIL'
                filename = f'{case}_{kind}.dcm'
                audit = save_dicom(a, ref, stage/'dicom'/filename,
                    series_description=f'{index} {kind.upper()} {case} - x1 DEMO',
                    derivation_description=f'Semisynthetic B + frozen Y x1; fixed full-fit M1; block16. {kind}. Baseline maximum pileup {row["baseline_maximum_fraction"]:.6f}.',
                    source_images=[baseline_ds] if baseline_ds is not None else None)
                reread = pydicom.dcmread(stage/'dicom'/filename)
                if index == 1: baseline_ds = reread
                assert audit['clipped_low'] == audit['clipped_high'] == 0
                assert np.array_equal(reread.pixel_array, np.rint(a).astype(np.uint16))
                assert (float(reread.WindowCenter), float(reread.WindowWidth)) == windows[kind]
                decoded.append(reread.pixel_array.astype(float))
                artifacts[kind] = dict(path='dicom/'+filename, sha256=sha(stage/'dicom'/filename), export=audit)
            assert windows['ghosted'] == windows['corrected']
            pixels = lift(valid, valid, base.shape).astype(bool)
            metrics = {k: error_metrics(a-base, pixels) for k, a in zip(('before','after'), decoded[1:])}
            for a, expected in ((decoded[1], signal), (decoded[2], signal-prediction)):
                assert np.max(np.abs((a-base)-expected)) <= .5+1e-8
            draw_case(case, decoded, valid, windows, metrics, row['baseline_maximum_fraction'], stage/f'{case}.png')
            records.append(dict(case=case, screening=row, windows=windows, metrics=metrics,
                model=provenance, parameters={k:fit[k] for k in ('alpha_full_descriptive','intercept_full_descriptive')}, artifacts=artifacts))
            print(case, metrics, flush=True)
        uids = [str(pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID) for p in (stage/'dicom').glob('*.dcm')]
        assert len(uids) == len(set(uids)) == 9
        report = dict(generated_at=stamp.isoformat(), code_sha256=sha(Path(__file__)),
            manifest_sha256=sha(manifest_path), scientific_definition='B+frozen Y x1; subtract frozen descriptive full-fit M1; no refit',
            selection='62 donor/baseline combinations screened on inputs; curated for display, not unbiased performance sampling',
            screening_method='block16; object proxy B<p75-1000 dilated 2 blocks; support from frozen maps; proxy for display only',
            display_definition='Baseline full-object percentile window; ghosted and corrected share identical narrow detail WC/WW. Shared error color scale.',
            rationale_for_output='Replace superseded display derivatives in the same stable package; display-only revision, scientific definition unchanged.',
            limitations=['Known donor fit, not independent testing', 'Baseline not certified clean',
                'Clipped baselines shown explicitly as illustrations only', 'RMSE measures added-signal error; it is not real ghost removal rate',
                '16x16 mean visualization; full-resolution DICOM has unaltered within-block baseline texture'],
            candidates=candidates, cases=records)
        (stage/'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (stage/'README.md').write_text('''# 汇报演示选样

首选 `G0506_B08`：5-light→6-dark 的实测残影，加到 8-light。
备选 `G0304_B30`：3-light→4-dark 的实测残影，加到 30-light。
真实背景对照 `G0506_B04`：5-light→6-dark 残影，加到 4-light。

前两组背景存在大量 55000 最大值堆积，只适合说明叠加与扣除流程，不证明真实影像恢复。
第三组未见明显最大值堆积，但残影容易被背景与噪声遮盖；结合下排误差图汇报。
图像筛选使用输入几何与背景，并经人工视觉挑选，不是无偏性能队列。

PNG 上排原 light 使用结构窗；加入残影图与校正后图使用完全相同的残影细节窄窗（WC/WW）。
每张图标明 WC/WW，均为 16×16 均值展示，下排误差图共享色标。
上排第 2、3 张可以直接比较残影变化；下排同色标误差图与数值提供量化对照。
DICOM 为全尺寸三系列，第 1 系列使用结构窗，第 2、3 系列共享细节窗口；未给残影乘增强系数，未重新拟合。
导入 dicom/ 下 9 个文件，按 Study 中 G0506_B08、G0304_B30、G0506_B04 分组看 1/2/3 系列。
若查看器沿用旧窗口，请对各系列手动设置 selection.json 内的 windows 对应项 [WC,WW]。
宽窗看物体结构、窄窗看残影是不同用途，不宜在一张窄窗截图中评判细节保真。

建议汇报表述：在实测残影半合成实验中，将冻结背景扣除后的信号以原幅度叠加到参考 light，
再用既有 M1 参数扣除，附加误差明显减小；底图不是真实 clean，供体参与过拟合，
因此尚不构成独立真实残影去除验证。

复现：`.venv/bin/python scripts/prepare_m1_presentation.py`。
只覆盖本目录同名演示派生产物；保留原 6 组查看文件和全部冻结证据。
''')
        OUT.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            dest = OUT/path.name
            if dest.is_dir(): shutil.rmtree(dest)
            shutil.move(str(path), dest)
    print(OUT)


if __name__ == '__main__':
    main()
