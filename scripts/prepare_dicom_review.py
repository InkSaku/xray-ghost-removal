"""Reproducible full-resolution, single-lag seamless review (not frozen M1).

Select the cohort using inputs only, then export SC review copies and diagnostics.
No ground truth: noise/edge measurements are proxies, never accuracy claims.
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
from src.models.seamless_ghost import remove_ghost_seamless
from src.utils.dicom_utils import save_dicom

RAW = ROOT / 'data/raw/残影图像'
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = screen_inputs()
    cohort = select_cohort(rows)
    # Freeze selection before running the remover. Never select on improvement.
    manifest = dict(method='seamless, nearest previous frame only, default parameters; clip_output=False',
                    not_frozen_M1=True, input_screen=rows,
                    cohort={str(k): v for k, v in sorted(cohort.items())},
                    selection_policy='Historical 4/6/18 + unchanged 1 + observed-maximum pile-up extrema (>0.1%) + weakest input association excluding pile-up',
                    limitations=['No clean ground truth', 'Endpoint pile-up does not prove physical overexposure',
                                 'Noise estimates are single-image proxies', 'Latest frozen M1 is block-domain only'],
                    code_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
                        ROOT/'src/models/seamless_ghost.py', ROOT/'src/models/physics_model.py', ROOT/'src/utils/dicom_utils.py']})
    (OUT/'selection.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    # A single study makes all selected images visible together despite missing source UIDs.
    study = generate_uid()
    originals = {}
    needed = sorted(set(cohort) | {i-1 for i in cohort if i > 1})
    for idx in needed:
        source = pydicom.dcmread(RAW/f'{idx}.dcm')
        a = source.pixel_array
        lo, hi = np.percentile(a, [0.5, 99.5])
        source.WindowCenter, source.WindowWidth = float((lo+hi+1)/2), float(max(hi-lo+1, 2))
        source.SeriesNumber, source.InstanceNumber = idx*10, 1
        path = OUT/'dicom'/f'{idx:02d}_original.dcm'
        audit = save_dicom(a, source, path, study_uid=study,
                          series_description=f'{idx:02d} ORIGINAL PIXELS - research',
                          derivation_description=f'Pixel-identical review copy of {idx}.dcm; source UIDs invalid/duplicated; file SHA256 in selection.json')
        originals[idx] = pydicom.dcmread(path)
        assert audit['clipped_low'] == audit['clipped_high'] == 0
        assert np.array_equal(originals[idx].pixel_array, a)
    results = []
    for idx, reasons in sorted(cohort.items()):
        before_ds = originals[idx]
        before = before_ds.pixel_array.astype(np.float64)
        previous = originals[idx-1].pixel_array.astype(np.float64) if idx > 1 else before
        air, prev_obj = detect_air_mask(before), detect_object_mask(previous)
        upper = 2**int(before_ds.BitsStored)-1
        if idx == 1:
            proposed, applied, reason = before.copy(), False, 'unchanged export control'
        else:
            res = remove_ghost_seamless(before, previous, clip_output=False)
            proposed, applied, reason = res.cleaned, res.applied, res.reason
        template = deepcopy(before_ds)
        template.SeriesNumber = idx*10+1
        path = OUT/'dicom'/f'{idx:02d}_processed.dcm'
        audit = save_dicom(proposed, template, path, study_uid=study,
                          series_description=f'{idx:02d} SEAMLESS LAG1 - research',
                          derivation_description=f'Single-lag seamless exploratory review; applied={applied}; {reason}; not frozen M1',
                          source_images=[before_ds] + ([originals[idx-1]] if idx > 1 else []))
        after_ds = pydicom.dcmread(path)
        after = after_ds.pixel_array.astype(np.float64)
        for key in ('Rows','Columns','BitsAllocated','BitsStored','PixelRepresentation',
                    'PhotometricInterpretation','WindowCenter','WindowWidth'):
            assert before_ds.get(key) == after_ds.get(key), key
        clean_air = binary_erosion(air & ~prev_obj, iterations=8)
        roi = choose_roi(before, clean_air, upper)
        noise = {'status': 'unverified: no valid unsaturated clean-air tile'}
        if roi:
            y,x,size = roi
            nb, na = noise_stats(before[y:y+size,x:x+size]), noise_stats(after[y:y+size,x:x+size])
            noise = dict(status='proxy only: input-selected lowest-variation tile', before=nb, after=na,
                         highpass_std_ratio=na['highpass_std']/nb['highpass_std'] if nb['highpass_std'] > 0 else None)
        obj = detect_object_mask(before)
        gb = np.hypot(*np.gradient(gaussian_filter(before, 1)))
        ga = np.hypot(*np.gradient(gaussian_filter(after, 1)))
        edge = obj & (gb > np.percentile(gb[obj], 90)) if obj.any() else obj
        edge_ratio = float(np.median(ga[edge])/np.median(gb[edge])) if edge.any() else None
        b0, b1 = blocks(before), blocks(after)
        ghost_blocks = blocks((air & prev_obj).astype(float)) > 0.95
        sd0, sd1 = (float(np.std(b[ghost_blocks])) if ghost_blocks.any() else None for b in (b0,b1))
        # Display windows are selected from the original only; both images share them.
        global_window = [float(before_ds.WindowCenter), float(before_ds.WindowWidth)]
        vals = before[air]
        med = float(np.median(vals)) if vals.size else float(np.median(before))
        spread = float(np.median(np.abs(vals-med))*1.4826) if vals.size else float(np.std(before))
        windows = [global_window, [med, max(6*spread, 2)]]
        preview(idx, before, after, roi, windows, OUT/f'{idx:02d}_comparison.png')
        result = dict(index=idx, reasons=reasons, applied=applied, reason=reason,
                      export=audit, noise_roi_y_x_size=roi, noise=noise, windows=windows,
                      input_high_endpoint_fraction=float(np.mean(before==upper)),
                      input_observed_maximum=float(before.max()),
                      input_observed_max_fraction=float(np.mean(before==before.max())),
                      output_above_input_maximum_pixels=int(np.sum(after>before.max())),
                      coefficient_at_cap=('alpha=0.0500' in reason),
                      output_high_endpoint_fraction=float(np.mean(after==upper)),
                      newly_high_endpoint_pixels=int(np.sum((before<upper)&(after==upper))),
                      newly_low_endpoint_pixels=int(np.sum((before>0)&(after==0))),
                      regions={name:region_stats(before,after,mask) for name,mask in {
                          'clean_air':clean_air,'candidate_ghost_air':air & prev_obj,
                          'current_object':obj,'object_overlap':obj & prev_obj,
                          'input_endpoints':(before==0)|(before==upper)}.items()},
                      edge_gradient_ratio_proxy=edge_ratio,
                      candidate_ghost_block_std_before=sd0,candidate_ghost_block_std_after=sd1,
                      viewer_status='pending', detail_preservation_status='unverified without ground truth',
                      processed_sha256=sha(path))
        results.append(result)
        print(f'frame {idx}: {reason}; clip={audit["clipped_low"]}/{audit["clipped_high"]}; noise={noise.get("highpass_std_ratio")}', flush=True)
    # Confirm raw files stayed unchanged and all exported instances are unique.
    assert all(sha(RAW/f'{r["index"]}.dcm') == r['sha256'] for r in rows)
    # Remove only superseded files owned by this stable review generator.
    expected = {f'{i:02d}_original.dcm' for i in needed} | {f'{i:02d}_processed.dcm' for i in cohort}
    for p in (OUT/'dicom').glob('*.dcm'):
        if p.name not in expected and p.stem.split('_')[0].isdigit() and p.stem.split('_')[-1] in ('original','processed'):
            p.unlink()
    for p in OUT.glob('*_comparison.png'):
        if p.stem.split('_')[0].isdigit() and int(p.stem.split('_')[0]) not in cohort:
            p.unlink()
    files = sorted((OUT/'dicom').glob('*.dcm'))
    ids = [str(pydicom.dcmread(p, stop_before_pixels=True).SOPInstanceUID) for p in files]
    assert len(ids) == len(set(ids))
    payload = dict(selection=manifest, results=results, raw_hashes_unchanged=True,
                   exported_instances=len(files), unique_sop_uids=True,
                   software_versions={m.__name__:m.__version__ for m in (np,pydicom,matplotlib)})
    (OUT/'review_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False))
    lines = ['# DICOM 第一批阅片检查', '',
        '方法：早期 seamless，仅前一帧，探索性全分辨率输出；不是冻结 M1 的恢复验证。',
        '原始数据只读。所有查看副本是 Secondary Capture，保留像素位深；原图副本像素完全一致。',
        '原始 30 张图 UID 重复且缺少 Study/Series UID；查看副本分配独立标识，处理图引用对应查看副本。',
        '来源文件和算法哈希见 selection.json，完整数值和局限见 review_results.json。', '',
        '|帧|执行处理|系数碰上限|原图最大值堆积|导出前低/高越界像素|背景高频标准差比|边缘梯度比|阅片状态|',
        '|---|---|---|---|---|---|---|---|']
    for r in results:
        ratio = r['noise'].get('highpass_std_ratio')
        edge = r['edge_gradient_ratio_proxy']
        nr = f'{ratio:.6f}' if ratio is not None else '未验证'
        er = f'{edge:.6f}' if edge is not None else '未验证'
        lines.append(f'|{r["index"]}|{r["applied"]}|{r["coefficient_at_cap"]}|{r["input_observed_max_fraction"]:.2%}|{r["export"]["clipped_low"]}/{r["export"]["clipped_high"]}|{nr}|{er}|待阅片|')
    lines += ['', '## 如何查看', '',
        '导入 dicom/ 全部文件。按帧号配对 ORIGINAL PIXELS 与 SEAMLESS LAG1；额外原图用于检查前序来源。',
        '首先使用两图共享的默认窗宽窗位；再按 review_results.json 中 windows 的第二组数值同步调窗。',
        '逐一检查物体重叠处、边缘、噪声纹理、亮暗斑和接缝，并记录结论。比较 PNG 的红框为同一背景 ROI。',
        '', '## 解释边界', '',
        '- 背景 ROI 从原图低变化空气区选取，可能低估全图噪声；邻差与高频统计只是代理。',
        '- 高频标准差比或边缘梯度比接近 1 不能证明噪声和细节完全不变。',
        '- 候选残影区的分块标准差包含背景和真实结构，不能当作残影去除率。',
        '- 顶格像素不能单独证明过曝；未发现顶格也不能排除前端饱和。',
        '- 最弱关联样本不是真实无残影对照；第 1 帧仅验证导出不改像素。',
        '- 未获得 clean/previous/ghosted 三元组前，不报告恢复准确率。',
        '- 本目录为当前稳定阅片输出；重运行覆盖同名结果。旧多帧结果不混入本包。']
    (OUT/'review.md').write_text('\n'.join(lines)+'\n')
    print(f'Written {OUT}: {len(results)} cases, {len(files)} DICOMs', flush=True)


if __name__ == '__main__':
    main()
