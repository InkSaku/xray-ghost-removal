"""Plateau/unclipped split model (Msplit) on the 10.30 RAW batch.

Design fixed in docs/plans/2026-10-30-raw-split-model.md before implementation.
Plateau blocks (previous frame at 55000) carry no intensity information, so their
ghost is a learned position map G; unclipped blocks keep an affine term on the mean
of their non-plateau pixels; boundary blocks mix both by plateau fraction s. G is an
empirical map, not a physical exposure coefficient.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ghost-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from xray_ghost_removal.common.paths import require_current_m1_output
from xray_ghost_removal.common.provenance import clean_json, package_sources
from xray_ghost_removal.m1.config import PRIMARY_BLOCK, TRIM_PERCENTILES
from xray_ghost_removal.m1.fitting import _weighted_lowpass
from xray_ghost_removal.research.raw_cohort import (
    CLEAN_DARK_COUNTERS, PLATEAU, Frame, clean_dark, fit_pooled, load_frames, pair_frames,
    plateau_strata, prediction_metrics, short_directory, training_indices,
)
from xray_ghost_removal.research.raw_m1 import leave_one_out as m1_leave_one_out

DESIGN = 'docs/plans/2026-10-30-raw-split-model.md'
MIN_MAP_SAMPLES = 3
FILL_SIGMAS = (2, 4, 8, 16, 32)
FILL_MIN_SUPPORT = 1e-3
PROMOTE_MAX_MEDIAN_RATIO = 0.80
PROMOTE_MIN_PAIR_FRACTION = 0.90
PIXEL_TERM_MIN_GAIN = 0.05
ROW_PROFILE_MIN_BLOCKS = 5
SHOWCASE = ('200000A5->200000A6', '20000059->2000005A')
MODELS = ('m1', 'map_only', 'msplit')


@dataclass
class PairData:
    x: np.ndarray      # whole-block mean of the previous frame minus clean dark (M1 input)
    xu: np.ndarray     # mean of non-plateau pixels minus clean dark; NaN on full plateau
    y: np.ndarray      # low-value frame minus clean dark
    s: np.ndarray      # plateau pixel fraction


def learn_map(ys: list[np.ndarray], ss: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Y over training pairs where the block is fully plateau.

    Blocks with fewer than MIN_MAP_SAMPLES samples are filled from observed blocks
    only, with increasing smoothing; observed values are never altered.
    """
    full = np.stack(ss) == 1.0
    count = full.sum(axis=0)
    total = np.where(full, np.stack(ys), 0.0).sum(axis=0)
    observed = count >= MIN_MAP_SAMPLES
    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(observed, total / count, np.nan)
    missing = ~observed
    for sigma in FILL_SIGMAS:
        if not missing.any():
            break
        low = _weighted_lowpass(g, observed, sigma)
        support = gaussian_filter(observed.astype(np.float64), sigma=sigma, mode='reflect')
        take = missing & (support > FILL_MIN_SUPPORT)
        g[take] = low[take]
        missing &= ~take
    if missing.any():
        raise ValueError('Position map could not be filled; too few plateau training blocks')
    return g, count, ~observed


def fit_split(train: list[PairData]) -> dict:
    g, count, filled = learn_map([d.y for d in train], [d.s for d in train])
    xu = [d.xu[d.s == 0] for d in train]
    yu = [d.y[d.s == 0] for d in train]
    a, b = fit_pooled(xu, yu)
    return {'G': g, 'a': a, 'b': b, 'c': float(np.mean(np.concatenate(yu))),
            'map_filled_blocks': int(filled.sum()), 'map_min_samples': int(count.min())}


def predict_split(params: dict, d: PairData, use_pixels: bool = True) -> np.ndarray:
    if use_pixels:
        unclipped = params['a'] * np.nan_to_num(d.xu) + params['b']
    else:
        unclipped = np.full(d.y.shape, params['c'])
    return d.s * params['G'] + (1.0 - d.s) * unclipped


def plateau_row_corr(y: np.ndarray, p: np.ndarray, s: np.ndarray) -> float | None:
    """Can the prediction follow the top-to-bottom ghost profile on plateau blocks?"""
    full = s == 1.0
    rows = [k for k in range(y.shape[0]) if full[k].sum() >= ROW_PROFILE_MIN_BLOCKS]
    if len(rows) < 3:
        return None
    ym = np.array([y[k][full[k]].mean() for k in rows])
    pm = np.array([p[k][full[k]].mean() for k in rows])
    if np.std(pm) <= 1e-9 or np.std(ym) <= 1e-9:
        return None
    return float(np.corrcoef(ym, pm)[0, 1])


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def _promotion(rows: list[dict], model: str) -> dict:
    ratio = np.array([r['s1'][model]['rmse'] / r['s1']['m1']['rmse'] for r in rows])
    better = int(np.sum(ratio < 1.0))
    needed = math.ceil(PROMOTE_MIN_PAIR_FRACTION * len(rows))
    strata = {name: {'model': _median(r['s1'][model][f'rmse_{name}'] for r in rows),
                     'm1': _median(r['s1']['m1'][f'rmse_{name}'] for r in rows)}
              for name in ('unclipped', 'boundary', 'plateau')}
    checks = {
        'median_ratio': bool(np.median(ratio) <= PROMOTE_MAX_MEDIAN_RATIO),
        'pairs_better_than_m1': bool(better >= needed),
        'unclipped_not_worse': bool(strata['unclipped']['model'] <= strata['unclipped']['m1']),
        'boundary_not_worse': bool(strata['boundary']['model'] <= strata['boundary']['m1']),
    }
    return {'median_rmse_ratio_vs_m1': float(np.median(ratio)), 'pairs_better_than_m1': better,
            'pairs_required': needed, 'median_stratum_rmse': strata, 'checks': checks,
            'passed': all(checks.values())}


def decide(rows: list[dict]) -> dict:
    msplit_u = _median(r['s1']['msplit']['rmse_unclipped'] for r in rows)
    map_u = _median(r['s1']['map_only']['rmse_unclipped'] for r in rows)
    gain = 1.0 - msplit_u / map_u
    pixel_term = {'msplit_median_unclipped_rmse': msplit_u, 'map_only_median_unclipped_rmse': map_u,
                  'gain': gain, 'kept': bool(gain >= PIXEL_TERM_MIN_GAIN)}
    candidate = 'msplit' if pixel_term['kept'] else 'map_only'
    promotion = _promotion(rows, candidate)
    return {'pairs': len(rows), 'pixel_term': pixel_term, 'candidate': candidate,
            'promotion': promotion, 'msplit_promotion_checks': _promotion(rows, 'msplit'),
            'decision': f'{candidate}_promoted' if promotion['passed'] else 'not_promoted'}


def build_data(frames: list[Frame]):
    pairs = pair_frames(frames)
    dark = clean_dark(frames)
    data = [PairData(x=p.high.block - dark, xu=p.high.unclipped_mean - dark,
                     y=p.low.block - dark, s=p.high.plateau_fraction) for p in pairs]
    return pairs, data


def analyze(frames: list[Frame]) -> tuple[dict, dict]:
    pairs, data = build_data(frames)
    groups = [p.directory for p in pairs]
    m1 = m1_leave_one_out([d.x for d in data], [d.y for d in data], groups, 'pair')
    fits = {}
    for scheme in ('pair', 'group'):
        for i in range(len(pairs)):
            train = training_indices(groups, i, scheme)
            key = (scheme, tuple(train))
            if key not in fits:
                fits[key] = fit_split([data[j] for j in train])
            fits[(scheme, i)] = fits[key]

    rows, residuals, showcase = [], {m: [] for m in MODELS}, {}
    for i, (pair, d) in enumerate(zip(pairs, data)):
        strata = plateau_strata(d.s)
        s1, s2 = fits[('pair', i)], fits[('group', i)]
        preds = {'m1': m1[i][0] * d.x + m1[i][1], 'map_only': predict_split(s1, d, False),
                 'msplit': predict_split(s1, d)}
        row = {'pair': pair.label, 'directory': pair.directory, 'high_frame': pair.high.file,
               'low_frame': pair.low.file, 'filename_gap_seconds': pair.gap_seconds,
               'flags': pair.flags,
               'parameters': {'m1_alpha': m1[i][0], 'm1_intercept': m1[i][1], 'a': s1['a'],
                              'b': s1['b'], 'c': s1['c'], 'map_filled_blocks': s1['map_filled_blocks'],
                              'map_min_samples': s1['map_min_samples']},
               's1': {}, 's2': {}}
        for model, p in preds.items():
            row['s1'][model] = {**prediction_metrics(d.y, p, strata),
                                'plateau_row_corr': plateau_row_corr(d.y, p, d.s)}
            residuals[model].append(d.y - p)
        for model, use in (('map_only', False), ('msplit', True)):
            p = predict_split(s2, d, use)
            row['s2'][model] = {**prediction_metrics(d.y, p, strata),
                                'plateau_row_corr': plateau_row_corr(d.y, p, d.s)}
        row['msplit_to_m1_rmse_ratio'] = row['s1']['msplit']['rmse'] / row['s1']['m1']['rmse']
        rows.append(row)
        if pair.label in SHOWCASE:
            showcase[pair.label] = {'data': d, 'preds': preds, 'directory': pair.directory}

    # G stability: one map per directory from all of its pairs, compared across directories
    # and with the mean M1 residual of the previous test.
    directories = sorted(set(groups))
    maps = {g: fit_split([d for d, gg in zip(data, groups) if gg == g])['G'] for g in directories}
    m1_mean_residual = np.mean(residuals['m1'], axis=0)
    corr = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    stability = {
        'between_directories': {f'{a} | {b}': corr(maps[a], maps[b])
                                for i, a in enumerate(directories) for b in directories[i + 1:]},
        'with_m1_mean_residual': {g: corr(maps[g], m1_mean_residual) for g in directories},
        'map_range_p1_p99': {g: np.percentile(maps[g], [1, 99]).tolist() for g in directories},
    }
    by_dir = {}
    for g in directories:
        sub = [r for r in rows if r['directory'] == g]
        by_dir[g] = {
            'pairs': len(sub),
            'median_ratio_msplit_vs_m1': _median(r['msplit_to_m1_rmse_ratio'] for r in sub),
            **{f'median_s1_r2_{m}': _median(r['s1'][m]['r2'] for r in sub) for m in MODELS},
            'median_s2_r2_msplit': _median(r['s2']['msplit']['r2'] for r in sub),
        }
    summary = {
        'primary': decide(rows),
        'sensitivity_without_flagged': decide([r for r in rows if not r['flags']]),
        'medians_all_pairs': {
            **{f's1_r2_{m}': _median(r['s1'][m]['r2'] for r in rows) for m in MODELS},
            **{f's1_rmse_{m}': _median(r['s1'][m]['rmse'] for r in rows) for m in MODELS},
            **{f's1_plateau_row_corr_{m}': _median(r['s1'][m]['plateau_row_corr'] for r in rows)
               for m in MODELS},
            's2_r2_msplit': _median(r['s2']['msplit']['r2'] for r in rows),
            's2_r2_map_only': _median(r['s2']['map_only']['r2'] for r in rows),
            'unclipped_slope_a': _median(r['parameters']['a'] for r in rows),
        },
        'by_directory': by_dir,
        'map_stability': stability,
        'map_filled_blocks_max': max(r['parameters']['map_filled_blocks'] for r in rows),
    }
    figure = {'rows': rows, 'maps': maps, 'showcase': showcase}
    return {'summary': summary, 'pairs': rows}, figure


def plot_diagnostics(figure: dict, path: Path) -> None:
    rows, maps = figure['rows'], figure['maps']
    directories = sorted(maps)
    colors = dict(zip(directories, ('#1f77b4', '#ff7f0e', '#2ca02c')))
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    for g in directories:
        sub = [r for r in rows if r['directory'] == g]
        ax[0, 0].scatter(range(len(sub)), [r['msplit_to_m1_rmse_ratio'] for r in sub], s=16,
                         color=colors[g], label=short_directory(g))
    ax[0, 0].axhline(PROMOTE_MAX_MEDIAN_RATIO, color='k', lw=.8, ls='--')
    ax[0, 0].axhline(1.0, color='k', lw=.8, ls=':')
    ax[0, 0].set(title='S1 RMSE: Msplit / M1 (dashed = 0.80 gate, dotted = no change)',
                 xlabel='pair order within directory')
    ax[0, 0].legend(fontsize=8)
    names = ('plateau', 'boundary', 'unclipped')
    width = 0.27
    for k, model in enumerate(MODELS):
        values = [_median(r['s1'][model][f'rmse_{n}'] for r in rows) for n in names]
        ax[0, 1].bar(np.arange(3) + (k - 1) * width, values, width, label=model)
    ax[0, 1].set_xticks(range(3), names)
    ax[0, 1].set(title='median S1 RMSE by stratum (counts)')
    ax[0, 1].legend(fontsize=8)
    floor = -0.5
    data = [[max(r['s1'][m]['r2'], floor) for r in rows] for m in MODELS] + \
        [[max(r['s2']['msplit']['r2'], floor) for r in rows]]
    ax[0, 2].boxplot(data)
    ax[0, 2].set_xticks([1, 2, 3, 4], ['M1 S1', 'map-only S1', 'Msplit S1', 'Msplit S2'])
    ax[0, 2].axhline(0.9, color='k', lw=.8, ls='--')
    ax[0, 2].set(title=f'R² per pair (values below {floor:g} drawn at floor)', ylim=(floor - .05, 1))
    lo = min(np.percentile(m, 1) for m in maps.values())
    hi = max(np.percentile(m, 99) for m in maps.values())
    for k, g in enumerate(directories):
        im = ax[1, k].imshow(maps[g], cmap='magma', vmin=lo, vmax=hi)
        ax[1, k].set(title=f'position map G | {short_directory(g)}', xticks=[], yticks=[])
    fig.colorbar(im, ax=ax[1, 2], fraction=.046, label='ghost on plateau (counts)')
    fig.suptitle('10.30 RAW | Msplit vs M1 | 16×16 blocks | leave-one-pair-out within directory')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_showcase(figure: dict, path: Path) -> None:
    cases = [(label, figure['showcase'][label]) for label in SHOWCASE if label in figure['showcase']]
    fig, ax = plt.subplots(len(cases), 6, figsize=(30, 5.4 * len(cases)), squeeze=False)
    for row, (label, case) in zip(ax, cases):
        d, preds = case['data'], case['preds']
        vmax = np.percentile(d.y, 99.5)
        row[0].imshow(d.x, cmap='gray', vmin=np.percentile(d.x, 1), vmax=d.x.max())
        row[0].set_title(f'{label} | {short_directory(case["directory"])}\nprevious frame (white = 55000)')
        for ax_, image, title in ((row[1], d.y, 'measured ghost Y'),
                                  (row[2], preds['m1'], 'M1 prediction'),
                                  (row[3], preds['msplit'], 'Msplit prediction')):
            ax_.imshow(image, cmap='magma', vmin=0, vmax=vmax)
            ax_.set_title(f'{title} (shared scale)')
        residual = d.y - preds['msplit']
        lim = np.percentile(np.abs(d.y - preds['m1']), 99)
        im = row[4].imshow(residual, cmap='coolwarm', vmin=-lim, vmax=lim)
        row[4].set_title('Msplit residual (scale of M1 residual)')
        fig.colorbar(im, ax=row[4], fraction=.046)
        for a in row[:5]:
            a.set_xticks([])
            a.set_yticks([])
        full = d.s == 1.0
        keep = [k for k in range(d.y.shape[0]) if full[k].sum() >= ROW_PROFILE_MIN_BLOCKS]
        for name, image, style in (('measured', d.y, '-'), ('M1', preds['m1'], '--'),
                                   ('Msplit', preds['msplit'], '-.')):
            row[5].plot(np.array(keep) * PRIMARY_BLOCK, [image[k][full[k]].mean() for k in keep],
                        style, label=name)
        row[5].set(title='plateau blocks: mean ghost per row', xlabel='image row (0 = top)',
                   ylabel='counts')
        row[5].legend()
        row[5].grid(alpha=.3)
    fig.suptitle('Msplit six-panel | pairs fixed in the design, display only')
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def write_outputs(result: dict, figure: dict, frames: list[Frame], output_dir: Path,
                  data_dir: Path, shape, dtype) -> None:
    require_current_m1_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'design': DESIGN,
        'input_root': str(data_dir),
        'assumed_layout': {'shape': list(shape), 'dtype': dtype, 'header_bytes': 0, 'confirmed': False},
        'definitions': {
            'block': PRIMARY_BLOCK, 'plateau_value': PLATEAU,
            'clean_dark_counters': [f'{c:08X}' for c in CLEAN_DARK_COUNTERS],
            'msplit': 's*G + (1-s)*(a*Xu + b)', 'map_only': 's*G + (1-s)*c',
            'm1': 'alpha*X + b (same as research raw-m1)',
            'map_min_samples': MIN_MAP_SAMPLES, 'fill_sigmas_blocks': list(FILL_SIGMAS),
            'trim_percentiles': list(TRIM_PERCENTILES),
            'thresholds': {'promote_max_median_ratio': PROMOTE_MAX_MEDIAN_RATIO,
                           'promote_min_pair_fraction': PROMOTE_MIN_PAIR_FRACTION,
                           'pixel_term_min_gain': PIXEL_TERM_MIN_GAIN},
        },
        'interpretation_notes': [
            'Y includes read noise and about 44 counts of drift between the two clean darks.',
            'G is an empirical plateau ghost map; the cause of the 55000 plateau is unconfirmed.',
            'Filename gaps are unverified; G has no time term.',
            'No recovery of exposed images is evaluated.',
        ],
        **result,
        'inputs': [{'file': f.file, 'sha256': f.sha256, 'high': f.high} for f in frames],
        'code': package_sources(),
    }
    (output_dir / 'results.json').write_text(
        json.dumps(clean_json(record), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'directory', 'filename_gap_s', 'flags', 'm1_rmse', 'map_only_rmse',
                         'msplit_rmse', 'msplit_to_m1_ratio', 'm1_r2', 'msplit_r2', 'msplit_s2_r2',
                         'a', 'b', 'msplit_plateau_row_corr'])
        for r in result['pairs']:
            s1 = r['s1']
            corr = s1['msplit']['plateau_row_corr']
            writer.writerow([r['pair'], r['directory'], f"{r['filename_gap_seconds']:.3f}", ';'.join(r['flags']),
                             f"{s1['m1']['rmse']:.2f}", f"{s1['map_only']['rmse']:.2f}",
                             f"{s1['msplit']['rmse']:.2f}", f"{r['msplit_to_m1_rmse_ratio']:.4f}",
                             f"{s1['m1']['r2']:.4f}", f"{s1['msplit']['r2']:.4f}",
                             f"{r['s2']['msplit']['r2']:.4f}", f"{r['parameters']['a']:.5f}",
                             f"{r['parameters']['b']:.2f}", '' if corr is None else f'{corr:.4f}'])
    plot_diagnostics(figure, output_dir / 'diagnostics.png')
    plot_showcase(figure, output_dir / 'six_panel.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('10.30例图'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/raw_split_v1'))
    parser.add_argument('--shape', type=int, nargs=2, default=(3072, 3072), metavar=('ROWS', 'COLS'))
    parser.add_argument('--dtype', default='<u2', choices=('<u2', '>u2'))
    args = parser.parse_args()
    frames = load_frames(args.data_dir, tuple(args.shape), args.dtype)
    result, figure = analyze(frames)
    write_outputs(result, figure, frames, args.output_dir, args.data_dir, tuple(args.shape), args.dtype)
    print(json.dumps(clean_json(result['summary']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
