"""Cross-pair M1 test on the 10.30 RAW batch.

Design fixed in docs/plans/2026-10-30-raw-m1-cross-pair.md before any cross-pair
fit. Y is the low-value frame minus two clean dark frames; it still contains read
noise and dark drift. Filename times are unverified and are only described.
"""
from __future__ import annotations

import argparse
import csv
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

from xray_ghost_removal.common.paths import require_current_m1_output
from xray_ghost_removal.common.provenance import clean_json, package_sources
from xray_ghost_removal.common.spatial import spatial_folds
from xray_ghost_removal.m1.config import PRIMARY_BLOCK, TRIM_PERCENTILES
from xray_ghost_removal.m1.fitting import strict_affine_oof
from xray_ghost_removal.research.raw_cohort import (
    CLEAN_DARK_COUNTERS, PLATEAU, Frame, clean_dark, fit_pooled, load_frames, pair_frames,
    plateau_strata, prediction_metrics, short_directory, training_indices,
)

DESIGN = 'docs/plans/2026-10-30-raw-m1-cross-pair.md'
TRANSFER_MAX_MEDIAN_RATIO = 1.25
TRANSFER_PAIR_RATIO = 1.50
TRANSFER_MIN_PAIR_FRACTION = 0.90
FORM_MIN_MEDIAN_OOF_R2 = 0.90
RESPONSE_BINS = 10


def leave_one_out(xs: list[np.ndarray], ys: list[np.ndarray], groups: list[str],
                  by: str) -> list[tuple[float, float]]:
    """M1 parameters for each held-out pair; its own y is never part of training."""
    params = []
    for i in range(len(groups)):
        train = training_indices(groups, i, by)
        params.append(fit_pooled([xs[j] for j in train], [ys[j] for j in train]))
    return params


def _partial_corr(residual: np.ndarray, other: np.ndarray, current: np.ndarray) -> float:
    """Correlation of residual with `other` after removing the current source."""
    c = current.ravel() - current.mean()
    o = other.ravel() - other.mean()
    o = o - c * (o @ c) / max(c @ c, 1e-12)
    r = residual.ravel() - residual.mean()
    return float(r @ o / math.sqrt(max((r @ r) * (o @ o), 1e-12)))


def _spearman(a, b) -> float | None:
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _gates(rows: list[dict]) -> dict:
    ratios = np.array([r['s1']['rmse'] / r['ceiling']['rmse'] for r in rows])
    oof = np.array([r['s0']['cv_r2'] for r in rows])
    needed = math.ceil(TRANSFER_MIN_PAIR_FRACTION * len(rows))
    within = int(np.sum(ratios <= TRANSFER_PAIR_RATIO))
    transfer = {
        'median_rmse_ratio': float(np.median(ratios)),
        'median_ratio_pass': bool(np.median(ratios) <= TRANSFER_MAX_MEDIAN_RATIO),
        'pairs_within_ratio': within, 'pairs_required': needed,
        'pair_fraction_pass': bool(within >= needed),
    }
    transfer['passed'] = transfer['median_ratio_pass'] and transfer['pair_fraction_pass']
    form = {'median_oof_r2': float(np.median(oof)),
            'passed': bool(np.median(oof) >= FORM_MIN_MEDIAN_OOF_R2)}
    return {'pairs': len(rows), 'transfer': transfer, 'affine_form': form}


def analyze(frames: list[Frame]) -> tuple[dict, dict]:
    pairs = pair_frames(frames)
    dark = clean_dark(frames)
    xs = [p.high.block - dark for p in pairs]
    ys = [p.low.block - dark for p in pairs]
    groups = [p.directory for p in pairs]
    folds = spatial_folds(dark.shape)
    s1 = leave_one_out(xs, ys, groups, 'pair')
    s2 = leave_one_out(xs, ys, groups, 'group')

    rows, residuals = [], []
    for i, pair in enumerate(pairs):
        x, y, s = xs[i], ys[i], pair.high.plateau_fraction
        strata = plateau_strata(s)
        oof = strict_affine_oof(x, y, np.ones(y.shape, dtype=bool), folds)
        ceiling = fit_pooled([x], [y])
        row = {
            'pair': pair.label, 'directory': pair.directory, 'high_frame': pair.high.file,
            'low_frame': pair.low.file, 'filename_gap_seconds': pair.gap_seconds,
            'flags': pair.flags,
            's0': {'cv_r2': oof['cv_r2'], 'alpha_fold_mean': oof['alpha_fold_mean'],
                   'rmse': float(np.sqrt(np.nanmean(oof['residual'] ** 2)))},
        }
        for name, (alpha, intercept) in (('ceiling', ceiling), ('s1', s1[i]), ('s2', s2[i])):
            row[name] = {'alpha': alpha, 'intercept': intercept,
                         **prediction_metrics(y, alpha * x + intercept, strata)}
        row['s1_rmse_ratio'] = row['s1']['rmse'] / row['ceiling']['rmse']
        # Post-hoc (added after the primary gates were seen): own fit on unclipped blocks only.
        u = strata['unclipped']
        try:
            ua, ub = fit_pooled([x[u]], [y[u]])
            row['posthoc_unclipped_fit'] = {
                'alpha': ua, 'intercept': ub, 'blocks': int(u.sum()),
                **{k: v for k, v in prediction_metrics(y[u], ua * x[u] + ub, {}).items()
                   if k in ('r2', 'rmse')}}
        except ValueError:
            row['posthoc_unclipped_fit'] = None
        residual = y - (s1[i][0] * x + s1[i][1])
        residuals.append(residual)
        # D4: older exposure versus the causally impossible next exposure.
        same = [j for j, g in enumerate(groups) if g == pair.directory]
        k = same.index(i)
        row['older_source_partial_corr'] = (
            _partial_corr(residual, xs[same[k - 1]], x) if k > 0 else None)
        row['next_source_partial_corr'] = (
            _partial_corr(residual, xs[same[k + 1]], x) if k + 1 < len(same) else None)
        rows.append(row)
    ratios = np.array([r['s1_rmse_ratio'] for r in rows])
    typical = int(np.argmin(np.abs(ratios - np.median(ratios))))
    examples = [(rows[typical]['pair'], residuals[typical])]

    # D2: binned residual against X on unclipped blocks only.
    unclipped_x = np.concatenate([xs[i][pairs[i].high.plateau_fraction == 0] for i in range(len(pairs))])
    unclipped_r = np.concatenate([residuals[i][pairs[i].high.plateau_fraction == 0] for i in range(len(pairs))])
    edges = np.percentile(unclipped_x, np.linspace(0, 100, RESPONSE_BINS + 1))
    which = np.clip(np.digitize(unclipped_x, edges[1:-1]), 0, RESPONSE_BINS - 1)
    response = [{'x_low': float(edges[b]), 'x_high': float(edges[b + 1]),
                 'x_mean': float(unclipped_x[which == b].mean()),
                 'mean_residual': float(unclipped_r[which == b].mean()),
                 'blocks': int((which == b).sum())} for b in range(RESPONSE_BINS)]

    # D3: does the cross-pair residual keep one spatial pattern?
    odd = np.mean(residuals[1::2], axis=0)
    even = np.mean(residuals[0::2], axis=0)
    mean_map = np.mean(residuals, axis=0)
    stable = float(np.corrcoef(odd.ravel(), even.ravel())[0, 1])
    pooled_var = float(np.mean([np.var(r) for r in residuals]))
    centered = [r - r.mean() for r in residuals]
    explained = 1.0 - float(np.mean([np.var(c - (mean_map - mean_map.mean())) for c in centered])) / \
        max(float(np.mean([np.var(c) for c in centered])), 1e-12)

    flagged = [r for r in rows if r['flags']]
    clean_rows = [r for r in rows if not r['flags']]
    by_dir = {}
    for directory in sorted(set(groups)):
        sub = [r for r in rows if r['directory'] == directory]
        by_dir[directory] = {
            'pairs': len(sub),
            'pooled_alpha_all_pairs': fit_pooled([xs[i] for i, g in enumerate(groups) if g == directory],
                                                 [ys[i] for i, g in enumerate(groups) if g == directory])[0],
            'median_s0_cv_r2': float(np.median([r['s0']['cv_r2'] for r in sub])),
            'median_s1_r2': float(np.median([r['s1']['r2'] for r in sub])),
            'median_s1_rmse_ratio': float(np.median([r['s1_rmse_ratio'] for r in sub])),
            'median_s2_r2': float(np.median([r['s2']['r2'] for r in sub])),
            'median_s2_rmse_ratio': float(np.median([r['s2']['rmse'] / r['ceiling']['rmse'] for r in sub])),
            'spearman_gap_vs_ceiling_alpha': _spearman(
                [r['filename_gap_seconds'] for r in sub], [r['ceiling']['alpha'] for r in sub]),
        }
    stratum = {name: float(np.median([r['s1'][f'rmse_{name}'] for r in rows
                                      if r['s1'][f'rmse_{name}'] is not None]))
               for name in ('plateau', 'boundary', 'unclipped')}
    unclipped_fits = [r['posthoc_unclipped_fit'] for r in rows if r['posthoc_unclipped_fit']]
    older = [r['older_source_partial_corr'] for r in rows if r['older_source_partial_corr'] is not None]
    nxt = [r['next_source_partial_corr'] for r in rows if r['next_source_partial_corr'] is not None]
    summary = {
        'primary': _gates(rows),
        'sensitivity_without_flagged': _gates(clean_rows),
        'flagged_pairs': [{'pair': r['pair'], 'flags': r['flags']} for r in flagged],
        'by_directory': by_dir,
        'medians_all_pairs': {
            's0_cv_r2': float(np.median([r['s0']['cv_r2'] for r in rows])),
            'ceiling_r2': float(np.median([r['ceiling']['r2'] for r in rows])),
            's1_r2': float(np.median([r['s1']['r2'] for r in rows])),
            's2_r2': float(np.median([r['s2']['r2'] for r in rows])),
            's1_bias': float(np.median([r['s1']['bias'] for r in rows])),
        },
        'diagnostics': {
            'd1_median_s1_rmse_by_stratum': stratum,
            'd1_boundary_to_unclipped_rmse': stratum['boundary'] / stratum['unclipped'],
            'd2_unclipped_response_bins': response,
            'd3_odd_even_mean_residual_map_corr': stable,
            'd3_variance_explained_by_mean_residual_map': explained,
            'd3_mean_pairwise_residual_variance': pooled_var,
            'd4_median_older_source_partial_corr': float(np.median(older)),
            'd4_median_next_source_partial_corr': float(np.median(nxt)),
        },
        'posthoc_unclipped_only': {
            'note': 'Added after the primary gates were seen; exploratory, not a gate.',
            'pairs_fitted': len(unclipped_fits),
            'median_alpha': float(np.median([f['alpha'] for f in unclipped_fits])),
            'alpha_iqr': np.percentile([f['alpha'] for f in unclipped_fits], [25, 75]).tolist(),
            'median_intercept': float(np.median([f['intercept'] for f in unclipped_fits])),
            'median_r2': float(np.median([f['r2'] for f in unclipped_fits])),
            'median_all_block_alpha': float(np.median([r['ceiling']['alpha'] for r in rows])),
        },
    }
    figure_data = {'rows': rows, 'response': response, 'mean_map': mean_map, 'examples': examples}
    return {'summary': summary, 'pairs': rows}, figure_data


def plot(figure_data: dict, path: Path) -> None:
    rows = figure_data['rows']
    directories = sorted({r['directory'] for r in rows})
    colors = dict(zip(directories, ('#1f77b4', '#ff7f0e', '#2ca02c')))
    # ASCII legend labels: the default font has no CJK glyphs. Directory names are in the JSON.
    short = {d: short_directory(d) for d in directories}
    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    ratio_cap, r2_floor = 3.0, -0.5
    for d in directories:
        sub = [r for r in rows if r['directory'] == d]
        ax[0, 0].scatter(range(len(sub)), [min(r['s1_rmse_ratio'], ratio_cap) for r in sub],
                         s=16, color=colors[d], label=short[d])
        ax[0, 2].scatter([r['filename_gap_seconds'] for r in sub], [r['ceiling']['alpha'] for r in sub],
                         s=16, color=colors[d], label=short[d])
    ax[0, 0].axhline(TRANSFER_MAX_MEDIAN_RATIO, color='k', lw=.8, ls='--')
    ax[0, 0].axhline(TRANSFER_PAIR_RATIO, color='k', lw=.8, ls=':')
    capped = sum(r['s1_rmse_ratio'] > ratio_cap for r in rows)
    ax[0, 0].set(title=f'S1 RMSE / own-fit RMSE ({capped} pair(s) above {ratio_cap:g} drawn at top)',
                 xlabel='pair order within directory', ylim=(0.9, ratio_cap + 0.1))
    ax[0, 0].legend(fontsize=8)
    ax[0, 2].legend(fontsize=8)
    data = [[r[k]['r2'] if k != 's0' else r['s0']['cv_r2'] for r in rows] for k in ('s0', 'ceiling', 's1', 's2')]
    below = sum(v < r2_floor for col in data for v in col)
    ax[0, 1].boxplot([[max(v, r2_floor) for v in col] for col in data])
    ax[0, 1].set_xticks([1, 2, 3, 4], ['S0 spatial OOF', 'own fit', 'S1 leave-pair', 'S2 leave-dir'])
    ax[0, 1].axhline(FORM_MIN_MEDIAN_OOF_R2, color='k', lw=.8, ls='--')
    ax[0, 1].set(title=f'R² per pair ({below} value(s) below {r2_floor:g} drawn at floor)',
                 ylim=(r2_floor - 0.05, 1))
    ax[0, 2].set(title='own-fit alpha vs filename gap (time unverified)', xlabel='seconds', xscale='log')
    resp = figure_data['response']
    ax[1, 0].plot([b['x_mean'] for b in resp], [b['mean_residual'] for b in resp], 'o-')
    ax[1, 0].axhline(0, color='k', lw=.8)
    ax[1, 0].set(title='S1 mean residual vs X (unclipped blocks)', xlabel='X = high − dark', ylabel='Y − prediction')
    m = figure_data['mean_map']
    lim = np.percentile(np.abs(m), 99)
    im = ax[1, 1].imshow(m, cmap='coolwarm', vmin=-lim, vmax=lim)
    ax[1, 1].set(title='mean S1 residual over 70 pairs', xticks=[], yticks=[])
    fig.colorbar(im, ax=ax[1, 1], fraction=.046)
    label, residual = figure_data['examples'][0]
    lim = np.percentile(np.abs(residual), 99)
    im = ax[1, 2].imshow(residual, cmap='coolwarm', vmin=-lim, vmax=lim)
    ax[1, 2].set(title=f'S1 residual, median-ratio pair {label}', xticks=[], yticks=[])
    fig.colorbar(im, ax=ax[1, 2], fraction=.046)
    fig.suptitle('10.30 RAW | M1 cross-pair test | 16×16 blocks | Y = low − clean dark (not noise-free)')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_outputs(result: dict, figure_data: dict, frames: list[Frame], output_dir: Path,
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
            'x': 'high block mean - mean clean dark', 'y': 'low block mean - mean clean dark',
            'trim_percentiles': list(TRIM_PERCENTILES),
            'thresholds': {'transfer_max_median_ratio': TRANSFER_MAX_MEDIAN_RATIO,
                           'transfer_pair_ratio': TRANSFER_PAIR_RATIO,
                           'transfer_min_pair_fraction': TRANSFER_MIN_PAIR_FRACTION,
                           'form_min_median_oof_r2': FORM_MIN_MEDIAN_OOF_R2},
        },
        'interpretation_notes': [
            'Y includes read noise and about 44 counts of drift between the two clean darks.',
            'X is clipped at 55000 on plateau blocks; alpha is not a coefficient of true exposure.',
            'Filename gaps are unverified; no decay constant is fitted.',
            'The own-fit ceiling uses the held-out target and is a reference, not a model.',
        ],
        **result,
        'inputs': [{'file': f.file, 'sha256': f.sha256, 'high': f.high} for f in frames],
        'code': package_sources(),
    }
    (output_dir / 'results.json').write_text(
        json.dumps(clean_json(record), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'directory', 'filename_gap_s', 'flags', 's0_cv_r2', 'ceiling_alpha',
                         'ceiling_r2', 's1_alpha', 's1_r2', 's1_rmse_ratio', 's2_r2',
                         'older_source_partial_corr', 'posthoc_unclipped_alpha'])
        for r in result['pairs']:
            writer.writerow([r['pair'], r['directory'], f"{r['filename_gap_seconds']:.3f}", ';'.join(r['flags']),
                             f"{r['s0']['cv_r2']:.4f}", f"{r['ceiling']['alpha']:.5f}", f"{r['ceiling']['r2']:.4f}",
                             f"{r['s1']['alpha']:.5f}", f"{r['s1']['r2']:.4f}", f"{r['s1_rmse_ratio']:.4f}",
                             f"{r['s2']['r2']:.4f}",
                             '' if r['older_source_partial_corr'] is None else f"{r['older_source_partial_corr']:.4f}",
                             '' if r['posthoc_unclipped_fit'] is None else f"{r['posthoc_unclipped_fit']['alpha']:.5f}"])
    plot(figure_data, output_dir / 'diagnostics.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('10.30例图'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/raw_m1_v1'))
    parser.add_argument('--shape', type=int, nargs=2, default=(3072, 3072), metavar=('ROWS', 'COLS'))
    parser.add_argument('--dtype', default='<u2', choices=('<u2', '>u2'))
    args = parser.parse_args()
    frames = load_frames(args.data_dir, tuple(args.shape), args.dtype)
    result, figure_data = analyze(frames)
    write_outputs(result, figure_data, frames, args.output_dir, args.data_dir, tuple(args.shape), args.dtype)
    print(json.dumps(clean_json(result['summary']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
