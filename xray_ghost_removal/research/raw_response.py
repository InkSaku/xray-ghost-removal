"""Unclipped-response shape on the 10.30 RAW batch: NLCSC-style saturation vs power law.

Design fixed in docs/plans/2026-10-31-raw-response-shape.md before implementation.
Only the unclipped term of Msplit changes: s*G + (1-s)*f(Xu). The NLCSC-style
curve borrows only the exposure-dependent amplitude of Starman 2012; its multi-
frame stored-charge tracking cannot be tested with one dark per exposure.
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
from xray_ghost_removal.m1.config import PRIMARY_BLOCK, TRIM_PERCENTILES
from xray_ghost_removal.m1.fitting import _training_keep
from xray_ghost_removal.research.raw_cohort import (
    CLEAN_DARK_COUNTERS, PLATEAU, Frame, load_frames, plateau_strata, prediction_metrics,
    short_directory, training_indices,
)
from xray_ghost_removal.research.raw_split import PairData, build_data, fit_split, predict_split

DESIGN = 'docs/plans/2026-10-31-raw-response-shape.md'
X_SCALE = 55000.0
GRIDS = {'msat': np.geomspace(2000.0, 500000.0, 40), 'mpow': np.geomspace(0.25, 4.0, 40)}
MODELS = ('msplit', 'msat', 'mpow')
REPLACE_MAX_MEDIAN_RATIO = 0.95
REPLACE_MIN_PAIR_FRACTION = 0.60
STRATUM_TOLERANCE = 1.01
IDENTIFIABLE_MIN_FRACTION = 0.90
RESPONSE_BINS = 10


def basis(model: str, x: np.ndarray, theta: float) -> np.ndarray:
    if model == 'msat':
        return 1.0 - np.exp(-x / theta)
    if model == 'mpow':
        return np.power(np.clip(x, 0.0, None) / X_SCALE, theta)
    raise ValueError(model)


def fit_shape(model: str, x: np.ndarray, y: np.ndarray) -> dict:
    """Grid over the nonlinear parameter; least squares for amplitude and offset."""
    keep, _ = _training_keep(x, y, np.ones(x.shape, dtype=bool), TRIM_PERCENTILES)
    x, y = x[keep], y[keep]
    best = None
    grid = GRIDS[model]
    for k, theta in enumerate(grid):
        design = np.column_stack([basis(model, x, theta), np.ones(x.size)])
        (c, b), *_ = np.linalg.lstsq(design, y, rcond=None)
        sse = float(np.sum((y - design @ np.array([c, b])) ** 2))
        if best is None or sse < best['sse']:
            best = {'theta': float(theta), 'c': float(c), 'b': float(b), 'sse': sse, 'index': k}
    best['at_grid_edge'] = best['index'] in (0, len(grid) - 1)
    return best


def predict_shape(model: str, fit: dict, split: dict, d: PairData) -> np.ndarray:
    unclipped = fit['c'] * basis(model, np.nan_to_num(d.xu), fit['theta']) + fit['b']
    return d.s * split['G'] + (1.0 - d.s) * unclipped


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def _replacement(rows: list[dict], model: str) -> dict:
    ratio = np.array([r['s1'][model]['rmse'] / r['s1']['msplit']['rmse'] for r in rows])
    better = int(np.sum(ratio < 1.0))
    needed = math.ceil(REPLACE_MIN_PAIR_FRACTION * len(rows))
    strata = {name: {'model': _median(r['s1'][model][f'rmse_{name}'] for r in rows),
                     'msplit': _median(r['s1']['msplit'][f'rmse_{name}'] for r in rows)}
              for name in ('unclipped', 'boundary', 'plateau')}
    interior = sum(not r['fits'][model]['at_grid_edge'] for r in rows)
    checks = {
        'median_ratio': bool(np.median(ratio) <= REPLACE_MAX_MEDIAN_RATIO),
        'pairs_better': bool(better >= needed),
        'unclipped_not_worse': bool(strata['unclipped']['model'] <= STRATUM_TOLERANCE * strata['unclipped']['msplit']),
        'boundary_not_worse': bool(strata['boundary']['model'] <= STRATUM_TOLERANCE * strata['boundary']['msplit']),
        'identifiable': bool(interior >= math.ceil(IDENTIFIABLE_MIN_FRACTION * len(rows))),
    }
    return {'median_rmse_ratio_vs_msplit': float(np.median(ratio)), 'pairs_better': better,
            'pairs_required': needed, 'interior_fits': interior, 'median_stratum_rmse': strata,
            'median_rmse': _median(r['s1'][model]['rmse'] for r in rows),
            'checks': checks, 'passed': all(checks.values())}


def decide(rows: list[dict]) -> dict:
    candidates = {m: _replacement(rows, m) for m in ('msat', 'mpow')}
    passed = [m for m in candidates if candidates[m]['passed']]
    winner = min(passed, key=lambda m: candidates[m]['median_rmse']) if passed else None
    return {'pairs': len(rows), 'candidates': candidates,
            'decision': f'{winner}_replaces_msplit' if winner else 'msplit_retained'}


def _binned(xs: list[np.ndarray], residuals: dict[str, list[np.ndarray]]) -> list[dict]:
    x = np.concatenate(xs)
    edges = np.percentile(x, np.linspace(0, 100, RESPONSE_BINS + 1))
    which = np.clip(np.digitize(x, edges[1:-1]), 0, RESPONSE_BINS - 1)
    flat = {m: np.concatenate(v) for m, v in residuals.items()}
    return [{'x_low': float(edges[k]), 'x_high': float(edges[k + 1]), 'x_mean': float(x[which == k].mean()),
             'blocks': int((which == k).sum()),
             **{f'mean_residual_{m}': float(flat[m][which == k].mean()) for m in flat}}
            for k in range(RESPONSE_BINS)]


def analyze(frames: list[Frame]) -> tuple[dict, dict]:
    pairs, data = build_data(frames)
    groups = [p.directory for p in pairs]
    cache, rows = {}, []
    by_stratum = {name: {'x': [], 'r': {m: [] for m in MODELS}} for name in ('unclipped', 'boundary')}
    curves = {}
    for i, (pair, d) in enumerate(zip(pairs, data)):
        train = tuple(training_indices(groups, i, 'pair'))
        if train not in cache:
            subset = [data[j] for j in train]
            xu = np.concatenate([t.xu[t.s == 0] for t in subset])
            yu = np.concatenate([t.y[t.s == 0] for t in subset])
            cache[train] = {'split': fit_split(subset),
                            **{m: fit_shape(m, xu, yu) for m in ('msat', 'mpow')}}
        fit = cache[train]
        split = fit['split']
        preds = {'msplit': predict_split(split, d),
                 'msat': predict_shape('msat', fit['msat'], split, d),
                 'mpow': predict_shape('mpow', fit['mpow'], split, d)}
        strata = plateau_strata(d.s)
        row = {'pair': pair.label, 'directory': pair.directory, 'filename_gap_seconds': pair.gap_seconds,
               'flags': pair.flags,
               'fits': {'msplit': {'a': split['a'], 'b': split['b']},
                        **{m: {k: fit[m][k] for k in ('theta', 'c', 'b', 'at_grid_edge')} for m in ('msat', 'mpow')}},
               's1': {m: prediction_metrics(d.y, p, strata) for m, p in preds.items()}}
        row['ratio_vs_msplit'] = {m: row['s1'][m]['rmse'] / row['s1']['msplit']['rmse'] for m in ('msat', 'mpow')}
        rows.append(row)
        for name in by_stratum:
            mask = strata[name]
            by_stratum[name]['x'].append(d.xu[mask])
            for m, p in preds.items():
                by_stratum[name]['r'][m].append((d.y - p)[mask])
        curves.setdefault(pair.directory, fit)

    summary = {
        'primary': decide(rows),
        'sensitivity_without_flagged': decide([r for r in rows if not r['flags']]),
        'selected_parameters': {
            m: {'median': _median(r['fits'][m]['theta'] for r in rows),
                'range': [min(r['fits'][m]['theta'] for r in rows), max(r['fits'][m]['theta'] for r in rows)],
                'at_grid_edge': sum(r['fits'][m]['at_grid_edge'] for r in rows),
                'grid': [float(GRIDS[m][0]), float(GRIDS[m][-1])]}
            for m in ('msat', 'mpow')},
        'medians_all_pairs': {f's1_{k}_{m}': _median(r['s1'][m][k] for r in rows)
                              for m in MODELS for k in ('rmse', 'r2')},
        'binned_residual': {name: _binned(v['x'], v['r']) for name, v in by_stratum.items()},
    }
    return {'summary': summary, 'pairs': rows}, {'rows': rows, 'curves': curves,
                                                 'binned': summary['binned_residual']}


def plot(figure: dict, path: Path) -> None:
    rows = figure['rows']
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    for k, m in enumerate(('msat', 'mpow')):
        ax[0, 0].scatter(np.arange(len(rows)) + 0.2 * k, [r['ratio_vs_msplit'][m] for r in rows], s=14, label=m)
    ax[0, 0].axhline(REPLACE_MAX_MEDIAN_RATIO, color='k', lw=.8, ls='--')
    ax[0, 0].axhline(1.0, color='k', lw=.8, ls=':')
    ax[0, 0].set(title='S1 RMSE / Msplit (dashed = 0.95 gate)', xlabel='pair index')
    ax[0, 0].legend(fontsize=8)
    names = ('plateau', 'boundary', 'unclipped')
    for k, m in enumerate(MODELS):
        values = [_median(r['s1'][m][f'rmse_{n}'] for r in rows) for n in names]
        ax[0, 1].bar(np.arange(3) + (k - 1) * 0.27, values, 0.27, label=m)
    ax[0, 1].set_xticks(range(3), names)
    ax[0, 1].set(title='median S1 RMSE by stratum (counts)')
    ax[0, 1].legend(fontsize=8)
    for k, (m, label) in enumerate((('msat', 'tau (log)'), ('mpow', 'p (log)'))):
        ax[0, 2].hist(np.log10([r['fits'][m]['theta'] for r in rows]), bins=20, alpha=.6,
                      label=f'{m}: log10 {label}')
    ax[0, 2].set(title='selected nonlinear parameter over 70 leave-one-out fits')
    ax[0, 2].legend(fontsize=8)
    for a, name in ((ax[1, 0], 'unclipped'), (ax[1, 1], 'boundary')):
        bins = figure['binned'][name]
        for m in MODELS:
            a.plot([b['x_mean'] for b in bins], [b[f'mean_residual_{m}'] for b in bins], 'o-', label=m)
        a.axhline(0, color='k', lw=.8)
        a.set(title=f'{name} blocks: mean residual vs Xu', xlabel='Xu (counts above clean dark)')
        a.legend(fontsize=8)
    x = np.linspace(0, 54000, 300)
    for g, fit in sorted(figure['curves'].items()):
        line, = ax[1, 2].plot(x, fit['split']['a'] * x + fit['split']['b'], lw=1, label=f'{short_directory(g)} linear')
        for m, style in (('msat', '--'), ('mpow', ':')):
            ax[1, 2].plot(x, fit[m]['c'] * basis(m, x, fit[m]['theta']) + fit[m]['b'], style, color=line.get_color())
    ax[1, 2].set(title='fitted f(Xu), one fold per directory (solid lin, -- sat, : pow)',
                 xlabel='Xu', ylabel='predicted ghost (counts)')
    ax[1, 2].legend(fontsize=7)
    fig.suptitle('10.30 RAW | unclipped response shape | Msplit vs NLCSC-style saturation vs power law')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
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
            'msplit': 's*G + (1-s)*(a*Xu + b)',
            'msat': 's*G + (1-s)*(c*(1 - exp(-Xu/tau)) + b)',
            'mpow': 's*G + (1-s)*(c*(Xu/55000)^p + b)',
            'grids': {m: [float(GRIDS[m][0]), float(GRIDS[m][-1]), len(GRIDS[m])] for m in GRIDS},
            'trim_percentiles': list(TRIM_PERCENTILES),
            'thresholds': {'replace_max_median_ratio': REPLACE_MAX_MEDIAN_RATIO,
                           'replace_min_pair_fraction': REPLACE_MIN_PAIR_FRACTION,
                           'stratum_tolerance': STRATUM_TOLERANCE,
                           'identifiable_min_fraction': IDENTIFIABLE_MIN_FRACTION},
        },
        'interpretation_notes': [
            'Only the exposure-dependent amplitude of NLCSC is tested; its multi-frame charge tracking needs time series this batch lacks.',
            'Xu is above clean dark in the stored-pixel domain; 55000 plateau cause is unconfirmed.',
            'Y includes read noise and about 44 counts of drift between the two clean darks.',
        ],
        **result,
        'inputs': [{'file': f.file, 'sha256': f.sha256, 'high': f.high} for f in frames],
        'code': package_sources(),
    }
    (output_dir / 'results.json').write_text(
        json.dumps(clean_json(record), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'directory', 'flags', 'msplit_rmse', 'msat_rmse', 'mpow_rmse',
                         'msat_ratio', 'mpow_ratio', 'msat_tau', 'mpow_p'])
        for r in result['pairs']:
            s1 = r['s1']
            writer.writerow([r['pair'], r['directory'], ';'.join(r['flags']),
                             f"{s1['msplit']['rmse']:.2f}", f"{s1['msat']['rmse']:.2f}", f"{s1['mpow']['rmse']:.2f}",
                             f"{r['ratio_vs_msplit']['msat']:.4f}", f"{r['ratio_vs_msplit']['mpow']:.4f}",
                             f"{r['fits']['msat']['theta']:.0f}", f"{r['fits']['mpow']['theta']:.3f}"])
    plot(figure, output_dir / 'diagnostics.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('10.30例图'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/raw_response_v1'))
    parser.add_argument('--shape', type=int, nargs=2, default=(3072, 3072), metavar=('ROWS', 'COLS'))
    parser.add_argument('--dtype', default='<u2', choices=('<u2', '>u2'))
    args = parser.parse_args()
    frames = load_frames(args.data_dir, tuple(args.shape), args.dtype)
    result, figure = analyze(frames)
    write_outputs(result, figure, frames, args.output_dir, args.data_dir, tuple(args.shape), args.dtype)
    print(json.dumps(clean_json(result['summary']['primary']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
