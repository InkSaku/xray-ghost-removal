"""Edge attenuation that depends on the brightness across the edge (Mcontrast), 10.30 RAW.

Design fixed in docs/plans/2026-10-31-raw-edge-contrast.md before implementation.
Plateau ghost is G*(1 - kappa0*E_lambda - kappa1*F_lambda), where F_lambda weights
each plateau pixel's edge decay by the value of its nearest non-plateau pixel / 55000.
The candidate was suggested by Medge diagnostics on the same 70 pairs; leave-one-
pair-out evaluation guards parameters, not the choice of candidate.
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
from xray_ghost_removal.common.raw import read_raw
from xray_ghost_removal.m1.config import PRIMARY_BLOCK, TRIM_PERCENTILES
from xray_ghost_removal.research.raw_cohort import (
    CLEAN_DARK_COUNTERS, PLATEAU, Frame, load_frames, plateau_strata, prediction_metrics,
    short_directory, training_indices,
)
from xray_ghost_removal.research.raw_edge import (
    LAMBDAS, MAX_ITERATIONS, KAPPA_TOLERANCE, _binned, _median, edge_features, fit_attenuation, fit_edge,
    predict_attenuated, predict_edge,
)
from xray_ghost_removal.research.raw_split import SHOWCASE, PairData, build_data, fit_split

DESIGN = 'docs/plans/2026-10-31-raw-edge-contrast.md'
REPLACE_MAX_MEDIAN_RATIO = 0.95
REPLACE_MIN_PAIR_FRACTION = 0.60
BOUNDARY_MAX_RATIO = 0.90
STRATUM_TOLERANCE = 1.01
DIRECTION_MIN_FRACTION = 0.90
NEAR_EDGE_LAMBDA, NEAR_EDGE_MIN_E = 8.0, 0.05


def fit_contrast(train: list[PairData], es: list[dict], fs: list[dict]) -> dict:
    base = fit_split(train)
    fits = {lam: fit_attenuation(train, [[e[lam], f[lam]] for e, f in zip(es, fs)], base) for lam in LAMBDAS}
    lam = min(fits, key=lambda k: fits[k]['sse'])
    return {**fits[lam], 'lambda': lam, 'at_grid_edge': lam in (LAMBDAS[0], LAMBDAS[-1])}


def decide(rows: list[dict]) -> dict:
    ratio = np.array([r['ratio_vs_medge'] for r in rows])
    better = int(np.sum(ratio < 1.0))
    needed = math.ceil(REPLACE_MIN_PAIR_FRACTION * len(rows))
    strata = {n: {'mcontrast': _median(r['s1']['mcontrast'][f'rmse_{n}'] for r in rows),
                  'medge': _median(r['s1']['medge'][f'rmse_{n}'] for r in rows)}
              for n in ('plateau', 'boundary', 'unclipped')}
    fits_needed = math.ceil(DIRECTION_MIN_FRACTION * len(rows))
    k1_positive = sum(r['fit']['kappa1'] > 0 for r in rows)
    bright_ok = sum(r['fit']['kappa0'] + r['fit']['kappa1'] <= 1 for r in rows)
    lam_ok = sum(not r['fit']['at_grid_edge'] for r in rows)
    checks = {
        'median_ratio': bool(np.median(ratio) <= REPLACE_MAX_MEDIAN_RATIO),
        'pairs_better': bool(better >= needed),
        'boundary_improved': bool(strata['boundary']['mcontrast'] <= BOUNDARY_MAX_RATIO * strata['boundary']['medge']),
        'plateau_not_worse': bool(strata['plateau']['mcontrast'] <= STRATUM_TOLERANCE * strata['plateau']['medge']),
        'unclipped_not_worse': bool(strata['unclipped']['mcontrast'] <= STRATUM_TOLERANCE * strata['unclipped']['medge']),
        'kappa1_positive': bool(k1_positive >= fits_needed),
        'bright_attenuation_at_most_one': bool(bright_ok >= fits_needed),
        'lambda_identifiable': bool(lam_ok >= fits_needed),
    }
    passed = all(checks.values())
    return {'pairs': len(rows), 'median_rmse_ratio_vs_medge': float(np.median(ratio)), 'pairs_better': better,
            'pairs_required': needed, 'kappa1_positive': k1_positive, 'bright_attenuation_ok': bright_ok,
            'lambda_interior': lam_ok, 'fits_required': fits_needed, 'median_stratum_rmse': strata,
            'checks': checks, 'passed': passed,
            'decision': 'mcontrast_replaces_medge' if passed else 'medge_retained'}


def analyze(frames: list[Frame], data_dir: Path, shape, dtype) -> tuple[dict, dict]:
    pairs, data = build_data(frames)
    groups = [p.directory for p in pairs]
    es, fs = [], []
    for p in pairs:
        pixels, _ = read_raw(data_dir / p.high.file, shape, dtype)
        e, f, _ = edge_features(pixels, neighbor=True)
        es.append(e)
        fs.append(f)
    rows, showcase = [], {}
    pools = {'bx': [], 'pv': [], 'br': {'medge': [], 'mcontrast': []}, 'pr': {'medge': [], 'mcontrast': []}}
    for i, (pair, d) in enumerate(zip(pairs, data)):
        train = training_indices(groups, i, 'pair')
        sub, sub_e, sub_f = [data[j] for j in train], [es[j] for j in train], [fs[j] for j in train]
        medge = fit_edge(sub, sub_e)['edge']
        contrast = fit_contrast(sub, sub_e, sub_f)
        lam = contrast['lambda']
        preds = {'medge': predict_edge(medge, d, es[i][medge['lambda']]),
                 'mcontrast': predict_attenuated(contrast, d, [es[i][lam], fs[i][lam]])}
        strata = plateau_strata(d.s)
        row = {'pair': pair.label, 'directory': pair.directory, 'flags': pair.flags,
               'fit': {'kappa0': contrast['kappas'][0], 'kappa1': contrast['kappas'][1], 'lambda': lam,
                       'at_grid_edge': contrast['at_grid_edge'], 'iterations': contrast['iterations'],
                       'converged': contrast['converged'],
                       'medge_kappa': medge['kappa'], 'medge_lambda': medge['lambda']},
               's1': {m: prediction_metrics(d.y, p, strata) for m, p in preds.items()}}
        row['ratio_vs_medge'] = row['s1']['mcontrast']['rmse'] / row['s1']['medge']['rmse']
        rows.append(row)
        near = strata['plateau'] & (es[i][NEAR_EDGE_LAMBDA] > NEAR_EDGE_MIN_E)
        with np.errstate(invalid='ignore', divide='ignore'):
            v = fs[i][NEAR_EDGE_LAMBDA] / es[i][NEAR_EDGE_LAMBDA]
        pools['bx'].append(d.xu[strata['boundary']])
        pools['pv'].append(v[near])
        for m, p in preds.items():
            pools['br'][m].append((d.y - p)[strata['boundary']])
            pools['pr'][m].append((d.y - p)[near])
        if pair.label in SHOWCASE:
            showcase[pair.label] = {m: d.y - p for m, p in preds.items()}
    cat = np.concatenate
    directories = sorted(set(groups))
    summary = {
        'primary': decide(rows),
        'sensitivity_without_flagged': decide([r for r in rows if not r['flags']]),
        'selected': {g: {k: _median(r['fit'][k] for r in rows if r['directory'] == g)
                         for k in ('kappa0', 'kappa1', 'lambda')} for g in directories},
        'medians_all_pairs': {f's1_{k}_{m}': _median(r['s1'][m][k] for r in rows)
                              for m in ('medge', 'mcontrast') for k in ('rmse', 'r2')},
        'fits_converged': sum(r['fit']['converged'] for r in rows),
        'fit_iterations_max': max(r['fit']['iterations'] for r in rows),
        'boundary_residual_by_xu': _binned(cat(pools['bx']), {m: cat(v) for m, v in pools['br'].items()}),
        'near_edge_plateau_residual_by_neighbor_brightness': _binned(cat(pools['pv']),
                                                                     {m: cat(v) for m, v in pools['pr'].items()}),
    }
    return {'summary': summary, 'pairs': rows}, {'rows': rows, 'summary': summary, 'showcase': showcase}


def plot(figure: dict, path: Path) -> None:
    rows, summary, showcase = figure['rows'], figure['summary'], figure['showcase']
    fig, ax = plt.subplots(2, 4, figsize=(24, 11))
    for g in sorted({r['directory'] for r in rows}):
        sub = [r for r in rows if r['directory'] == g]
        ax[0, 0].scatter(range(len(sub)), [r['ratio_vs_medge'] for r in sub], s=16, label=short_directory(g))
    ax[0, 0].axhline(REPLACE_MAX_MEDIAN_RATIO, color='k', lw=.8, ls='--')
    ax[0, 0].axhline(1.0, color='k', lw=.8, ls=':')
    ax[0, 0].set(title='S1 RMSE: Mcontrast / Medge (dashed = 0.95 gate)', xlabel='pair order within directory')
    ax[0, 0].legend(fontsize=8)
    names = ('plateau', 'boundary', 'unclipped')
    for k, m in enumerate(('medge', 'mcontrast')):
        ax[0, 1].bar(np.arange(3) + (k - 0.5) * 0.38,
                     [summary['primary']['median_stratum_rmse'][n][m] for n in names], 0.38, label=m)
    ax[0, 1].set_xticks(range(3), names)
    ax[0, 1].set(title='median S1 RMSE by stratum (counts)')
    ax[0, 1].legend(fontsize=8)
    for a, key, xlabel in ((ax[0, 2], 'boundary_residual_by_xu', 'Xu of unclipped pixels in boundary blocks'),
                           (ax[0, 3], 'near_edge_plateau_residual_by_neighbor_brightness',
                            'neighbor brightness v (near-edge plateau blocks)')):
        for m in ('medge', 'mcontrast'):
            a.plot([b['x_mean'] for b in summary[key]], [b[f'mean_residual_{m}'] for b in summary[key]], 'o-', label=m)
        a.axhline(0, color='k', lw=.8)
        a.set(title=f'mean residual vs {xlabel.split(" (")[0]}', xlabel=xlabel, ylabel='counts')
        a.legend(fontsize=8)
    col = 0
    for label in SHOWCASE:
        if label not in showcase:
            continue
        lim = np.percentile(np.abs(showcase[label]['medge']), 99)
        for m in ('medge', 'mcontrast'):
            im = ax[1, col].imshow(showcase[label][m], cmap='coolwarm', vmin=-lim, vmax=lim)
            ax[1, col].set(title=f'{label} | {m} residual (shared scale)', xticks=[], yticks=[])
            fig.colorbar(im, ax=ax[1, col], fraction=.046)
            col += 1
    fig.suptitle('10.30 RAW | edge attenuation by neighbor brightness (Mcontrast) vs Medge | leave-one-pair-out')
    fig.tight_layout()
    fig.savefig(path, dpi=90)
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
            'mcontrast': 's*G*(1 - kappa0*E_lambda - kappa1*F_lambda) + (1-s)*(a*Xu + b)',
            'f_lambda': 'block mean over plateau pixels of exp(-d/lambda) * v, v = nearest non-plateau pixel / 55000',
            'lambdas_px': list(LAMBDAS), 'max_iterations': MAX_ITERATIONS, 'kappa_tolerance': KAPPA_TOLERANCE,
            'trim_percentiles': list(TRIM_PERCENTILES),
            'thresholds': {'replace_max_median_ratio': REPLACE_MAX_MEDIAN_RATIO,
                           'replace_min_pair_fraction': REPLACE_MIN_PAIR_FRACTION,
                           'boundary_max_ratio': BOUNDARY_MAX_RATIO, 'stratum_tolerance': STRATUM_TOLERANCE,
                           'direction_min_fraction': DIRECTION_MIN_FRACTION},
        },
        'interpretation_notes': [
            'The candidate was suggested by Medge diagnostics on the same 70 pairs; evidence is same-session exploratory.',
            'G remains an empirical plateau map; the cause of the 55000 plateau is unconfirmed.',
        ],
        **result,
        'inputs': [{'file': f.file, 'sha256': f.sha256, 'high': f.high} for f in frames],
        'code': package_sources(),
    }
    (output_dir / 'results.json').write_text(
        json.dumps(clean_json(record), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'directory', 'flags', 'medge_rmse', 'mcontrast_rmse', 'ratio',
                         'medge_boundary_rmse', 'mcontrast_boundary_rmse', 'kappa0', 'kappa1', 'lambda_px'])
        for r in result['pairs']:
            s1 = r['s1']
            writer.writerow([r['pair'], r['directory'], ';'.join(r['flags']),
                             f"{s1['medge']['rmse']:.2f}", f"{s1['mcontrast']['rmse']:.2f}", f"{r['ratio_vs_medge']:.4f}",
                             f"{s1['medge']['rmse_boundary']:.2f}", f"{s1['mcontrast']['rmse_boundary']:.2f}",
                             f"{r['fit']['kappa0']:.4f}", f"{r['fit']['kappa1']:.4f}", r['fit']['lambda']])
    plot(figure, output_dir / 'diagnostics.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('10.30例图'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/raw_edge_contrast_v1'))
    parser.add_argument('--shape', type=int, nargs=2, default=(3072, 3072), metavar=('ROWS', 'COLS'))
    parser.add_argument('--dtype', default='<u2', choices=('<u2', '>u2'))
    args = parser.parse_args()
    shape = tuple(args.shape)
    frames = load_frames(args.data_dir, shape, args.dtype)
    result, figure = analyze(frames, args.data_dir, shape, args.dtype)
    write_outputs(result, figure, frames, args.output_dir, args.data_dir, shape, args.dtype)
    print(json.dumps(clean_json(result['summary']['primary']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
