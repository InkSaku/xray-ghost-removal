"""Plateau edge attenuation (Medge) on the 10.30 RAW batch.

Design fixed in docs/plans/2026-10-31-raw-plateau-edge.md before implementation.
Plateau ghost is G scaled by (1 - kappa * E_lambda), where E_lambda is the block
mean of exp(-d/lambda) over plateau pixels and d is each plateau pixel's distance
to the nearest non-plateau pixel. d is a regressor computed from the previous
frame only; it never excludes a pixel or block.
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
from scipy.ndimage import distance_transform_edt

from xray_ghost_removal.common.paths import require_current_m1_output
from xray_ghost_removal.common.provenance import clean_json, package_sources
from xray_ghost_removal.common.raw import read_raw
from xray_ghost_removal.common.spatial import block_mean
from xray_ghost_removal.m1.config import PRIMARY_BLOCK, TRIM_PERCENTILES
from xray_ghost_removal.research.raw_cohort import (
    CLEAN_DARK_COUNTERS, PLATEAU, Frame, load_frames, plateau_strata, prediction_metrics,
    short_directory, training_indices,
)
from xray_ghost_removal.research.raw_split import (
    SHOWCASE, PairData, build_data, fit_split, learn_map, predict_split,
)

DESIGN = 'docs/plans/2026-10-31-raw-plateau-edge.md'
LAMBDAS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
MAX_ITERATIONS = 300
KAPPA_TOLERANCE = 1e-4
REPLACE_MAX_MEDIAN_RATIO = 0.95
REPLACE_MIN_PAIR_FRACTION = 0.60
BOUNDARY_MAX_RATIO = 0.90
STRATUM_TOLERANCE = 1.01
IDENTIFIABLE_MIN_FRACTION = 0.90
BINS = 10


def edge_features(pixels: np.ndarray, block: int = PRIMARY_BLOCK, neighbor: bool = False):
    """E_lambda per block and the mean plateau-pixel distance per block (NaN without plateau).

    With neighbor=True also returns F_lambda: the block mean over plateau pixels of
    exp(-d/lambda) * v, where v is the value of the nearest non-plateau pixel / 55000.
    """
    plateau = pixels == PLATEAU
    if neighbor:
        distance, indices = distance_transform_edt(plateau, return_indices=True)
        brightness = pixels[indices[0], indices[1]].astype(np.float64) / PLATEAU
    else:
        distance = distance_transform_edt(plateau)
    fraction = block_mean(plateau.astype(np.float64), block)
    over_plateau = lambda v: np.where(fraction > 0, block_mean(np.where(plateau, v, 0.0), block) / fraction, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        decay = {lam: np.exp(-distance / lam) for lam in LAMBDAS}
        features = {lam: over_plateau(decay[lam]) for lam in LAMBDAS}
        mean_distance = np.where(fraction > 0, block_mean(np.where(plateau, distance, 0.0), block) / fraction, np.nan)
        if neighbor:
            weighted = {lam: over_plateau(decay[lam] * brightness) for lam in LAMBDAS}
            return features, weighted, mean_distance
    return features, mean_distance


def predict_attenuated(fit: dict, d: PairData, features: list[np.ndarray]) -> np.ndarray:
    """s*G*(1 - sum_j kappa_j*feature_j) + (1-s)*(a*Xu + b)."""
    attenuation = 1.0 - sum(k * f for k, f in zip(fit['kappas'], features))
    unclipped = fit['a'] * np.nan_to_num(d.xu) + fit['b']
    return d.s * fit['G'] * attenuation + (1.0 - d.s) * unclipped


def predict_edge(fit: dict, d: PairData, e: np.ndarray) -> np.ndarray:
    return predict_attenuated({**fit, 'kappas': [fit['kappa']]}, d, [e])


def fit_attenuation(train: list[PairData], features: list[list[np.ndarray]], base: dict) -> dict:
    """Alternate G (weighted map) and the kappas (least squares), starting at all kappas = 0.

    G and the kappas are strongly coupled, so convergence is slow; iterate until no
    kappa moves more than KAPPA_TOLERANCE.
    """
    kappas, g = np.zeros(len(features[0])), base['G']
    targets = [d.y - (1.0 - d.s) * (base['a'] * np.nan_to_num(d.xu) + base['b']) for d in train]
    multipliers = lambda k: [1.0 - sum(kj * fj for kj, fj in zip(k, fs)) for fs in features]
    for iteration in range(1, MAX_ITERATIONS + 1):
        g, _, _ = learn_map([d.y for d in train], [d.s for d in train], multipliers(kappas))
        q = np.column_stack([np.concatenate([(-d.s * g * fs[j])[d.s > 0] for d, fs in zip(train, features)])
                             for j in range(len(kappas))])
        r = np.concatenate([(t - d.s * g)[d.s > 0] for d, t in zip(train, targets)])
        previous = kappas
        kappas, *_ = np.linalg.lstsq(q, r, rcond=None)
        if np.max(np.abs(kappas - previous)) < KAPPA_TOLERANCE:
            break
    g, _, _ = learn_map([d.y for d in train], [d.s for d in train], multipliers(kappas))
    fit = {'G': g, 'kappas': [float(k) for k in kappas], 'a': base['a'], 'b': base['b'],
           'iterations': iteration, 'converged': bool(np.max(np.abs(kappas - previous)) < KAPPA_TOLERANCE)}
    fit['sse'] = float(sum(np.sum(((d.y - predict_attenuated(fit, d, fs)) ** 2)[d.s > 0])
                           for d, fs in zip(train, features)))
    return fit


def fit_edge_lambda(train: list[PairData], es: list[np.ndarray], base: dict) -> dict:
    fit = fit_attenuation(train, [[e] for e in es], base)
    return {**fit, 'kappa': fit['kappas'][0]}


def fit_edge(train: list[PairData], es: list[dict[float, np.ndarray]]) -> dict:
    base = fit_split(train)
    fits = {lam: fit_edge_lambda(train, [e[lam] for e in es], base) for lam in LAMBDAS}
    lam = min(fits, key=lambda k: fits[k]['sse'])
    best = {**fits[lam], 'lambda': lam, 'at_grid_edge': lam in (LAMBDAS[0], LAMBDAS[-1])}
    return {'edge': best, 'split': base}


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def decide(rows: list[dict]) -> dict:
    ratio = np.array([r['ratio_vs_msplit'] for r in rows])
    better = int(np.sum(ratio < 1.0))
    needed = math.ceil(REPLACE_MIN_PAIR_FRACTION * len(rows))
    strata = {n: {'medge': _median(r['s1']['medge'][f'rmse_{n}'] for r in rows),
                  'msplit': _median(r['s1']['msplit'][f'rmse_{n}'] for r in rows)}
              for n in ('plateau', 'boundary', 'unclipped')}
    fits_needed = math.ceil(IDENTIFIABLE_MIN_FRACTION * len(rows))
    kappa_ok = sum(0 < r['fit']['kappa'] <= 1 for r in rows)
    lambda_ok = sum(not r['fit']['at_grid_edge'] for r in rows)
    checks = {
        'median_ratio': bool(np.median(ratio) <= REPLACE_MAX_MEDIAN_RATIO),
        'pairs_better': bool(better >= needed),
        'boundary_improved': bool(strata['boundary']['medge'] <= BOUNDARY_MAX_RATIO * strata['boundary']['msplit']),
        'plateau_not_worse': bool(strata['plateau']['medge'] <= STRATUM_TOLERANCE * strata['plateau']['msplit']),
        'unclipped_not_worse': bool(strata['unclipped']['medge'] <= STRATUM_TOLERANCE * strata['unclipped']['msplit']),
        'kappa_in_range': bool(kappa_ok >= fits_needed),
        'lambda_identifiable': bool(lambda_ok >= fits_needed),
    }
    passed = all(checks.values())
    return {'pairs': len(rows), 'median_rmse_ratio_vs_msplit': float(np.median(ratio)), 'pairs_better': better,
            'pairs_required': needed, 'kappa_in_range': kappa_ok, 'lambda_interior': lambda_ok,
            'fits_required': fits_needed, 'median_stratum_rmse': strata, 'checks': checks, 'passed': passed,
            'decision': 'medge_replaces_msplit' if passed else 'msplit_retained'}


def _binned(x: np.ndarray, residuals: dict[str, np.ndarray]) -> list[dict]:
    edges = np.percentile(x, np.linspace(0, 100, BINS + 1))
    which = np.clip(np.digitize(x, edges[1:-1]), 0, BINS - 1)
    return [{'x_mean': float(x[which == k].mean()), 'blocks': int((which == k).sum()),
             **{f'mean_residual_{m}': float(v[which == k].mean()) for m, v in residuals.items()}}
            for k in range(BINS)]


def analyze(frames: list[Frame], data_dir: Path, shape, dtype) -> tuple[dict, dict]:
    pairs, data = build_data(frames)
    groups = [p.directory for p in pairs]
    es, distances = [], []
    for p in pairs:
        pixels, _ = read_raw(data_dir / p.high.file, shape, dtype)
        e, dist = edge_features(pixels)
        es.append(e)
        distances.append(dist)
    rows, showcase = [], {}
    pools = {'boundary_x': [], 'plateau_d': [], 'boundary_r': {'msplit': [], 'medge': []},
             'plateau_r': {'msplit': [], 'medge': []}}
    for i, (pair, d) in enumerate(zip(pairs, data)):
        train = training_indices(groups, i, 'pair')
        fit = fit_edge([data[j] for j in train], [es[j] for j in train])
        edge = fit['edge']
        preds = {'msplit': predict_split(fit['split'], d), 'medge': predict_edge(edge, d, es[i][edge['lambda']])}
        strata = plateau_strata(d.s)
        row = {'pair': pair.label, 'directory': pair.directory, 'filename_gap_seconds': pair.gap_seconds,
               'flags': pair.flags,
               'fit': {'kappa': edge['kappa'], 'lambda': edge['lambda'], 'at_grid_edge': edge['at_grid_edge'],
                       'iterations': edge['iterations'], 'converged': edge['converged']},
               's1': {m: prediction_metrics(d.y, p, strata) for m, p in preds.items()}}
        row['ratio_vs_msplit'] = row['s1']['medge']['rmse'] / row['s1']['msplit']['rmse']
        rows.append(row)
        pools['boundary_x'].append(d.xu[strata['boundary']])
        pools['plateau_d'].append(distances[i][strata['plateau']])
        for m, p in preds.items():
            pools['boundary_r'][m].append((d.y - p)[strata['boundary']])
            pools['plateau_r'][m].append((d.y - p)[strata['plateau']])
        if pair.label in SHOWCASE:
            showcase[pair.label] = {m: d.y - p for m, p in preds.items()}
    cat = lambda v: np.concatenate(v)
    directories = sorted(set(groups))
    summary = {
        'primary': decide(rows),
        'sensitivity_without_flagged': decide([r for r in rows if not r['flags']]),
        'selected': {g: {'kappa_median': _median(r['fit']['kappa'] for r in rows if r['directory'] == g),
                         'lambda_values': sorted({r['fit']['lambda'] for r in rows if r['directory'] == g}),
                         'lambda_median': _median(r['fit']['lambda'] for r in rows if r['directory'] == g)}
                     for g in directories},
        'medians_all_pairs': {f's1_{k}_{m}': _median(r['s1'][m][k] for r in rows)
                              for m in ('msplit', 'medge') for k in ('rmse', 'r2')},
        'selected_fits_converged': sum(r['fit']['converged'] for r in rows),
        'selected_fit_iterations_max': max(r['fit']['iterations'] for r in rows),
        'boundary_residual_by_xu': _binned(cat(pools['boundary_x']), {m: cat(v) for m, v in pools['boundary_r'].items()}),
        'plateau_residual_by_mean_distance_px': _binned(cat(pools['plateau_d']),
                                                        {m: cat(v) for m, v in pools['plateau_r'].items()}),
    }
    return {'summary': summary, 'pairs': rows}, {'rows': rows, 'summary': summary, 'showcase': showcase}


def plot(figure: dict, path: Path) -> None:
    rows, summary, showcase = figure['rows'], figure['summary'], figure['showcase']
    fig, ax = plt.subplots(2, 4, figsize=(24, 11))
    directories = sorted({r['directory'] for r in rows})
    for g in directories:
        sub = [r for r in rows if r['directory'] == g]
        ax[0, 0].scatter(range(len(sub)), [r['ratio_vs_msplit'] for r in sub], s=16, label=short_directory(g))
    ax[0, 0].axhline(REPLACE_MAX_MEDIAN_RATIO, color='k', lw=.8, ls='--')
    ax[0, 0].axhline(1.0, color='k', lw=.8, ls=':')
    ax[0, 0].set(title='S1 RMSE: Medge / Msplit (dashed = 0.95 gate)', xlabel='pair order within directory')
    ax[0, 0].legend(fontsize=8)
    names = ('plateau', 'boundary', 'unclipped')
    for k, m in enumerate(('msplit', 'medge')):
        ax[0, 1].bar(np.arange(3) + (k - 0.5) * 0.38,
                     [summary['primary']['median_stratum_rmse'][n][m] for n in names], 0.38, label=m)
    ax[0, 1].set_xticks(range(3), names)
    ax[0, 1].set(title='median S1 RMSE by stratum (counts)')
    ax[0, 1].legend(fontsize=8)
    for a, key, xlabel in ((ax[0, 2], 'boundary_residual_by_xu', 'Xu of unclipped pixels in boundary blocks'),
                           (ax[0, 3], 'plateau_residual_by_mean_distance_px', 'mean plateau-pixel distance to edge (px)')):
        bins = summary[key]
        for m in ('msplit', 'medge'):
            a.plot([b['x_mean'] for b in bins], [b[f'mean_residual_{m}'] for b in bins], 'o-', label=m)
        a.axhline(0, color='k', lw=.8)
        a.set(title=f'mean residual vs {xlabel.split(" (")[0]}', xlabel=xlabel, ylabel='counts')
        a.legend(fontsize=8)
    ax[0, 3].set_xscale('log')
    col = 0
    for label in SHOWCASE:
        if label not in showcase:
            continue
        lim = np.percentile(np.abs(showcase[label]['msplit']), 99)
        for m in ('msplit', 'medge'):
            im = ax[1, col].imshow(showcase[label][m], cmap='coolwarm', vmin=-lim, vmax=lim)
            ax[1, col].set(title=f'{label} | {m} residual (shared scale)', xticks=[], yticks=[])
            fig.colorbar(im, ax=ax[1, col], fraction=.046)
            col += 1
    fig.suptitle('10.30 RAW | plateau edge attenuation Medge vs Msplit | leave-one-pair-out within directory')
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
            'medge': 's*G*(1 - kappa*E_lambda) + (1-s)*(a*Xu + b)',
            'e_lambda': 'block mean over plateau pixels of exp(-d/lambda), d = distance to nearest non-plateau pixel',
            'lambdas_px': list(LAMBDAS), 'max_iterations': MAX_ITERATIONS, 'kappa_tolerance': KAPPA_TOLERANCE, 'trim_percentiles': list(TRIM_PERCENTILES),
            'thresholds': {'replace_max_median_ratio': REPLACE_MAX_MEDIAN_RATIO,
                           'replace_min_pair_fraction': REPLACE_MIN_PAIR_FRACTION,
                           'boundary_max_ratio': BOUNDARY_MAX_RATIO, 'stratum_tolerance': STRATUM_TOLERANCE,
                           'identifiable_min_fraction': IDENTIFIABLE_MIN_FRACTION},
        },
        'interpretation_notes': [
            'Edge attenuation does not distinguish near-threshold true exposure from spatial dilution.',
            'G remains an empirical plateau map; the cause of the 55000 plateau is unconfirmed.',
            'Filename gaps are unverified; no time term is modelled.',
        ],
        **result,
        'inputs': [{'file': f.file, 'sha256': f.sha256, 'high': f.high} for f in frames],
        'code': package_sources(),
    }
    (output_dir / 'results.json').write_text(
        json.dumps(clean_json(record), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'directory', 'flags', 'msplit_rmse', 'medge_rmse', 'ratio',
                         'msplit_boundary_rmse', 'medge_boundary_rmse', 'kappa', 'lambda_px'])
        for r in result['pairs']:
            s1 = r['s1']
            writer.writerow([r['pair'], r['directory'], ';'.join(r['flags']),
                             f"{s1['msplit']['rmse']:.2f}", f"{s1['medge']['rmse']:.2f}", f"{r['ratio_vs_msplit']:.4f}",
                             f"{s1['msplit']['rmse_boundary']:.2f}", f"{s1['medge']['rmse_boundary']:.2f}",
                             f"{r['fit']['kappa']:.4f}", r['fit']['lambda']])
    plot(figure, output_dir / 'diagnostics.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('10.30例图'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/raw_edge_v1'))
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
