"""Ghost laws on unclipped blocks of the 10.30 RAW batch: position and plateau proximity.

Design fixed in docs/plans/2026-10-31-raw-unclipped-laws.md before implementation.
Only blocks with no plateau pixel (s = 0) are analysed, where the previous frame
records exposure without the 55000 clip. Y and every regressor are centered per
pair, so per-pair offsets such as clean-dark drift never enter a law. The blurred
plateau fraction is a regressor only; it never removes a block from analysis.
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
from scipy.ndimage import distance_transform_edt, gaussian_filter

from xray_ghost_removal.common.paths import require_current_m1_output
from xray_ghost_removal.common.provenance import clean_json, package_sources
from xray_ghost_removal.m1.config import PRIMARY_BLOCK, TRIM_PERCENTILES
from xray_ghost_removal.m1.fitting import _training_keep
from xray_ghost_removal.research.raw_cohort import (
    CLEAN_DARK_COUNTERS, PLATEAU, Frame, load_frames, short_directory, training_indices,
)
from xray_ghost_removal.research.raw_split import build_data, fit_split

DESIGN = 'docs/plans/2026-10-31-raw-unclipped-laws.md'
SIGMAS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
MODELS = ('mconst', 'mmap', 'mspill')
GATE_MAX_MEDIAN_RATIO = 0.95
GATE_MIN_PAIR_FRACTION = 0.60
SPILL_MIN_FIT_FRACTION = 0.90
SHAPE_AGREE, SHAPE_DISAGREE = 0.7, 0.3
DISTANCE_BINS = ((1, 2), (2, 4), (4, 8), (8, math.inf))


@dataclass
class Unclipped:
    label: str
    directory: str
    gap_seconds: float
    flags: list[str]
    rows: np.ndarray       # normalized to [-1, 1]
    cols: np.ndarray
    xu: np.ndarray
    y: np.ndarray
    distance: np.ndarray   # blocks to the nearest block containing any plateau pixel
    blur: dict[float, np.ndarray]


def position_basis(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(rows), rows, cols, rows ** 2, rows * cols, cols ** 2])


def _center(values: np.ndarray) -> np.ndarray:
    return values - values.mean(axis=0)


def features(model: str, u: Unclipped, sigma: float | None = None) -> np.ndarray:
    """Per-pair centered regressors."""
    if model == 'mconst':
        raw = u.xu[:, None]
    elif model == 'mmap':
        raw = position_basis(u.rows, u.cols) * u.xu[:, None]
    elif model == 'mspill':
        raw = np.column_stack([u.xu, u.blur[sigma]])
    else:
        raise ValueError(model)
    return _center(raw)


def fit_centered(model: str, train: list[Unclipped], sigma: float | None = None) -> dict:
    x = np.vstack([features(model, u, sigma) for u in train])
    y = np.concatenate([_center(u.y) for u in train])
    xu = np.concatenate([u.xu for u in train])
    keep, _ = _training_keep(xu, y, np.ones(y.shape, dtype=bool), TRIM_PERCENTILES)
    beta, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
    sse = float(np.sum((y[keep] - x[keep] @ beta) ** 2))
    return {'model': model, 'sigma': sigma, 'beta': beta, 'sse': sse}


def fit_model(model: str, train: list[Unclipped]) -> dict:
    if model != 'mspill':
        return fit_centered(model, train)
    fits = [fit_centered(model, train, s) for s in SIGMAS]
    best = min(fits, key=lambda f: f['sse'])
    best['at_grid_edge'] = best['sigma'] in (SIGMAS[0], SIGMAS[-1])
    return best


def residual(fit: dict, u: Unclipped) -> np.ndarray:
    return _center(u.y) - features(fit['model'], u, fit['sigma']) @ fit['beta']


def build_unclipped(frames: list[Frame]) -> list[Unclipped]:
    pairs, data = build_data(frames)
    out = []
    for pair, d in zip(pairs, data):
        mask = d.s == 0
        h, w = d.s.shape
        rr, cc = np.indices((h, w))
        distance = distance_transform_edt(mask)
        out.append(Unclipped(
            label=pair.label, directory=pair.directory, gap_seconds=pair.gap_seconds, flags=pair.flags,
            rows=(2.0 * rr[mask] / (h - 1) - 1.0), cols=(2.0 * cc[mask] / (w - 1) - 1.0),
            xu=d.xu[mask], y=d.y[mask], distance=distance[mask],
            blur={s: gaussian_filter(d.s, sigma=s, mode='nearest')[mask] for s in SIGMAS}))
    return out


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def _gate(rows: list[dict], model: str) -> dict:
    ratio = np.array([r['rmse'][model] / r['rmse']['mconst'] for r in rows])
    better = int(np.sum(ratio < 1.0))
    needed = math.ceil(GATE_MIN_PAIR_FRACTION * len(rows))
    out = {'median_ratio_vs_mconst': float(np.median(ratio)), 'pairs_better': better,
           'pairs_required': needed,
           'checks': {'median_ratio': bool(np.median(ratio) <= GATE_MAX_MEDIAN_RATIO),
                      'pairs_better': bool(better >= needed)}}
    if model == 'mspill':
        fits_needed = math.ceil(SPILL_MIN_FIT_FRACTION * len(rows))
        positive = sum(r['fits']['mspill']['gamma'] > 0 for r in rows)
        interior = sum(not r['fits']['mspill']['at_grid_edge'] for r in rows)
        out.update({'gamma_positive': positive, 'sigma_interior': interior, 'fits_required': fits_needed})
        out['checks'].update({'gamma_positive': positive >= fits_needed,
                              'sigma_identifiable': interior >= fits_needed})
    out['passed'] = all(out['checks'].values())
    return out


def decide(rows: list[dict]) -> dict:
    return {'pairs': len(rows), 'position_law': _gate(rows, 'mmap'), 'proximity_law': _gate(rows, 'mspill')}


def bridge(units: list[Unclipped], frames: list[Frame], directory: str, grid: tuple[int, int]) -> dict:
    """Uncentered Mmap on all pairs of a directory, compared with its plateau map G."""
    subset = [u for u in units if u.directory == directory]
    x = np.vstack([np.column_stack([position_basis(u.rows, u.cols) * u.xu[:, None], np.ones(u.xu.size)])
                   for u in subset])
    y = np.concatenate([u.y for u in subset])
    xu = np.concatenate([u.xu for u in subset])
    keep, _ = _training_keep(xu, y, np.ones(y.shape, dtype=bool), TRIM_PERCENTILES)
    beta, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
    h, w = grid
    rr, cc = np.indices(grid)
    slope = (position_basis(2.0 * rr.ravel() / (h - 1) - 1.0, 2.0 * cc.ravel() / (w - 1) - 1.0)
             @ beta[:6]).reshape(grid)
    pairs, data = build_data(frames)
    g = fit_split([d for p, d in zip(pairs, data) if p.directory == directory])['G']
    corr = float(np.corrcoef(slope.ravel(), g.ravel())[0, 1])
    positive = slope > 0
    exposure = np.where(positive, (g - beta[6]) / np.where(positive, slope, 1.0), np.nan)
    multiple = exposure[positive] / PLATEAU
    shape = 'agree' if corr >= SHAPE_AGREE else 'disagree' if corr <= SHAPE_DISAGREE else 'inconclusive'
    return {'slope_map_vs_G_corr': corr, 'shape_verdict': shape, 'intercept': float(beta[6]),
            'slope_p1_p50_p99': np.percentile(slope, [1, 50, 99]).tolist(),
            'slope_positive_fraction': float(positive.mean()),
            'implied_exposure_multiple_of_55000_p5_p50_p95':
                np.percentile(multiple, [5, 50, 95]).tolist() if multiple.size else None,
            '_slope': slope, '_G': g}


def _spearman(a, b) -> float | None:
    if len(a) < 3:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def analyze(frames: list[Frame]) -> tuple[dict, dict]:
    units = build_unclipped(frames)
    groups = [u.directory for u in units]
    cache, rows = {}, []
    bins = {m: {b: [] for b in DISTANCE_BINS} for m in MODELS}
    for i, u in enumerate(units):
        train = tuple(training_indices(groups, i, 'pair'))
        if train not in cache:
            subset = [units[j] for j in train]
            cache[train] = {m: fit_model(m, subset) for m in MODELS}
        fits = cache[train]
        res = {m: residual(fits[m], u) for m in MODELS}
        own = fit_centered('mconst', [u])
        rows.append({
            'pair': u.label, 'directory': u.directory, 'filename_gap_seconds': u.gap_seconds,
            'flags': u.flags, 'unclipped_blocks': int(u.y.size),
            'own_centered_slope': float(own['beta'][0]),
            'rmse': {m: float(np.sqrt(np.mean(res[m] ** 2))) for m in MODELS},
            'centered_y_std': float(np.std(u.y)),
            'fits': {'mconst': {'a': float(fits['mconst']['beta'][0])},
                     'mmap': {'beta': fits['mmap']['beta'].tolist()},
                     'mspill': {'a': float(fits['mspill']['beta'][0]), 'gamma': float(fits['mspill']['beta'][1]),
                                'sigma': fits['mspill']['sigma'], 'at_grid_edge': fits['mspill']['at_grid_edge']}},
        })
        for m in MODELS:
            for lo, hi in DISTANCE_BINS:
                sel = (u.distance >= lo) & (u.distance < hi)
                bins[m][(lo, hi)].append(res[m][sel])

    directories = sorted(set(groups))
    grid = frames[0].block.shape
    bridges = {g: bridge(units, frames, g, grid) for g in directories}
    intensity = {g: {'median_own_slope': _median(r['own_centered_slope'] for r in rows if r['directory'] == g),
                     'own_slope_iqr': np.percentile([r['own_centered_slope'] for r in rows if r['directory'] == g],
                                                    [25, 75]).tolist(),
                     'spearman_gap_vs_slope': _spearman(
                         [r['filename_gap_seconds'] for r in rows if r['directory'] == g],
                         [r['own_centered_slope'] for r in rows if r['directory'] == g])}
                 for g in directories}
    distance_residuals = {m: [{'distance_blocks': f'{lo}-{hi}', 'blocks': int(sum(v.size for v in vals)),
                               'mean_residual': float(np.concatenate(vals).mean())}
                              for (lo, hi), vals in bins[m].items()] for m in MODELS}
    summary = {
        'primary': decide(rows),
        'sensitivity_without_flagged': decide([r for r in rows if not r['flags']]),
        'medians_all_pairs': {**{f'centered_rmse_{m}': _median(r['rmse'][m] for r in rows) for m in MODELS},
                              'centered_y_std': _median(r['centered_y_std'] for r in rows),
                              'mspill_sigma': _median(r['fits']['mspill']['sigma'] for r in rows),
                              'mspill_gamma': _median(r['fits']['mspill']['gamma'] for r in rows),
                              'mconst_a': _median(r['fits']['mconst']['a'] for r in rows)},
        'intensity_by_directory': intensity,
        'residual_by_distance': distance_residuals,
        'bridge': {g: {k: v for k, v in b.items() if not k.startswith('_')} for g, b in bridges.items()},
    }
    return {'summary': summary, 'pairs': rows}, {'rows': rows, 'bridges': bridges,
                                                 'distance': distance_residuals}


def plot(figure: dict, path: Path) -> None:
    rows, bridges = figure['rows'], figure['bridges']
    directories = sorted(bridges)
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    for k, m in enumerate(('mmap', 'mspill')):
        ax[0, 0].scatter(np.arange(len(rows)) + 0.2 * k, [r['rmse'][m] / r['rmse']['mconst'] for r in rows],
                         s=14, label=f'{m} / mconst')
    ax[0, 0].axhline(GATE_MAX_MEDIAN_RATIO, color='k', lw=.8, ls='--')
    ax[0, 0].axhline(1.0, color='k', lw=.8, ls=':')
    ax[0, 0].set(title='centered RMSE ratio, leave-one-pair-out (dashed = 0.95 gate)', xlabel='pair index')
    ax[0, 0].legend(fontsize=8)
    labels = [d['distance_blocks'] for d in figure['distance']['mconst']]
    for k, m in enumerate(MODELS):
        ax[0, 1].bar(np.arange(len(labels)) + (k - 1) * 0.27,
                     [d['mean_residual'] for d in figure['distance'][m]], 0.27, label=m)
    ax[0, 1].axhline(0, color='k', lw=.8)
    ax[0, 1].set_xticks(range(len(labels)), labels)
    ax[0, 1].set(title='mean centered residual by distance to plateau (blocks)', ylabel='counts')
    ax[0, 1].legend(fontsize=8)
    for g in directories:
        sub = [r for r in rows if r['directory'] == g]
        ax[0, 2].scatter([r['filename_gap_seconds'] for r in sub], [r['own_centered_slope'] for r in sub],
                         s=16, label=short_directory(g))
    ax[0, 2].set(title='per-pair centered slope vs filename gap (time unverified)', xscale='log',
                 xlabel='seconds', ylabel='slope')
    ax[0, 2].legend(fontsize=8)
    for k, g in enumerate(directories):
        b = bridges[g]
        im = ax[1, k].imshow(b['_slope'], cmap='viridis')
        ax[1, k].set(title=f'slope map a(row,col) | {short_directory(g)}\ncorr with G = {b["slope_map_vs_G_corr"]:.2f} '
                           f'({b["shape_verdict"]})', xticks=[], yticks=[])
        fig.colorbar(im, ax=ax[1, k], fraction=.046)
    fig.suptitle('10.30 RAW | unclipped blocks only | per-pair centered | position and proximity laws')
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
            'block': PRIMARY_BLOCK, 'plateau_value': PLATEAU, 'unit': 'blocks with s = 0, centered per pair',
            'clean_dark_counters': [f'{c:08X}' for c in CLEAN_DARK_COUNTERS],
            'mconst': 'a*Xu', 'mmap': 'sum_k beta_k*phi_k(row,col)*Xu, phi = 1,r,c,r^2,rc,c^2',
            'mspill': 'a*Xu + gamma*gaussian(s, sigma)', 'sigmas_blocks': list(SIGMAS),
            'trim_percentiles': list(TRIM_PERCENTILES),
            'thresholds': {'max_median_ratio': GATE_MAX_MEDIAN_RATIO, 'min_pair_fraction': GATE_MIN_PAIR_FRACTION,
                           'spill_min_fit_fraction': SPILL_MIN_FIT_FRACTION,
                           'shape_agree': SHAPE_AGREE, 'shape_disagree': SHAPE_DISAGREE},
        },
        'interpretation_notes': [
            'Centering per pair removes offsets, so absolute ghost level is not modelled here.',
            'The implied exposure assumes a linear ghost and extrapolates the unclipped slope map onto the plateau.',
            'Filename gaps are unverified; no decay constant is fitted.',
        ],
        **result,
        'inputs': [{'file': f.file, 'sha256': f.sha256, 'high': f.high} for f in frames],
        'code': package_sources(),
    }
    (output_dir / 'results.json').write_text(
        json.dumps(clean_json(record), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output_dir / 'pairs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pair', 'directory', 'filename_gap_s', 'flags', 'unclipped_blocks', 'own_slope',
                         'rmse_mconst', 'rmse_mmap', 'rmse_mspill', 'spill_sigma', 'spill_gamma'])
        for r in result['pairs']:
            writer.writerow([r['pair'], r['directory'], f"{r['filename_gap_seconds']:.3f}", ';'.join(r['flags']),
                             r['unclipped_blocks'], f"{r['own_centered_slope']:.5f}",
                             *(f"{r['rmse'][m]:.2f}" for m in MODELS),
                             r['fits']['mspill']['sigma'], f"{r['fits']['mspill']['gamma']:.2f}"])
    plot(figure, output_dir / 'diagnostics.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('10.30例图'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/raw_unclipped_v1'))
    parser.add_argument('--shape', type=int, nargs=2, default=(3072, 3072), metavar=('ROWS', 'COLS'))
    parser.add_argument('--dtype', default='<u2', choices=('<u2', '>u2'))
    args = parser.parse_args()
    frames = load_frames(args.data_dir, tuple(args.shape), args.dtype)
    result, figure = analyze(frames)
    write_outputs(result, figure, frames, args.output_dir, args.data_dir, tuple(args.shape), args.dtype)
    print(json.dumps(clean_json(result['summary']['primary']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
