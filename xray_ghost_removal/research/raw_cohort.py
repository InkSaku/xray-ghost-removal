"""10.30 RAW cohort shared by the RAW model studies: frames, pairs, clean dark, scoring.

Frame roles are observed classes, not confirmed light/dark acquisitions. The clean
dark is predeclared by counter; filename times are unverified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from xray_ghost_removal.common.raw import parse_raw_name, read_raw
from xray_ghost_removal.common.spatial import block_mean
from xray_ghost_removal.m1.config import PRIMARY_BLOCK, TRIM_PERCENTILES
from xray_ghost_removal.m1.fitting import _fit_affine, _training_keep

PLATEAU = 55000
HIGH_FRAME_PLATEAU_FRACTION = 0.5
CLEAN_DARK_COUNTERS = (0x20000039, 0x200000C0)
LONG_GAP_SECONDS = 10.0


@dataclass
class Frame:
    file: str
    directory: str
    timestamp: datetime
    counter: int
    sha256: str
    block: np.ndarray = field(repr=False)
    plateau_fraction: np.ndarray = field(repr=False)
    # Mean of the pixels that are not on the plateau; NaN where a block is fully plateau.
    unclipped_mean: np.ndarray | None = field(default=None, repr=False)

    @property
    def high(self) -> bool:
        return float(np.mean(self.plateau_fraction)) > HIGH_FRAME_PLATEAU_FRACTION


@dataclass
class Pair:
    directory: str
    high: Frame
    low: Frame
    previous_counter: int | None
    flags: list[str]

    @property
    def gap_seconds(self) -> float:
        return (self.low.timestamp - self.high.timestamp).total_seconds()

    @property
    def label(self) -> str:
        return f'{self.high.counter:08X}->{self.low.counter:08X}'


def load_frames(data_dir: Path, shape: tuple[int, int], dtype: str) -> list[Frame]:
    files = sorted(data_dir.rglob('*.raw'))
    if not files:
        raise ValueError(f'No .raw files under {data_dir}')
    frames = []
    for path in files:
        pixels, digest = read_raw(path, shape, dtype)
        timestamp, counter = parse_raw_name(path)
        values = pixels.astype(np.float64)
        # Pixel-level threshold count per block; no morphology of any kind.
        plateau = pixels == PLATEAU
        fraction = block_mean(plateau.astype(np.float64), PRIMARY_BLOCK)
        unclipped_sum = block_mean(np.where(plateau, 0.0, values), PRIMARY_BLOCK)
        with np.errstate(invalid='ignore', divide='ignore'):
            unclipped = np.where(fraction < 1.0, unclipped_sum / (1.0 - fraction), np.nan)
        frames.append(Frame(
            file=path.relative_to(data_dir).as_posix(), directory=path.parent.name,
            timestamp=timestamp, counter=counter, sha256=digest,
            block=block_mean(values, PRIMARY_BLOCK), plateau_fraction=fraction,
            unclipped_mean=unclipped,
        ))
    return frames


def pair_frames(frames: list[Frame]) -> list[Pair]:
    """Adjacent high->low files inside one directory; clean darks never pair."""
    pairs = []
    for directory in sorted({f.directory for f in frames}):
        rows = sorted((f for f in frames if f.directory == directory), key=lambda f: f.file)
        for i, (a, b) in enumerate(zip(rows, rows[1:])):
            if not (a.high and not b.high) or b.counter in CLEAN_DARK_COUNTERS:
                continue
            previous = rows[i - 1].counter if i > 0 else None
            flags = []
            if previous is None or a.counter - previous != 1:
                flags.append('preceded_by_counter_gap')
            if (b.timestamp - a.timestamp).total_seconds() > LONG_GAP_SECONDS:
                flags.append('long_filename_gap')
            pairs.append(Pair(directory, a, b, previous, flags))
    return pairs


def clean_dark(frames: list[Frame]) -> np.ndarray:
    darks = [f for f in frames if f.counter in CLEAN_DARK_COUNTERS]
    if len(darks) != len(CLEAN_DARK_COUNTERS) or any(f.high for f in darks):
        raise ValueError('Expected both predeclared clean dark frames as low-value frames')
    return np.mean([f.block for f in darks], axis=0)


def short_directory(directory: str) -> str:
    """ASCII figure label, e.g. '250ma/200ms/window 3000ms'; default fonts lack CJK glyphs."""
    return '/'.join(w for w in directory.split() if w.endswith(('ma', 'ms')) and 'kv' not in w) \
        .replace('曝光时间', 'window ')


def training_indices(groups: list[str], held_out: int, by: str) -> list[int]:
    """by='pair': other pairs of the same group; by='group': every pair of other groups."""
    group = groups[held_out]
    if by == 'pair':
        return [j for j, g in enumerate(groups) if g == group and j != held_out]
    if by == 'group':
        return [j for j, g in enumerate(groups) if g != group]
    raise ValueError(by)


def fit_pooled(xs: list[np.ndarray], ys: list[np.ndarray]) -> tuple[float, float]:
    x = np.concatenate([v.ravel() for v in xs])
    y = np.concatenate([v.ravel() for v in ys])
    keep, _ = _training_keep(x, y, np.ones(x.shape, dtype=bool), TRIM_PERCENTILES)
    return _fit_affine(x[keep], y[keep])


def plateau_strata(fraction: np.ndarray) -> dict[str, np.ndarray]:
    return {'plateau': fraction == 1.0, 'boundary': (fraction > 0) & (fraction < 1),
            'unclipped': fraction == 0}


def prediction_metrics(y: np.ndarray, p: np.ndarray, strata: dict[str, np.ndarray]) -> dict:
    r = y - p
    total = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    out = {
        'r2': 1.0 - float(np.sum(r ** 2)) / total,
        'rmse': float(np.sqrt(np.mean(r ** 2))),
        'ncc': float(np.corrcoef(y.ravel(), p.ravel())[0, 1]),
        'bias': float(np.mean(r)),
        'centered_rmse': float(np.std(r)),
    }
    for name, mask in strata.items():
        out[f'rmse_{name}'] = float(np.sqrt(np.mean(r[mask] ** 2))) if mask.any() else None
        out[f'blocks_{name}'] = int(mask.sum())
    return out
