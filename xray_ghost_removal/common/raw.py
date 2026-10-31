"""Headerless RAW reading; the pixel layout is a caller assumption, not a vendor fact."""
from __future__ import annotations
from datetime import datetime
import hashlib
from pathlib import Path
import numpy as np


def parse_raw_name(path: Path) -> tuple[datetime, int]:
    """Filename time and hexadecimal counter, e.g. 20261029-141219-777-20000069.raw.

    The time is whatever the exporter wrote; it is not a validated readout time.
    """
    stem = Path(path).stem
    return datetime.strptime(stem[:19], '%Y%m%d-%H%M%S-%f'), int(stem.rsplit('-', 1)[1], 16)


def read_raw(path: Path, shape: tuple[int, int], dtype: str) -> tuple[np.ndarray, str]:
    """Return read-only pixels and the SHA-256 of the exact file bytes."""
    data = Path(path).read_bytes()
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(data) != expected:
        raise ValueError(f'Unexpected byte count: {path}; expected {expected}')
    return np.frombuffer(data, dtype=dtype).reshape(shape), hashlib.sha256(data).hexdigest()
