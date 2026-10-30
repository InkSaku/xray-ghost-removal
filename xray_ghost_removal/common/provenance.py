"""Hashing and JSON conversion for reproducible research records."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def package_sources() -> dict[str, Any]:
    """Record every package source so extracted dependencies stay attributable."""
    from .paths import PROJECT_ROOT
    files = {str(path.relative_to(PROJECT_ROOT)): sha256(path)
             for path in sorted((PROJECT_ROOT / 'xray_ghost_removal').rglob('*.py'))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {'sha256': digest, 'files': files}
