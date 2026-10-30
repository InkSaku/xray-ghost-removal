"""Repository paths; independent of module nesting and working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require_current_m1_output(path: Path) -> None:
    """Prevent a current no-expansion fit from replacing historical evidence."""
    destination = path.resolve()
    for name in ('pseudo_ghost_mechanism_v1', 'mhinge_cross_pair_v1',
                 'background_freeze_audit_v1', 'fixed_pattern_identifiability_v1',
                 'pair_transfer_v1'):
        frozen = (PROJECT_ROOT / 'outputs' / name).resolve()
        if destination == frozen or frozen in destination.parents:
            raise ValueError(f'Historical evidence is read-only: {frozen}. Use outputs/m1 or a temporary directory for the current mask definition.')
