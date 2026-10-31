"""One lazy command dispatcher for inherited methods, M1, and research."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
import os
import sys

from .common.paths import PROJECT_ROOT


@dataclass(frozen=True)
class Command:
    module: str
    description: str
    accepts_arguments: bool = True


COMMANDS = {
    'data': {
        'inspect-raw': Command('data.raw_intake', 'Inventory and preview a RAW batch under an explicit layout assumption.'),
        'export-raw-dicom': Command('data.raw_dicom', 'Package an inventoried RAW batch for bulk DICOM import into Horos.'),
    },
    'legacy': {
        'remove': Command('legacy.remove', 'Run the inherited seamless remover (nearest previous frame by default).'),
        'inpaint': Command('legacy.inpaint', 'Run the inherited diffusion-inpainting baseline.'),
        'linear-fit': Command('legacy.analysis', 'Revisit the original 30-frame affine analysis and figure.', False),
        'train': Command('legacy.train', 'Train the inherited synthetic U-Net; requires the legacy extra.'),
        'diagnose': Command('legacy.diagnose', 'Diagnose synthetic-to-real U-Net transfer; requires a checkpoint.', True),
    },
    'm1': {
        'fit': Command('m1.pipeline', 'Estimate background and fit strict spatial OOF M1 without mask expansion.'),
        'review': Command('m1.review', 'Export three existing frozen OOF pairs for DICOM review.', False),
        'simulate': Command('m1.semisynthetic', 'Rebuild the six seen-donor semisynthetic demonstrations.', False),
        'presentation': Command('m1.presentation', 'Rebuild the selected three semisynthetic review cases.', False),
    },
    'research': {
        'freeze-audit': Command('research.audits.freeze', 'Historical BG sensitivity audit; the old dilation definition cannot be rerun.'),
        'crossfit-audit': Command('research.audits.crossfit', 'Historical BG crossfit audit; requires a compatible frozen definition.'),
        'coverage-audit': Command('research.audits.coverage', 'Historical fixed-pattern coverage audit; requires a compatible frozen definition.'),
        'pairing': Command('research.pairing', 'Revisit the existing dark/light pairing and robustness study.'),
        'candidates': Command('research.candidates', 'Compare blur and quadratic candidates on frozen M1 inputs.'),
        'nonlinearity': Command('research.nonlinearity', 'Evaluate the predefined intensity-response candidates.'),
        'three-pair': Command('research.three_pair', 'Rebuild the descriptive three-pair intensity comparison.'),
        'transfer': Command('research.transfer', 'Evaluate bidirectional pair transfer and target calibration.'),
        'hinge-audit': Command('research.hinge_audit', 'Audit the predeclared cross-pair hinge candidate.'),
        'exposure': Command('research.exposure', 'Analyze exposure associations in the frozen 13-pair cohort.'),
        'exposure-audit': Command('research.exposure_audit', 'Read-only audit of the original DICOM files and exposure records.'),
        'exposure-plot': Command('research.exposure_plot', 'Plot the four existing repeated exposure settings.', False),
        'raw-m1': Command('research.raw_m1', 'Test M1 cross-pair prediction on the 10.30 RAW batch against clean darks.'),
        'raw-split': Command('research.raw_split', 'Compare the plateau/unclipped split model with M1 on the 10.30 RAW batch.'),
    },
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    groups = parser.add_subparsers(dest='group', required=True)
    for group, commands in COMMANDS.items():
        group_parser = groups.add_parser(group)
        actions = group_parser.add_subparsers(dest='command', required=True)
        for name, command in commands.items():
            actions.add_parser(name, add_help=False, help=command.description)
    args, remaining = parser.parse_known_args(argv)
    command = COMMANDS[args.group][args.command]
    if not command.accepts_arguments:
        if remaining in (['-h'], ['--help']):
            print(f'usage: python run.py {args.group} {args.command}\n\n{command.description}\nPaths are relative to the repository; defaults are documented in README and DATA.')
            return
        if remaining:
            parser.error(f'{args.group} {args.command} accepts no arguments')
    # Modules are imported only after command validation. In particular --help must
    # never execute a no-argument export or load optional training dependencies.
    previous_argv, previous_cwd = sys.argv, os.getcwd()
    try:
        os.chdir(PROJECT_ROOT)
        sys.argv = [f'xray {args.group} {args.command}', *remaining]
        import_module('xray_ghost_removal.' + command.module).main()
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
