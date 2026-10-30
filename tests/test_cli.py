"""CLI routing, architecture boundaries, and frozen-evidence protection."""
import ast
from importlib.util import find_spec
from pathlib import Path
import sys

import pytest

from xray_ghost_removal import cli
from xray_ghost_removal.common.paths import PROJECT_ROOT, require_current_m1_output


@pytest.mark.parametrize('group,name', [(group, name)
    for group, commands in cli.COMMANDS.items() for name, command in commands.items()
    if not command.accepts_arguments])
def test_no_argument_help_does_not_import_or_execute(group, name, monkeypatch, capsys):
    def forbidden_import(*args):
        raise AssertionError('Help must not import or execute a no-argument experiment')
    monkeypatch.setattr(cli, 'import_module', forbidden_import)
    cli.main([group, name, '--help'])
    assert 'usage:' in capsys.readouterr().out


def test_dispatch_restores_cwd_and_arguments_on_failure(tmp_path, monkeypatch):
    original_argv = sys.argv
    monkeypatch.chdir(tmp_path)
    def failing_import(name):
        assert Path.cwd() == PROJECT_ROOT
        assert sys.argv[1:] == ['--shape', '16', '16']
        raise RuntimeError('test failure')
    monkeypatch.setattr(cli, 'import_module', failing_import)
    with pytest.raises(RuntimeError, match='test failure'):
        cli.main(['data', 'inspect-raw', '--shape', '16', '16'])
    assert Path.cwd() == tmp_path and sys.argv is original_argv


def test_current_m1_cannot_overwrite_frozen_evidence(tmp_path):
    for name in ('pseudo_ghost_mechanism_v1', 'mhinge_cross_pair_v1',
                 'background_freeze_audit_v1', 'fixed_pattern_identifiability_v1', 'pair_transfer_v1'):
        with pytest.raises(ValueError, match='read-only'):
            require_current_m1_output(PROJECT_ROOT / 'outputs' / name / 'nested')
    require_current_m1_output(PROJECT_ROOT / 'outputs/m1')
    require_current_m1_output(tmp_path)
    alias = tmp_path / 'alias'
    alias.symlink_to(PROJECT_ROOT / 'outputs/mhinge_cross_pair_v1', target_is_directory=True)
    with pytest.raises(ValueError, match='read-only'):
        require_current_m1_output(alias)


def test_commands_exist_and_dependency_directions_stay_clear():
    for commands in cli.COMMANDS.values():
        for command in commands.values():
            assert find_spec('xray_ghost_removal.' + command.module) is not None
    package = PROJECT_ROOT / 'xray_ghost_removal'
    allowed = {'common': {'common'}, 'legacy': {'common', 'legacy'},
               'm1': {'common', 'm1'}, 'research': {'common', 'm1', 'research'},
               'data': {'common', 'data'}}
    for layer, dependencies in allowed.items():
        for path in (package / layer).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert not node.module.startswith(('src.', 'scripts.')), path
                    if node.module.startswith('xray_ghost_removal.'):
                        assert node.module.split('.')[1] in dependencies, (path, node.module)
