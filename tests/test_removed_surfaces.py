"""Every surface removed in 0.14.0 and 0.15.0, pinned as ABSENT.

Owner ruling 2026-08-20: everything ships in 0.14.0, no deferrals. These names each had
a deprecation shim and a test asserting the shim warned; those tests are replaced by
these, because a removal needs pinning exactly as much as a deprecation did -- otherwise
a future refactor can quietly reintroduce an alias nobody meant to keep.

The migration for each is in the CHANGELOG's "Upgrading from 0.13.x" section, and for the
0.15.0 removals under that release's "Removed".
"""
import importlib

import pytest

import simplipy
from simplipy import Mode, SimpliPyEngine
from conftest import acj_config_path, require_or_skip


@pytest.fixture(scope='module')
def eng() -> SimpliPyEngine:
    require_or_skip(acj_config_path(), 'needs the shipped acj triple')
    return SimpliPyEngine.from_config(acj_config_path())


class TestModeAliasesAreGone:
    @pytest.mark.parametrize('name', ['SOUND', 'LOSSY'])
    def test_the_enum_attribute_is_gone(self, name: str) -> None:
        with pytest.raises(AttributeError):
            getattr(Mode, name)

    def test_the_members_that_remain(self) -> None:
        assert [m.name for m in Mode] == ['f64', 'real', 'permissive']

    @pytest.mark.parametrize('spelling', ['sound', 'lossy', 'SOUND', 'LOSSY'])
    def test_the_string_spellings_are_refused(self, spelling: str, eng) -> None:
        with pytest.raises(ValueError, match='unknown mode'):
            eng.simplify('x0 + 0', mode=spelling)


class TestRenamedCallablesAreGone:
    def test_engine_parse_is_gone(self) -> None:
        assert not hasattr(SimpliPyEngine, 'parse')
        assert hasattr(SimpliPyEngine, 'read_infix'), 'the replacement must survive'

    @pytest.mark.parametrize('name', ['normalize_skeleton', 'normalize_expression'])
    def test_the_normalization_aliases_are_gone(self, name: str) -> None:
        mod = importlib.import_module('simplipy.normalization')
        with pytest.raises(AttributeError):
            getattr(mod, name)
        with pytest.raises(AttributeError):
            getattr(simplipy, name)

    def test_the_surviving_normalization_names(self) -> None:
        assert callable(simplipy.to_skeleton) and callable(simplipy.to_expression)

    def test_mask_values_keep_structure_is_gone(self) -> None:
        mod = importlib.import_module('simplipy.masking')
        with pytest.raises(AttributeError):
            getattr(mod, 'mask_values_keep_structure')
        assert callable(mod.mask_fittable), 'the replacement must survive'

    def test_the_substitute_constants_misspelling_is_gone(self) -> None:
        mod = importlib.import_module('simplipy.utils')
        with pytest.raises(AttributeError):
            getattr(mod, 'substitude_constants')
        assert callable(mod.substitute_constants)

    def test_numbers_to_constant_stays_gone(self) -> None:
        mod = importlib.import_module('simplipy.utils')
        with pytest.raises(AttributeError):
            getattr(mod, 'numbers_to_constant')


class TestRemovedParameters:
    def test_simplify_has_no_form(self, eng) -> None:
        with pytest.raises(TypeError, match='form'):
            eng.simplify('x0 + 0', form='infix')

    def test_simplify_has_no_node_budget(self, eng) -> None:
        with pytest.raises(TypeError, match='node_budget'):
            eng.simplify('x0 + 0', node_budget=8)

    def test_max_passes_survives_with_its_guard(self, eng) -> None:
        assert eng.simplify('x0 + 0', max_passes=4) == 'x0'
        for bad in (-1, 0):
            with pytest.raises(ValueError, match='at least 1'):
                eng.simplify('x0 + 0', max_passes=bad)

    def test_load_has_no_path(self) -> None:
        with pytest.raises(TypeError, match='path'):
            SimpliPyEngine.load(path='acj-4-3')


class TestRemovedIn015:
    """0.15.0 removes without an alias cycle: a removed keyword raises ``TypeError``."""

    def test_simplify_has_no_work_budget(self, eng) -> None:
        for work in (None, 'default', 10_000):
            with pytest.raises(TypeError, match='work'):
                eng.simplify('x0 + 0', work=work)
        assert eng.simplify('x0 + 0', effort=0) == 'x0', 'effort= stays the search bound'

    def test_the_work_default_is_gone(self) -> None:
        from simplipy import _core
        assert not hasattr(simplipy, 'DEFAULT_WORK')
        assert 'DEFAULT_WORK' not in simplipy.__all__
        assert not hasattr(importlib.import_module('simplipy.engine'), 'DEFAULT_WORK')
        assert not hasattr(_core, 'DEFAULT_WORK')

    @pytest.mark.parametrize('entry', ['ac_simplify_in_mode', 'ac_simplify_infix_in_mode'])
    def test_the_core_takes_no_work_budget(self, eng, entry) -> None:
        with pytest.raises(TypeError, match='work_budget'):
            getattr(eng._core, entry)(['x0'], work_budget=10, rule_mode='default')

    def test_evaluate_constants_has_no_form(self, eng) -> None:
        for form in ('infix', 'tagged', 'explicit'):
            with pytest.raises(TypeError, match='form'):
                eng.evaluate_constants(['+', 'tan', '1', 'x0'], form=form)
        # it answers in the input's dialect
        assert eng.evaluate_constants(['+', 'tan', '1', 'x0']) == ['+', '1.5574077246549023', 'x0']
        assert eng.evaluate_constants('tan(1) + x0') == '1.5574077246549023 + x0'

    def test_find_rules_has_no_save_every(self, eng) -> None:
        from simplipy.mining import RuleMiner
        # prune=True is refused before any mining, so a find_rules that still took the
        # keyword fails this test at once with that ValueError instead of starting a mine
        with pytest.raises(TypeError, match='save_every'):
            RuleMiner(eng).find_rules(save_every=10, prune=True)

    @pytest.mark.parametrize('flag', ['-s', '--save-every'])
    def test_the_cli_has_no_save_every(self, flag, capsys) -> None:
        from simplipy.__main__ import main
        with pytest.raises(SystemExit) as exit_info:
            main(['find-rules', '-e', 'nonexistent_engine_xyz', '-c', 'dummy.yaml',
                  '-o', 'out.json', flag, '10'])
        assert exit_info.value.code == 2  # argparse refuses the flag before anything runs
        assert 'unrecognized arguments' in capsys.readouterr().err


def test_the_compatibility_policy_is_the_hard_cut() -> None:
    """The published policy matches what the releases do: a removal is a hard cut,
    announced in the release that makes it, and the page carries no open question."""
    import os
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    page = open(os.path.join(root, 'docs', 'compatibility.md'), encoding='utf-8').read()
    versioning = page.split('## Versioning', 1)[1].split('\n## ', 1)[0]
    assert 'A removal is a hard cut, announced in the release that makes it' in versioning
    assert 'PENDING' not in page
    assert not re.search(r'warning for at\s+least one minor release', page)
