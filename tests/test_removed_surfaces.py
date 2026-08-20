"""Every surface removed in 0.14.0, pinned as ABSENT.

Owner ruling 2026-08-20: everything ships in 0.14.0, no deferrals. These names each had
a deprecation shim and a test asserting the shim warned; those tests are replaced by
these, because a removal needs pinning exactly as much as a deprecation did -- otherwise
a future refactor can quietly reintroduce an alias nobody meant to keep.

The migration for each is in the CHANGELOG's "Upgrading from 0.13.x" section.
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
        assert [m.name for m in Mode] == ['f64', 'real', 'corpus']

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
        assert eng.simplify('x0 + 0', max_passes=4) is not None
        with pytest.raises(ValueError, match='non-negative'):
            eng.simplify('x0 + 0', max_passes=-1)

    def test_load_has_no_path(self) -> None:
        with pytest.raises(TypeError, match='path'):
            SimpliPyEngine.load(path='acj-4-3')
