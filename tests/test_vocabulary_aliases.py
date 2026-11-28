"""The aliases the catalogs write (the 2026-11-26 vocabulary request): ``Abs`` is ``abs`` and
``div(a, b)`` is ``a/b`` in every config the repository ships or tests with, and the spellings
that were already read (``ln``, ``arcsin``, ``arccos``, ``pow(a, b)``, ``power(a, b)``) keep
reading as their targets. An alias is a reading only: it is resolved by the reader, so the
canonical state, the answer and the price are the target's."""

import glob
import os
import warnings

import pytest
import yaml

from simplipy import SimpliPyEngine
from conftest import acj_config_path, legacy_from_config

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS = sorted(glob.glob(os.path.join(_ROOT, 'assets', 'engines', '*', 'config.yaml'))
                  + glob.glob(os.path.join(_ROOT, 'tests', 'fixtures', '*', 'config.yaml'))
                  + [os.path.join(_ROOT, 'tests', 'fixtures', 'legacy_vocab_config.yaml')])

# (written, the target's spelling)
_ALIASES = [
    ('Abs(x0)', 'abs(x0)'),
    ('Abs(x0 - x1)', 'abs(x0 - x1)'),
    ('div(8, 2 + pow(x1, 2) + pow(x2, 2))', '8/(2 + x1**2 + x2**2)'),
    ('div(x0, x1)', 'x0/x1'),
    ('ln(x0)', 'log(x0)'),
    ('arcsin(x0)', 'asin(x0)'),
    ('arccos(x0)', 'acos(x0)'),
    ('pow(x1, 2)', 'x1**2'),
    ('power(x0, 3)', 'x0**3'),
]


@pytest.mark.parametrize('path', _CONFIGS, ids=lambda p: os.path.relpath(p, _ROOT))
def test_every_config_declares_the_aliases(path: str) -> None:
    with open(path) as f:
        operators = yaml.safe_load(f)['operators']
    assert 'Abs' in operators['abs']['alias']
    assert 'div' in operators['/']['alias']


def _engines() -> list[tuple[str, SimpliPyEngine]]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return [('acj-4', legacy_from_config(acj_config_path())),
                ('acj-4-llm_0.15', SimpliPyEngine.from_config(
                    os.path.join(_ROOT, 'assets', 'engines', 'acj-4-llm_0.15', 'config.yaml')))]


@pytest.fixture(scope='module', params=['acj-4', 'acj-4-llm_0.15'])
def engine(request) -> SimpliPyEngine:
    return dict(_engines())[request.param]


@pytest.mark.parametrize('written, target', _ALIASES)
def test_an_alias_reads_as_its_target(engine: SimpliPyEngine, written: str, target: str) -> None:
    assert engine.read_infix(written) == engine.read_infix(target)
    # the strict readers of the conversions and of simplify read it too
    assert engine.to_prefix(written) == engine.read_infix(target)


@pytest.mark.parametrize('written, target', _ALIASES)
@pytest.mark.parametrize('mode', ['strict', 'permissive'])
def test_an_alias_answers_and_prices_as_its_target(engine: SimpliPyEngine, written: str, target: str,
                                                   mode: str) -> None:
    assert engine.simplify(written, mode=mode) == engine.simplify(target, mode=mode)
    assert engine.complexity(written, mode=mode) == engine.complexity(target, mode=mode)


def test_div_takes_two_arguments(engine: SimpliPyEngine) -> None:
    with pytest.raises(ValueError, match='div takes 2 arguments, got 1'):
        engine.read_infix('div(x0)')
