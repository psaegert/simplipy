"""A mode's mine reads everything in that mode: its marks are the mode's serve answer, its
sources are the mode's canonical projections, its rules are read in the mode's context, and
the law the engine owns is refused by the predicate the rule load applies."""
import json

import pytest
import yaml

from simplipy import SimpliPyEngine

from conftest import acj_config_path

MODES = ('default', 'real', 'permissive')
DISTRIBUTIVE = (['+', '*', '?0', '?1', '*', '?2', '?1'], ['*', '+', '?0', '?2', '?1'])


@pytest.fixture(scope='module')
def eng():
    # A RULE-FREE engine over the shipped vocabulary: the mine's own situation.
    shipped = SimpliPyEngine.from_config(acj_config_path())
    return SimpliPyEngine(operators=shipped._operators_config, rules=[])


def test_the_mark_is_the_serve_answer_of_the_mode(eng):
    mark = eng._core.mine_mark
    # f64 folds the ground transcendental; real leaves it to a rule
    assert mark(['cos', '0'], rule_mode='default') == ['1']
    assert mark(['cos', '0'], rule_mode='real') == ['cos', '0']
    # real's constructor collapses the inverse pair; f64's saturation guard keeps it
    assert mark(['atanh', 'tanh', 'x0'], rule_mode='real') == ['x0']
    assert mark(['atanh', 'tanh', 'x0'], rule_mode='default') == ['atanh', 'tanh', 'x0']
    # the serve answer runs the search: every mode factors
    for mode in MODES:
        assert mark(['+', 'x0', '*', 'x0', 'x1'], rule_mode=mode) == ['*', 'x0', '+', 'x1', '1']


def test_sources_are_the_canonical_projections_of_the_mode(eng):
    enumerated = [['+', 'cos', '0', 'sin', 'x0'], ['+', '1', 'sin', 'x0'],
                  ['atanh', 'tanh', 'x0'], ['cos', '0']]
    # f64 reads `cos 0` as 1: one class for the first two, and `cos 0` is an atom
    assert eng._core.mine_sources(enumerated, rule_mode='default') == [
        ['+', 'sin', 'x0', '1'], ['atanh', 'tanh', 'x0']]
    # real keeps `cos 0`, and `atanh tanh x0` is the atom x0
    assert eng._core.mine_sources(enumerated, rule_mode='real') == [
        ['+', 'cos', '0', 'sin', 'x0'], ['+', 'sin', 'x0', '1'], ['cos', '0']]


@pytest.mark.parametrize('mode', MODES)
def test_rules_are_read_in_the_mode(eng, mode):
    # f64 folds `cos 0` in the pattern too, so the rule states what the constructor does
    served = eng._core.ac_translate_preview([(['*', 'cos', '0', '?0'], ['?0'])], rule_mode=mode)
    assert (served == []) == (mode == 'default'), served


@pytest.mark.parametrize('mode', MODES)
def test_the_miner_refuses_the_law_the_load_refuses(tmp_path, eng, mode):
    """One predicate: `law_instance` names the engine-owned law exactly where the rule load
    drops the rule as "engine-owned-law"."""
    assert eng._core.law_instance(*DISTRIBUTIVE, rule_mode=mode) == 'engine-owned-law'
    assert eng._core.law_instance(['*', '?0', '1'], ['?0'], rule_mode=mode) == 'canon'
    assert eng._core.law_instance(['sin', '?0'], ['?0'], rule_mode=mode) is None
    (tmp_path / 'rules.json').write_text(json.dumps([list(DISTRIBUTIVE)]))
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(
        {'operators': eng._operators_config, 'rules': 'rules.json',
         'rules_real': 'rules.json', 'rules_permissive': 'rules.json'}))
    loaded = SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'))
    assert loaded._core.ac_rules_drop_census(rule_mode=mode).get('engine-owned-law') == 1
