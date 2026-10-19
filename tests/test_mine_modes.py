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


@pytest.mark.parametrize('name,args', [
    ('ac_simplify_in_mode', (['x0'],)), ('ac_simplify_infix_in_mode', (['x0'],)),
    ('ac_simplify_explore', (['x0'],)), ('prettify', (['x0'],)), ('ac_complexity', (['x0'],)),
    ('ac_complexity_certified', (['x0'],)), ('ac_rules_info_in_mode', ()),
    ('ac_served_rules_in_mode', ()), ('mode_rules_len', ()),
    ('find_rule', (['x0'], 1, None, [['x0']], ['x0'], [1.0], 1)),
    ('ac_simplify_suppressed', (['x0'],)),
])
def test_every_mode_taking_entry_names_its_mode(eng, name, args):
    """No hidden default in the FFI: every entry that reads a mode takes it explicitly."""
    with pytest.raises(TypeError, match='rule_mode'):
        getattr(eng._core, name)(*args)


def test_the_mode_less_twins_are_retired(eng):
    """One entry per question: the mode-less twins of the `_in`/`_in_mode` entries are gone,
    and no signature in the extension defaults a mode."""
    import os
    import re
    for name in ('ac_simplify', 'ac_simplify_infix', 'ac_ordered_below', 'ac_rules_info',
                 'ac_served_rules', 'set_rules', 'ac_judge'):
        assert not hasattr(eng._core, name), name
    lib = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'rust', 'lib.rs')).read()
    assert not re.findall(r'signature = \([^)]*(rule_mode="|wildcard_all=)', lib)


#: (source, its law candidate as a rule): factorings over a function base, `_` and `!` sorts.
FACTORINGS = [
    (['+', '*', 'sin', 'x0', 'x1', '*', 'sin', 'x0', 'x2'],
     ['+', '*', 'sin', '?0', '?1', '*', 'sin', '?0', '?2'], ['*', 'sin', '?0', '+', '?1', '?2']),
    (['+', '*', 'exp', 'x0', 'x1', '*', 'exp', 'x0', 'x2'],
     ['+', '*', 'exp', '?0', '?1', '*', 'exp', '?0', '?2'], ['*', 'exp', '?0', '+', '?1', '?2']),
    (['+', '*', 'cos', 'x0', 'x1', 'cos', 'x0'],
     ['+', '*', 'cos', '?0', '?1', 'cos', '?0'], ['*', 'cos', '?0', '+', '?1', '1']),
    (['+', '*', 'log', 'x0', 'x1', '*', 'log', 'x0', 'x2'],
     ['+', '*', 'log', '?0', '?1', '*', 'log', '?0', '?2'], ['*', 'log', '?0', '+', '?1', '?2']),
    (['+', '*', 'x0', 'x1', '*', 'x0', 'x2'],
     ['+', '*', '_0', '_1', '*', '_0', '_2'], ['*', '_0', '+', '_1', '_2']),
    (['+', '*', 'x0', 'x1', '*', 'x0', 'x2'],
     ['+', '*', '!0', '!1', '*', '!0', '!2'], ['*', '!0', '+', '!1', '!2']),
    (['+', '/', 'x1', 'x0', '/', 'x2', 'x0'],
     ['+', '/', '?1', '?0', '/', '?2', '?0'], ['/', '+', '?1', '?2', '?0']),
]


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('source,lhs,rhs', FACTORINGS)
def test_the_law_gate_takes_the_searchs_licence(eng, mode, source, lhs, rhs):
    """G8 (S1): the gate names the distributive law exactly where the search performs it on
    the rule's instances. The search factors `sin(x0)*x1 + sin(x0)*x2` in every mode, and the
    gate's own pattern licence used to refuse every function base."""
    instance = [{'?0': 'x0', '?1': 'x1', '?2': 'x2'}.get(t, t) for t in rhs]
    searched = eng._core.mine_mark(source, rule_mode=mode) == eng._core.mine_mark(instance, rule_mode=mode)
    owned = eng._core.law_instance(lhs, rhs, rule_mode=mode) == 'engine-owned-law'
    if lhs[2] in ('_0', '!0'):  # a wide sort: the search must factor EVERY binding
        assert owned == (mode == 'permissive' or lhs[2] == '!0'), (mode, lhs)
    else:
        assert owned == searched, (mode, lhs, searched)


def test_each_mode_mines_from_its_own_set():
    """Without `reset_rules`, a mode's mine starts from that mode's own set, never from
    another mode's: the f64 set's rule does not seed the real and permissive files (S9)."""
    ops = {
        "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
        "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
        "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
        "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    }
    f64_only = (('log', 'exp', 'exp', '?0'), ('exp', '?0'))
    real_only = (('exp', 'log', 'exp', '?0'), ('exp', '?0'))
    engine = SimpliPyEngine(operators=ops, rules=[f64_only], rules_real=[real_only],
                            rules_permissive=[])
    engine.find_rules(max_source_pattern_length=2, max_target_pattern_length=2, dummy_variables=1,
                      X=128, seed=3, reset_rules=False, promote_sorts=False, symbolic_gate=False,
                      judge=lambda pairs: ['core'] * len(pairs))
    assert f64_only in engine.simplification_rules
    assert f64_only not in engine.real_simplification_rules
    assert real_only in engine.real_simplification_rules
    assert f64_only not in engine.permissive_simplification_rules
    assert real_only not in engine.permissive_simplification_rules
