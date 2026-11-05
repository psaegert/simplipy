"""A mode's mine reads everything in that mode: its marks are the mode's serve answer, its
sources are the mode's canonical projections, its rules are read in the mode's context, and
the law the engine owns is refused by the predicate the rule load applies."""
import json

import pytest
import yaml

from simplipy import SimpliPyEngine

from conftest import acj_config_path
from conftest import legacy_from_config  # noqa: E402

MODES = ('default', 'permissive')
DISTRIBUTIVE = (['+', '*', '?0', '?1', '*', '?2', '?1'], ['*', '+', '?0', '?2', '?1'])


@pytest.fixture(scope='module')
def eng():
    # A RULE-FREE engine over the shipped vocabulary: the mine's own situation.
    shipped = legacy_from_config(acj_config_path())
    return SimpliPyEngine(operators=shipped._operators_config, rules=[])


def test_the_mark_is_the_serve_answer_of_the_mode(eng):
    mark = eng._core.mine_mark
    # f64 folds the ground transcendental; permissive reads the exact value from the table
    assert mark(['cos', '0'], rule_mode='default') == ['1']
    assert mark(['cos', '0'], rule_mode='permissive') == ['1']
    # permissive collapses the inverse pair; f64's saturation guard keeps it
    assert mark(['atanh', 'tanh', 'x0'], rule_mode='permissive') == ['x0']
    assert mark(['atanh', 'tanh', 'x0'], rule_mode='default') == ['atanh', 'tanh', 'x0']
    # the serve answer runs the search: every mode factors
    for mode in MODES:
        assert mark(['+', 'x0', '*', 'x0', 'x1'], rule_mode=mode) == ['*', 'x0', '+', 'x1', '1']


def test_sources_are_the_canonical_projections_of_the_mode(eng):
    enumerated = [['+', 'cos', '0', 'sin', 'x0'], ['+', '1', 'sin', 'x0'],
                  ['atanh', 'tanh', 'x0'], ['cos', '0']]
    # no reading folds `cos 0` (the walk does, as a step); f64 keeps `atanh tanh x0`, and so does
    # permissive, whose sources are read as its rule sides are (certified licences)
    for mode in MODES:
        assert eng._core.mine_sources(enumerated, rule_mode=mode) == [
            ['+', 'cos', '0', 'sin', 'x0'], ['+', 'sin', 'x0', '1'], ['atanh', 'tanh', 'x0'],
            ['cos', '0']], mode


@pytest.mark.parametrize('mode', MODES)
def test_rules_are_read_in_the_mode(eng, mode):
    # f64 keeps the pattern (tanh saturates in floats), and so does permissive, whose rule sides
    # are read with certified licences
    served = eng._core.ac_translate_preview([(['atanh', 'tanh', '?0'], ['?0'])], rule_mode=mode)
    assert served != [], (mode, served)


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
         'rules_permissive': 'rules.json'}))
    loaded = SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'))
    assert loaded._core.ac_rules_drop_census(rule_mode=mode).get('engine-owned-law') == 1


#: `exp(a) exp(b) = exp(a + b)` stated as a rule, in either direction.
EXP_LAW = [
    (['*', 'exp', '?0', 'exp', '?1'], ['exp', '+', '?0', '?1']),
    (['exp', '+', '?0', '*', '2', '?1'], ['*', 'exp', '?0', 'pow', 'exp', '?1', '2']),
    (['/', 'exp', '_0', 'exp', '_1'], ['exp', '-', '_0', '_1']),
    (['*', 'np.e', 'exp', '!0'], ['exp', '+', '!0', '1']),
]


@pytest.mark.parametrize('mode', MODES)
def test_a_rule_stating_the_exp_law_is_refused_in_either_direction(tmp_path, eng, mode):
    """The product constructor collects powers of e, so both sides of a rule stating the law
    are one canonical state: the miner names it `canon`, and the load refuses every such row."""
    for lhs, rhs in EXP_LAW:
        assert eng._core.law_instance(lhs, rhs, rule_mode=mode) == 'canon', (lhs, rhs)
    (tmp_path / 'rules.json').write_text(json.dumps(EXP_LAW))
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(
        {'operators': eng._operators_config, 'rules': 'rules.json',
         'rules_permissive': 'rules.json'}))
    loaded = SimpliPyEngine.from_config(str(tmp_path / 'config.yaml'))
    assert loaded._core.ac_served_rules_in_mode(rule_mode=mode) == []
    assert loaded._core.ac_rules_drop_census(rule_mode=mode) == {'arithmetic-subsumed': len(EXP_LAW)}


@pytest.mark.parametrize('name,args', [
    ('ac_simplify_in_mode', (['x0'],)), ('ac_simplify_infix_in_mode', (['x0'],)),
    ('ac_simplify_explore', (['x0'],)), ('ac_complexity', (['x0'],)),
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
    """G8: the gate names the distributive law exactly where the search performs it on
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
    another mode's: the f64 set's rule does not seed the permissive file, nor the permissive
    set's the f64 file."""
    ops = {
        "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
        "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
        "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
        "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    }
    in_f64_set = (('log', 'exp', 'exp', '?0'), ('exp', '?0'))
    in_permissive_set = (('exp', 'log', 'exp', '?0'), ('exp', '?0'))
    engine = SimpliPyEngine(operators=ops, rules=[in_f64_set],
                            rules_permissive=[in_permissive_set])
    engine.find_rules(max_source_pattern_length=2, max_target_pattern_length=2, dummy_variables=1,
                      X=128, seed=3, reset_rules=False, promote_sorts=False, symbolic_gate=False,
                      judge=lambda pairs: ['core'] * len(pairs))
    assert in_f64_set in engine.simplification_rules
    assert in_permissive_set not in engine.simplification_rules
    assert in_f64_set not in engine.permissive_simplification_rules
    assert in_permissive_set in engine.permissive_simplification_rules


def test_the_refund_reads_the_constructors_dedup(monkeypatch):
    """The promotion refund maps each probe to its row through the constructor's own dedup
    fold. A replica of it ("shortest target") disagreed with the fold's tie-break (the serve
    ordering), and every probe then fell back to the slow rebuild. With a tie-break of the
    test's own, only the probes of the shared bucket fall back."""
    import simplipy.utils
    from simplipy.promotion._refund import refund
    monkeypatch.setattr(simplipy.utils, 'dedup_target_wins',
                        lambda engine, new, kept, rule_mode: len(new) > len(kept))
    ops = {
        "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
        "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
        "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
        "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
    }
    rules = [(('neg', 'neg', '?0'), ('?0',)), (('neg', 'neg', '?0'), ('*', '1', '?0')),
             (('log', 'exp', '?0'), ('?0',)), (('exp', 'log', 'exp', '?0'), ('exp', '?0')),
             (('neg', 'exp', 'log', '?0'), ('neg', '?0'))]
    stages = []

    class Spy:
        def stage(self, name, **kw):
            stages.append((name, kw))

        def track(self, iterable, name, total=None):
            return iterable

    refund(list(rules), ops, set(), progress=Spy(), rule_mode='default')
    done = [kw for name, kw in stages if name.endswith('derivability probe: done')][-1]
    assert done['fallbacks'] <= 2, done  # the replica made it 5: every probe


_OFFER_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "tan": {"realization": "np.tan", "alias": [], "inverse": "atan", "arity": 1, "precedence": 3, "commutative": False},
    "atan": {"realization": "np.arctan", "alias": [], "inverse": "tan", "arity": 1, "precedence": 3, "commutative": False},
    "abs": {"realization": "np.abs", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False},
}
# Exact in float64 and true on R, and beyond the length-2 horizon of the mine below, so
# permissive's own scan never reaches it: only the offered f64 file can bring it. (`abs` is no
# entry of permissive's own list.)
OFFERED = (('*', 'abs', '?0', 'abs', '?0'), ('*', '?0', '?0'))


def _offer_mine(tmp_path, offer):
    engine = SimpliPyEngine(operators=_OFFER_OPERATORS, rules=[OFFERED], rules_permissive=[])
    engine.find_rules(max_source_pattern_length=2, max_target_pattern_length=2, dummy_variables=1,
                      X=128, seed=3, reset_rules=False, promote_sorts=False, symbolic_gate=False,
                      judge=lambda pairs: ['core'] * len(pairs), offer_final_rules=offer,
                      output_file=str(tmp_path / 'rules_f64.json'))
    side = json.load(open(tmp_path / 'rules_permissive.json.provenance.json'))
    return engine, side


def test_the_permissive_mine_is_offered_the_finished_file(tmp_path):
    """The permissive mine runs last and passes the finished f64 file through its proposal lane,
    where each rule is certified by permissive's own acceptance: an f64 rule its own scan never
    reaches enters the permissive file where it holds there, and the sidecar pins the file."""
    import hashlib
    engine, side = _offer_mine(tmp_path, offer=True)
    assert OFFERED in engine.simplification_rules
    assert OFFERED in engine.permissive_simplification_rules
    offered = side['offered']
    assert set(offered['files']) == {'f64'}
    for key, name in (('f64', 'rules_f64.json'),):
        pin = offered['files'][key]
        assert pin['file'] == name
        assert pin['sha256'] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert pin['count'] == len(json.load(open(tmp_path / name)))
    assert side['params']['offer_final_rules'] is True
    assert sum(offered['outcomes'].values()) == offered['count']
    entry, = [e for e in offered['trail'] if e['source'] == ['*', 'abs', 'x0', 'abs', 'x0']]
    assert entry['verdict'] == 'certified' and entry['target'] == ['*', 'x0', 'x0'], entry


def test_the_offer_can_be_declined(tmp_path):
    engine, side = _offer_mine(tmp_path, offer=False)
    assert OFFERED not in engine.permissive_simplification_rules
    assert 'offered' not in side and side['params']['offer_final_rules'] is False


_GROUND_OPERATORS = {
    "atan2": {"realization": "simplipy.operators.atan2", "alias": [], "arity": 2, "precedence": 2,
              "commutative": False},
    "atanh": {"realization": "simplipy.operators.atanh", "alias": [], "arity": 1, "precedence": 2,
              "commutative": False},
}
# True in f64 (atan2(c, -inf) is +-pi) and live there, where the rule fires before the pass
# folds `atanh(-1)`; permissive reads `atanh(-1)` as -inf before any rule can fire.
GROUND_OFFERED = (('atan2', '<constant>', 'atanh', '-1'), ('<constant>',))


def test_an_f64_rule_is_offered_at_permissives_reading_of_its_left_side(tmp_path):
    """Permissive reads a ground pole in a left side as its infinity before any rule can fire,
    so the f64 rule `atan2(C, atanh(-1)) -> C` would never match there. The offered lane adopts
    each f64 rule at permissive's reading of its left side, where the rule fires, and the
    permissive file covers the f64 file."""
    engine = SimpliPyEngine(operators=_GROUND_OPERATORS, rules=[GROUND_OFFERED],
                            rules_permissive=[])
    engine.find_rules(max_source_pattern_length=2, max_target_pattern_length=2,
                      dummy_variables=1, X=128, seed=3, reset_rules=False, promote_sorts=False,
                      symbolic_gate=False, judge=lambda pairs: ['core'] * len(pairs),
                      offer_final_rules=True, output_file=str(tmp_path / 'rules_f64.json'))
    side = json.load(open(tmp_path / 'rules_permissive.json.provenance.json'))
    assert GROUND_OFFERED in engine.simplification_rules
    reading = ['atan2', '<constant>', 'float("-inf")']
    entry, = [e for e in side['offered']['trail'] if e['source'] == reading]
    assert entry['verdict'] == 'certified' and entry['target'] == ['<constant>'], entry
    assert (tuple(reading), ('<constant>',)) in engine.permissive_simplification_rules
    assert side['gates']['covers_f64'] == []


def test_the_f64_scan_reads_the_range_standard(eng):
    """The scan's f64 check applies the judge's standard (`numeric::RangeKind`, one
    definition): on a row where an operation of either side overflows or underflows, the
    true value is accepted beside the evaluator's. The mine's own matrix carries such rows
    (exp overflows past 709.78 and underflows below -745.13). Permissive reads the true value
    everywhere and is unchanged."""
    import numpy as np
    from simplipy.mining import RuleMiner
    X = RuleMiner(eng)._mining_sample_x(1024, 2, np.random.default_rng(42))
    assert (abs(X[:, 0]) > 709.78).sum() > 0 and (X[:, 0] < -745.13).sum() > 0

    def judged(source, target, mode):
        return eng._core.judge_pair(source, target, ['x0', 'x1'], X.flatten().tolist(),
                                    X.shape[0], rule_mode=mode, rtol=1e-11, atol=1e-12)[0]

    pairs = {  # pair -> (f64, permissive)
        (('rootn', 'exp', 'x0', '2'), ('exp', '/', 'x0', '2')): (True, True),
        # the fit arm: the fitted constant |C| on the rows past the overflow
        (('rootn', '*', 'pow', '<constant>', '2', 'exp', 'x0', '2'),
         ('*', '<constant>', 'exp', '/', 'x0', '2')): (True, True),
        # tanh saturates to 1.0 inside the range, and atanh(1.0) is a pole: no event
        (('atanh', 'tanh', 'x0'), ('x0',)): (False, True),
        (('sin', 'x0'), ('x0',)): (False, False),
    }
    for (source, target), want in pairs.items():
        got = (judged(list(source), list(target), 'default'),
               judged(list(source), list(target), 'permissive'))
        assert got == want, (source, target, got)


def test_the_f64_mine_reads_the_range_standard(tmp_path):
    """A tiny mine: f64 mints `rootn(exp(x), 2) -> exp(x/2)`, which differs from the
    evaluator only past exp's overflow and underflow. `atanh(tanh(x)) -> x` stays out, since
    tanh saturates inside the range. `log(exp(x)) -> x` stays out too: the scan asks the judge's
    agreement (8 units in the last place) and its sides differ by 34 at the ordinary point
    0.0137421, so the judge never sees it."""
    ops = {
        "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
        "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
        "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
        "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
        "tanh": {"realization": "np.tanh", "alias": [], "inverse": "atanh", "arity": 1, "precedence": 3, "commutative": False},
        "atanh": {"realization": "np.arctanh", "alias": [], "inverse": "tanh", "arity": 1, "precedence": 3, "commutative": False},
        "rootn": {"realization": "simplipy.operators.rootn", "alias": [], "inverse": None, "arity": 2, "precedence": 3, "commutative": False},
    }
    engine = SimpliPyEngine(operators=ops, rules=[])
    engine.find_rules(max_source_pattern_length=4, max_target_pattern_length=4, dummy_variables=1,
                      extra_internal_terms=['2'], X=512, seed=7, judge_processes=1,
                      output_file=str(tmp_path / 'rules_f64.json'), promote_sorts=False)
    f64 = [(tuple(lhs), tuple(rhs)) for lhs, rhs in json.load(open(tmp_path / 'rules_f64.json'))]
    assert (('rootn', 'exp', '?0', '2'), ('exp', '/', '?0', '2')) in f64
    assert not [r for r in f64 if r[0][:2] == ('atanh', 'tanh') and len(r[0]) == 3]
    assert not [r for r in f64 if r[0] == ('log', 'exp', '?0')]
    side = json.load(open(tmp_path / 'rules_f64.json.provenance.json'))
    assert [['log', 'exp', 'x0'], ['x0']] not in side['per_length']['3']['excluded']
