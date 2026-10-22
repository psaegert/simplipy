"""The 0.15 heads (floor, ceil, max, min, mod, erf, harmonic, atan2, besselj) in the engine:
functions of literals fold under each mode's convention, every layer without a sound treatment
of a head refuses it, and the miner mines with them."""
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.verify import _contract, _monitor

ACJ2_4 = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4', 'config.yaml')
MODES = ('f64', 'real', 'permissive')


@pytest.fixture(scope='module')
def bare():
    """The acj2-4 operators with an empty rule set per mode: the core on its own."""
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_real=[], rules_permissive=[])


@pytest.mark.parametrize('mode', MODES)
def test_step_and_lattice_heads_of_literals_fold(bare, mode):
    s = lambda t: bare.simplify(t.split(), mode=mode)  # noqa: E731
    assert s('floor 2.5') == ['2']
    assert s('ceil -2.5') == ['-2']
    assert s('max 1 / 1 3') == ['1']
    assert s('mod 7 -3') == ['-2']
    assert s('mod 1 0') == ['float("nan")']
    assert s('floor float("inf")') == ['float("inf")']
    # a transcendental head of a literal stays written where its float is dearer
    assert s('erf 1') == ['erf', '1']


def test_f64_reads_a_literal_at_its_float_and_real_at_its_rational(bare):
    assert bare.simplify(['floor', '0.99999999999999999'], mode='f64') == ['1']
    assert bare.simplify(['floor', '0.99999999999999999'], mode='real') == ['0']
    assert bare.simplify(['erf', '0'], mode='f64') == ['0']
    assert bare.simplify(['besselj', '0', '0'], mode='f64') == ['1']
    # erf is odd in float64 bit for bit, so its sign leaves it
    assert bare.simplify(['erf', 'neg', 'x0'], mode='f64') == ['neg', 'erf', 'x0']


def test_the_evaluator_folds_with_the_deployed_kernels(bare):
    import simplipy.operators as ops
    ev = bare._core.evaluate_constant_subtree
    for toks, f, args in ((['erf', '0.5'], ops.erf, (0.5,)), (['harmonic', '2.5'], ops.harmonic, (2.5,)),
                          (['besselj', '1', '2'], ops.besselj, (1.0, 2.0)),
                          (['atan2', '-3', '0.5'], ops.atan2, (-3.0, 0.5)),
                          (['mod', '-7', '3'], ops.mod, (-7.0, 3.0))):
        want = float(f(*[np.array([a]) for a in args])[0])
        assert float(ev(toks)) == want, toks


def test_judge_lane_for_heads_without_a_contract_kernel():
    j = _contract.judge_rule
    ok = j('floor floor _0'.split(), 'floor _0'.split())
    assert ok['verdict'] == 'F64-ONLY' and ok['tier'] == 'f64', ok
    assert j('max _0 _0'.split(), '_0'.split())['tier'] == 'f64'
    # a law of the reals that float64 breaks (x = -1e-17): no tier at all
    bad = j('floor + _0 1'.split(), '+ floor _0 1'.split())
    assert bad['verdict'] == 'F64-ONLY' and bad['tier'] == 'reject', bad
    # a constant needs the contract's witness: refused
    assert j('floor <constant>'.split(), '<constant>'.split())['verdict'] == 'UNSUPPORTED-SHAPE'
    # a head the judge does not know at all stays refused
    assert j('gamma _0'.split(), '_0'.split())['verdict'] == 'UNSUPPORTED-SHAPE'
    assert _contract._tier('F64-ONLY', None) == 'reject'


def test_monitor_and_oracle_refuse_unknown_heads():
    assert 'floor' in _monitor.unknown_tokens(['floor', 'x0'])
    with pytest.raises(_monitor.Unresolved):
        _monitor._zero_signbit('floor', [0.0])
    from simplipy.promotion._hp_equiv import evaluate
    with pytest.raises(ValueError):
        evaluate(['floor', 'x0'], {'x0': 1.5}, [])


def test_a_small_mine_mines_new_head_rules():
    """End to end, in every mode: rules the float64 comparison certifies appear in the f64 and
    permissive files; the real file holds none (the contract cannot judge the heads)."""
    full = yaml.safe_load(open(ACJ2_4))['operators']
    ops = {k: full[k] for k in ('neg', 'floor', 'ceil', 'max')}
    e = SimpliPyEngine(operators=ops, rules=[], rules_real=[], rules_permissive=[])
    e.find_rules(max_source_pattern_length=3, max_target_pattern_length=3, dummy_variables=2,
                 extra_internal_terms=['0', '1'], X=128, seed=3, reset_rules=True,
                 promote_sorts=False, judge_processes=1)

    def has(rules, lhs, rhs):
        strip = lambda t: [x if x[0] not in '?_!$' or not x[1:].isdigit() else '#' + x[1:]  # noqa: E731
                           for x in t]
        return any(strip(list(a)) == lhs.split() and strip(list(b)) == rhs.split() for a, b in rules)

    for rules in (e.simplification_rules, e.permissive_simplification_rules):
        assert has(rules, 'floor floor #0', 'floor #0'), rules
        assert has(rules, 'max #0 #0', '#0'), rules
    heads = {'floor', 'ceil', 'max'}
    assert not any(heads & set(list(a) + list(b)) for a, b in e.real_simplification_rules)
