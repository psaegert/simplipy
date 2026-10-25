"""Sort promotion decides a rule's sort from the rule alone, and decisively.

Every bar drew its generic valuations from one numpy stream shared by the whole ladder, so a
rule's probes depended on how many rules were judged before it, and float64's rounding band
at a draw decided its sort: `atan(sinh(x)) -> asin(tanh(x))` passed the `!` bar on 4 of 12
seeds. Its exact arbiter re-judged every valuation, and at the `1e20` atom that needs 10^20
digits, so an arbitration never promoted. Now the valuations are fixed (the atoms, the generic
probes with float64's hard points, the rule-correlated points) and the arbiter decides exactly
the valuations float64 left open.
"""
import os

import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.promotion import _f64_eval, _ladder, _overturn, _pointwise, promote

ACJ2_4 = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4', 'config.yaml')
MODES = ('f64', 'real', 'permissive')


@pytest.fixture(scope='module')
def bare():
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    eng = SimpliPyEngine(operators=ops, rules=[], rules_real=[], rules_permissive=[])
    _f64_eval.configure(eng)
    return eng


def test_the_valuations_are_a_function_of_the_slots():
    assert _pointwise.valuations_for(['_0']) == _pointwise.valuations_for(['_0'])
    assert _pointwise.valuations_for(['_0', '_1', '_2']) == _pointwise.valuations_for(['_0', '_1', '_2'])
    assert _ladder.atom_valuations(list('abcd')) == _ladder.atom_valuations(list('abcd'))
    # the hard points: tanh's float64 saturation band and exp's overflow edge, both signs
    for v in ('18.5', '(-18.5)', '19.5', '709.5', '(-710.5)'):
        assert v in _pointwise.GENERIC


def test_a_saturating_identity_is_decided_exactly(bare):
    """float64 reads `asin(tanh(18.5))` in its rounding band; the arbiter decides it there."""
    lhs, rhs = 'atan sinh ?0'.split(), 'asin tanh ?0'.split()
    assert _pointwise.judge(lhs, rhs, _ladder.finite_only(_pointwise.valuations_for(['?0'])))[0] \
        == 'UNDECIDED'
    assert _ladder.judge_bang(lhs, rhs) == ('PROMOTE', 'exact-arbiter')
    # a false rule stays refuted, and a disagreement at inf is never arbitrated
    assert _ladder.judge_bang('atan sinh ?0'.split(), 'asin sin ?0'.split())[0] == 'DEMOTE'
    assert not _overturn.arbitrable('DEMOTE', ({'?0': 'float("inf")'}, 1.0, 2.0))


def _by_rule(kept):
    """kept rules keyed by their `?`-respelled source"""
    return {' '.join('?' + t[1:] if t[:1] in '_!$' else t for t in lhs): ' '.join(lhs) + ' -> ' + ' '.join(rhs)
            for lhs, rhs in kept}


def test_a_rules_sort_does_not_depend_on_the_rules_before_it(bare):
    rules = [(r.split(' -> ')[0].split(), r.split(' -> ')[1].split()) for r in (
        'atan sinh ?0 -> asin tanh ?0', 'atanh tanh ?0 -> ?0', 'sin acos tanh ?0 -> inv cosh ?0',
        '+ ?0 * 0 ?1 -> ?0', 'max ?0 acos <constant> -> max <constant> ?0',
        'acos <constant> -> <constant>')]
    for mode in MODES:
        together = _by_rule(promote(rules, bare, rule_mode=mode, run_positive_controls=False)[0])
        assert together == _by_rule(promote(rules[::-1], bare, rule_mode=mode,
                                            run_positive_controls=False)[0]), mode
        for rule in rules:
            alone = _by_rule(promote([rule], bare, rule_mode=mode, run_positive_controls=False)[0])
            key = ' '.join(rule[0])
            if key in together:
                assert alone[key] == together[key], (mode, alone[key], together[key])
        assert together['atan sinh ?0'] in ('atan sinh _0 -> asin tanh _0',
                                            'atan sinh !0 -> asin tanh !0'), (mode, together)
