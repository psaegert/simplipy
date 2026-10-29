"""Sort promotion asks the judge, rung by rung, and decides a rule's sort from the rule alone.

Each rung is the rule with every slot respelled at one sort (`_`, then `!`), and it is licensed
where the judge's tier at that sort holds in the mode and no named policy refuses it
(`verify.licensed`). Promotion has no evaluator or probe list of its own, so a rule's sort is a
function of the rule and the mode: never of the rules judged before it.
"""
import os
import time

import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.promotion import promote
from simplipy.verify import licensed, verify_rule

ACJ2_4 = os.path.join(os.path.dirname(__file__), '..', 'assets', 'engines', 'acj2-4', 'config.yaml')
MODES = ('f64', 'real', 'permissive')


@pytest.fixture(scope='module')
def bare():
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_real=[], rules_permissive=[])


def _by_rule(kept):
    """kept rules keyed by their `?`-respelled source"""
    return {' '.join('?' + t[1:] if t[:1] in '_!$' else t for t in lhs): ' '.join(lhs) + ' -> ' + ' '.join(rhs)
            for lhs, rhs in kept}


RULES = [(r.split(' -> ')[0].split(), r.split(' -> ')[1].split()) for r in (
    'atan sinh ?0 -> asin tanh ?0', 'atanh tanh ?0 -> ?0', 'sin acos tanh ?0 -> inv cosh ?0',
    '+ ?0 * 0 ?1 -> ?0', 'max ?0 acos <constant> -> max <constant> ?0',
    'acos <constant> -> <constant>', 'ceil inv cosh ?0 -> 1')]


def test_a_rules_sort_does_not_depend_on_the_rules_before_it(bare):
    for mode in MODES:
        together = _by_rule(promote(RULES, bare, rule_mode=mode)[0])
        assert together == _by_rule(promote(RULES[::-1], bare, rule_mode=mode)[0]), mode
        for rule in RULES:
            alone = _by_rule(promote([rule], bare, rule_mode=mode)[0])
            key = ' '.join(rule[0])
            if key in together:
                assert alone[key] == together[key], (mode, alone[key], together[key])


def _strongest(lhs, rhs, mode, engine):
    """The sort the rungs give: the first of `_` and `!` the judge licenses, else `?`."""
    for sort in '_!':
        spell = [sort + t[1:] if t[:1] == '?' and t[1:].isdigit() else t for t in lhs + rhs]
        rung_l, rung_r = spell[:len(lhs)], spell[len(lhs):]
        if licensed(rung_l, rung_r, mode, engine, detail=verify_rule(rung_l, rung_r))['licensed']:
            return ' '.join(rung_l) + ' -> ' + ' '.join(rung_r)
    return ' '.join(lhs) + ' -> ' + ' '.join(rhs)


@pytest.mark.parametrize('mode', MODES)
def test_each_rule_ships_at_the_strongest_sort_the_judge_licenses(bare, mode):
    """`ceil(1/cosh(x)) -> 1` holds for every real x, and at x = +-inf the left side is
    ceil(0) = 0: the `_` rung is refused in f64 and real, the `!` rung (a finite subtree) holds.
    permissive binds a slot to real values only, so the `_` rung holds there."""
    kept = _by_rule(promote(RULES, bare, rule_mode=mode)[0])
    for lhs, rhs in RULES:
        key = ' '.join(lhs)
        if key in kept and '?' in key:
            assert kept[key] == _strongest(lhs, rhs, mode, bare), (mode, key, kept[key])
    want = 'ceil inv cosh _0 -> 1' if mode == 'permissive' else 'ceil inv cosh !0 -> 1'
    assert kept['ceil inv cosh ?0'] == want, (mode, kept)


def test_each_rung_is_asked_of_the_judge(bare):
    """The rungs go through the judge the caller passes, at their sorts: a judge that refuses
    everything leaves every rule at `?` (the positive controls ask the rule judge itself)."""
    asked = []

    def refusing(pairs):
        asked.extend((tuple(lhs), tuple(rhs)) for lhs, rhs in pairs)
        return ['reject'] * len(pairs)

    kept, report = promote(RULES, bare, rule_mode='real', judge=refusing)
    assert ('ceil', 'inv', 'cosh', '_0') in [lhs for lhs, _ in asked]
    assert ('ceil', 'inv', 'cosh', '!0') in [lhs for lhs, _ in asked]
    assert not report['promoted_any'] and not report['promoted_bang']
    assert all(not any(t[:1] in '_!$' for t in lhs) for lhs, _ in kept)


def test_a_miscalibrated_rule_judge_stops_promotion(bare, monkeypatch):
    """The positive controls ask the rule judge: one that holds `x + 1 -> x` raises."""
    import simplipy.verify as verify
    real = verify.verify_rule
    monkeypatch.setattr(verify, 'verify_rule', lambda lhs, rhs, *a, **k: dict(
        real(lhs, rhs, *a, **k), tier='core'))
    with pytest.raises(RuntimeError, match='positive control'):
        promote(RULES, bare, rule_mode='real')


def test_a_rule_without_slots_passes_through_at_once(bare):
    """A rule without slots has no rung: promotion asks nothing of it. These constant
    collapses (from an acj-5-4 mine) read a periodic function of an astronomically large
    argument, and an evaluator of promotion's own once sat in one of them for 26 hours."""
    offenders = [(('atanh', 'neg', 'cos', 'exp', '<constant>'), ('<constant>',)),
                 (('asin', 'sin', 'abs', 'sinh', '<constant>'), ('<constant>',)),
                 (('acosh', 'sin', 'pow', '(-8)', '<constant>'), ('float("nan")',)),
                 (('log', 'exp', 'cos', 'sinh', '<constant>'), ('<constant>',))]
    asked = []

    def judge(pairs):
        asked.extend((tuple(lhs), tuple(rhs)) for lhs, rhs in pairs)
        return ['reject'] * len(pairs)      # the seeds, refused here, are asked about too

    start = time.monotonic()
    kept, _ = promote(offenders, bare, rule_mode='real', run_positive_controls=False,
                      judge=judge)
    assert time.monotonic() - start < 5.0
    assert [(tuple(lhs), tuple(rhs)) for lhs, rhs in kept] == offenders
    assert not set(asked) & set(offenders)
