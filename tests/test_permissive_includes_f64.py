"""Permissive includes f64: a rule holds in permissive where it holds in f64, or where
permissive's own criterion holds (the reals' values at float resolution, with its extensions).
The judge's tiers, evidence, the mint's pair judgment and the coverage gate each read it."""
import os

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine
from simplipy.verify import _contract, licensed, verify_rule

REPO = os.path.join(os.path.dirname(__file__), '..')


@pytest.fixture(scope='module')
def acj():
    """A rule-less engine over the acj2-4 vocabulary (the 0.15 heads included)."""
    config = os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')
    return SimpliPyEngine(operators=yaml.safe_load(open(config))['operators'], rules=[])


@pytest.mark.parametrize('lhs, rhs', [
    # in float64, cos(asin(1.0)) is 6.1e-17 and the left side is 1; in the reals it is 0 at
    # pi/2, and so on all of (0, 1] where the slot is bound to asin(ceil(x))
    ('ceil abs cos ?0', '1'),
    ('ceil abs cos !0', '1'),
    # float64's asin(1.0) falls short of pi/2, and tan of it is finite; in the reals tan(pi/2)
    # has no value at x = 1
    ('erf sinh atanh _0', 'erf tan asin _0'),
    ('tanh sinh atanh _0', 'tanh tan asin _0'),
    # float64's 1/0 is inf, so both sides are 0 at x0 = 0; in the reals 1/0 has no value there
    ('* 2 / tanh ?0 + 1 pow tanh ?0 2', '/ 2 + tanh ?0 inv tanh ?0'),
])
def test_a_rule_f64_holds_holds_in_permissive(acj, lhs, rhs) -> None:
    """Permissive's reading includes f64's by construction: `permissive` holds wherever
    `realised` does, whatever permissive's own criterion reads (`permissive_own`)."""
    d = verify_rule(lhs.split(), rhs.split())
    assert d['realised'] is True and d['permissive'] is True and d['permissive_own'] is False, d
    assert d['tier'] == 'core'
    assert d['basis'] == {'f64': 'evidence', 'permissive': 'evidence'}
    for mode in ('f64', 'permissive'):
        assert licensed(lhs.split(), rhs.split(), mode, acj, detail=d)['licensed'], mode


def test_a_rule_neither_mode_holds_stays_refused(acj) -> None:
    """`ceil(x0) - floor(x0) -> 1`: the left side is 0 at every integer, in float64 too."""
    d = verify_rule('- ceil ?0 floor ?0'.split(), ['1'])
    assert (d['realised'], d['permissive'], d['tier']) == (False, False, 'reject'), d
    for mode in ('f64', 'permissive'):
        assert not licensed('- ceil ?0 floor ?0'.split(), ['1'], mode, acj,
                            detail=d)['licensed']


def test_permissive_evidence_is_float64s_or_the_reals(acj) -> None:
    """`(-1)^erf(C)` is finite in float64 on |C| >= 5.92, where erf rounds to +-1, and in the
    reals at C = 0 alone: evidence in f64, none in the reals, and so evidence in permissive.
    `(-1)^erf(C) -> C` is licensed in permissive."""
    from simplipy import _core
    lhs, rhs = 'pow (-1) erf <constant>'.split(), ['<constant>']
    arity = dict(_contract.ARITY)
    assert _core.table_evidence(lhs, arity, True) is True
    assert _core.table_evidence(lhs, arity, False) is False
    assert _core.table_mode_evidence(lhs, arity, 'default') is True
    assert _core.table_mode_evidence(lhs, arity, 'permissive') is True
    # and a source with evidence in neither reading has none in either mode
    assert _core.table_mode_evidence('pow (-1) <constant>'.split(), arity, 'permissive') is False
    d = verify_rule(lhs, rhs)
    assert d['tier'] == 'core'
    assert licensed(lhs, rhs, 'permissive', acj, detail=d) == {
        'licensed': True, 'holds': True, 'tier': 'core', 'refused_by': []}


def test_permissive_evidence_is_that_of_a_reading_the_rule_holds_in(acj) -> None:
    """`tan(atan2(x0, 0)) -> nan` holds by permissive's own criterion alone, vacuously: off
    x0 = 0 the left side is tan(pi/2), no value in the reals, and 1.633e16 in float64, where the
    rule is false. Float64's evidence licenses nothing there: refused by evidence."""
    lhs, rhs = 'tan atan2 ?0 0'.split(), ['float("nan")']
    d = verify_rule(lhs, rhs)
    assert (d['realised'], d['permissive'], d['tier']) == (False, True, 'permissive'), d
    got = licensed(lhs, rhs, 'permissive', acj, detail=d)
    assert got == {'licensed': False, 'holds': True, 'tier': 'permissive',
                   'refused_by': ['evidence']}, got


def test_the_mode_reading_includes_f64() -> None:
    """Permissive holds where either reading holds, is refused where both refuse, and reaches no
    answer otherwise."""
    readings = (True, False, None)
    assert [_contract.mode_reading(f, p, 'permissive') for f in readings for p in readings] == [
        True, True, True, True, False, None, True, None, None]
    assert [_contract.mode_reading(f, p, 'f64') for f in readings for p in readings] == [
        True, True, True, False, False, False, None, None, None]


def test_the_permissive_mint_certifies_what_f64s_lane_certifies(acj) -> None:
    """The mint is a pre-filter of the judge: permissive's judge holds `ceil(abs(cos(x0))) -> 1`
    because f64 holds it, so permissive's pair judgment certifies it, through f64's lane
    (its own reading of the special points refuses it at pi/2)."""
    X = _contract.mine_rows(512, 2, np.random.default_rng(0))
    for mode in ('default', 'permissive'):
        ok, why = acj._core.judge_pair('ceil abs cos x0'.split(), ['1'], ['x0', 'x1'],
                                       X.flatten().tolist(), X.shape[0], rtol=1e-11,
                                       atol=1e-12, rule_mode=mode)
        assert ok, (mode, why)
    # a pair neither lane certifies is refused in both
    for mode in ('default', 'permissive'):
        ok, _ = acj._core.judge_pair('- ceil x0 floor x0'.split(), ['1'], ['x0', 'x1'],
                                     X.flatten().tolist(), X.shape[0], rtol=1e-11, atol=1e-12,
                                     rule_mode=mode)
        assert not ok, mode


def test_the_coverage_gate_names_an_f64_rule_permissive_leaves_dearer(acj) -> None:
    """On each f64 rule's generic instance, permissive's answer must cost no more, in
    permissive's price, than the rule's right side: an empty permissive set leaves
    `ceil(abs(cos(x0)))` as it is, dearer than 1, and the set holding the rule covers it."""
    from simplipy.mining import RuleMiner, _uncovered_f64_rules
    rule = (('ceil', 'abs', 'cos', '!0'), ('1',))
    miner = RuleMiner(acj)
    miner._install('permissive', [])
    try:
        assert _uncovered_f64_rules(acj, [rule]) == [
            [list(rule[0]), list(rule[1]), ['ceil', 'abs', 'cos', 'x0'],
             ['ceil', 'abs', 'cos', 'x0']]]
        miner._install('permissive', [rule])
        assert _uncovered_f64_rules(acj, [rule]) == []
    finally:
        miner._install('permissive', [])
