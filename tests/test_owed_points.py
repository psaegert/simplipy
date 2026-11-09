"""The points a mode owes a reading at. The battery takes each slot's atoms because they are
exceptional (zero, the units, the infinities and the rest of each sort's exact values), and a
ground rule has its one point: a mode whose reading leaves one of them unread refuses the rule
there, with uncertainty, and names the point (`unread_owed`). The grid and the other sampled
points keep the coverage quota. And the table reads a Bessel head at a zero argument exactly,
whether the order is written or bound to a slot."""
from fractions import Fraction

import pytest
import yaml

from simplipy import SimpliPyEngine, _core
from simplipy.verify import _contract, verify_rule
from test_new_heads import ACJ2_4

NEW = {'erfc': 1, 'bessely': 2, 'besseli': 2, 'besselk': 2}


def reading(lhs: str, rhs: str, **at) -> tuple | None:
    """The core's float-resolution reading of `lhs` against `rhs`, each named slot at its value."""
    names, extra = {}, {}
    trees = [_contract.parse(side.split()) for side in (lhs, rhs)]
    tokens = [_contract._tree_tokens(t, names, extra) for t in trees]
    order = list(names)
    specs = [_contract._spec(extra[n] if n in extra else at[n]) for n in order]
    return _core.table_float_resolution(tokens[0], tokens[1], order, specs,
                                        dict(_contract.ARITY))


@pytest.mark.parametrize('head', ['besselj', 'besseli'])
def test_a_bessel_head_at_a_zero_argument_reads_exactly(head) -> None:
    """`J_0(0) = I_0(0) = 1` exactly, so `atanh` of it is +inf: read at an order bound to 0 as at
    a written one (an approximate 1 left atanh inside its pole's band, and the point unread)."""
    assert reading(f'atanh {head} _0 0', '0', _0=Fraction(0)) == ('doubles', float('inf'), 0.0)
    assert reading(f'atanh {head} 0 0', '0') == ('doubles', float('inf'), 0.0)
    assert reading(f'atanh {head} _0 0', 'float("inf")', _0=Fraction(0)) == \
        ('doubles', float('inf'), float('inf'))
    spec, unresolved, *_ = _core.TableExpr([head, 'x0', '0'], ['x0'], dict(_contract.ARITY)) \
        .eval_real([('q', '0', '1')], 50)
    assert not unresolved and tuple(spec) == ('q', '1', '1')


@pytest.mark.parametrize('lhs', ['atanh besselj _0 0', 'atanh besseli _0 0',
                                 'atanh besseli ?0 0', 'atanh besseli 0 0'])
def test_atanh_of_a_bessel_head_at_zero_is_not_zero(lhs) -> None:
    """At order 0 the left side is atanh(1) = +inf: refused in every mode."""
    d = verify_rule(lhs.split(), ['0'])
    assert d['tier'] == 'reject' and d['permissive'] is False, d


def test_the_fully_numeric_form_takes_its_table_value() -> None:
    """P5: `atanh(besseli(0, 0))` is +inf, and no rule changes it, even the false one installed
    by hand (the load does not judge): the fold reads the table's value first."""
    ops = yaml.safe_load(open(ACJ2_4))['operators']
    for name, arity in NEW.items():
        ops[name] = {'realization': f'simplipy.operators.{name}', 'alias': [], 'arity': arity,
                     'commutative': False}
    false = [[['atanh', head, '_0', '0'], ['0']] for head in ('besseli', 'besselj')]
    engine = SimpliPyEngine(operators=ops, rules=[], rules_permissive=false)
    for mode in ('f64', 'permissive'):
        assert engine.simplify('atanh(besseli(0, 0))', mode=mode) == 'inf'
        assert engine.simplify('atanh(besselj(0, 0))', mode=mode) == 'inf'


@pytest.mark.parametrize('lhs', ['floor erf exp _0', 'floor tanh * 100 abs _0'])
def test_a_bounded_function_inside_its_band_tells_the_ladder_its_distance(lhs) -> None:
    """`erf(exp(3))` is 1 - 1.7e-177 and `tanh(300)` 1 - 2.7e-261, inside the band of 1 at fifty
    digits: their distance to 1 sizes the next rung, where `floor` of them reads 0. The rule holds
    by permissive's own criterion at every owed point; float64 rounds them to 1.0 (f64 refuses)."""
    d = verify_rule(lhs.split(), ['0'])
    assert d['permissive'] is True and 'unread_owed' not in d, d
    assert d['realised'] is False, d


def unread_at(monkeypatch, value) -> None:
    """Leave the reals' ladder and permissive's float-resolution reading unread wherever a slot is
    bound to `value`."""
    point_verdict, float_resolution = _contract._point_verdict, _contract._float_resolution

    def hit(env) -> bool:
        return any(float(v) == value for v in env().values())

    monkeypatch.setattr(_contract, '_point_verdict',
                        lambda tl, tr, env, owed=False: (None, False) if hit(env)
                        else point_verdict(tl, tr, env, owed))
    monkeypatch.setattr(_contract, '_float_resolution',
                        lambda tl, tr, env, extends=False: None if hit(env)
                        else float_resolution(tl, tr, env, extends))


def test_an_owed_point_left_unread_refuses_permissives_own_reading(monkeypatch) -> None:
    """`sin(_0)*1 -> sin(_0)` holds at every point; left unread at the atom 1/2 alone, permissive's
    own reading reaches no answer (4.10, `permissive_own`) and names the point. It still holds
    in permissive through f64."""
    lhs, rhs = '* 1 sin _0'.split(), 'sin _0'.split()
    assert verify_rule(lhs, rhs)['permissive'] is True
    unread_at(monkeypatch, 0.5)
    d = verify_rule(lhs, rhs)
    assert d['permissive_own'] is None and {'_0': 0.5} in d['unread_owed']['permissive'], d
    assert d['realised'] is True and d['permissive'] is True and d['tier'] == 'core', d


def test_a_sampled_point_left_unread_keeps_the_quota(monkeypatch) -> None:
    """A grid point left unread counts neither way: the rule still holds."""
    lhs, rhs = '* 1 sin _0'.split(), 'sin _0'.split()
    unread_at(monkeypatch, _contract.GRID[60])
    d = verify_rule(lhs, rhs)
    assert d['permissive'] is True and 'unread_owed' not in d, d


def test_a_constants_magnitude_sample_left_unread_keeps_the_quota() -> None:
    """`exp(x0 + C) -> D*exp(x0)`: at the span sample C = 3000 the witness e^3000 is read to
    fewer digits than the rungs, and its points are unread. The constants' span and saturation
    magnitudes are samples, as the slots' magnitude ladder is: the rule holds in permissive."""
    assert 3000.0 in _contract.MAGNITUDE_CONSTS
    d = verify_rule('exp + x0 <constant>'.split(), '* <constant> exp x0'.split())
    assert d['permissive'] is True and 'unread_owed' not in d, d


def test_an_owed_point_left_unread_refuses_f64(monkeypatch) -> None:
    """In f64 an owed point is unread where an operation left the range, float64's sides differ
    and no rung reads the true value: f64 reaches no answer there."""
    lhs, rhs = '* 1 sin _0'.split(), 'sin _0'.split()
    d_eval, point_verdict = _contract.d_eval, _contract._point_verdict

    def at_half(env) -> bool:
        return any(float(v) == 0.5 for v in env.values())

    def leaving_the_range(tree, env, in_range=None):
        v = d_eval(tree, env, in_range)
        if at_half(env) and in_range is not None:
            in_range[0] = False
            return v + (1.0 if tree is not None and tree[0] == '*' else 0.0)
        return v

    monkeypatch.setattr(_contract, 'd_eval', leaving_the_range)
    monkeypatch.setattr(_contract, '_point_verdict',
                        lambda tl, tr, env, owed=False: (None, False) if at_half(env())
                        else point_verdict(tl, tr, env, owed))
    d = verify_rule(lhs, rhs)
    assert d['realised'] is None and {'_0': 0.5} in d['unread_owed']['f64'], d


def test_the_f64_bar_keeps_a_reading_without_an_answer(monkeypatch) -> None:
    """The bar is ANDed into each of strict's readings (F6's and the agreement in kind): it
    refuses where it fails, and a reading that reached no answer stays without one (it read as a
    pass)."""
    monkeypatch.setattr(_core, 'f64_bar_sorted', lambda *a, **k: (True, 10, None, []))
    lhs, rhs = 'erfc _0'.split(), 'erfc _0'.split()

    def judged(before) -> dict:
        return {'realised_f6': before, 'realised_kinds': before, 'verdict': 'CERTIFIED',
                'permissive_own': True}

    for before, after in ((None, None), (True, True), (False, False)):
        r = _contract._with_f64_bar(lhs, rhs, judged(before))
        assert r['realised'] is after, (before, r)
    monkeypatch.setattr(_core, 'f64_bar_sorted', lambda *a, **k: (False, 10, [0.5], []))
    r = _contract._with_f64_bar(lhs, rhs, judged(None))
    assert r['realised'] is False, r
