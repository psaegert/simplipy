"""Strict reads a zero's sign where it changes a value on a region.

F6 compares two zeros as one value, so a rule that only changes a zero's sign looked sound,
although an enclosing `1/x` or `atan2` reads the sign: `0*asin(x0) -> 0*acos(x0)` is -0 against
+0 for every x0 < 0, and `1/` of it -inf against +inf. A rule holds in strict only if, besides
its agreement in float64 (a) or in kind (b), (i) float64 never gives its two sides zeros of
opposite sign, and (ii) (b) never admits a difference at which one side's value turns on a zero's
sign while the other's does not, on a set of positive measure. A difference at isolated points is
admitted, as the engine's constructors treat a zero. Permissive reads one zero and is unchanged.
"""
import math

import pytest

from simplipy.verify import _contract

INF = 'float("inf")'


def judge(lhs, rhs):
    return _contract.judge_rule(lhs.split(), rhs.split())


def strict(detail):
    return _contract.judged_reading(detail, 'f64')


def permissive(detail):
    return _contract.judged_reading(detail, 'permissive')


#: Refused in strict, each with the event that refuses it and its region.
REFUSED = [
    # (i): -0 against +0 for x0 < 0
    ('* 0 asin x0', '* 0 acos x0', 'opposite-zeros'),
    # (i): ceil of a value in (-1, 0] is -0, for every x0 that is no integer
    ('ceil mod x0 -1', '0', 'opposite-zeros'),
    # (ii): float64 gives -pi for x0 < 0, the reals' one zero pi; (b) admitted it as both finite
    ('atan2 * 0 x0 -1', 'np.pi', 'turns-on-a-zero-sign'),
    # (i) at a slot that matches any subexpression: -0 where `_0` binds `ceil(x1)` on (-1, 0)
    ('log exp _0', '_0', 'opposite-zeros'),
    ('log exp atanh _0', 'atanh _0', 'opposite-zeros'),
    # (i) on float64's plateau of exp: exp(x0) is 1.0 for |x0| < 1.1e-16, where the left side is
    # +0 and the right side -0
    ('log exp neg x0', 'neg log exp x0', 'opposite-zeros'),
]


@pytest.mark.parametrize('lhs, rhs, event', REFUSED)
def test_strict_refuses_a_zero_sign_changed_on_a_region(lhs, rhs, event) -> None:
    d = judge(lhs, rhs)
    assert strict(d) is False and d['zero_sign_refused'] is True, d
    # float64 with the two zeros one value holds it: the refusal is a zero's sign alone
    assert d['realised_one_zero'] is True, d
    assert event in {e for e, _ in d.get('zero_sign') or []} | \
        {(d.get('zero_sign_bar') or {}).get('event')}, d


@pytest.mark.parametrize('lhs, rhs, event', REFUSED)
def test_permissive_holds_each_refused_rule(lhs, rhs, event) -> None:
    """Permissive reads one zero: each rule strict refuses here stays permissive's."""
    d = judge(lhs, rhs)
    assert permissive(d) is True and d['tier'] == 'permissive', d


@pytest.mark.parametrize('lhs, rhs', [('neg acosh erf _0', 'acosh erf _0'),
                                      ('neg acosh tanh _0', 'acosh tanh _0')])
def test_permissive_keeps_a_rule_it_held_through_f64_with_float64s_evidence(lhs, rhs) -> None:
    """Permissive includes f64 with the two zeros one value, evidence and all. Past erf's (tanh's)
    saturation float64 gives -0 against +0, so strict refuses the rule; the reals give the left
    side no value anywhere (erf < 1), so only float64's evidence licenses it, and permissive
    keeps it as it held it before strict read a zero's sign: in the licence verify-mine reads
    and in the mine's, which reads a tier."""
    from simplipy.mining import _licences
    from simplipy.verify import _vocabulary_engine, licensed
    d = judge(lhs, rhs)
    assert strict(d) is False and d['realised_one_zero'] is True, d
    engine = _vocabulary_engine()
    assert licensed(lhs.split(), rhs.split(), 'permissive', engine)['licensed'] is True
    assert licensed(lhs.split(), rhs.split(), 'f64', engine)['licensed'] is False
    pair = (tuple(lhs.split()), tuple(rhs.split()))
    tiers = _licences(engine, 'permissive', [pair], lambda pairs: [d['tier'] for _ in pairs])
    assert tiers[pair]['licensed'] is True, tiers


@pytest.mark.parametrize('lhs, rhs', [
    # +0 against -0 at x0 = -0 alone
    ('log exp x0', 'x0'),
    # +0 against -0 at x0 = 1 alone
    ('log inv x0', 'neg log x0'),
    # +0 against -0 at x0 = -0 alone, where floor(-0 + 0.5) is +0
    ('floor + floor x0 0.5', 'floor x0'),
])
def test_strict_admits_a_zero_sign_changed_at_an_isolated_point(lhs, rhs) -> None:
    d = judge(lhs, rhs)
    assert strict(d) is True and d['tier'] == 'core', d
    assert not d.get('zero_sign') and not d.get('zero_sign_bar'), d


def test_a_value_turning_on_a_zero_sign_is_read_only_where_float64_differs() -> None:
    """(ii) restricts (b): where float64's two sides agree, a zero's sign one side reads is no
    change float64 can see. `1/(0*x1)` and `inf/x1` are the same value at every x1."""
    d = judge('inv * 0 x1', f'/ {INF} x1')
    assert strict(d) is True, d


def test_the_range_standard_comes_first() -> None:
    """Past x0 = 27.3 `erfc` underflows to 0, so float64 gives `erf(x0) - 1` +0 against
    `-erfc(x0)`'s -0; the true value of the underflow, which carries no zero's sign, stands."""
    d = judge('- erf x0 1', 'neg erfc x0')
    assert strict(d) is True, d
    # bound to any subexpression, the slot can be +inf on a region, where both sides are exact
    d = judge('- erf _0 1', 'neg erfc _0')
    assert strict(d) is False and permissive(d) is True, d


def test_a_fitted_constant_carries_the_left_sides_zero_sign() -> None:
    """In f64 a fitted constant is a double, -0.0 included: the witness of `max(C, -0.0) -> C` at
    C = -1.5 is -0.0, the source's own value, and the rule holds."""
    d = judge('max <constant> -0.0', '<constant>')
    assert strict(d) is True and d['tier'] == 'core', d
    lhs, rhs = _contract.variables_as_dummies(['max', '<constant>', '-0.0'], ['<constant>'])
    tl = _contract.parse(lhs, '<C_L>', number=True)
    tr = _contract.parse(rhs, '<C_R>', number=True)
    w = _contract.fit_witness(tl, tr, set(), {'<C_L0>': -1.5}, None, '<C_R0>')
    assert w == 0 and math.copysign(1.0, w) == -1.0, w
    for lhs, rhs in (('atan2 <constant> float("inf")', '<constant>'),
                     ('inv bessely <constant> 0', '<constant>')):
        d = judge(lhs, rhs)
        assert strict(d) is True, (lhs, rhs, d)
