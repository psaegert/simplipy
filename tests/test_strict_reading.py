"""Strict (`f64`) holds a rule with variables where float64's two sides agree (F6), or where the reals
hold the rule at every point where they give both sides a value and float64's two sides agree in
kind -- both finite -- there, and as F6 asks at every other point. The second admits the
identities float64 only rounds, however far a jump or a pole magnifies the rounding; it admits
nothing false in the reals, and no difference where the reals have no value to round. Where the
reals give one side no value, float64 decides by F6 alone. The judge decides it in one place
(`_contract.strict_reading`, `kind_holds`), which every reader of the verdict reads."""
import pytest

from simplipy.verify import MODE_TIERS, _contract, effective_tier, verify_rule

ADMITTED = [
    ('sin + x0 np.pi', 'neg sin x0'),
    ('tan + x0 np.pi', 'tan x0'),
    ('exp + x0 log 2', '* 2 exp x0'),
    ('- + x0 1 x0', '1'),
    ('- pow + x0 1 2 pow x0 2', '+ * 2 x0 1'),
    ('floor + x0 1', '+ floor x0 1'),
    ('floor tanh abs x0', '0'),
    ('sinh asinh x0', 'x0'),
]

#: True in the reals wherever they give both sides a value, and float64 agrees by F6 at every point
#: where they give one side none.
ADMITTED_POINTWISE = [
    # at t = -inf (`t = log(floor(x))` on [0, 1)) the left side has no real value (1/0) and the
    # right side is +inf; float64 gives +inf on both sides
    ('inv pow 3 _0', 'pow / 1 3 _0'),
    ('pow / 1 3 _0', 'inv pow 3 _0'),
    # at _0 = pi/2 the reals give the left side no value (tan(pi/2)); float64 gives
    # 6.123233995736766e-17 on both sides at its double
    ('cos atan tan _0', 'abs cos _0'),
]

REFUSED = [
    # +inf against x0 from x0 = 19.06: a change of kind
    ('atanh tanh x0', 'x0'),
    # NaN against x0 for x0 < 0: a change of kind on a set of positive measure
    ('exp log x0', 'x0'),
    # false in the reals
    ('sin + x0 1', 'sin x0'),
    ('- pow + x0 1 2 pow x0 2', '* 2 x0'),
    # 1 against -1 on x0 > 0, where the reals give neither side a value (1/0): F6 there
    ('tanh / 1 - abs x0 x0', 'tanh neg / 1 - x0 abs x0'),
]

#: False in the reals where no affordable rung reads them apart, and refuted at float resolution:
#: for x0 > 74.7 atan(atan(x0)) passes 1, while erf(x0) lies within 1e-2400 of it.
FALSE_PAST_THE_LADDER = [
    ('ceil atan atan x0', 'ceil erf x0'),
    ('floor atan atan x0', 'floor erf x0'),
    ('ceil atan atan !0', 'ceil erf !0'),
    ('floor atan atan !0', 'floor erf !0'),
]


@pytest.mark.parametrize('lhs, rhs', ADMITTED)
def test_strict_admits_the_identities_float64_only_rounds(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['realised'] is True and r['tier'] == 'core', r
    assert r['verdict'] in _contract.HOLDS_IN_THE_REALS, r
    assert r['realised_kinds'] is True, r


@pytest.mark.parametrize('lhs, rhs', ADMITTED_POINTWISE)
def test_where_the_reals_give_one_side_no_value_float64_decides(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['reals_valued_holds'] is True and r['realised_kinds'] is True, r
    assert r['realised'] is True and r['tier'] == 'core', r


@pytest.mark.parametrize('lhs, rhs', FALSE_PAST_THE_LADDER)
def test_a_counterexample_at_float_resolution_refutes_the_reals(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert (r['verdict'], r['clause']) == ('KILL', 'a-float-resolution'), r
    assert r['reals_valued_holds'] is False and r['realised'] is False, r
    assert r['tier'] == 'reject', r


@pytest.mark.parametrize('lhs, rhs', REFUSED)
def test_strict_refuses_a_change_of_kind_and_whatever_is_false_in_the_reals(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['realised'] is not True and effective_tier(r) not in ('core',), r


def test_the_two_readings_and_the_reals_decide_strict() -> None:
    read = _contract.strict_reading
    assert read(True, False, False) is True               # (a) alone
    assert read(False, True, True) is True                # (b)
    assert read(False, True, False) is False              # false in the reals
    assert read(False, False, True) is False              # a change of kind
    assert read(False, True, None) is None                # the reals reached no answer
    assert read(None, None, True) is None
    # a fully numeric rule's one point: the reals' verdict is their reading there
    assert [_contract.reals_hold(v) for v in ('CERTIFIED', 'ENGINE-MISALIGN', 'KILL',
                                              'UNRESOLVED-COVERAGE')] == [True, True, False, None]


def test_the_range_standard_reads_overflowed_terms_through_their_true_value() -> None:
    """At x = 800 both terms of `besseli(1, -x) + besseli(1, x)` overflow and float64 gives
    -inf + inf = NaN; the true value 0 is accepted there. A kernel's NaN is no range event:
    `besseli(1, x)` is NaN for 0 < |x| <= 2.2250738585072014e-308, where the true value is
    finite, so the rule gives a value where float64's left side has none on an interval, and
    strict refuses it; permissive, on the reals' values, holds it."""
    from simplipy import _core
    from simplipy import operators
    lhs, rhs = '+ besseli 1 neg x0 besseli 1 x0'.split(), ['0']
    tl, tr = _contract.parse(lhs, number=True), _contract.parse(rhs, number=True)
    v, _ = _contract._point_verdict(tl, tr, lambda: {'?0': _contract.mpf(800)})
    assert v == 'eq'
    assert _core.range_event('besseli', [1.0, 800.0], float(operators.besseli(1.0, 800.0)))
    tiny = 2.2250738585072014e-308
    assert operators.besseli(1.0, tiny) != operators.besseli(1.0, tiny)        # NaN
    assert not _core.range_event('besseli', [1.0, tiny], float(operators.besseli(1.0, tiny)))
    r = verify_rule(lhs, rhs)
    assert r['realised'] is False and r['permissive'] is True and r['tier'] == 'permissive', r


def test_the_second_reading_reads_kind_only_where_the_reals_give_both_sides_a_value() -> None:
    r = verify_rule('tanh / 1 - abs x0 x0'.split(), 'tanh neg / 1 - x0 abs x0'.split())
    assert r['verdict'] in _contract.HOLDS_IN_THE_REALS, r
    assert (r['realised_f6'], r['realised_kinds'], r['tier']) == (False, False, 'permissive'), r
    assert _contract.kind_holds('REAL-CHANGE', lambda: True) is True
    assert _contract.kind_holds('REAL-CHANGE', lambda: False) is False
    assert _contract.kind_holds('REAL-CHANGE', lambda: None) is None
    assert _contract.kind_holds('INF-CHANGE', lambda: True) is False


def test_a_ground_rule_is_read_on_each_modes_value() -> None:
    # one point: float64's 1.0000000000000002 is no 1, but the reals give both sides the value 1
    # and float64's two values are both finite, so f64 holds it too
    r = verify_rule('exp sin np.pi'.split(), ['1'])
    assert r['realised'] is True and r['tier'] == 'core', r
    # where float64's kind differs from the reals', only float64's own double holds in f64
    r = verify_rule('tan asin 1'.split(), ['0'])
    assert r['realised'] is False, r


@pytest.mark.parametrize('lhs, rhs', [ADMITTED[0], ADMITTED[5]])
def test_the_mine_licenses_what_strict_holds(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert effective_tier(r) in MODE_TIERS['f64'] and effective_tier(r) in MODE_TIERS['permissive']
