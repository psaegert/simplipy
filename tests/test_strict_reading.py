"""Strict (`f64`) holds a rule with variables where float64's two sides agree (F6), or where the rule
is true in the reals and float64's two sides agree in kind -- both finite -- at every point where
the reals give both sides a value, and as F6 asks at every other point. The second admits the
identities float64 only rounds, however far a jump or a pole magnifies the rounding; it admits
nothing false in the reals, and no difference where the reals have no value to round. The judge
decides it in one place (`_contract.strict_reading`, `kind_holds`), which every reader of the
verdict reads."""
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


@pytest.mark.parametrize('lhs, rhs', ADMITTED)
def test_strict_admits_the_identities_float64_only_rounds(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['realised'] is True and r['tier'] == 'core', r
    assert r['verdict'] in _contract.HOLDS_IN_THE_REALS, r
    assert r['realised_kinds'] is True, r


@pytest.mark.parametrize('lhs, rhs', REFUSED)
def test_strict_refuses_a_change_of_kind_and_whatever_is_false_in_the_reals(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['realised'] is not True and effective_tier(r) not in ('core',), r


def test_the_two_readings_and_the_reals_decide_strict() -> None:
    read = _contract.strict_reading
    assert read(True, False, 'KILL') is True              # (a) alone
    assert read(False, True, 'CERTIFIED') is True         # (b)
    assert read(False, True, 'ENGINE-MISALIGN') is True
    assert read(False, True, 'KILL') is False             # false in the reals
    assert read(False, False, 'CERTIFIED') is False       # a change of kind
    assert read(False, True, 'UNRESOLVED-COVERAGE') is None
    assert read(None, None, 'CERTIFIED') is None


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
