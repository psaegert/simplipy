"""Float64 gives an isolated pole an infinity, and strict reads the point. Where a divisor, the
operand of `inv` or the base of a negative power crosses zero between two of the judge's grid
points, the judge solves for the zero and reads float64 at the doubles nearest it
(`_contract._pole_probes`), so a pole no battery point holds is read too. Strict does not promise
the sign of the infinity at an isolated pole: there +inf and -inf agree, as the two zeros do, while
a change of kind (an infinity against a finite value) and an infinity's sign on a region stay
refused. The reals give a division by zero no value, so the reading is strict's alone; permissive
is unaffected."""
import pytest

from simplipy.verify import _contract, verify_rule

#: An orientation moved through an odd negative power: -inf against +inf at the pole alone.
HELD = [
    ('neg inv - x0 3.5', 'inv - 3.5 x0'),
    ('neg inv - x0 1.7', 'inv - 1.7 x0'),
    ('* -2 inv - x0 3.5', '/ 2 - 3.5 x0'),
    ('neg pow - x0 3.5 -3', 'pow - 3.5 x0 -3'),
    # a double root never changes sign: its factor's zero is solved
    ('inv * - x1 3.5 - 3.5 x1', 'neg inv pow - x1 3.5 2'),
    # the sign of the infinity read through an operation: -pi/2 against pi/2 at x0 = 3.5 alone
    ('atan neg inv - x0 3.5', 'atan inv - 3.5 x0'),
]


@pytest.mark.parametrize('lhs, rhs', HELD)
def test_strict_reads_an_infinity_at_an_isolated_pole_up_to_its_sign(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['realised'] is True and r['tier'] == 'core', r
    assert 'POLE-SIGN' in r['f64_kinds'], r


#: Refused: the sign of an infinity on a region, and a change of kind at a pole.
REFUSED = [
    # float64 lands on the zero at fl(log 2) and at its neighbours, where exp is 2.0 too: no
    # isolated point
    ('neg inv - exp x0 2', 'inv - 2 exp x0', 'INF-CHANGE'),
    # +inf against -inf for every x0 > 0, and -inf against +inf for every x0 < 0
    ('inv * 0 x0', '/ float("-inf") x0', 'INF-CHANGE'),
    # 1 against -1 for every x0 > 0, where the reals give neither side a value
    ('tanh inv - abs x0 x0', 'tanh neg inv - x0 abs x0', 'REAL-CHANGE'),
    # +inf against 8.2e15 at x0 = np.pi: a change of kind, not of sign
    ('inv sin - np.pi x0', 'inv sin x0', 'INF-CHANGE'),
    # a slot bound to any subexpression guards no point: `_0` binds `3 + 0*x1` on a region
    ('neg inv - _0 3', 'inv - 3 _0', 'INF-CHANGE'),
]


@pytest.mark.parametrize('lhs, rhs, kind', REFUSED)
def test_strict_reads_an_infinitys_sign_on_a_region_and_a_change_of_kind(lhs, rhs, kind) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['realised'] is False and r['tier'] == 'permissive', r
    assert any(k == kind for k, _ in r['unrealised']), r


def test_a_rule_that_keeps_the_infinity_holds() -> None:
    # at x0 = +-0, sin gives a zero of x0's sign, and neg carries it: the same infinity
    r = verify_rule('neg inv sin x0'.split(), 'inv neg sin x0'.split())
    assert r['realised'] is True and r['tier'] == 'core', r
    assert 'POLE-SIGN' not in (r.get('f64_kinds') or ()), r


def test_a_pole_without_a_sign_to_read_is_no_event() -> None:
    """The reading is the core's (`_core.table_pole_sign`): no event where the sides agree or differ
    otherwise than by a pole's sign; else whether the point is isolated along the guarded names."""
    tl = _contract.parse(['neg', 'inv', '-', '?0', '3'], number=True)
    tr = _contract.parse(['inv', '-', '3', '?0'], number=True)
    guards = frozenset({'?0'})
    assert _contract.pole_sign(tl, tr, {'?0': 3.0}, guards) is True
    assert _contract.pole_sign(tl, tr, {'?0': 3.0}, frozenset()) is False
    assert _contract.pole_sign(tl, tr, {'?0': 2.0}, guards) is None
    flat = _contract.parse(['inv', '-', '?0', '3'], number=True)
    assert _contract.pole_sign(flat, tr, {'?0': 2.0}, guards) is None   # 1 against -1: no pole


def test_the_poles_of_a_side() -> None:
    def probes(tokens):
        return _contract._pole_probes(_contract.parse(tokens.split()))
    got = probes('neg inv - x0 3.5')
    assert len(got) == 1 and got[0][0] == '-' and got[0][1] == ('slot', '?0'), got
    # a bare slot's zero is a point of every battery: no probe
    assert probes('/ 1 x0') == []
    # a product's factors carry its zero
    got = probes('inv * - x1 3.5 - 3.5 x1')
    assert len(got) == 3 and got[0][0] == '*', got
