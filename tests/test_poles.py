"""Float64 gives an isolated pole an infinity, and strict reads the point. Where a divisor, the
operand of `inv` or the base of a negative power crosses zero between two of the judge's grid
points, the judge solves for the zero and reads float64 at the doubles nearest it
(`_contract._pole_probes`), so a pole no battery point holds is read too. Strict does not promise
the sign of the infinity at an isolated pole: there +inf and -inf agree, as the two zeros do. Where
the pole takes both signs beside it, strict promises no value at all: the point counts as one where
the left side has none. A change of kind elsewhere (an infinity against a finite value) and an
infinity's sign on a region stay refused. The reals give a division by zero no value, so the
reading is strict's alone; permissive is unaffected."""
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


@pytest.mark.parametrize('lhs, rhs', [
    # +inf against 0 at x0 = 3 alone, where 1/(x0 - 3) takes both signs beside the pole
    ('+ inv - x0 3 inv - 3 x0', '0'),
    ('+ inv - x0 1.7 inv - 1.7 x0', '0'),
])
def test_strict_promises_no_value_at_an_isolated_two_signed_pole(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['realised'] is True and r['tier'] == 'core', r
    assert 'TWO-SIGNED-POLE' in r['f64_kinds'], r


def test_a_slot_that_binds_any_part_isolates_no_two_signed_pole() -> None:
    # `_0` binds `3 + 0*x1` on a region, where the left side is +inf against 0
    r = verify_rule('+ inv - _0 3 inv - 3 _0'.split(), ['0'])
    assert r['realised'] is False and r['tier'] == 'permissive', r


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


@pytest.mark.parametrize('lhs, rhs', [
    # float64's sin(x0) is exactly -1 on about 10^8 doubles around -pi/2, its cos(x0) exactly -1
    # around pi: each run lies around a zero of the reals (of even order), and is its point, where
    # an infinity's sign is open
    ('inv - neg sin ?0 1', 'neg inv + sin ?0 1'),
    ('inv - neg cos ?0 1', 'neg inv + cos ?0 1'),
    # float64's exp is 2.0 at fl(log 2) and at its neighbours: a run around log 2, its point
    ('neg inv - exp x0 2', 'inv - 2 exp x0'),
    # three doubles where float64's divisor is exactly 0 around a simple zero of the reals
    ('inv - * -23.918005637790476 sin + * 0.0082535 ?0 11.865502713421696 23.60396315341633',
     'neg inv + * 23.918005637790476 sin + * 0.0082535 ?0 11.865502713421696 23.60396315341633'),
])
def test_a_run_of_doubles_around_a_zero_of_the_reals_is_its_point(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert (r['realised'], r['tier']) == (True, 'core'), r


@pytest.mark.parametrize('lhs, rhs', [
    # float64's tanh(x0) is exactly 1 past 19.06, where the reals have no zero of tanh(x0) - 1:
    # a region, where the two infinities part
    ('inv - tanh ?0 1', 'neg inv - 1 tanh ?0'),
    # 0*asin(x0) is zero on a region of the reals
    ('* 0 asin ?0', '* 0 acos ?0'),
])
def test_a_run_without_a_zero_of_the_reals_is_a_region(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['tier'] != 'core', r


def test_the_reals_place_a_run() -> None:
    def run(tokens, at):
        return _contract.zero_run(_contract.parse(tokens.split(), number=True), {'?0': at}, '?0')
    assert run('+ sin ?0 1', -1.5707963267948966) == 'even'
    assert run('- tanh ?0 1', 20.0) is False
    assert run('- ?0 3', 3.0) == 'single'
    assert run('- ?0 3', 2.0) is None
