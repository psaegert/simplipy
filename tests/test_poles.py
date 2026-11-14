"""Float64 gives an isolated pole an infinity, and that infinity is a value: strict reads it. Where a
divisor, the operand of `inv` or the base of a negative power crosses zero between two of the
judge's grid points, the judge solves for the zero and reads float64 at the doubles nearest it
(`_contract._pole_probes`), so a pole no battery point holds is read too. The reals give a division
by zero no value, so the reading is strict's alone; permissive is unaffected."""
import pytest

from simplipy.verify import _contract, verify_rule

#: An orientation moved through an odd negative power: -inf against +inf at the pole.
REFUSED = [
    ('neg inv - x0 3.5', 'inv - 3.5 x0'),
    ('neg inv - x0 1.7', 'inv - 1.7 x0'),
    ('* -2 inv - x0 3.5', '/ 2 - 3.5 x0'),
    ('neg pow - x0 3.5 -3', 'pow - 3.5 x0 -3'),
    # a double root never changes sign: its factor's zero is solved
    ('inv * - x1 3.5 - 3.5 x1', 'neg inv pow - x1 3.5 2'),
    # float64 lands on the zero: exp(0.6931471805599453) is 2.0
    ('neg inv - exp x0 2', 'inv - 2 exp x0'),
]


@pytest.mark.parametrize('lhs, rhs', REFUSED)
def test_strict_reads_the_infinity_at_a_pole_off_the_battery(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert r['realised'] is False and r['tier'] == 'permissive', r
    assert any(kind == 'INF-CHANGE' for kind, _ in r['unrealised']), r


def test_a_rule_that_keeps_the_infinity_holds() -> None:
    # at x0 = +-0, sin gives a zero of x0's sign, and neg carries it: the same infinity
    r = verify_rule('neg inv sin x0'.split(), 'inv neg sin x0'.split())
    assert r['realised'] is True and r['tier'] == 'core', r


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
