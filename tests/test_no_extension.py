"""Clause (b) measures the cells where the left side has no value and the right side has one. Where
the right side has no value wherever the left side has none, there is nothing to measure, and a
structural certificate shows it (`_contract._no_extension`) where the cells would run past the
interval layer's horizon (a literal past it, `x0/299792458`) or its budget (`log(cos(x1))` beside a
second variable). Every node of the left side that can be the first undefined one occurs on the
right side, under heads that keep it undefined."""
import pytest

from simplipy.verify import _contract, verify_rule


@pytest.mark.parametrize('lhs, rhs', [
    ('/ x0 299792458', '/ x0 299792458'),
    ('+ cos x0 log cos x1', '+ cos x0 log cos x1'),
    # the same rule, its literal spelled otherwise, and its sum reordered
    ('/ x1 2.99792458e8', '/ x1 299792458'),
    ('+ log cos x2 cos x1', '+ cos x1 log cos x2'),
])
def test_a_rule_whose_right_side_is_undefined_where_its_left_side_is_holds(lhs, rhs) -> None:
    r = verify_rule(lhs.split(), rhs.split())
    assert (r['verdict'], r['tier'], r['realised']) == ('CERTIFIED', 'core', True), r


@pytest.mark.parametrize('lhs, rhs, f64, shown', [
    ('/ ?0 299792458', '/ ?0 299792458', False, True),
    ('+ log cos ?2 cos ?1', '+ cos ?1 log cos ?2', True, True),
    # an extension: the left side's `log(x)` has no value on x < 0, the right side does
    ('exp log ?0', '?0', False, False),
    ('/ ?0 ?0', '1', False, False),
    # float64's `max` drops a NaN: its operand's undefinedness does not reach the right side's top
    ('log ?0', 'max log ?0 0', True, False),
    # exp of a real is finite in the reals, so their difference has a value; in float64 both
    # overflow to +inf, and inf - inf has none: that node must occur on the right side too
    ('- exp ?0 exp ?1', '- exp ?1 exp ?0', False, True),
    ('- exp ?0 exp ?1', '- exp ?1 exp ?0', True, False),
])
def test_the_certificate(lhs, rhs, f64, shown) -> None:
    assert _contract._no_extension(lhs.split(), rhs.split(), f64) is shown


def test_an_extension_is_still_measured() -> None:
    assert verify_rule('exp log x0'.split(), ['x0'])['verdict'] == 'KILL'
    assert verify_rule('/ x0 x0'.split(), ['1'])['verdict'] == 'TOLERATED'
