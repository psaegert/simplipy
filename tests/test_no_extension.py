"""Clause (b) measures the cells where the left side has no value and the right side has one. Where
that set is null by the sides' structure, a certificate shows it (`_contract._no_extension`) where
the cells would run past the interval layer's horizon (a literal past it, `x0/299792458`) or its
budget (`log(cos(x1))` beside a second variable, the level sets of an exponent). Every node of the
left side that can be the first undefined one occurs on the right side, under heads that keep it
undefined, or is a power `b^e` the right side holds as `b^(q*e)` for a rational q > 0, with `e`
nonconstant entire: the extension then lies in the level sets {e in Z/q}, a null set."""
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
    # a merged power: x0^x1 has no value where x0 < 0 and x1 is no integer, x0^(2*x1) none there
    # unless 2*x1 is one, so the extension lies on the lines x1 in Z/2, in both readings
    ('* pow ?0 ?1 pow ?0 ?1', 'pow ?0 * 2 ?1', False, True),
    ('* pow ?0 ?1 pow ?0 ?1', 'pow ?0 * 2 ?1', True, True),
    ('* pow ?0 ?1 * pow ?0 ?1 pow ?0 ?1', 'pow ?0 * 3 ?1', False, True),
    # an exponent constant on whole intervals has fat level sets: the cells decide
    ('* pow ?0 / floor ?1 2 pow ?0 / floor ?1 2', 'pow ?0 floor ?1', False, False),
    # another base: nothing covers the left side's power
    ('* pow ?0 ?1 pow ?0 ?1', 'pow abs ?0 * 2 ?1', False, False),
])
def test_the_certificate(lhs, rhs, f64, shown) -> None:
    assert _contract._no_extension(lhs.split(), rhs.split(), f64) is shown


def test_an_extension_is_still_measured() -> None:
    assert verify_rule('exp log x0'.split(), ['x0'])['verdict'] == 'KILL'
    assert verify_rule('/ x0 x0'.split(), ['1'])['verdict'] == 'TOLERATED'


def test_a_merge_on_null_level_sets_holds_and_a_fat_one_does_not() -> None:
    # float64 gives NaN against -2 at (-2, 0.5), on the null set x0 < 0, x1 in Z/2 \ Z only
    r = verify_rule('* pow x0 x1 pow x0 x1'.split(), 'pow x0 * 2 x1'.split())
    assert (r['realised'], r['tier']) == (True, 'core'), r
    # x0^(floor(x1)/2) has no value on whole intervals of x1 at x0 < 0, where x0^floor(x1) has
    r = verify_rule('* pow x0 / floor x1 2 pow x0 / floor x1 2'.split(), 'pow x0 floor x1'.split())
    assert (r['verdict'], r['clause'], r['realised']) == ('KILL', 'b-positive-measure', False), r


@pytest.mark.parametrize('lhs, rhs', [
    ('pow x1 + 1 2', 'pow x1 3'),
    ('rootn x1 + 2 1', 'rootn x1 3'),
    ('pow x1 / 6 2', 'pow x1 3'),
    ('rootn x1 * 3 1', 'rootn x1 3'),
])
def test_float64_reads_a_fully_numeric_exponent_by_its_double(lhs, rhs) -> None:
    """float64 computes `1 + 2` to the double 3 before the power or the root reads it, so the two
    sides have a value at every x1: no cell of f64's clause (b) parts them, and the rule holds in
    both modes, as the sum is exactly 3."""
    r = verify_rule(lhs.split(), rhs.split())
    assert (r['tier'], r['realised'], r['f64_measure']) == ('core', True, 0.0), r


def test_a_fully_numeric_exponent_still_reads_an_extension() -> None:
    """`rootn(x1, 1 + 1)` is the square root, with no value for x1 < 0, where `abs(x1)` has one."""
    r = verify_rule('rootn x1 + 1 1'.split(), 'abs x1'.split())
    assert r['realised'] is False and r['tier'] == 'reject', r
