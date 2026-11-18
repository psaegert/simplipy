"""Clause (b) measures the cells where the left side has no value and the right side has one. Where
that set is null by the sides' structure, a certificate shows it (`_contract._no_extension`) where
the cells would run past the interval layer's horizon (a literal past it, `x0/299792458`) or its
budget (`log(cos(x1))` beside a second variable, the level sets of an exponent). Every node of the
left side that can be the first undefined one occurs on the right side, under heads that keep it
undefined, or is a power `b^e` the right side holds as `b^(q*e)` for a rational q > 0, with `e`
nonconstant entire: the extension then lies in the level sets {e in Z/q}, a null set. Where the
whole left side occurs on the right side so, up to the order of a commutative operation's operands
(a rule that respells a side's literals or orders them otherwise), no node needs covering: the right
side has no value wherever the left side has none."""
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
    # a rational function of its atoms: x0/x0 has no value where its divisor x0 is zero, a null
    # set; in float64 a rational function rounds and overflows on whole regions: the cells decide
    ('/ ?0 ?0', '1', False, True),
    ('/ ?0 ?0', '1', True, False),
    # its atoms x1, log(x3) and sin(x2): log(x3) is on the right side, the divisor sin(x2) is a
    # nonzero polynomial of one analytic atom
    ('* - / * ?1 log ?3 sin ?2 ?1 sin ?2', '* ?1 - log ?3 sin ?2', False, True),
    ('* tan ?0 / ?1 ?1', 'tan ?0', False, True),
    # an atom with no value on a region, which the right side does not hold
    ('- log ?0 log ?0', '0', False, False),
    ('+ ?1 - log ?0 log ?0', '?1', False, False),
    # a divisor that is zero: x1 - x1, and sqrt(x0)^2 - x0, whose atoms x0 and sqrt(x0) read one
    # variable, so a nonzero polynomial of them can vanish
    ('/ ?0 - ?1 ?1', '?0', False, False),
    ('+ 1 / 0 - pow rootn ?0 2 2 ?0', '1', False, False),
    ('+ 1 / 0 - + pow sin ?0 2 pow cos ?0 2 1', '1', False, False),
    # an atom of a constant operand, infinite everywhere it has a value
    ('* log - ?0 ?0 / ?1 ?1', 'log - ?0 ?0', False, False),
    # a `_` slot binds any part, a constant or a step function among them
    ('/ _0 _0', '1', False, False),
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
    # the whole left side on the right, its literals respelled: in float64 `max` drops a NaN inside
    # it, and still neither side has a value where the other has none
    ('max erf ceil besseli ?0 1 * 2.0 ?0', 'max erf ceil besseli ?0 1 * 2 ?0', True, True),
    ('max erf ceil besseli ?0 1 * 2.0 ?0', 'max erf ceil besseli ?0 1 * 2 ?0', False, True),
    ('bessely ceil ?0 3.0', '+ 0 bessely ceil ?0 3', True, True),
    # the operands of a sum, a product, a max or a min in another order, literals respelled
    ('max sinh ?1 harmonic ?0', 'max harmonic ?0 sinh ?1', True, True),
    ('+ 2.0 inv tan besselj log ?1 ?1', '+ inv tan besselj log ?1 ?1 2', True, True),
    ('min harmonic besselj ?1 3.0 0', 'min 0 harmonic besselj ?1 3', True, True),
    # no other head's operands: atan2 is not commutative
    ('atan2 harmonic ?0 ?1', 'atan2 ?1 harmonic ?0', True, False),
    # a literal of a quarter of a million digits orders as any other
    ('* 2.827881176741212e-244756 ?0', '* ?0 2.827881176741212e-244756', False, True),
])
def test_the_certificate(lhs, rhs, f64, shown) -> None:
    assert _contract._no_extension(lhs.split(), rhs.split(), f64) is shown


@pytest.mark.parametrize('lhs, rhs', [
    ('max erf ceil besseli x0 1 * 2.0 x0', 'max erf ceil besseli x0 1 * 2 x0'),
    ('erf / x1 bessely sinh sinh 3.0 asin x1', 'erf / x1 bessely sinh sinh 3 asin x1'),
    ('besseli x0 harmonic besselk x0 3.0', 'besseli x0 harmonic besselk x0 3'),
    ('max sinh x1 harmonic x0', 'max harmonic x0 sinh x1'),
    ('min rootn 1 cosh exp x1 exp np.pi', 'min exp np.pi rootn 1 cosh exp x1'),
])
def test_a_side_respelled_holds_in_strict(lhs, rhs) -> None:
    # float64's cells past the layer's budget left the measure undecided, and strict refused
    r = verify_rule(lhs.split(), rhs.split())
    assert (r['tier'], r['realised'], r['f64_measure']) == ('core', True, 0.0), r


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


def test_a_rational_function_of_its_atoms_is_measured_by_its_divisors() -> None:
    # (x1*log(x3)/sin(x2) - x1)*sin(x2) -> x1*(log(x3) - sin(x2)): the left side has no value
    # where log(x3) has none, as the right side, and where sin(x2) = 0, a null set; the cells ran
    # past the interval layer's budget and left the measure undecided
    r = verify_rule('* - / * x1 log x3 sin x2 x1 sin x2'.split(), '* x1 - log x3 sin x2'.split())
    assert r['verdict'] != 'UNDECIDED-MEASURE' and r['measure'] == 0.0, r


def test_a_fully_numeric_exponent_still_reads_an_extension() -> None:
    """`rootn(x1, 1 + 1)` is the square root, with no value for x1 < 0, where `abs(x1)` has one."""
    r = verify_rule('rootn x1 + 1 1'.split(), 'abs x1'.split())
    assert r['realised'] is False and r['tier'] == 'reject', r
