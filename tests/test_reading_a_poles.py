"""Reading A at poles: the value at a point where the input jumps is not compared, and at a one-signed limit the
answer keeps that limit.

A negative odd power's pole is two-signed (its sides disagree), so the sign of its infinity is no value the answer
keeps: the inverse's orientation is the product's choice in every mode, as every other sign carrier's is
(`sign_trade_flip`, `sign_place` in rust/ac/expr.rs). The class of a sum is decided as a whole: a factor belongs to
it by its parity in the sum (`zero_read`), the sites over one sum flip together, and where the product is even in
the sum its pole is one-signed and its value there is the limit, which the answer keeps; where a member is defined
on one side of the zero only (a log, an even root, a fractional or symbolic power), the class is written in the
orientation positive on that side and the value there is that side's limit (`zero_limit`); where the value is not
read, the class takes its cheapest orientation. Like terms are like up to sign, so `x + f - f`
cancels whichever orientation each `f` arrived in. A lossy context (permissive) keeps no value at a pole.
"""
import math

import numpy as np
import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

CONFIG = acj_config_path()
MODES = ('f64', 'real', 'permissive')


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj config not staged')
    return SimpliPyEngine.from_config(CONFIG, modes='all')


def simp(eng, infix, mode):
    return list(eng.simplify(list(eng.to_prefix(infix)), mode=mode))


@pytest.mark.parametrize('mode', MODES)
def test_a_lone_inverse_takes_either_orientation(eng, mode):
    # The pole at x1 = 3 is two-signed: -1/(x1 - 3) and 1/(3 - x1) are one value class.
    assert simp(eng, '-1/(x1 - 3)', mode) == simp(eng, '1/(3 - x1)', mode)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('pair', [('atan(1/(3 - x1))', '-atan(1/(x1 - 3))'),
                                  ('exp(-1/(3 - x1))', 'exp(1/(x1 - 3))')])
def test_a_jump_through_a_function_takes_either_orientation(eng, mode, pair):
    # atan and exp of a lone inverse jump at x1 = 3 (from -pi/2 to pi/2, from 0 to inf): the value there is not
    # compared, so both spellings are one input.
    a, b = pair
    assert simp(eng, a, mode) == simp(eng, b, mode)


@pytest.mark.parametrize('mode', MODES)
def test_fractions_over_a_class_combine_up_to_sign(eng, mode):
    # The two inverses carry one class in its two orientations; FACTOR reads the shared base up to sign.
    want = simp(eng, '(x1 - x4)/(x2 - x3)', mode)
    assert simp(eng, 'x1/(x2 - x3) + x4/(x3 - x2)', mode) == want
    assert simp(eng, 'x1/(x2 - x3) - x4/(x2 - x3)', mode) == want


@pytest.mark.parametrize('mode', ('f64', 'real'))
@pytest.mark.parametrize('src', ['-1/(rootn(x1 - x2, 3)*(x1 - x2))', '1/(rootn(x1 - x2, 3)*(x1 - x2))'])
def test_a_one_signed_pole_keeps_its_infinity(eng, mode, src):
    # At x1 = x2 the product's pole has order -4/3: one-signed, so its infinity is the limit. A member that traded
    # its sign alone would flip that infinity (the gen4 row 391 shape).
    ans = list(eng.simplify(list(eng.to_prefix(src)), mode=mode))
    f_src = eng.as_callable(list(eng.to_prefix(src)), ['x1', 'x2'])
    f_ans = eng.as_callable(ans, ['x1', 'x2'])
    with np.errstate(all='ignore'):
        at, near = f_ans(1.0, 1.0), f_ans(1.0 + 1e-9, 1.0)
        assert math.isinf(at) and at == f_src(1.0, 1.0)
        assert np.sign(near) == np.sign(at)


@pytest.mark.parametrize('mode', MODES)
def test_a_class_with_two_odd_members_is_written_in_one_orientation(eng, mode):
    # (x1 - 3)(3 - x1) is -(x1 - 3)^2: the class's two odd members meet in one orientation, the sign in the
    # coefficient, and the value at x1 = 3 is the limit, -inf.
    expected = list(eng.simplify(['/', '-1', 'pow', '-', 'x1', '3', '2'], mode=mode))
    assert simp(eng, '1/((x1 - 3)*(3 - x1))', mode) == expected


# Classes whose f - f the finiteness licence admits in every mode (`- x3 tan x2` and `- x3 rootn x2 3` are refused
# outside permissive, as before the class trade).
CANCEL_CLASSES = ['- x3 x2', '- x3 sin x2', '- sin x2 x3', '- x3 atan x2', '+ x3 sin x2', '- x3 * 2 sin x2',
                  '- sin x3 sin x2', '- x3 pow x2 3', '- x3 sinh * 2 x2', '- x3 sin * 2 x2']
CANCEL_SHAPES = ['inv {c}', 'pow {c} -3', 'sinh inv {c}', 'inv sinh {c}', 'inv rootn {c} 3', '* x4 inv {c}']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('cls', CANCEL_CLASSES)
def test_x_plus_f_minus_f_cancels_whatever_orientation_f_takes(eng, mode, cls):
    # The sign of -f may go into f's class (a lone inverse trades its sign): the sum collects its terms up to sign.
    for shape in CANCEL_SHAPES:
        f = shape.format(c=cls)
        for src in (f'- + x1 {f} {f}', f'+ x1 - {f} {f}', f'- + x1 * 2 {f} * 2 {f}'):
            assert list(eng.simplify(src.split(), mode=mode)) == ['x1'], (src, mode)


# One-signed poles whose members hold the sum inside a product or a nested odd carrier: the product is even in the
# sum, so its value at x1 = 3 is the limit (+inf, the source's value there too).
ONE_SIGNED = ['inv * - 3 x1 sin * x1 - 3 x1', 'inv * - 3 x1 rootn * x1 - 3 x1 3', 'inv * - 3 x1 sin * x1 * x2 - 3 x1',
              'inv * - 3 x1 sinh * x2 - 3 x1', '/ x2 * - 3 x1 sin * x1 - 3 x1', 'inv * - 3 x1 rootn sin - 3 x1 3',
              '+ x2 inv * - 3 x1 sin * x1 - 3 x1']


@pytest.mark.parametrize('mode', ('f64', 'real'))
@pytest.mark.parametrize('src', ONE_SIGNED)
def test_a_class_member_inside_a_product_keeps_the_limit(eng, mode, src):
    t = src.split()
    ans = list(eng.simplify(list(t), mode=mode))
    f_src = eng.as_callable(t, ['x1', 'x2'])
    f_ans = eng.as_callable(ans, ['x1', 'x2'])
    with np.errstate(all='ignore'):
        lim = [f_src(3.0 + h, 1.5) for h in (1e-7, -1e-7)]
        assert all(v > 1e5 for v in lim)
        assert f_src(3.0, 1.5) == math.inf
        assert f_ans(3.0, 1.5) == math.inf, ' '.join(ans)


@pytest.mark.parametrize('src', ['+ / x2 - x1 3 inv * - 3 x1 rootn * x1 - 3 x1 3',
                                 '+ / x2 - x1 3 inv * - 3 x1 rootn sin - 3 x1 3'])
def test_the_up_to_sign_retry_keeps_a_class_whole(eng, src):
    # The rule `_0/_2 + _1/_2 -> (_0 + _1)/_2` binds its repeated wildcard up to sign. The second term's class
    # (3 - x1) also sits inside the cube root. The retry orients a term only as a whole and only where reading A
    # allows it: `cbrt(sin(3 - x1))` is read at the zero and the term is oriented with it, while `cbrt(x1 (3 - x1))`
    # holds a factor of unknown sign, so its term keeps its orientation. Either way the inverse never moves alone,
    # and the answer keeps the limit at x1 = 3, +inf.
    over = [['+', '*', '_0', 'pow', '_2', '-1', '*', '_1', 'pow', '_2', '-1'], ['*', '+', '_0', '_1', 'pow', '_2', '-1']]
    rule = SimpliPyEngine(eng._operators_config, [over])
    t = src.split()
    ans = list(rule.simplify(list(t), effort=0, mode='f64'))
    f_ans = eng.as_callable(ans, ['x1', 'x2'])
    with np.errstate(all='ignore'):
        assert f_ans(3.0, 1.5) == math.inf, ' '.join(ans)


# The answer's value at x1 = 3 is the source's limit there (one-sided where the source is defined on one side only).
# The source's own value there need not be: `-(3 - x1)` is -0 at its zero but positive beside it, so
# `1/(cbrt(-2 (x1 - 3)) * -(3 - x1))` is +inf at x1 = 3 where its limit is -inf.
#   * A member defined on one side of the zero (the class's own sum under a square root) fixes the side: the class is
#     written in the orientation positive there, so the value at the zero is that side's limit.
#   * A class is a zero set: `tanh(2 (3 - x1))` is built over `6 - 2 x1`, a multiple of `x1 - 3`, and `2 x1 - 6`
#     beside `x1 - 3` is one class with it.
#   * An odd root reads like an odd function, also over a factor of unknown sign: `1/(cbrt(x2 (x1 - 3)) (3 - x1))`
#     is `-1/(cbrt(x2 (x1 - 3)) (x1 - 3))`, -inf at x1 = 3 like its limit.
CLASS_AT_ITS_ZERO = ['inv * pow - 3 x1 / 1 2 sin - 3 x1', 'inv * rootn - 3 x1 2 tanh - 3 x1',
                     'inv * tanh * 2 - 3 x1 * -1 - 3 x1',
                     'inv * rootn * - x1 3 -2 3 * -1 - 3 x1', 'inv * * -1 - x1 3 - * 2 x1 6',
                     'inv * - - x1 3 pow - x1 3 3 - * x2 x1 * 3 x2', 'inv * rootn * x2 - x1 3 3 - 3 x1']


@pytest.mark.parametrize('mode', ('f64', 'real'))
@pytest.mark.parametrize('src', CLASS_AT_ITS_ZERO)
def test_the_value_at_the_zero_of_a_class_is_kept(eng, mode, src):
    t = src.split()
    ans = list(eng.simplify(list(t), mode=mode))
    f_src = eng.as_callable(t, ['x1', 'x2'])
    f_ans = eng.as_callable(ans, ['x1', 'x2'])
    with np.errstate(all='ignore'):
        beside = [v for v in (f_src(3.0 - 1e-9, 1.5), f_src(3.0 + 1e-9, 1.5)) if not math.isnan(v)]
        assert beside and all(abs(v) > 1e6 for v in beside)
        assert len({math.copysign(1.0, v) for v in beside}) == 1  # one-sided, or two sides that agree
        limit = math.copysign(math.inf, beside[0])
        assert f_ans(3.0, 1.5) == limit, ' '.join(ans)


# A member defined on one side of the class's zero (a log, a symbolic power, or a sum such members dominate there)
# fixes the class's orientation: positive on that side, so every spelling of the function gets one answer and its
# value at x1 = 3 is that side's limit.
ONE_SIDED = [(['/ log - x1 3 - 3 x1', 'neg / log - x1 3 - x1 3', '/ * log - x1 3 - 3 x1 pow - x1 3 2'], 1.0),
             (['/ pow - 3 x1 x2 - x1 3', 'neg / pow - 3 x1 x2 - 3 x1'], -1.0),
             (['/ pow - x1 3 x2 - 3 x1', 'neg / pow - x1 3 x2 - x1 3'], 1.0),
             # a sum whose one-sided terms are infinite at the zero and the rest finite reads as them
             (['/ + log - x1 3 x2 - 3 x1', 'neg / + log - x1 3 x2 - x1 3'], 1.0),
             (['/ + pow - 3 x1 / -1 2 * x2 - x1 3 - x1 3', 'neg / + pow - 3 x1 / -1 2 * x2 - x1 3 - 3 x1'], -1.0),
             (['/ * * x2 rootn - 3 x1 3 pow rootn - 3 x1 3 x2 * - x1 3 pow - + x1 x2 1 7',
               'neg / * * x2 rootn - x1 3 3 pow rootn - 3 x1 3 x2 * - x1 3 pow - + x1 x2 1 7'], None)]


@pytest.mark.parametrize('mode', ('f64', 'real'))
@pytest.mark.parametrize('spellings,side', ONE_SIDED)
def test_a_one_sided_member_fixes_the_orientation(eng, mode, spellings, side):
    answers = {' '.join(eng.simplify(s.split(), mode=mode)) for s in spellings}
    assert len(answers) == 1, answers
    if side is None:
        return
    t, ans = spellings[0].split(), answers.pop().split()
    f_src = eng.as_callable(t, ['x1', 'x2'])
    f_ans = eng.as_callable(ans, ['x1', 'x2'])
    with np.errstate(all='ignore'):
        beside = f_src(3.0 + side * 1e-9, -0.5)
        assert abs(beside) > 1e6 and math.isnan(f_src(3.0 - side * 1e-9, -0.5))
        assert f_ans(3.0, -0.5) == math.copysign(math.inf, beside), ' '.join(ans)


# A sum that holds the class among terms free of it (`S - S^3` written flat, `(x1 - 3)^3 + 3 - x1`) is not read at the
# zero, so its product keeps no value there and takes its cheapest orientation: one answer for every spelling of the
# function, whose value at x1 = 3 need not be the limit (a registered residual).
@pytest.mark.parametrize('mode', ('f64', 'real'))
def test_an_unread_class_takes_its_cheapest_orientation(eng, mode):
    spellings = ['inv * - - x1 3 pow - x1 3 3 * -1 - x1 3', 'inv * - pow - x1 3 3 - x1 3 - x1 3',
                 'neg inv * - - x1 3 pow - x1 3 3 - x1 3']
    answers = {' '.join(eng.simplify(s.split(), mode=mode)) for s in spellings}
    assert len(answers) == 1, answers


# An odd root keeps the sign of zero, as IEEE-754 `rootn` and every odd function do, so a canonical form that moves a
# negation under the root (`rootn(-x2*(x1 - 3), 3)` for `rootn((3 - x1)*x2, 3)`) keeps the infinity at the class's
# zero beside an odd function of the class, a polynomial factor holding it, or a shared-factor multiple of it.
SIGN_OF_ZERO_UNDER_AN_ODD_ROOT = ['inv * rootn * - 3 x1 x2 3 sinh * x2 - 3 x1',      # 1/(cbrt((3-x1)*x2)*sinh(x2*(3-x1)))
                                  'inv * rootn * x2 - 3 x1 3 - 9 pow x1 2',          # 1/(cbrt(x2*(3-x1))*(9-x1^2))
                                  'inv * rootn * - 3 x1 x2 3 - * 3 x2 * x1 x2']      # 1/(cbrt((3-x1)*x2)*(3*x2-x1*x2))


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('src', SIGN_OF_ZERO_UNDER_AN_ODD_ROOT)
def test_a_nested_product_keeps_the_sign_of_zero_under_an_odd_root(eng, mode, src):
    t = src.split()
    ans = list(eng.simplify(list(t), mode=mode))
    f_src = eng.as_callable(t, ['x1', 'x2'])
    f_ans = eng.as_callable(ans, ['x1', 'x2'])
    with np.errstate(all='ignore'):
        for x2 in (0.7, 1.5):
            assert f_src(3.0, x2) == math.inf  # the limit from both sides
            assert f_ans(3.0, x2) == math.inf, ' '.join(ans)
            assert f_ans(np.array([3.0]), np.array([x2]))[0] == math.inf, ' '.join(ans)


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('fn', ['sin', 'atan'])
def test_an_odd_function_of_a_negation_reads_as_the_negated_term(eng, mode, fn):
    # `x1 + sin(-x1)` is `x1 - sin(x1)`: one class with `sin(x1) - x1`, so their product is minus a square.
    src = f'* - {fn} x1 x1 + x1 {fn} neg x1'.split()
    want = list(eng.simplify(f'neg pow - x1 {fn} x1 2'.split(), mode=mode))
    assert list(eng.simplify(src, mode=mode)) == want


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('fn', ['sin', 'atan'])
def test_an_odd_function_of_a_negation_reads_back_as_itself(eng, mode, fn):
    # Apart from the reading above, so the sign placement waiting for D1 masks no fixed point.
    for src in (f'* - {fn} x1 x1 + x1 {fn} neg x1', f'neg pow - x1 {fn} x1 2'):
        ans = list(eng.simplify(src.split(), mode=mode))
        assert list(eng.simplify(ans, mode=mode)) == ans, src


# A second read of an answer gives the answer back. A negated sum the sign folds into (one holding an infinity) is
# joined as `mul()` joins it, and a pinned class freezes only the sites that may vanish at its zero (a sum with a pole
# of its own there, `(x9 - 1)^(-1/4) - x9`, flips exactly).
SECOND_READ = ['* / x8 - x8 + x8 - - x13 x16 x8 - * x8 - - log + - x16 x14 x14 * x2 x13 + x8 atanh / x2 x2 x13',
               '+ + + x9 x9 x9 - - x9 asin / - rootn - x9 / x9 x9 -4 x9 - x9 pow / x9 x9 4 x9']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('src', SECOND_READ)
def test_an_answer_reads_back_as_itself(eng, mode, src):
    ans = list(eng.simplify(src.split(), mode=mode))
    assert list(eng.simplify(ans, mode=mode)) == ans


# The search's distributive law keeps a one-signed pole joined where splitting it would leave two
# addends infinite at the pole with signs that differ (review P3): `(1 + x3 - x1 + 5/3 exp(x2))/abs(x3)`
# split as `5/3 exp(x2) abs(1/x3) + (x3 + 1 - x1) abs(1/x3)` is NaN at `x3 = 0` on the whole slab
# `1 < x1 < 1 + 5/3 exp(x2)`, where the product is `+inf`. A lossy mode keeps no value at a pole.
POLE_DISTRIBUTIONS = ['(8.3921 + (x3 - x1)*8.3921 + 8.3921*exp(x2)*5/3)/abs(x3)',
                      '(8.3921 + (x3 - x1)*8.3921 + 8.3921*exp(x2)*5/3)/x3^2']


@pytest.mark.parametrize('mode', ('f64', 'real'))
@pytest.mark.parametrize('src', POLE_DISTRIBUTIONS)
def test_the_search_keeps_a_one_signed_pole_joined(eng, mode, src):
    t = eng.to_prefix(src)
    ans = simp(eng, src, mode)
    f_src = eng.as_callable(t, ['x1', 'x2', 'x3'])
    f_ans = eng.as_callable(ans, ['x1', 'x2', 'x3'])
    with np.errstate(all='ignore'):
        for x1 in (0.5, 1.5, 2.5):  # 1 + x3 - x1 + 5/3 exp(x2) > 0 at x2 = x3 = 0
            assert f_src(x1, 0.0, 0.0) == math.inf
            assert f_ans(x1, 0.0, 0.0) == math.inf, ' '.join(ans)


@pytest.mark.parametrize('mode', MODES)
def test_the_search_still_splits_a_pole_one_addend_cancels(eng, mode):
    # `x2*x2^-1` is 1: one addend keeps the pole, and the sum takes the product's value there.
    assert simp(eng, '(x1 + x2 + x3)/x2', mode) == simp(eng, '(x1 + x3)/x2 + 1', mode)
