"""Reading A at poles: the value at a point where the input jumps is not compared, and at a one-signed limit the
answer keeps that limit.

A negative odd power's pole is two-signed (its sides disagree), so the sign of its infinity is no value the answer
keeps: the inverse's orientation is the product's choice in every mode, as every other sign carrier's is
(`sign_trade_flip`, `sign_place` in rust/ac/expr.rs). The class of a sum is decided as a whole: a factor belongs to
it by its parity in the sum (`zero_read`), the sites over one sum flip together, and where the product is even in
the sum its pole is one-signed and its value there is the limit, which the answer keeps; where that value is not
read, the class keeps the input's orientation (`class_pinned`). Like terms are like up to sign, so `x + f - f`
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
