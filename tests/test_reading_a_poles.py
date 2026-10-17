"""Reading A at poles: the value at a point where the input jumps is not compared, and at a one-signed limit the
answer keeps that limit.

A negative odd power's pole is two-signed (its sides disagree), so the sign of its infinity is no value the answer
keeps: the inverse's orientation is the product's choice in every mode, as every other sign carrier's is
(`sign_trade_flip`, `sign_place` in rust/ac/expr.rs). Where the product's pole over the same sum is one-signed --
the members over that sum add to a total order with an even numerator, `1/(cbrt(u) u)` at order -4/3 -- the sign of
its infinity is the limit there, which the answer keeps, so no member trades its sign (`two_signed_pole`). A lossy
context (permissive) keeps neither value.
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
