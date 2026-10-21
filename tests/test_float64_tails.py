"""float64's `tanh` and `atan` reach their limits on whole tails.

`tanh(u)` is exactly +-1 for every |u| >= 18.99 and `atan(u)` exactly +-fl(pi/2) for every
|u| >= 5.8e15, in the deployed evaluator as in the engine's. So in `f64` a sum that cancels the
limit, `x2*tanh(x1) - x2`, is +0 on a whole region and its inverse +inf there. The zero-set
certificate read the real functions, which never reach their limits, and let a sign move through
such a sum under an inverse: `-1/(x2*tanh(x1) - x2)` became `1/((1 - tanh(x1))*x2)`, +inf where
the input is -inf. `f64` reads each function on its float64 range; `real` and `permissive` keep the
real reading.
"""
import numpy as np
import pytest

import simplipy
from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

CONFIG = acj_config_path()
TAILS = (25.0, -25.0, 1e17, -1e17)  # past both functions' thresholds


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj config not staged')
    return SimpliPyEngine.from_config(CONFIG, modes='all')


def evaluate(eng, tokens, names, X):
    """The deployed realization's values at the rows of X."""
    code = eng.prefix_to_infix(tokens, realization=True)
    env = {name: X[:, i] for i, name in enumerate(names)}
    with np.errstate(all='ignore'):
        values = eval(code, {'simplipy': simplipy, 'np': np}, env)
    return np.asarray(values, dtype=float) + np.zeros(len(X))


def tail_points(n_vars):
    """Generic points, and the same points with one variable moved onto a tail."""
    base = np.random.default_rng(0).uniform(-4, 4, size=(64, n_vars))
    rows = [base]
    for i in range(n_vars):
        for t in TAILS:
            moved = base.copy()
            moved[:, i] = t
            rows.append(moved)
    return np.concatenate(rows)


@pytest.mark.parametrize('expression', [
    '* -1 inv - * x2 tanh x1 x2',
    '* -1 inv - tanh x1 1',
    '* -1 inv + tanh - x1 x3 1',
    '* -1 inv - tanh tanh x1 tanh 1',
    '* -1 inv - atan x1 1.5707963267948966',
    '* -1 inv - * x2 atan x1 * x2 1.5707963267948966',
    '/ / x9 + x2 + x9 x9 - * x2 tanh - x8 * / x9 * x9 / x9 asin x2 x8 x2',
])
def test_f64_keeps_the_infinities_of_a_zero_on_a_tail(eng, expression):
    tokens = expression.split()
    answer = eng.simplify(tokens, mode='f64')
    names = sorted(eng.expression_variables(tokens))
    X = tail_points(len(names))
    a, b = evaluate(eng, tokens, names, X), evaluate(eng, answer, names, X)
    assert np.isinf(a).any(), 'the input has no infinity on its tails'
    same = (np.isnan(a) == np.isnan(b)) & np.where(np.isinf(a) | np.isinf(b), a == b, True)
    assert same.all(), f'{" ".join(answer)} at {X[~same][0]}: {a[~same][0]} vs {b[~same][0]}'


@pytest.mark.parametrize('mode, expected', [
    ('f64', '/ -1 - * x2 tanh x1 x2'),
    ('real', '/ inv - 1 tanh x1 x2'),
    ('permissive', '/ inv - 1 tanh x1 x2'),
])
def test_only_f64_reads_the_tails(eng, mode, expected):
    assert ' '.join(eng.simplify('* -1 inv - * x2 tanh x1 x2'.split(), mode=mode)) == expected


@pytest.mark.parametrize('mode', ('f64', 'real', 'permissive'))
@pytest.mark.parametrize('expression, expected', [
    # the tails do not cancel: x2 - x1 and -x2 - x1, fl(pi/2) - x1 are zero on null sets
    ('* -1 inv - * x2 tanh x1 x1', 'inv - x1 * x2 tanh x1'),
    ('* -1 inv - atan x1 x1', 'inv - x1 atan x1'),
])
def test_a_sum_whose_tails_do_not_cancel_still_moves_its_sign(eng, mode, expression, expected):
    assert ' '.join(eng.simplify(expression.split(), mode=mode)) == expected
