"""A sum that vanishes on a region keeps float64's zero sign where it is inverted, in `f64`.

float64 adds to `+0` wherever a sum cancels, however the sum is written: `|x1| - x1` is `+0` for
every `x1 >= 0`, so `-(|x1| - x1)` is `-0` there and `1/-(|x1| - x1)` is `-inf`. The canonical form
reads a sum and its negation as one sum turned around, and `f64` wrote that inverse as
`1/(x1 - |x1|)`, `+inf` there. Where the zero-set certificate cannot show a sum's zero set null and
the sum is inverted, `f64` now keeps the sign with the inverse, also through an odd power, an odd
function or an odd root of the sum and in a product holding it. `real` reads the one zero of the
reals, whose inverse is `+inf`, and `permissive` a value up to the sign of a zero.
"""
import itertools

import numpy as np
import pytest

import simplipy
from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

CONFIG = acj_config_path()
VALUES = (0.0, 0.5, 2.0, 3.7, -1.0, -2.5)


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


@pytest.mark.parametrize('expression, expected', [
    ('inv neg - abs x1 x1', '/ -1 - abs x1 x1'),
    ('inv neg pow - + x1 pow - x2 abs x2 2 abs x1 3', '/ -1 pow - + x1 pow - x2 abs x2 2 abs x1 3'),
    ('inv * x2 neg - abs x1 x1', '/ neg inv - abs x1 x1 x2'),
    ('inv neg sin - abs x1 x1', '/ -1 sin - abs x1 x1'),
    ('inv neg rootn - abs x1 x1 3', '/ -1 rootn - abs x1 x1 3'),
])
def test_f64_keeps_the_infinity_of_an_inverted_region_zero(eng, expression, expected):
    tokens = expression.split()
    answer = eng.simplify(tokens, mode='f64')
    names = sorted(eng.expression_variables(tokens))
    X = np.array(list(itertools.product(VALUES, repeat=len(names))))
    a, b = evaluate(eng, tokens, names, X), evaluate(eng, answer, names, X)
    assert np.isinf(a).any(), 'the input has no infinity on these points'
    same = (np.isnan(a) == np.isnan(b)) & np.where(np.isinf(a) | np.isinf(b), a == b, True)
    assert same.all(), f'{" ".join(answer)} at {X[~same][0]}: {a[~same][0]} vs {b[~same][0]}'
    assert ' '.join(answer) == expected


@pytest.mark.parametrize('mode', ('real', 'permissive'))
def test_real_and_permissive_read_the_sum_turned_around(eng, mode):
    assert ' '.join(eng.simplify('inv neg - abs x1 x1'.split(), mode=mode)) == 'inv - x1 abs x1'
