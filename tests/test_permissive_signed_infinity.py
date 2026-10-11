"""Permissive keeps the sign of an infinity times a sum whose sign varies.

The search's distribute move turned `-inf*(x1 - 1/2)` into `inf - inf*x1`, which is NaN for every
`x1 > 0` where the product is +-inf; the chain then fills that NaN as permissive allows
(`inf - inf*x1 -> inf`), so the answer was `inf`, wrong on `x1 > 1/2`. Wrapped in a bounded
function the same step returned a wrong finite value (`1 + tanh(inf*(x1 - 1/2))` became `0`,
which is 2 on `x1 > 1/2`). The lossy blanket licence ("every piece finite almost everywhere") is
false everywhere for a piece that spells an infinity, so the move takes the sound licence there.

Every answer below must agree with its input wherever the input is defined (permissive may give
a value where the input has none, never change one it has).
"""
import warnings

import numpy as np
import pytest

from simplipy import SimpliPyEngine

INF, NINF = 'float("inf")', 'float("-inf")'

CASES = [
    f'* {NINF} - x1 / 1 2',                       # the reported second call
    f'/ - 0.5 x1 / exp x2 {INF}',                 # the reported first call
    f'+ 1 tanh * {INF} - x1 / 1 2',               # a wrong finite value: 0 for 1 + sign
    f'* x2 + 1 tanh * {INF} - x1 1',
    f'tanh * {INF} - pow x1 2 3',
    f'* {INF} * x1 - x2 1',                       # became -inf*x1
    f'+ * 2 x1 * {NINF} - x1 1',
    f'log * {NINF} * + x1 - x2 1 - x2 x1',        # became nan (the distributed form alone)
    f'+ x3 tanh * {NINF} * x3 - x2 1',
]

VARS = ['x1', 'x2', 'x3']


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load('acj-5-4-llm', install=True)


def values(engine: SimpliPyEngine, expr: list[str], points: np.ndarray) -> np.ndarray:
    f = engine.as_callable(list(expr), variables=VARS)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        v = np.asarray(f(*points.T), dtype=np.float64)
    return np.broadcast_to(v, (points.shape[0],))


def agrees_where_defined(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pointwise: the input is undefined (NaN), or the answer has the input's value."""
    finite = np.isfinite(a) & np.isfinite(b)
    close = np.zeros(a.shape, dtype=bool)
    close[finite] = np.abs(a[finite] - b[finite]) <= 1e-9 + 1e-9 * np.abs(a[finite])
    return np.isnan(a) | close | (~np.isfinite(a) & (a == b))


@pytest.fixture(scope='module')
def points() -> np.ndarray:
    g = np.random.default_rng(0)
    return np.vstack([g.uniform(-4, 4, size=(512, 3)), g.uniform(-0.6, 0.6, size=(128, 3))])


@pytest.mark.parametrize('expr', CASES)
@pytest.mark.parametrize('effort', [0, 1, 4, None])
def test_the_answer_keeps_every_defined_value(engine, points, expr, effort) -> None:
    t = expr.split()
    a = values(engine, t, points)
    answer = engine.simplify(t, mode='permissive', effort=effort)
    ok = agrees_where_defined(a, values(engine, answer, points))
    assert ok.all(), (expr, effort, answer, points[~ok][0])
    # and so does a second call on the answer (the reported route)
    again = engine.simplify(answer, mode='permissive', effort=effort)
    ok = agrees_where_defined(a, values(engine, again, points))
    assert ok.all(), (expr, effort, answer, again, points[~ok][0])


@pytest.mark.parametrize('expr', CASES)
def test_the_search_leaves_these_alone(engine, expr) -> None:
    # No sound move applies to these, so the search ends where the chain does.
    t = expr.split()
    off = engine.simplify(t, mode='permissive', effort=0)
    for effort in (1, 4, None):
        assert engine.simplify(t, mode='permissive', effort=effort) == off


@pytest.mark.parametrize('expr, expected', [
    ('* x1 + x1 / 1 x1', '+ pow x1 2 1'),       # a finite factor times a sum
    ('* + x1 x2 - x1 x2', '- pow x1 2 pow x2 2'),  # a sum times a sum
])
def test_finite_pieces_still_distribute(engine, expr, expected) -> None:
    # The move itself is kept wherever no piece spells an infinity, in every mode.
    t = expr.split()
    for mode in ('f64', 'real', 'permissive'):
        assert ' '.join(engine.simplify(t, mode=mode, effort=0)) != expected, mode  # the search did it
        assert ' '.join(engine.simplify(t, mode=mode)) == expected, mode
