"""The zero and one laws: each a named step of the core that carries its own premise, per mode.

A law applies where its premise holds of its operand, and each mode states its own premise:
`f64` reads float64's values, `permissive` cancels like terms without a definedness check. A
fully numeric operand never reaches a law, since its value is read first (a part without one is
NaN). Each row is an input the law reaches, with the answer in each mode, on an engine without
rules, so every answer is the constructors' and the pass's own: where the premise holds the law
applies, and where it fails the input keeps the operand the law would have removed.
"""
import os
import warnings

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine, _core

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INF = 'float("inf")'
NAN = 'float("nan")'

#: (law, input, f64 answer, permissive answer)
ROWS = [
    # t + 0 = t. f64 keeps the zero where the other term may be -0 on a region: `ceil(x0) + 0` is
    # +0 on (-1, 0), where `ceil(x0)` is -0, and an inverse reads the sign.
    ('add.zero', '+ x0 0', 'x0', 'x0'),
    ('add.zero', '- + x0 2 2', 'x0', 'x0'),
    ('add.zero', '+ ceil x0 0', '+ ceil x0 0', 'ceil x0'),
    ('add.zero', 'inv + ceil x0 0', 'inv + ceil x0 0', 'inv ceil x0'),
    # 1*t = t
    ('mul.one', '* 1 x0', 'x0', 'x0'),
    # 0*t = 0. f64: t finite almost everywhere; `0*log(x0)` is NaN for every x0 < 0, and
    # `0*tanh(floor(x0)/floor(x0))` on all of [0, 1).
    ('mul.zero', '* 0 x0', '0', '0'),
    ('mul.zero', '* 0 log x0', '* 0 log x0', '0'),
    ('mul.zero', '* 0 tanh / floor x0 floor x0', '* 0 tanh / floor x0 floor x0', '0'),
    ('mul.zero', '+ x0 * 0 log x0', '+ x0 * 0 log x0', 'x0'),
    # a fully numeric operand is valued first: float64's tan(pi/2) is finite, the reals' has
    # no value
    ('mul.zero', '* 0 tan asin 1', '0', NAN),
    # f64: and t never negative, as float64's `0*t` is -0 wherever t < 0, which an inverse reads;
    # the zero carries the product's sign, and goes where nothing reads it (the answer, a sum
    # whose other terms are not -0 on a region, a head blind to the sign). permissive: one zero,
    # whose inverse has no value.
    ('mul.zero', 'inv * 0 x0', f'/ {INF} x0', NAN),
    ('mul.zero', 'inv * 0 pow x0 2', f'/ {INF} pow x0 2', NAN),
    ('mul.zero', f'inv * x1 / x0 {INF}', f'/ {INF} * x0 x1', NAN),
    ('mul.zero', '/ 1 / x0 float("-inf")', '/ float("-inf") x0', NAN),
    ('mul.zero', f'/ x2 / x1 / {INF} x0', f'/ * {INF} x2 * x0 x1', NAN),
    ('mul.zero', '/ 1 + * 0 x0 * 0 x1', 'inv + * 0 x0 * 0 x1', NAN),
    ('mul.zero', 'inv + ceil x1 * 0 x0', 'inv + ceil x1 * 0 x0', 'inv ceil x1'),
    ('mul.zero', '* -1 * 0 exp x0', '-0.0', '0'),
    ('mul.zero', '+ x1 * 0 x0', 'x1', 'x1'),
    ('mul.zero', '+ x1 * 0 log x0', '+ x1 * 0 log x0', 'x1'),
    ('mul.zero', '+ * 0 x0 * 0 x1', '0', '0'),
    ('mul.zero', '* 2 * 0 x0', '0', '0'),
    ('mul.zero', 'exp * 0 x0', '1', '1'),
    ('mul.zero', 'rootn * 0 x0 2', '0', '0'),
    ('mul.zero', 'pow * 0 x0 1.5', '0', '0'),
    # a head that keeps a zero keeps its sign too
    ('mul.zero', 'inv atan * 0 x0', f'/ {INF} x0', NAN),
    ('mul.zero', 'inv floor * 0 x0', f'/ {INF} x0', NAN),
    ('mul.zero', 'inv rootn * 0 x0 3', f'/ {INF} x0', NAN),
    ('mul.zero', 'inv pow * 0 x0 0.5', f'/ {INF} x0', NAN),
    ('mul.zero', 'inv * x1 atanh * 0 x0', f'/ {INF} * x0 x1', NAN),
    ('mul.zero', '+ x1 atan * 0 x0', 'x1', 'x1'),
    ('mul.zero', 'atan2 * 0 x0 -1', 'atan2 * 0 x0 -1', 'np.pi'),
    # t - t = 0. f64: t finite almost everywhere; `log(x0) - log(x0)` is NaN for every x0 < 0.
    ('add.cancel', '- x0 x0', '0', '0'),
    ('add.cancel', '- * 2 sin x0 sin x0', 'sin x0', 'sin x0'),
    ('add.cancel', '- log x0 log x0', '- log x0 log x0', '0'),
    ('add.cancel', '- exp tan asin 1 exp tan asin 1', NAN, NAN),
    # t/t = 1. f64: t finite and nonzero almost everywhere, and the exponents' branch cut.
    ('mul.cancel', '/ x0 x0', '1', '1'),
    ('mul.cancel', '/ pow x0 2 x0', 'x0', 'x0'),
    ('mul.cancel', '/ floor x0 floor x0', '/ floor x0 floor x0', '1'),
    ('mul.cancel', '/ pow x0 0.5 pow x0 0.5', '/ rootn x0 2 pow x0 / 1 2', '1'),
    ('mul.cancel', '/ exp tan asin 1 exp tan asin 1', NAN, NAN),
    # t^0 = 1, numpy's at every t, NaN and the infinities included
    ('pow.zero', 'pow x0 0', '1', '1'),
    ('pow.zero', 'pow log x0 0', '1', '1'),
    ('pow.zero', 'pow tan asin 1 0', '1', NAN),
    # t^1 = t
    ('pow.one', 'pow x0 1', 'x0', 'x0'),
    # 1^t = 1, numpy's at every t, NaN included
    ('pow.base_one', 'pow 1 x0', '1', '1'),
    ('pow.base_one', 'pow 1 tan asin 1', '1', NAN),
    # rootn(t, 1) = t
    ('rootn.one', 'rootn x0 1', 'x0', 'x0'),
    # inf + t = inf. f64: t finite almost everywhere, at construction; `inf + log(x0)` is NaN
    # for every x0 < 0. permissive: after the like terms, so a part with a variable beside an
    # infinity is one term to them; f64 reads each of these as float64 does (NaN).
    ('add.infinity_absorbs', f'+ {INF} x0', INF, INF),
    ('add.infinity_absorbs', f'+ x0 * {INF} x0', f'* {INF} x0', f'* {INF} x0'),
    ('add.infinity_absorbs', f'+ {INF} log x0', f'+ log x0 {INF}', INF),
    ('add.infinity_absorbs', f'sin + x0 {INF}', NAN, NAN),
    ('add.infinity_absorbs', f'/ + x0 {INF} + x0 {INF}', NAN, '1'),
    ('add.infinity_absorbs', f'* + x0 {INF} inv + x0 {INF}', NAN, '1'),
    ('add.infinity_absorbs', f'/ * 2 + x0 {INF} + x0 {INF}', NAN, '2'),
    ('add.infinity_absorbs', f'- + x0 {INF} + x0 {INF}', NAN, '0'),
    ('add.infinity_absorbs', f'- + log x0 {INF} + log x0 {INF}', NAN, '0'),
    ('add.infinity_absorbs', f'- * x1 + x0 {INF} * x1 + x0 {INF}',
     f'+ * {INF} x1 * float("-inf") x1', '0'),
    # fully numeric: no value in either mode
    ('add.infinity_absorbs', f'/ {INF} {INF}', NAN, NAN),
]


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    with open(os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')) as f:
        ops = yaml.safe_load(f)['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


def test_the_core_names_its_laws() -> None:
    assert _core.laws() == [
        ('add.zero', 't + 0 = t'), ('mul.one', '1*t = t'), ('mul.zero', '0*t = 0'),
        ('add.cancel', 't - t = 0'), ('mul.cancel', 't/t = 1'), ('pow.zero', 't^0 = 1'),
        ('pow.one', 't^1 = t'), ('pow.base_one', '1^t = 1'), ('rootn.one', 'rootn(t, 1) = t'),
        ('add.infinity_absorbs', 'inf + t = inf')]


def test_every_law_has_its_rows() -> None:
    assert {law for law, *_ in ROWS} == {name for name, _ in _core.laws()}


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in ROWS])
@pytest.mark.parametrize('effort', [0, None], ids=['effort 0', 'default effort'])
def test_each_mode_applies_the_law_where_its_premise_holds(
        engine, law, expr, f64, permissive, effort) -> None:
    for mode, want in (('f64', f64), ('permissive', permissive)):
        got = ' '.join(engine.simplify(expr.split(), mode=mode, effort=effort))
        assert got == want, (law, mode, expr)


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in ROWS])
def test_each_answer_reads_back_to_itself(engine, law, expr, f64, permissive) -> None:
    for mode, answer in (('f64', f64), ('permissive', permissive)):
        assert ' '.join(engine.simplify(answer.split(), mode=mode, effort=0)) == answer, (law, mode)


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in ROWS])
def test_the_f64_answer_keeps_float64s_values(engine, law, expr, f64, permissive) -> None:
    """Float64's value of the f64 answer is the input's wherever the input has one, an
    infinity's sign included (the answer's own zero may be either zero: nothing reads it)."""
    names = ['x0', 'x1', 'x2']
    points = np.random.default_rng(0).uniform(-3, 3, size=(2000, 3))

    def values(tokens):
        f = engine.as_callable(tokens.split(), variables=names)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v = np.asarray(f(*points.T), dtype=np.float64)
        return np.broadcast_to(v, (len(points),))

    a, b = values(expr), values(f64)
    same = np.isnan(a) | (a == b) | (np.isfinite(a) & np.isclose(a, b, rtol=1e-12, atol=0))
    assert same.all(), (law, expr, f64, points[~same][0])
