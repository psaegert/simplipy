"""The laws of the catalogue: each a named step of the core that carries its own premise, per mode.

A law applies where its premise holds of its operand, and each mode states its own premise:
`f64` reads float64's values, `permissive` cancels like terms without a definedness check. A
fully numeric operand never reaches a law, since its value is read first (a part without one is
NaN). Each row is an input the law reaches, with the answer in each mode, on an engine without
rules, so every answer is the constructors' and the pass's own: where the premise holds the law
applies, and where it fails the input keeps the operand the law would have removed.

The absorptions of a fitted constant hold of families (for every value of the constants on the
left there is one on the right), so their rows have no float64 agreement to check: the rule
judge reads them (`tests/test_differential.py`).
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
    # a zero product over a sum keeps the sum, whose sign its zero has, and spreads where the sum
    # around it reads no zero; one the constructors build spreads as a written one does, so the
    # answer is its own simplification
    ('mul.zero', 'inv + * 0 - x0 x1 * 0 x2', 'inv + * 0 x2 * 0 - x0 x1', NAN),
    ('mul.zero', '+ * 0 + * 2 x1 rootn x1 2 x2', '+ x2 * 0 rootn x1 2', 'x2'),
    ('mul.zero',
     '- x1 - x1 + abs - x1 x1 / + - + pow x1 0.5 x1 x1 + x1 x1 / - x1 / x1 x1 - x1 x1',
     '* 0 rootn x1 2', NAN),
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

C = '<constant>'
#: (law, input, f64 answer, permissive answer) of the absorptions of a fitted constant
ABSORB_ROWS = [
    # c1 + r*c2 + q = c
    ('absorb.sum', f'+ x0 + {C} 5', f'+ x0 {C}', f'+ x0 {C}'),
    ('absorb.sum', f'+ x0 + {C} * 2 {C}', f'+ x0 {C}', f'+ x0 {C}'),
    ('absorb.sum', f'+ {C} * 2 {C}', C, C),
    # r*c = c for a nonzero rational r; 0*c is the exact 0, and an infinity is no rational
    ('absorb.product', f'* 5 {C}', C, C),
    ('absorb.product', f'* 5 * {C} x0', f'* {C} x0', f'* {C} x0'),
    ('absorb.product', f'* 0 {C}', '0', '0'),
    ('absorb.product', f'* {INF} {C}', f'* {INF} {C}', f'* {INF} {C}'),
    # -k*(s + c) = k*(-s + c)
    ('absorb.sum_sign', f'* -1 + x0 {C}', f'- {C} x0', f'- {C} x0'),
    ('absorb.sum_sign', f'* x1 * -2 + x0 {C}', f'* 2 * x1 - {C} x0', f'* 2 * x1 - {C} x0'),
    # c*(a - b) = c*(b - a): the cheaper orientation
    ('absorb.refit_sign', f'* {C} - x1 x0', f'* {C} - x0 x1', f'* {C} - x0 x1'),
    ('absorb.refit_sign', f'* {C} - 1 x1', f'* {C} - x1 1', f'* {C} - x1 1'),
    # c + g = c and c*g = c for a ground g, finite (and nonzero in a product)
    ('absorb.ground_shift', f'+ x0 + {C} cosh 1', f'+ x0 {C}', f'+ x0 {C}'),
    ('absorb.ground_shift', f'* x0 * {C} sin 1', f'* {C} x0', f'* {C} x0'),
    ('absorb.ground_shift', f'- {C} cosh 1', C, C),
    # L(c) = c wherever L stands: a fitted constant is finite, so the values of c at which L has
    # none (log(0), 1/0, atanh(1)) are no members, and an overflow is a range event. f64 reads
    # L's evidence: (-1)^c, finite only at the integers, has none.
    ('absorb.collapse', f'log {C}', C, C),
    ('absorb.collapse', f'pow 2 {C}', C, C),
    ('absorb.collapse', f'inv log {C}', C, C),
    ('absorb.collapse', f'+ x0 log {C}', f'+ x0 {C}', f'+ x0 {C}'),
    ('absorb.collapse', f'+ x0 inv {C}', f'+ x0 {C}', f'+ x0 {C}'),
    ('absorb.collapse', f'exp - x0 log {C}', f'exp + x0 {C}', f'exp + x0 {C}'),
    ('absorb.collapse', f'* x0 tanh log {C}', f'* {C} x0', f'* {C} x0'),
    ('absorb.collapse', f'tanh + x0 tanh log {C}', f'tanh + x0 {C}', f'tanh + x0 {C}'),
    ('absorb.collapse', f'tanh + x0 exp {C}', f'tanh + x0 {C}', f'tanh + x0 {C}'),
    ('absorb.collapse', f'tanh + x0 log {C}', f'tanh + x0 {C}', f'tanh + x0 {C}'),
    ('absorb.collapse', f'exp + x0 log {C}', f'exp + x0 {C}', f'exp + x0 {C}'),
    ('absorb.collapse', f'tanh + x0 inv {C}', f'tanh + x0 {C}', f'tanh + x0 {C}'),
    ('absorb.collapse', f'tanh + x0 atanh {C}', f'tanh + x0 {C}', f'tanh + x0 {C}'),
    ('absorb.collapse', f'tanh + x0 pow 0 {C}', f'tanh + x0 {C}', f'tanh + x0 {C}'),
    ('absorb.collapse', f'pow -1 {C}', f'pow -1 {C}', C),
    ('absorb.collapse', f'pow -1 floor {C}', f'pow -1 {C}', C),
    # L(c) = NaN where L is NaN at every c; `asin(cosh(c))` is pi/2 at c = 0
    ('absorb.nan_family', f'acosh acos cosh {C}', NAN, NAN),
    ('absorb.nan_family', f'asin cosh {C}', C, C),
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
        ('add.infinity_absorbs', 'inf + t = inf'), ('absorb.sum', 'c1 + r*c2 + q = c'),
        ('absorb.product', 'r*c = c'), ('absorb.sum_sign', '-k*(s + c) = k*(-s + c)'),
        ('absorb.refit_sign', 'c*(a - b) = c*(b - a)'),
        ('absorb.ground_shift', 'c + g = c, c*g = c'), ('absorb.collapse', 'L(c) = c'),
        ('absorb.nan_family', 'L(c) = NaN')]


def test_every_law_has_its_rows() -> None:
    assert {law for law, *_ in ROWS + ABSORB_ROWS} == {name for name, _ in _core.laws()}


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in ROWS + ABSORB_ROWS])
@pytest.mark.parametrize('effort', [0, None], ids=['effort 0', 'default effort'])
def test_each_mode_applies_the_law_where_its_premise_holds(
        engine, law, expr, f64, permissive, effort) -> None:
    for mode, want in (('f64', f64), ('permissive', permissive)):
        got = ' '.join(engine.simplify(expr.split(), mode=mode, effort=effort))
        assert got == want, (law, mode, expr)


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in ROWS + ABSORB_ROWS])
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
