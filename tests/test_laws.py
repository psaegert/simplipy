"""The laws of the catalogue: each a named step of the core that carries its own premise, per mode.

A law applies where its premise holds of its operand, and each mode states its own premise:
`f64` reads float64's values, `permissive` cancels like terms without a definedness check. A
fully numeric operand never reaches a law, since its value is read first (a part without one is
NaN). Each row is an input the law reaches, with the answer in each mode, on an engine without
rules, so every answer is the constructors' and the pass's own: where the premise holds the law
applies, and where it fails the input keeps the operand the law would have removed.

The numbers' laws (`num.*`) are how a fully numeric part is read and written: each mode writes
its value as the cheapest value it admits, so their rows are not checked against float64's value
of the input. The absorptions of a fitted constant hold of families (for every value of the
constants on the left there is one on the right), so their rows have no float64 agreement to
check: the rule judge reads them (`tests/test_differential.py`).
"""
import os
import warnings

import numpy as np
import pytest
import yaml

from simplipy import SimpliPyEngine, _core

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INF = 'float("inf")'
NINF = 'float("-inf")'
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
    # a - b = a + (-1)*b, a/b = a*b^-1, neg and inv, read at the parse
    ('parse.sub', '- x0 x1', '- x0 x1', '- x0 x1'),
    ('parse.sub', '- x0 neg x1', '+ x0 x1', '+ x0 x1'),
    ('parse.div', '/ x0 x1', '/ x0 x1', '/ x0 x1'),
    ('parse.div', '/ x0 / x1 x2', '/ * x0 x2 x1', '/ * x0 x2 x1'),
    ('parse.neg_inv', 'neg neg x0', 'x0', 'x0'),
    ('parse.neg_inv', 'inv inv x0', 'x0', 'x0'),
    # a numeral beyond the exact reader: float64 reads `1e309` as an infinity
    ('parse.signed_literal', '+ x0 -1e309', NINF, '- x0 1e309'),
    # the bags: one flat sum, product or lattice bag in the canonical order
    ('add.flatten', '+ + x0 x1 x2', '+ x0 + x1 x2', '+ x0 + x1 x2'),
    ('add.flatten', '+ x2 + x1 x0', '+ x0 + x1 x2', '+ x0 + x1 x2'),
    ('mul.flatten', '* x2 * x1 x0', '* x0 * x1 x2', '* x0 * x1 x2'),
    ('lattice.bag', 'max max x0 x1 x2', 'max x0 max x1 x2', 'max x0 max x1 x2'),
    ('lattice.bag', 'max x0 x0', 'x0', 'x0'),
    ('lattice.bag', f'max x0 {NAN}', NAN, NAN),
    ('add.collect', '+ * 2 sin x0 * 3 sin x0', '* 5 sin x0', '* 5 sin x0'),
    ('add.collect', '+ x0 x0', '* 2 x0', '* 2 x0'),
    # r*(a + b) + c = r*a + r*b + c. f64: float64's sum is +0 wherever it cancels, so a negative
    # r spreads only where the sum holds no sign or nothing reads its zero: the answer itself, a
    # head blind to the sign, a sum another addend keeps off -0 (`ceil(x2)` is -0 on (-1, 0), `x2`
    # only at a point).
    ('add.unfactor', '+ x2 * 2 + x0 x1', '+ * 2 x0 + * 2 x1 x2', '+ * 2 x0 + * 2 x1 x2'),
    ('add.unfactor', '+ x2 * -2 + x0 x1', '- - x2 * 2 x0 * 2 x1', '- - x2 * 2 x0 * 2 x1'),
    ('add.unfactor', '+ ceil x2 * -2 + ceil x0 floor x1', '- - ceil x2 * 2 ceil x0 * 2 floor x1',
     '- - ceil x2 * 2 ceil x0 * 2 floor x1'),
    ('add.unfactor', 'exp + ceil x2 * -2 + ceil x0 floor x1',
     'exp - - ceil x2 * 2 ceil x0 * 2 floor x1', 'exp - - ceil x2 * 2 ceil x0 * 2 floor x1'),
    ('add.unfactor', 'inv + ceil x2 * -2 + ceil x0 floor x1', 'inv - ceil x2 * 2 + ceil x0 floor x1',
     '/ -1 - + * 2 ceil x0 * 2 floor x1 ceil x2'),
    ('add.unfactor', 'inv + x2 * -2 + ceil x0 floor x1', '/ -1 - + * 2 ceil x0 * 2 floor x1 x2',
     '/ -1 - + * 2 ceil x0 * 2 floor x1 x2'),
    ('add.unfactor', 'inv + * -1 - ceil x0 ceil x1 ceil x2', 'inv - ceil x2 - ceil x0 ceil x1',
     'inv - + ceil x1 ceil x2 ceil x0'),
    ('add.unfactor', 'inv + * -2 + ceil x0 floor x1 * 2 floor x1', '/ neg inv ceil x0 2',
     '/ neg inv ceil x0 2'),
    # u*a + u*b = u*(a + b), -a - b = -(a + b). f64: a sign leaves or enters a sum only where it
    # holds none or nothing reads its zero: float64's `-ceil(x0) - floor(x1)` is +0 wherever the
    # sum cancels, and the negation `-(ceil(x0) + floor(x1))` is -0 there, which an inverse or
    # atan2 reads, and the answer, `exp` or a sign carried up to the answer does not.
    ('add.content', '+ * 2 x0 * 2 x1', '* 2 + x0 x1', '* 2 + x0 x1'),
    ('add.content', '- neg x0 x1', 'neg + x0 x1', 'neg + x0 x1'),
    ('add.content', '- neg ceil x0 floor x1', 'neg + ceil x0 floor x1', 'neg + ceil x0 floor x1'),
    ('add.content', 'exp - neg ceil x0 floor x1', 'exp neg + ceil x0 floor x1',
     'exp neg + ceil x0 floor x1'),
    ('add.content', 'sin - neg ceil x0 floor x1', 'neg sin + ceil x0 floor x1',
     'neg sin + ceil x0 floor x1'),
    ('add.content', 'inv sin - neg ceil x0 floor x1', 'inv sin - neg ceil x0 floor x1',
     '/ -1 sin + ceil x0 floor x1'),
    ('add.content', 'atan2 - neg ceil x0 floor x1 -1', 'atan2 - neg ceil x0 floor x1 -1',
     'atan2 neg + ceil x0 floor x1 -1'),
    ('add.content', 'inv - neg ceil x0 floor x1', 'inv - neg ceil x0 floor x1',
     '/ -1 + ceil x0 floor x1'),
    ('add.content', 'inv + * -2 ceil x0 * -2 floor x1', '/ inv - neg ceil x0 floor x1 2',
     '/ neg inv + ceil x0 floor x1 2'),
    ('add.content', 'inv * 2 - neg ceil x0 floor x1', '/ inv - neg ceil x0 floor x1 2',
     '/ neg inv + ceil x0 floor x1 2'),
    ('add.content', 'inv neg - floor x0 floor x1', '/ -1 - floor x0 floor x1',
     'inv - floor x1 floor x0'),
    ('add.content', 'inv + ceil x2 neg - floor x0 floor x1', 'inv - ceil x2 - floor x0 floor x1',
     'inv - + ceil x2 floor x1 floor x0'),
    ('mul.exp_merge', '* exp x0 exp x1', 'exp + x0 x1', 'exp + x0 x1'),
    ('mul.exp_merge', '* exp x0 np.e', 'exp + x0 1', 'exp + x0 1'),
    ('mul.rootn_join', '* rootn x0 3 rootn x1 3', 'rootn * x0 x1 3', 'rootn * x0 x1 3'),
    # f64 keeps `log(x0)` beside the infinity (not finite almost everywhere)
    ('mul.infinity_sign', f'* -2 + log x0 {INF}', f'* 2 neg + log x0 {INF}', NINF),
    ('pow.exp_compose', 'pow np.e x0', 'exp x0', 'exp x0'),
    ('pow.exp_compose', 'rootn exp 2 3', 'exp / 2 3', 'exp / 2 3'),
    ('pow.refused_power', 'pow * x0 - x1 abs x1 -3', '* inv pow x0 3 inv pow - x1 abs x1 3',
     '* inv pow x0 3 inv pow - x1 abs x1 3'),
    ('fun.rootn_index', 'rootn x0 -3', 'inv rootn x0 3', 'inv rootn x0 3'),
    ('fun.rootn_index', 'rootn rootn x0 3 2', 'rootn x0 6', 'rootn x0 6'),
]

#: (law, input, f64 answer, permissive answer) of the numbers: each mode's value of a fully numeric
#: part, which f64 writes as float64's double or, where float64's kind agrees, as the real value
#: (`floor(tanh(100))` is 0, where float64 reads 1), so their rows are not checked against
#: float64's values of the input.
GROUND_ROWS = [
    ('num.literal_arith', '+ 0.1 0.2', '0.3', '0.3'),
    ('num.literal_arith', '+ x0 + 0.1 0.2', '+ x0 0.3', '+ x0 0.3'),
    ('num.literal_arith', 'rootn -8 3', '-2', '-2'),
    ('num.abs_literal', 'abs -3', '3', '3'),
    ('num.exp_one', '* x0 exp 1', '* np.e x0', '* np.e x0'),
    # IEEE's pole in f64; no value in the reals
    ('num.zero_pole', 'pow 0 -1', INF, NAN),
    ('num.inf_arith', f'* -2 {INF}', NINF, NINF),
    ('num.inf_arith', f'* 0 {INF}', NAN, NAN),
    ('num.inf_arith', f'+ x0 + {INF} {NINF}', NAN, NAN),
    ('num.inf_power', f'pow {INF} -1', '0', '0'),
    ('num.inf_power', f'pow {NINF} -1', '-0.0', '0'),
    ('num.literal_base_inf_exponent', f'pow 0.5 {INF}', '0', '0'),
    ('num.literal_base_inf_exponent', f'* x0 pow 3 {NINF}', '0', '0'),
    # the table's value in permissive: no value at pow(1, inf)
    ('num.table_fold', f'pow 1 {INF}', '1', NAN),
    ('num.table_fold', f'mod 0 {INF}', '0', NAN),
    ('num.ground_class_fold', 'atanh 2', NAN, NAN),
    ('num.ground_class_fold', '+ x0 log 0', NINF, NINF),
    ('num.ground_class_fold', '/ 1 ceil -0.5', NINF, NAN),
    ('num.ground_float_fold', 'floor tanh 100', '0', '0'),
    ('num.ground_float_fold', 'exp sin np.pi', '1', '1'),
    ('num.rootn_index', 'rootn x0 0', NAN, NAN),
    ('num.rootn_index', 'rootn x0 np.pi', NAN, NAN),
    # the parity of the written integer, read exactly in f64
    ('num.pow_parity', f'pow {NINF} 10000000000000000001', INF, NINF),
    ('num.besselj_literals', 'besselj 0 0', '1', '1'),
    ('num.besselj_literals', 'besselj 1.5 0', '-0.0', '0'),
    ('num.besselj_literals', f'besselj {INF} 2', '0', '0'),
    ('num.lattice_numbers', 'max 2 3', '3', '3'),
    ('num.lattice_numbers', 'max x0 max 2 3', 'max 3 x0', 'max 3 x0'),
    ('num.literal_to_float', '* x0 / 200000000000000000000000000000 426738538271436458205631863649',
     '* 0.46867105279529636 x0', '* 0.4686710527952964 x0'),
    ('num.signed_zero', '+ x0 -0.0', 'x0', 'x0'),
    ('num.signed_zero', 'pow -0.0 -1', NINF, NAN),
    ('num.signed_zero', '* -1 0', '-0.0', '0'),
    ('num.ground_first', '* 0 tan asin 1', '0', NAN),
    ('num.ground_first', '/ tan asin 1 tan asin 1', '1', NAN),
    ('num.ground_first', f'/ {INF} {INF}', NAN, NAN),
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
        ('absorb.nan_family', 'L(c) = NaN'), ('num.literal_arith', 'r1 + r2 and r1*r2 are exact'),
        ('num.abs_literal', 'abs(r) = r or -r'), ('num.exp_one', 'exp(1) = e'),
        ('num.zero_pole', '0^q = inf, q < 0'),
        ('num.inf_arith', 'c*inf = sign(c)*inf, 0*inf = inf - inf = NaN'),
        ('num.inf_power', 'inf^q = inf (q > 0), inf^q = 0 (q < 0)'),
        ('num.literal_base_inf_exponent', 'b^inf = 0 (abs(b) < 1), b^-inf = 0 (abs(b) > 1)'),
        ('num.table_fold', "f(r1, r2) = the table's value"),
        ('num.ground_class_fold', 'g = NaN or g = +-inf, g ground'),
        ('num.ground_float_fold', 'g = its cheapest admitted value, g ground'),
        ('num.rootn_index', 'rootn(t, n) = NaN, n no nonzero integer'),
        ('num.pow_parity', '(-inf)^k = (-1)^k*inf'),
        ('num.besselj_literals', 'J_nu(0), J_(+-inf)(x) at literals'),
        ('num.lattice_numbers', 'max(r1, r2) = the greater'),
        ('num.literal_to_float', 'r = its shorter float'),
        ('num.signed_zero', 't + (-0) = t, (-0)^-k = -inf (k odd)'),
        ('num.ground_first', 'a fully numeric part is its value'),
        ('parse.sub', 'a - b = a + (-1)*b'), ('parse.div', 'a/b = a*b^-1'),
        ('parse.neg_inv', 'neg a = (-1)*a, inv a = a^-1'), ('parse.signed_literal', '-r = (-1)*r'),
        ('add.flatten', '(a + b) + c = a + (b + c)'), ('mul.flatten', '(a*b)*c = a*(b*c)'),
        ('lattice.bag', 'max(max(a, b), c) = max(a, b, c), max(a, a) = a'),
        ('add.collect', 'r1*t + r2*t = (r1 + r2)*t'),
        ('add.unfactor', 'r*(a + b) + c = r*a + r*b + c'),
        ('add.content', 'u*a + u*b = u*(a + b), -a - b = -(a + b)'),
        ('mul.exp_merge', 'exp(a)*exp(b) = exp(a + b)'),
        ('mul.rootn_join', 'rootn(a, n)*rootn(b, n) = rootn(a*b, n)'),
        ('mul.infinity_sign', '-k*(s + inf) = k*(-s - inf)'),
        ('pow.exp_compose', 'e^t = exp(t), exp(a)^b = exp(a*b)'),
        ('pow.refused_power', '(a*b)^-k = ((a*b)^k)^-1'),
        ('fun.rootn_index', 'rootn(t, -n) = 1/rootn(t, n)')]


def test_every_law_has_its_rows() -> None:
    assert {law for law, *_ in ROWS + GROUND_ROWS + ABSORB_ROWS} == {
        name for name, _ in _core.laws()}


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in ROWS + GROUND_ROWS + ABSORB_ROWS])
@pytest.mark.parametrize('effort', [0, None], ids=['effort 0', 'default effort'])
def test_each_mode_applies_the_law_where_its_premise_holds(
        engine, law, expr, f64, permissive, effort) -> None:
    for mode, want in (('f64', f64), ('permissive', permissive)):
        got = ' '.join(engine.simplify(expr.split(), mode=mode, effort=effort))
        assert got == want, (law, mode, expr)


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in ROWS + GROUND_ROWS + ABSORB_ROWS])
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
