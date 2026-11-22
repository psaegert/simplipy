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
    ('add.cancel', '- inv floor x0 inv floor x0', '- inv floor x0 inv floor x0', '0'),
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
    # only at a point). permissive spreads every product, and its terms then group by their
    # shared number again (`add.group`).
    ('add.unfactor', '+ x2 * 2 + x0 x1', '+ * 2 x0 + * 2 x1 x2', '+ x2 * 2 + x0 x1'),
    ('add.unfactor', '+ x2 * -2 + x0 x1', '- - x2 * 2 x0 * 2 x1', '- x2 * 2 + x0 x1'),
    ('add.unfactor', '+ ceil x2 * -2 + ceil x0 floor x1', '- - ceil x2 * 2 ceil x0 * 2 floor x1',
     '- ceil x2 * 2 + ceil x0 floor x1'),
    ('add.unfactor', 'exp + ceil x2 * -2 + ceil x0 floor x1',
     'exp - - ceil x2 * 2 ceil x0 * 2 floor x1', 'exp - ceil x2 * 2 + ceil x0 floor x1'),
    ('add.unfactor', 'inv + ceil x2 * -2 + ceil x0 floor x1', 'inv - ceil x2 * 2 + ceil x0 floor x1',
     'inv - ceil x2 * 2 + ceil x0 floor x1'),
    # beside x2, which is not -0 on a region, the product spreads: the sum's zero is isolated
    ('add.unfactor', 'inv + x2 * -2 + ceil x0 floor x1', '/ -1 - + * 2 ceil x0 * 2 floor x1 x2',
     'inv - x2 * 2 + ceil x0 floor x1'),
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
    # c*a + c*b + d = c*(a + b) + d. permissive writes a number its terms share once, the terms
    # grouped by the exact value of their coefficient up to sign (a literal term's coefficient is
    # itself), the sign in the coefficient; a magnitude of one is no number to share, and 4 is not
    # 2. f64 keeps the terms distributed.
    ('add.group', '+ + * 2 x0 * 2 x1 x2', '+ * 2 x0 + * 2 x1 x2', '+ x2 * 2 + x0 x1'),
    ('add.group', '+ - * 3 x0 * 3 x1 x2', '- + * 3 x0 x2 * 3 x1', '+ x2 * 3 - x0 x1'),
    ('add.group', '- - x2 * 2 x0 * 2 x1', '- - x2 * 2 x0 * 2 x1', '- x2 * 2 + x0 x1'),
    ('add.group', '+ + + * 2 x0 * 2 x1 2 x2', '+ * 2 x0 + * 2 x1 + x2 2', '+ x2 * 2 + x0 + x1 1'),
    ('add.group', '+ * 2 + x0 x1 x2', '+ * 2 x0 + * 2 x1 x2', '+ x2 * 2 + x0 x1'),
    ('add.group', '+ + * 2 x0 * 2 x1 * 4 x2', '+ * 2 x0 + * 2 x1 * 4 x2', '+ * 4 x2 * 2 + x0 x1'),
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

#: (law, input, f64 answer, permissive answer) of the functions' identities. Float64's round trip
#: through a function and its inverse drifts by many units in the last place while both sides stay
#: finite, so the f64 answer is checked against the input's values in kind: the same infinity, both
#: finite, or both without a value. Under an inverse, `ceil(x0)` is -0 on (-1, 0).
FUNCTION_ROWS = [
    # atanh(tanh(t)) = t; f64: |t| at most 19, where float64's tanh is no +-1
    ('fun.atanh_tanh', 'atanh tanh * 19 sin x0', '* 19 sin x0', '* 19 sin x0'),
    ('fun.atanh_tanh', 'atanh tanh * 20 sin x0', 'atanh tanh * 20 sin x0', '* 20 sin x0'),
    ('fun.atanh_tanh', 'atanh tanh x0', 'atanh tanh x0', 'x0'),
    ('fun.atanh_tanh', 'inv atanh tanh ceil sin x0', 'inv ceil sin x0', 'inv ceil sin x0'),
    # log(exp(t)) = t; f64: t is not -0 on a region, log(exp(-0)) being +0
    ('fun.log_exp', 'log exp x0', 'x0', 'x0'),
    ('fun.log_exp', 'floor log exp x0', 'floor x0', 'floor x0'),
    # f64: log(exp(t)) is +0 wherever 0 < |t| < 1.1e-16, so the pair stays under a head that reads
    # a zero (an inverse, log), through the heads that carry one (sin), and collapses under a head
    # blind to it (exp)
    ('fun.log_exp', 'sin log exp x0', 'sin x0', 'sin x0'),
    ('fun.log_exp', 'inv log exp x0', 'inv log exp x0', 'inv x0'),
    ('fun.log_exp', 'log log exp x0', 'log log exp x0', 'log x0'),
    ('fun.log_exp', 'inv sin log exp x0', 'inv sin log exp x0', 'inv sin x0'),
    ('fun.log_exp', 'inv exp log exp x0', 'inv exp x0', 'inv exp x0'),
    ('fun.log_exp', 'inv inv log exp x0', 'inv inv log exp x0', 'x0'),
    ('fun.log_exp', 'inv log exp ceil x0', 'inv log exp ceil x0', 'inv ceil x0'),
    # an odd negative power is -0 where its base is negative: at -inf (`log(ceil(x0))` on (-1, 0])
    # the pair stays, and a base that is never negative keeps it from -0
    ('fun.log_exp', 'log exp pow log ceil x0 -3', 'log exp inv pow log ceil x0 3',
     'inv pow log ceil x0 3'),
    ('fun.log_exp', 'log exp pow abs log ceil x0 -3', 'inv pow abs log ceil x0 3',
     'inv pow abs log ceil x0 3'),
    ('fun.asinh_sinh', 'asinh sinh x0', 'x0', 'x0'),
    ('fun.asinh_sinh', 'inv asinh sinh ceil x0', 'inv ceil x0', 'inv ceil x0'),
    ('fun.sinh_asinh', 'sinh asinh * 1000 sin x0', '* 1000 sin x0', '* 1000 sin x0'),
    ('fun.sinh_asinh', 'inv sinh asinh ceil x0', 'inv ceil x0', 'inv ceil x0'),
    # an odd head over a negated inverse collapses at once
    ('fun.sinh_asinh', 'sinh * min -1 cos 1 asinh x0', 'neg x0', 'neg x0'),
    # acosh(cosh(t)) = |t|; permissive cancels the pair, ignoring the branch. In f64 the pair is
    # flat at 0 (0 wherever |t| < 1.5e-8), and stays under a head that reads a zero
    ('fun.acosh_cosh', 'acosh cosh * 0.5 sin x0', 'abs / sin x0 2', '/ sin x0 2'),
    ('fun.acosh_cosh', 'inv acosh cosh ceil x0', 'inv acosh cosh ceil x0', 'inv ceil x0'),
    ('fun.acosh_cosh', 'exp acosh cosh x0', 'exp abs x0', 'exp x0'),
    ('fun.acosh_cosh', 'inv acosh cosh x0', 'inv acosh cosh x0', 'inv x0'),
    # exp(log(t)) = t; t >= 0, and in f64 not -0 on a region (numpy's sqrt keeps ceil's -0)
    ('fun.exp_log', 'exp log abs x0', 'abs x0', 'abs x0'),
    ('fun.exp_log', 'exp log x0', 'exp log x0', 'x0'),
    ('fun.exp_log', 'inv exp log pow ceil x0 0.5', 'inv exp log pow ceil x0 / 1 2',
     'inv pow ceil x0 / 1 2'),
    # log(1/t) = -log(t); f64: +0 against -0 at t = 1, so over a monomial t alone, which is 1 only
    # at points, whose zero's sign is no value kept, under every head (an inverse, an odd negative
    # power); the pair stays over a t float64 rounds to 1 on a region (exp(x0), ceil(x0), a sum).
    # permissive reads one zero: everywhere. Both at the root's walk, after exp(log(1/t)) = 1/t
    ('fun.log_inverse', 'log inv x0', 'neg log x0', 'neg log x0'),
    ('fun.log_inverse', 'inv inv log inv x0', 'neg log x0', 'neg log x0'),
    ('fun.log_inverse', 'log inv abs x0', 'neg log abs x0', 'neg log abs x0'),
    ('fun.log_inverse', 'sin log inv x0', 'neg sin log x0', 'neg sin log x0'),
    ('fun.log_inverse', 'exp log inv x0', 'exp neg log x0', 'inv x0'),
    ('fun.log_inverse', 'exp log inv abs x0', 'inv abs x0', 'inv abs x0'),
    ('fun.log_inverse', 'inv log inv x0', '/ -1 log x0', '/ -1 log x0'),
    ('fun.log_inverse', 'pow log inv x0 -3', '/ -1 pow log x0 3', '/ -1 pow log x0 3'),
    ('fun.log_inverse', 'inv sin log inv x0', '/ -1 sin log x0', '/ -1 sin log x0'),
    ('fun.log_inverse', 'log abs inv x0', 'neg log abs x0', 'neg log abs x0'),
    ('fun.log_inverse', 'log inv abs x0', 'neg log abs x0', 'neg log abs x0'),
    ('fun.log_inverse', 'log inv exp x0', 'log inv exp x0', 'neg x0'),
    ('fun.log_inverse', 'log inv ceil x0', 'log inv ceil x0', 'neg log ceil x0'),
    ('fun.log_inverse', 'log inv + x0 x1', 'log inv + x0 x1', 'neg log + x0 x1'),
    ('fun.log_inverse', 'inv log inv + x0 x1', 'inv log inv + x0 x1', '/ -1 log + x0 x1'),
    ('fun.unit_inverses', 'cos acos sin x0', 'sin x0', 'sin x0'),
    ('fun.unit_inverses', 'tanh atanh cos x0', 'cos x0', 'cos x0'),
    ('fun.unit_inverses', 'cos acos x0', 'cos acos x0', 'x0'),
    # permissive's own: an inverse pair cancels in both orders, ignoring domain and branch
    ('perm.inverse_pairs', 'asin sin x0', 'asin sin x0', 'x0'),
    ('perm.inverse_pairs', 'atan tan x0', 'atan tan x0', 'x0'),
    # |t| = t; t >= 0, and in f64 not -0 on a region
    ('fun.abs_nonnegative', 'abs exp x0', 'exp x0', 'exp x0'),
    ('fun.abs_nonnegative', 'abs sin x0', 'abs sin x0', 'abs sin x0'),
    ('fun.abs_nonnegative', 'inv abs pow ceil x0 0.5', 'inv abs pow ceil x0 / 1 2',
     'inv pow ceil x0 / 1 2'),
    # a negative power of a base that is never negative is never -0
    ('fun.abs_nonnegative', 'abs * 2 abs / -3 abs + x0 tan / x0 cos log - 1 * x0 x0',
     '/ 6 abs + x0 tan / x0 cos log - 1 pow x0 2', '/ 6 abs + x0 tan / x0 cos log - 1 pow x0 2'),
    ('fun.half_period', 'sin + x0 np.pi', 'neg sin x0', 'neg sin x0'),
    ('fun.half_period', 'cos - x0 np.pi', 'neg cos x0', 'neg cos x0'),
    ('fun.half_period', 'tan + x0 np.pi', 'tan x0', 'tan x0'),
    ('fun.half_period', 'sin - np.pi x0', 'sin x0', 'sin x0'),
    ('fun.besselj_reflection', 'besselj -3 x0', 'neg besselj 3 x0', 'neg besselj 3 x0'),
    ('fun.besselj_reflection', 'besselj -2 x0', 'besselj 2 x0', 'besselj 2 x0'),
    # the dual bag absorbs, the absorber takes, the identity drops; f64: the dropped members no
    # NaN almost everywhere
    ('lattice.absorb', 'max x0 min x0 x1', 'x0', 'x0'),
    ('lattice.absorb', 'max x0 min x0 log x1', 'max x0 min x0 log x1', 'x0'),
    ('lattice.absorb', 'max float("-inf") x0', 'x0', 'x0'),
    ('lattice.absorb', 'max float("inf") log x0', 'max float("inf") log x0', 'float("inf")'),
]

#: (law, input, f64 answer, permissive answer) of the laws of powers, each where its premise holds
#: and where it fails
POWER_ROWS = [
    # (a*b)^n = a^n*b^n; an odd negative power keeps a product that vanishes on a region with no
    # sign of its own (an even power of a sum) in f64
    ('pow.distribute', 'pow * x0 x1 2', '* pow x0 2 pow x1 2', '* pow x0 2 pow x1 2'),
    ('pow.distribute', 'pow * x0 x1 -3', '* inv pow x0 3 inv pow x1 3',
     '* inv pow x0 3 inv pow x1 3'),
    ('pow.distribute', 'pow * x1 pow - x0 abs x0 2 -3', 'inv * pow x1 3 pow - x0 abs x0 6',
     '* inv pow x1 3 inv pow - x0 abs x0 6'),
    # a divisor's product is read as its members' inverses where pow.distribute holds
    ('parse.inverse_of_product', '/ x2 * x0 x1', '/ x2 * x0 x1', '/ x2 * x0 x1'),
    ('parse.inverse_of_product', '/ x2 * x1 pow - x0 abs x0 2', '/ x2 * x1 pow - x0 abs x0 2',
     '/ * x2 inv pow - x0 abs x0 2 x1'),
    # (t^r)^s = t^(r*s) where both sides agree on t < 0
    ('pow.compose', 'pow pow x0 2 3', 'pow x0 6', 'pow x0 6'),
    ('pow.compose', 'pow pow x0 2 0.5', 'rootn pow x0 2 2', 'x0'),
    ('pow.compose', 'pow pow x0 3 0.5', 'pow x0 / 3 2', 'pow x0 / 3 2'),
    ('pow.compose', 'pow pow abs x0 2 0.5', 'abs x0', 'abs x0'),
    # rootn(t, n)^(+-n) = t^(+-1), n odd or t >= 0
    ('pow.root_inverse', 'pow rootn x0 3 3', 'x0', 'x0'),
    ('pow.root_inverse', 'pow rootn x0 3 -3', 'inv x0', 'inv x0'),
    ('pow.root_inverse', 'rootn pow x0 3 3', 'x0', 'x0'),
    ('pow.root_inverse', 'pow rootn x0 2 2', 'pow rootn x0 2 2', 'x0'),
    ('pow.root_inverse', 'pow rootn abs x0 2 2', 'abs x0', 'abs x0'),
    # rootn(t, m)^s = t^(s/m) for a non-integer s; an integer s keeps the root, whose class keeps
    # one spelling
    ('pow.odd_root_compose', 'pow rootn x0 3 0.5', 'rootn x0 6', 'rootn x0 6'),
    ('pow.odd_root_compose', 'pow rootn x0 3 2', 'pow rootn x0 3 2', 'pow rootn x0 3 2'),
    ('pow.odd_root_compose', 'pow rootn abs x0 3 2', 'pow rootn abs x0 3 2',
     'pow rootn abs x0 3 2'),
    # an even unit fraction over an index of no known parity
    ('pow.unknown_parity_root', 'pow rootn x0 9007199254740993 -0.5',
     'inv rootn x0 18014398509481986', 'inv rootn x0 18014398509481986'),
    # a literal base absorbs the exponent's sign
    ('pow.literal_base_sign', 'pow 2 neg x0', 'pow / 1 2 x0', 'pow / 1 2 x0'),
    ('pow.literal_base_sign', 'inv pow 3 x0', 'pow / 1 3 x0', 'pow / 1 3 x0'),
    ('pow.literal_base_sign', 'pow inv x0 float("inf")', 'pow x0 float("-inf")',
     'pow x0 float("-inf")'),
    # t^(1/2) = rootn(t, 2); f64's spelled square root keeps a zero's sign of its own
    ('pow.half_root', 'pow x0 0.5', 'rootn x0 2', 'rootn x0 2'),
    ('pow.half_root', 'pow ceil x0 0.5', 'pow ceil x0 / 1 2', 'rootn ceil x0 2'),
    ('pow.half_root', 'pow ceil x0 -0.5', 'inv rootn ceil x0 2', 'inv pow ceil x0 / 1 2'),
    # a power of a non-integer exponent is >= 0 wherever it has a value
    ('fun.abs_nonnegative', 'abs pow x0 / 1 3', 'pow x0 / 1 3', 'pow x0 / 1 3'),
    # t^a*t^b = t^(a+b) where the branch cut allows it
    ('mul.collect', '* pow x0 2 pow x0 3', 'pow x0 5', 'pow x0 5'),
    ('mul.collect', '* pow x0 0.5 pow x0 0.5', '* rootn x0 2 rootn x0 2', 'x0'),
    ('mul.collect', '* rootn x0 2 x0', 'pow x0 / 3 2', 'pow x0 / 3 2'),
    ('mul.collect', '* pow abs x0 0.5 pow abs x0 0.5', 'abs x0', 'abs x0'),
    # t^y*t^y = t^(2y): t >= 0 or y's level sets null, and in f64 t not -0 on a region
    ('mul.collect_symbolic', '* pow x0 x1 pow x0 x1', 'pow x0 * 2 x1', 'pow x0 * 2 x1'),
    ('mul.collect_symbolic', '* pow abs x0 x1 pow abs x0 x1', 'pow abs x0 * 2 x1',
     'pow abs x0 * 2 x1'),
    ('mul.collect_symbolic', '* pow ceil x0 x1 pow ceil x0 x1', '* pow ceil x0 x1 pow ceil x0 x1',
     'pow ceil x0 * 2 x1'),
    ('mul.collect_symbolic', '* pow x0 floor x1 pow x0 floor x1',
     '* pow x0 floor x1 pow x0 floor x1', 'pow x0 * 2 floor x1'),
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

#: (law, input, f64 answer, permissive answer) of the laws of signs and orientation. In f64 a sum
#: that may vanish on a region keeps the sign of its zero where it is read; at an isolated zero the
#: sign, and the sign of the infinity a pole makes of it, is no value kept (`-1/(x0 - 3)` is
#: `1/(3 - x0)`), and a one-signed pole keeps its limit; permissive reads no value at a pole.
SIGN_ROWS = [
    # f(-S) = -f(S): the cheaper placement; under an inverse a sum that vanishes on a region holds
    # its sign
    ('class.sign_trade', 'neg sin - x0 x1', 'sin - x1 x0', 'sin - x1 x0'),
    ('class.sign_trade', '* -2 sin - x0 3', '* 2 sin - 3 x0', '* 2 sin - 3 x0'),
    ('class.sign_trade', 'inv * -2 sin + ceil x0 floor x1', '/ neg inv sin + ceil x0 floor x1 2',
     '/ neg inv sin + ceil x0 floor x1 2'),
    # f(-S)^-k = -f(S)^-k, k odd; f64: an isolated zero's sign is no value kept
    ('class.pole_orientation', 'neg inv - x0 3', 'inv - 3 x0', 'inv - 3 x0'),
    ('class.pole_orientation', 'neg inv sin - x0 3', 'inv sin - 3 x0', 'inv sin - 3 x0'),
    ('class.pole_orientation', 'neg pow - x0 3 -3', 'inv pow - 3 x0 3', 'inv pow - 3 x0 3'),
    ('class.pole_orientation', 'neg inv - -1 pow x0 2', 'inv + pow x0 2 1', 'inv + pow x0 2 1'),
    # S^a*(-S)^b = (-1)^b*S^(a+b); a negative total keeps a one-signed pole's limit
    ('class.opposite_sums', '* - x0 3 - 3 x0', 'neg pow - x0 3 2', 'neg pow - x0 3 2'),
    ('class.opposite_sums', 'inv * - x0 3 - 3 x0', '/ -1 pow - x0 3 2', '/ -1 pow - x0 3 2'),
    ('class.opposite_sums', '* pow - x0 3 -1 pow - 3 x0 -2', 'inv pow - x0 3 3',
     'inv pow - x0 3 3'),
    # like terms up to the orientation of their classes; a two-signed pole's infinity is an
    # isolated zero's sign
    ('class.like_terms_up_to_sign', '+ sin - x0 x1 sin - x1 x0', '0', '0'),
    ('class.like_terms_up_to_sign', '+ inv - x0 3 inv - 3 x0', '0', '0'),
    ('pow.even_orientation', 'pow - x1 x0 2', 'pow - x0 x1 2', 'pow - x0 x1 2'),
    ('pow.even_orientation', 'pow sin * -2 x0 2', 'pow sin * 2 x0 2', 'pow sin * 2 x0 2'),
    # (-F)^s = flip(F)^s; f64: an odd negative s keeps a sign of a region's zero
    ('pow.negated_site', 'pow neg - x0 3 3', 'pow - 3 x0 3', 'pow - 3 x0 3'),
    ('pow.negated_site', 'inv pow neg + ceil x0 floor x1 3', '/ -1 pow + ceil x0 floor x1 3',
     '/ -1 pow + ceil x0 floor x1 3'),
    ('fun.even', 'cos neg x0', 'cos x0', 'cos x0'),
    ('fun.even', 'cos - x1 x0', 'cos - x0 x1', 'cos - x0 x1'),
    ('fun.odd_sign', 'sin neg x0', 'neg sin x0', 'neg sin x0'),
    ('fun.odd_sign', 'sin * -2 x0', 'neg sin * 2 x0', 'neg sin * 2 x0'),
    # f64 alone: permissive's one zero gives atan2(0, -1) = pi
    ('fun.atan2_sign', 'atan2 neg x0 x1', 'neg atan2 x0 x1', 'atan2 neg x0 x1'),
]

#: (law, input, f64 answer, permissive answer) over the scipy heads (`scipy_engine`). An odd
#: order of besseli is odd but at a zero, where the kernel gives +0 for either sign, so in f64 it
#: moves a sign only where its argument vanishes on no region.
SCIPY_ROWS = [
    ('fun.besseli_parity', 'besseli 2 neg x0', 'besseli 2 x0', 'besseli 2 x0'),
    ('fun.besseli_parity', 'besseli 0 - x1 x0', 'besseli 0 - x0 x1', 'besseli 0 - x0 x1'),
    ('fun.besseli_parity', 'besseli 1 * -2 x0', 'neg besseli 1 * 2 x0', 'neg besseli 1 * 2 x0'),
    ('fun.besseli_parity', 'besseli 1 neg ceil x0', 'besseli 1 neg ceil x0',
     'neg besseli 1 ceil x0'),
]


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    with open(os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')) as f:
        ops = yaml.safe_load(f)['operators']
    return SimpliPyEngine(operators=ops, rules=[], rules_permissive=[])


@pytest.fixture(scope='module')
def rule_engine() -> SimpliPyEngine:
    """acj2-4's operators and two rules, the same in each mode."""
    with open(os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')) as f:
        ops = yaml.safe_load(f)['operators']
    rules = [[['+', 'pow', 'sin', '_0', '2', 'pow', 'cos', '_0', '2'], ['1']],
             [['-', '1', '*', '2', 'pow', 'sin', '_0', '2'], ['cos', '*', '2', '_0']]]
    return SimpliPyEngine(operators=ops, rules=rules, rules_permissive=rules)


@pytest.fixture(scope='module')
def scipy_engine() -> SimpliPyEngine:
    """acj2-4's operators and the scipy heads, with an empty rule set per mode."""
    with open(os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')) as f:
        ops = yaml.safe_load(f)['operators']
    for name, arity in {'erfc': 1, 'bessely': 2, 'besseli': 2, 'besselk': 2}.items():
        ops[name] = {'realization': f'simplipy.operators.{name}', 'alias': [], 'arity': arity,
                     'commutative': False}
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
        ('add.group', 'c*a + c*b + d = c*(a + b) + d'),
        ('mul.exp_merge', 'exp(a)*exp(b) = exp(a + b)'),
        ('mul.rootn_join', 'rootn(a, n)*rootn(b, n) = rootn(a*b, n)'),
        ('mul.infinity_sign', '-k*(s + inf) = k*(-s - inf)'),
        ('pow.exp_compose', 'e^t = exp(t), exp(a)^b = exp(a*b)'),
        ('pow.refused_power', '(a*b)^-k = ((a*b)^k)^-1'),
        ('fun.rootn_index', 'rootn(t, -n) = 1/rootn(t, n)'),
        ('pow.distribute', '(a*b)^n = a^n*b^n'),
        ('parse.inverse_of_product', '1/(a*b) = (1/a)*(1/b)'),
        ('pow.compose', '(t^r)^s = t^(r*s)'), ('pow.root_inverse', 'rootn(t, n)^n = t'),
        ('pow.odd_root_compose', 'rootn(t, m)^s = t^(s/m)'),
        ('pow.unknown_parity_root', 'rootn(t, m)^(-1/n) = 1/rootn(rootn(t, m), n)'),
        ('pow.literal_base_sign', 'r^(-t) = (1/r)^t'),
        ('pow.half_root', 't^(1/(2*k)) = rootn(t, 2*k)'),
        ('mul.collect', 't^a*t^b = t^(a+b)'), ('mul.collect_symbolic', 't^y*t^y = t^(2*y)'),
        ('fun.atanh_tanh', 'atanh(tanh(t)) = t'), ('fun.log_exp', 'log(exp(t)) = t'),
        ('fun.asinh_sinh', 'asinh(sinh(t)) = t'), ('fun.sinh_asinh', 'sinh(asinh(t)) = t'),
        ('fun.acosh_cosh', 'acosh(cosh(t)) = abs(t)'), ('fun.exp_log', 'exp(log(t)) = t'),
        ('fun.log_inverse', 'log(1/t) = -log(t), log(abs(1/t)) = -log(abs(t))'),
        ('fun.unit_inverses', 'cos(acos(t)) = t, sin(asin(t)) = t, tanh(atanh(t)) = t'),
        ('perm.inverse_pairs', 'f(g(t)) = t, g the inverse of f'),
        ('fun.abs_nonnegative', 'abs(t) = t'),
        ('fun.half_period', 'sin(t + pi) = -sin(t), cos(t + pi) = -cos(t), tan(t + pi) = tan(t)'),
        ('fun.besselj_reflection', 'J_(-n)(t) = (-1)^n J_n(t)'),
        ('lattice.absorb', 'max(a, min(a, b)) = a, max(inf, a) = inf, max(-inf, a) = a'),
        ('class.sign_trade', 'f(-S) = -f(S)'),
        ('class.pole_orientation', 'f(-S)^-k = -f(S)^-k, k odd'),
        ('class.opposite_sums', 'S^a*(-S)^b = (-1)^b*S^(a+b)'),
        ('class.like_terms_up_to_sign', 'u*f(S) + v*f(-S) = (u - v)*f(S)'),
        ('pow.even_orientation', '(-S)^(2*k) = S^(2*k)'), ('pow.negated_site', '(-F)^s = flip(F)^s'),
        ('fun.even', 'g(-t) = g(t)'), ('fun.odd_sign', 'f(-u) = -f(u)'),
        ('fun.atan2_sign', 'atan2(-y, x) = -atan2(y, x)'),
        ('fun.besseli_parity', 'I_n(-t) = (-1)^n I_n(t)'),
        ('search.distribute', 'A*(B + C) = A*B + A*C'),
        ('search.pow_expand', '(a + b)^n = its expansion, 2 <= n <= 6'),
        ('search.factor', 'c*q1 + c*q2 = c*(q1 + q2)'),
        ('search.reciprocal_join', 'A^-1*B^-1 = (A*B)^-1, f*A^-1 = (A/f)^-1'),
        ('search.composite_step', 'a refused rebuild is kept where the whole descends'),
        ('settle.rule', 'a mined rule L -> T fires where L binds'),
        ('settle.negation_twin', 'L -> T entails -L -> -T'),
        ('perm.reciprocal_projection', 'a^-1*b^-1 = (a*b)^-1, as written')]


#: add.group's rows beyond three variables or holding a fitted constant (the float64 check of
#: `ROWS` reads x0, x1 and x2): two shared numbers, a term with a fitted constant, and terms of
#: magnitude one beside a group.
GROUP_ROWS = [
    ('add.group', '+ + * 2 x0 * 2 x1 + * 3 x2 * 3 x3', '+ * 2 x0 + * 2 x1 + * 3 x2 * 3 x3',
     '+ * 2 + x0 x1 * 3 + x2 x3'),
    ('add.group', '+ + * 2 x0 * 2 x1 <constant>', '+ * 2 x0 + * 2 x1 <constant>',
     '+ * 2 + x0 x1 <constant>'),
    ('add.group', '+ + neg x0 neg x1 + * 2 x2 * 2 x3', '- - + * 2 x2 * 2 x3 x0 x1',
     '- - * 2 + x2 x3 x0 x1'),
]

ALL_ROWS = ROWS + GROUND_ROWS + FUNCTION_ROWS + POWER_ROWS + ABSORB_ROWS + SIGN_ROWS + GROUP_ROWS

#: (law, input, f64 answer, permissive answer) of the search's laws, at the default effort (`effort=0`
#: does not search). The distributive law's two sides differ only where a piece it multiplies into
#: two or more addends, or collects out of them, is infinite: in f64 such a piece is infinite only
#: on a null set and may have no value on a region, and at each pole one addend keeps the pole,
#: each addend that keeps it carries two-signed poles only (its infinity's sign an isolated
#: zero's), or those addends take one strict sign; permissive reads no value at a pole.
SEARCH_ROWS = [
    ('search.distribute', '* x2 + x2 / + x1 1 x2', '+ x1 + pow x2 2 1', '+ x1 + pow x2 2 1'),
    # one move whose constructors cancel x0/x0: 1/x0 keeps its pole in one addend
    ('search.distribute', '* inv x0 + 1 x0', '+ inv x0 1', '+ inv x0 1'),
    # a two-signed pole may split, but the split is dearer
    ('search.distribute', '/ + x1 x2 x0', '/ + x1 x2 x0', '/ + x1 x2 x0'),
    # -inf at x1 = 0 where x2 < 0, the limit from x1 > 0, where x2/x1 + 2/sqrt(x1) is NaN
    ('search.distribute', '/ + x2 * 2 rootn x1 2 x1', '/ + x2 * 2 rootn x1 2 x1',
     '+ / 2 pow x1 / 1 2 / x2 x1'),
    ('search.pow_expand', '- pow + x0 1 2 pow x0 2', '+ * 2 x0 1', '+ * 2 x0 1'),
    # +inf at x0 = 0, a one-signed pole, where the expansion is NaN for x1 < 0
    ('search.pow_expand', '- pow + inv x0 x1 2 pow x1 2', '- pow + x1 inv x0 2 pow x1 2',
     '+ inv pow x0 2 / * 2 x1 x0'),
    ('search.factor', '+ * x0 x1 * x0 x2', '* x0 + x1 x2', '* x0 + x1 x2'),
    # a content with no value on a region
    ('search.factor', '+ * x1 rootn x0 2 rootn x0 2', '* rootn x0 2 + x1 1', '* rootn x0 2 + x1 1'),
    ('search.factor', '- log x0 log x0', '* 0 log x0', '0'),
    # composite results through one move: the content's pole is no addend's
    ('search.factor', '- / + x1 x2 x0 / x2 x0', '/ x1 x0', '/ x1 x0'),
    # the pole's sign at the sum's zero is an isolated zero's
    ('search.factor', 'inv + x1 * x1 x2', '/ inv + x2 1 x1', '/ inv + x2 1 x1'),
    # the classes' orientations at the pole x2 = x3 (class.pole_orientation)
    ('search.factor', '+ / x1 - x2 x3 / x4 - x3 x2', '/ - x1 x4 - x2 x3', '/ - x1 x4 - x2 x3'),
    ('search.reciprocal_join', 'inv * + x0 1 - x0 1', '* inv - x0 1 inv + x0 1', 'inv - pow x0 2 1'),
    # tan(x0) has a zero, where the left side is 0 and 2/(1/tan(x0) - tan(x0)) has no value
    ('search.reciprocal_join', '/ * 2 tan x0 - 1 pow tan x0 2', '/ * 2 tan x0 - 1 pow tan x0 2',
     '/ * 2 tan x0 - 1 pow tan x0 2'),
    # f64 distributes the inverse over the product (pow.distribute); permissive's answer writes it
    # joined
    ('perm.reciprocal_projection', 'inv * x0 floor x1', '/ inv floor x1 x0', 'inv * x0 floor x1'),
]

#: (law, input, f64 answer, permissive answer) of the settling's laws, on an engine with two rules
#: (`rule_engine`): sin^2 + cos^2 = 1, and 1 - 2*sin^2 = cos(2*t), whose twin -(1 - 2*sin^2) =
#: -cos(2*t) the load mints.
RULE_ROWS = [
    ('settle.rule', '+ pow sin x0 2 pow cos x0 2', '1', '1'),
    ('settle.rule', '* 3 + pow sin x1 2 pow cos x1 2', '3', '3'),
    ('settle.negation_twin', '- * 2 pow sin x0 2 1', 'neg cos * 2 x0', 'neg cos * 2 x0'),
    ('settle.negation_twin', '+ x1 - * 2 pow sin x0 2 1', '- x1 cos * 2 x0', '- x1 cos * 2 x0'),
]


#: The search's composite step is an acceptance with no identity of its own; a rebuild the settling
#: refuses and the search then keeps is met nowhere in srbf's ground truths, a tenth of the
#: training-data inputs or a tenth of the predictions, so its proposal is pinned in the core's own
#: tests (`ac::search`, `the_composite_step_proposes_a_refused_rebuild_where_its_node_stands`).
ACCEPTANCES = {'search.composite_step'}


def test_every_law_has_its_rows() -> None:
    rows = ALL_ROWS + SCIPY_ROWS + SEARCH_ROWS + RULE_ROWS
    assert {law for law, *_ in rows} | ACCEPTANCES == {name for name, _ in _core.laws()}


def rows_on(rows, fixture):
    return [pytest.param(fixture, *row, id=f'{row[0]}: {row[1]}') for row in rows]


@pytest.mark.parametrize('fixture, law, expr, f64, permissive',
                         rows_on(ALL_ROWS, 'engine') + rows_on(SCIPY_ROWS, 'scipy_engine')
                         + rows_on(RULE_ROWS, 'rule_engine'))
@pytest.mark.parametrize('effort', [0, None], ids=['effort 0', 'default effort'])
def test_each_mode_applies_the_law_where_its_premise_holds(
        request, fixture, law, expr, f64, permissive, effort) -> None:
    engine = request.getfixturevalue(fixture)
    for mode, want in (('f64', f64), ('permissive', permissive)):
        got = ' '.join(engine.simplify(expr.split(), mode=mode, effort=effort))
        assert got == want, (law, mode, expr)


@pytest.mark.parametrize('fixture, law, expr, f64, permissive',
                         rows_on(ALL_ROWS, 'engine') + rows_on(SCIPY_ROWS, 'scipy_engine')
                         + rows_on(RULE_ROWS, 'rule_engine'))
def test_each_answer_reads_back_to_itself(request, fixture, law, expr, f64, permissive) -> None:
    engine = request.getfixturevalue(fixture)
    for mode, answer in (('f64', f64), ('permissive', permissive)):
        assert ' '.join(engine.simplify(answer.split(), mode=mode, effort=0)) == answer, (law, mode)


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in SEARCH_ROWS])
def test_the_search_applies_its_laws_where_their_premises_hold(engine, law, expr, f64, permissive) -> None:
    for mode, want in (('f64', f64), ('permissive', permissive)):
        got = ' '.join(engine.simplify(expr.split(), mode=mode))
        assert got == want, (law, mode, expr)
        # the answer is its own simplification
        assert ' '.join(engine.simplify(want.split(), mode=mode)) == want, (law, mode)


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in SEARCH_ROWS])
def test_the_searchs_f64_answer_keeps_float64s_values(engine, law, expr, f64, permissive) -> None:
    """Float64's value of the search's f64 answer is the input's wherever the input has one, an
    infinity's sign included, but at the isolated zeros of its pieces (x0, x1 and x2 - x3 at 0),
    where a pole's infinity is no value kept (reading A)."""
    names = ['x0', 'x1', 'x2', 'x3', 'x4']
    rng = np.random.default_rng(0)
    points = rng.uniform(-3, 3, size=(2000, 5))
    for col in range(4):
        points[col * 100:(col + 1) * 100, col] = 0.0
    points[400:500, 2] = points[400:500, 3]
    points[500:600, 1:3] = [1.0, -2.0]

    def values(tokens):
        f = engine.as_callable(tokens.split(), variables=names)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v = np.asarray(f(*points.T), dtype=np.float64)
        return np.broadcast_to(v, (len(points),))

    a, b = values(expr), values(f64)
    isolated = np.zeros(len(points), dtype=bool)
    isolated[:600] = True
    same = (np.isnan(a) | (a == b) | (np.isfinite(a) & np.isclose(a, b, rtol=1e-12, atol=0))
            | (isolated & np.isinf(a)))
    assert same.all(), (law, expr, f64, points[~same][0])


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


@pytest.mark.parametrize('law, expr, f64, permissive', [
    pytest.param(*row, id=f'{row[0]}: {row[1]}') for row in FUNCTION_ROWS])
def test_the_f64_answer_keeps_float64s_kinds(engine, law, expr, f64, permissive) -> None:
    """Float64's value of the f64 answer is the input's in kind wherever the input has one: the
    same infinity (its sign included), or both finite."""
    names = ['x0', 'x1', 'x2']
    points = np.random.default_rng(0).uniform(-3, 3, size=(2000, 3))

    def values(tokens):
        f = engine.as_callable(tokens.split(), variables=names)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v = np.asarray(f(*points.T), dtype=np.float64)
        return np.broadcast_to(v, (len(points),))

    a, b = values(expr), values(f64)
    same = np.isnan(a) | (a == b) | (np.isfinite(a) & np.isfinite(b))
    assert same.all(), (law, expr, f64, points[~same][0])


@pytest.mark.parametrize('fixture, law, expr, f64, permissive',
                         rows_on(SIGN_ROWS, 'engine') + rows_on(SCIPY_ROWS, 'scipy_engine'))
def test_the_f64_answer_keeps_float64s_values_at_its_zeros(
        request, fixture, law, expr, f64, permissive) -> None:
    """Float64's value of the f64 answer is the input's wherever the input has one, an infinity's
    sign included, on the regions where its sums vanish too (points in (-1, 0) and [0, 1), where
    `ceil` and `floor` are `-0` and `+0`); at the isolated zeros of its sums (the integer points of
    the plane, where x0 = 3, x1 = 5 and x0 = x1 are met) a pole's infinity is no value kept
    (reading A), and the answer's own zero may be either zero, as nothing reads it."""
    engine = request.getfixturevalue(fixture)
    names = ['x0', 'x1', 'x2']
    grid = np.array([(a, b, 0.0) for a in np.arange(-3.0, 7.0) for b in np.arange(-3.0, 7.0)])
    halves = np.array([(a, b, 0.0) for a in (-0.5, 0.25) for b in (-0.5, 0.25)])
    points = np.vstack([grid, halves, np.random.default_rng(0).uniform(-3, 3, size=(500, 3))])

    def values(tokens):
        f = engine.as_callable(tokens.split(), variables=names)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v = np.asarray(f(*points.T), dtype=np.float64)
        return np.broadcast_to(v, (len(points),))

    a, b = values(expr), values(f64)
    isolated = np.zeros(len(points), dtype=bool)
    isolated[:len(grid)] = True
    same = (np.isnan(a) | (a == b) | (np.isfinite(a) & np.isclose(a, b, rtol=1e-12, atol=0))
            | (isolated & np.isinf(a)))
    assert same.all(), (law, expr, f64, points[~same][0])


@pytest.mark.parametrize('expr, f64', [
    # srbf's ground truths #164 and #584: log(1/x1) collapses where nothing reads its zero's sign
    ('* * -3.47355 x1 + * * x1 - x1 3.47355 log / 1 x1 * 3.47355 x1',
     '* -3.47355 * pow x1 2 + * log x1 - 3.47355 x1 3.47355'),
    ('* + neg x1 / x1 x2 + + - * x1 x2 x1 log / 1 x1 / 1 * x1 x2',
     '* - inv x2 1 - inv x2 * x1 + log x1 * x1 - 1 x2'),
])
def test_the_log_of_an_inverse_collapses_in_srbfs_ground_truths(engine, expr, f64) -> None:
    assert ' '.join(engine.simplify(expr.split(), mode='f64')) == f64
