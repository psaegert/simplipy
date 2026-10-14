"""A product is decided on the exponent totals of its classes, whatever bracketing and order spelled it.

The parse builds a product once from all of its members: `* a * b c`, `* * a b c` and `<mul> a b c </mul>` are one
product, and no smaller product a spelling nests is decided on its own first. `mul()` then reads each class {B, -B}
(B a sum, or an odd function of one) as a table of exponent totals per orientation and decides where the movable
totals go: all of the positive mass (an odd amount moved carries a sign) and the even part of the negative mass, a
fractional power's integer part counted as mass; never a lone `B^-1` (the sign of the infinity at the zero;
permissive, which reads a function up to its values at a zero, moves it too), the rest of a fractional power or a
symbolic one. The certificate-free reading and the certified one run the same decision. So every bracketing and order of the same members gives one answer, one
certificate-free price and one certified price, and inputs whose tables agree give one answer. Every answer is
simplified again.
"""
import itertools
import random

import numpy as np
import pytest

from simplipy.engine import SimpliPyEngine
from conftest import acj_config_path

CONFIG = acj_config_path()
MODES = ('f64', 'real', 'permissive')
BAG_TOKENS = {'<add>', '</add>', '<mul>', '</mul>', '<sub>', '<div>'}


@pytest.fixture(scope='module')
def eng():
    from conftest import require_or_skip
    require_or_skip(CONFIG, 'acj config not staged')
    return SimpliPyEngine.from_config(CONFIG, modes='all')


S, N = '<add> 3 <sub> x1 </add>', '<add> x1 <sub> 3 </add>'       # 3 - x1 and x1 - 3
T, TN = 'tanh <add> x1 <sub> x2 </add>', 'tanh <add> x2 <sub> x1 </add>'
SIN, SINN = 'sin <add> x2 <sub> 0.5 </add>', 'sin <add> 0.5 <sub> x2 </add>'
R, RN = 'rootn <add> x2 <sub> exp x1 </add> 3', 'rootn <add> exp x1 <sub> x2 </add> 3'


def trees(fs):
    if len(fs) == 1:
        yield fs[0]
        return
    for i in range(1, len(fs)):
        for a in trees(fs[:i]):
            for b in trees(fs[i:]):
                yield ['*'] + a + b


def spellings(members):
    ms = [m.split() for m in members]
    for order in itertools.permutations(ms):
        yield ['<mul>'] + [t for m in order for t in m] + ['</mul>']
        yield from trees(list(order))


# Products whose smaller products, decided first, used to keep apart what the whole product collects: an inner
# `(3 - x1)^-2 / (3 - x1)` merged to an odd negative power whose even part could no longer meet the root over
# `x1 - 3`; an inner `tanh(x1 - x2) * -1` became `tanh(x2 - x1)` beside `tanh(x1 - x2)`; a bare sum cancelled
# against one inverse inside and left its sign stuck on the other.
BAGS = {
    'S^-2 S^-1 sqrt(N)': ['inv pow ' + S + ' 2', 'inv ' + S, 'pow ' + N + ' 0.5'],
    'S 1/N 1/S': [S, 'inv ' + N, 'inv ' + S],
    'tanh tanh -1': [T, T, '-1'],
    'tanh 1/tanh(-) tanh(-)^2': [T, 'inv ' + TN, 'pow ' + TN + ' 2'],
    'inf S sqrt(N)': ['float("inf")', S, 'pow ' + N + ' 0.5'],
    'const S N': ['<constant>', S, N],
    'sin(-)^3 sin 2': ['pow ' + SINN + ' 3', SIN, '2'],
    'S^3 N^-2 sqrt(S) x2': ['pow ' + S + ' 3', 'inv pow ' + N + ' 2', 'pow ' + S + ' 0.5', 'x2'],
    # an odd root's members collect to its radicand, a factor outside the class: decided again from there
    'cbrt^2 cbrt^-3 cbrt(-)^2 cbrt(-)^-1': ['pow ' + RN + ' 2', 'inv pow ' + RN + ' 3', 'pow ' + R + ' 2', 'inv ' + R],
}


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('bag', sorted(BAGS))
def test_every_bracketing_and_order_is_one_product(eng, mode, bag):
    answers = {0: set(), None: set()}
    bare, certified = set(), set()
    for s in spellings(BAGS[bag]):
        for effort in answers:
            answers[effort].add(' '.join(eng.simplify(s, mode=mode, effort=effort)))
        bare.add(eng.complexity(s, mode=mode, certified=False))
        certified.add(eng.complexity(s, mode=mode))
    assert len(bare) == 1 and len(certified) == 1, (bare, certified)
    for effort, got in answers.items():
        # the tagged and the explicit spelling of the one answer
        assert len({eng.complexity(a.split(), mode=mode) for a in got}) == 1 and len(got) <= 2, got
        for a in got:
            assert list(eng.simplify(a.split(), mode=mode, effort=effort)) == a.split()


# Inputs whose tables agree give one answer: an odd negative power is a lone inverse beside an even power, an odd
# function of the negated sum the negated function, and an even power of an odd function of a sum takes the
# orientation its sum takes.
EQUAL = [
    ('<mul> inv pow ' + S + ' 3 pow ' + N + ' 0.5 </mul>', '<mul> inv pow ' + S + ' 2 inv ' + S + ' pow ' + N + ' 0.5 </mul>'),
    ('<mul> ' + T + ' ' + TN + ' </mul>', '<mul> -1 pow ' + T + ' 2 </mul>'),
    ('pow ' + TN + ' 2', 'pow ' + T + ' 2'),
    ('<mul> ' + S + ' ' + N + ' </mul>', '<mul> -1 ' + S + ' ' + S + ' </mul>'),
    ('<mul> inv pow ' + S + ' 2 inv pow ' + S + ' 3 inv ' + N + ' </mul>', '<mul> inv pow ' + S + ' 5 inv ' + N + ' </mul>'),
    # a fractional power's integer part is mass, however the powers split it
    ('<mul> pow ' + N + ' 1.5 pow ' + N + ' 3.5 </mul>', '<mul> pow ' + N + ' 0.5 pow ' + N + ' 4.5 </mul>'),
    ('<mul> ' + S + ' pow ' + N + ' 1.5 pow ' + N + ' 0.5 </mul>', '<mul> -1 pow ' + N + ' 2.5 pow ' + N + ' 0.5 </mul>'),
    # an odd power of an odd function of a sum takes a sign as the function does
    ('<mul> -1 pow ' + T + ' 3 </mul>', 'pow ' + TN + ' 3'),
    ('<mul> pow ' + R + ' 8 ' + RN + ' -1 </mul>', 'pow ' + R + ' 9'),
]


@pytest.mark.parametrize('effort', (0, None))
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('a, b', EQUAL)
def test_equal_tables_give_one_answer(eng, mode, effort, a, b):
    x = list(eng.simplify(a.split(), mode=mode, effort=effort))
    y = list(eng.simplify(b.split(), mode=mode, effort=effort))
    assert x == y
    assert list(eng.simplify(x, mode=mode, effort=effort)) == x


# At the zero of a class's sum a negative odd power is an infinity whose sign the orientation decides; a negative
# even power is `+inf` either way. The answer keeps every value, the sign of every infinity included.
POLES = [
    '<mul> inv pow ' + S + ' 3 inv pow ' + N + ' 2 </mul>',
    '<mul> inv ' + S + ' inv pow ' + N + ' 2 pow ' + N + ' 0.5 </mul>',
    '<mul> inv ' + T + ' inv pow ' + TN + ' 2 ' + TN + ' </mul>',
    '<mul> inv ' + S + ' ' + N + ' inv pow ' + N + ' 3 </mul>',
    '<mul> -2 inv pow ' + SIN + ' 3 pow ' + SINN + ' 2 </mul>',
    '<mul> pow ' + N + ' -2.5 ' + S + ' </mul>',
    '<mul> pow ' + N + ' 2.5 inv ' + S + ' inv pow ' + S + ' 2 </mul>',
    # `cbrt(x2 - exp(x1))^-1 * cbrt(x2 - exp(x1))^-2` collects to `1/(x2 - exp(x1))`, outside the class whose table
    # counts those members: it is counted once
    '<mul> inv ' + R + ' inv pow ' + R + ' 2 pow ' + RN + ' 0.5 pow ' + RN + ' 4 </mul>',
    # a divisor that is a product joins the product its quotient builds: each member keeps its own pole
    '/ x2 * - 3 x1 - x1 3',
    '/ x2 * sin - 3 x1 sin - x1 3',
    'inv * - x1 3 - 3 x1',
    '<mul> x2 <div> <mul> ' + S + ' ' + N + ' </mul> </mul>',
]


# A divisor that is a product, an inverted product and the product of the inverses are one product: members over
# one base merge before a negated sum gives its sign away, an infinite member licenses the inverse of each member,
# and the inverted members meet the rest of the product in one decision.
DIVISORS = [
    ('/ x2 * - 3 x1 - x1 3', '* x2 * inv - 3 x1 inv - x1 3'),
    ('inv * sin - x1 3 sin - 3 x1', '* inv sin - x1 3 inv sin - 3 x1'),
    ('<mul> x2 <div> <mul> ' + S + ' ' + N + ' </mul> </mul>', '<mul> x2 <div> ' + S + ' ' + N + ' </mul>'),
    ('pow * - 3 x1 - x1 3 -3', '* pow - 3 x1 -3 pow - x1 3 -3'),
    ('/ 1 * - - 1 x1 x2 pow - - 1 x1 x2 / 1 3', '* inv - - 1 x1 x2 inv pow - - 1 x1 x2 / 1 3'),
    ('/ * pow - 3 x1 2 pow - x1 3 / 1 3 * rootn - + x1 x2 1 3 - x2 float("inf")',
     '* pow - 3 x1 2 * pow - x1 3 / 1 3 * inv rootn - + x1 x2 1 3 inv - x2 float("inf")'),
    ('/ rootn - 3 x1 3 * pow rootn - 3 x1 3 2 rootn - 3 x1 3',
     '* rootn - 3 x1 3 * inv pow rootn - 3 x1 3 2 inv rootn - 3 x1 3'),
    ('/ * pow rootn - - 1 x1 x2 3 3 rootn - * 2 x1 * 3 x2 3 * pow rootn - 3 x1 5 3 pow rootn - x1 3 5 3',
     '* pow rootn - - 1 x1 x2 3 3 * rootn - * 2 x1 * 3 x2 3 * inv pow rootn - 3 x1 5 3 inv pow rootn - x1 3 5 3'),
    ('/ * - * 2 x1 x2 - x1 <constant> * - x2 * 2 x1 x3', '* - * 2 x1 x2 * - x1 <constant> * inv - x2 * 2 x1 inv x3'),
]


@pytest.mark.parametrize('effort', (0, None))
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('a, b', DIVISORS)
def test_a_divisor_product_is_decided_with_its_quotient(eng, mode, effort, a, b):
    x = list(eng.simplify(a.split(), mode=mode, effort=effort))
    y = list(eng.simplify(b.split(), mode=mode, effort=effort))
    assert eng.to_prefix(x) == eng.to_prefix(y)
    assert list(eng.simplify(x, mode=mode, effort=effort)) == x


# A negative power is zero only where its base is infinite, so the inverse of a quotient distributes as the quotient's
# members do: `x14 / (x12 / (x14 sqrt(x12)))` is `x14^2 / sqrt(x12)` in every mode.
INVERTED_QUOTIENTS = [
    ('/ x14 / x12 * x14 rootn x12 2', '/ pow x14 2 rootn x12 2'),
    ('inv * inv x14 pow x12 -0.5', '* x14 rootn x12 2'),
]


@pytest.mark.parametrize('effort', (0, None))
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('a, b', INVERTED_QUOTIENTS)
def test_an_inverted_quotient_distributes(eng, mode, effort, a, b):
    x = list(eng.simplify(a.split(), mode=mode, effort=effort))
    assert x == list(eng.simplify(b.split(), mode=mode, effort=effort))
    assert list(eng.simplify(x, mode=mode, effort=effort)) == x


# A placement can leave two odd roots of one index, and a collection a factor over another base (`cbrt(u)^3` is `u`
# beside `u`): the product is built again, so its answer reads back as itself.
@pytest.mark.parametrize('effort', (0, None))
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('src', [
    '* rootn - x2 x3 3 * inv rootn - * 2 x1 * 3 x2 3 pow rootn - * 3 x2 * 2 x1 3 2',
    '<mul> pow rootn - * 3 x2 * 2 x1 5 3 pow rootn - * 2 x1 * 3 x2 5 -2 rootn - * 3 x2 * 2 x1 5 '
    'rootn - * 2 x1 * 3 x2 5 </mul>',
    '<mul> rootn - + x1 x2 1 5 pow rootn - * 3 x2 * 2 x1 5 -1 pow rootn - * 2 x1 * 3 x2 5 3 '
    'pow rootn - * 2 x1 * 3 x2 5 -1 </mul>',
    '<mul> pow rootn - + x1 x2 1 3 -1 rootn - - 1 x1 x2 3 pow rootn - + x1 x2 1 3 4 pow rootn - + x1 x2 1 3 3 </mul>',
])
def test_a_product_its_decision_leaves_unjoined_is_built_again(eng, mode, effort, src):
    answer = list(eng.simplify(src.split(), mode=mode, effort=effort))
    assert list(eng.simplify(answer, mode=mode, effort=effort)) == answer


# A sum holding a `<constant>` takes a product's sign into its other terms (`-(x1 + c)` is `-x1 + c'`, never
# `x1 + c'`), whatever factor the sign leaves: `(2 x1 - x2)(x1 - c)` is `(x1 + c')(2 x1 - x2)`, and
# `(2 x1 - x2)(x1 - c) / (x2 - 2 x1)` is `c' - x1`, never `x1 + c'`.
@pytest.mark.parametrize('effort', (0, None))
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('src, expected', [
    ('* - x2 x1 - x1 <constant>', '* + x1 <constant> - x2 x1'),
    ('* -1 * - x2 * 2 x1 - x1 <constant>', '* + x1 <constant> - * 2 x1 x2'),
    ('/ * - * 2 x1 x2 - x1 <constant> - x2 * 2 x1', '- <constant> x1'),
])
def test_a_constant_sum_takes_the_sign_into_its_terms(eng, mode, effort, src, expected):
    answer = list(eng.simplify(src.split(), mode=mode, effort=effort))
    assert eng.to_prefix(answer) == eng.to_prefix(expected.split())
    assert list(eng.simplify(answer, mode=mode, effort=effort)) == answer


@pytest.mark.parametrize('mode', ('f64', 'real'))
@pytest.mark.parametrize('src', POLES)
def test_the_answer_keeps_every_value_at_the_zeros(eng, mode, src):
    k = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    xs = [np.concatenate([np.full_like(k, 3.0), k, k + 0.25]),          # 3 - x1 = 0
          np.concatenate([k, k, np.full_like(k, 0.5)]),                # x1 - x2 = 0, x2 - 1/2 = 0
          np.concatenate([k, np.full_like(k, 0.5), k])]
    answer = list(eng.simplify(src.split(), mode=mode))

    def values(tokens):
        with np.errstate(all='ignore'):
            v = eng.as_callable(tokens, ['x1', 'x2', 'x3'])(*xs)
        return np.broadcast_to(np.asarray(v, dtype=float), xs[0].shape)

    before, after = values(src.split()), values(answer)
    finite, infinite = np.isfinite(before), np.isinf(before)
    # where the source is defined the answer agrees; where it is an infinity, the same one (a
    # certified cancellation may fill a NaN the source has at a zero, `S/S` is 1 there)
    assert np.allclose(before[finite], after[finite], rtol=1e-9, atol=1e-12), (before, after, answer)
    assert np.array_equal(before[infinite], after[infinite]), (before, after, answer)
    assert list(eng.simplify(answer, mode=mode)) == answer


# A tagged answer either carries a bag or is spelled exactly as the explicit printer spells the same state: a
# bag-free list reads back as explicit notation, so any other spelling changed on a second call (`neg inv x1` came
# back as `/ -1 x1`, `pow x1 -2` as `inv pow x1 2`).
def test_a_tagged_answer_without_a_bag_reads_back_as_itself(eng):
    rng = random.Random(20261012)
    atoms = ['x1', 'x2', 'sin x1', 'pow x1 2', 'pow x1 3', '<add> x1 <sub> 3 </add>', 'exp x2', 'inv x1']
    numbers = ['-1', '-2', '2', '3']
    seen = 0
    for _ in range(300):
        top = [rng.choice(atoms) for _ in range(rng.randint(0, 2))]
        if rng.random() < 0.6:
            top.append(rng.choice(numbers))
        div = [rng.choice(atoms) for _ in range(rng.randint(0, 2))]
        while len(top) + len(div) < 2:
            top.append('x1')
        src = ('<mul> ' + ' '.join(top) + (' <div> ' + ' '.join(div) if div else '') + ' </mul>').split()
        for mode in MODES:
            for effort in (0, None):
                answer = list(eng.simplify(src, mode=mode, effort=effort))
                if BAG_TOKENS.intersection(answer):
                    continue
                seen += 1
                assert list(eng.simplify(answer, mode=mode, effort=effort)) == answer, (src, answer)
    assert seen > 100


@pytest.mark.parametrize('src, expected', [
    ('<mul> -1 <div> x1 </mul>', '<mul> -1 <div> x1 </mul>'),
    ('<mul> -1 x3 <div> pow x1 2 x3 </mul>', '<mul> -1 <div> pow x1 2 </mul>'),
    ('<mul> x1 <div> pow x1 3 </mul>', 'inv pow x1 2'),
    ('<mul> -1 sin x1 </mul>', 'sin neg x1'),
])
def test_tagged_spellings_that_read_back(eng, src, expected):
    for effort in (0, None):
        answer = list(eng.simplify(src.split(), effort=effort))
        assert ' '.join(answer) == expected
        assert list(eng.simplify(answer, effort=effort)) == answer
