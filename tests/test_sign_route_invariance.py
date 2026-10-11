"""A product's sign placement is a function of its value class, not of the route that built it.

`sign_place` (rust/ac/expr.rs) prices every orientation of a product's sign-trade sites and
builds the cheapest. Two holes let the route decide instead, and in both the state and the parse
of its own print were two states of one value (the debug build's serialization-stability
assertion failed, and a second call could return another answer):

* WIDE ORBITS. Up to six sites are enumerated; until `sign_place_wide`, a wider product kept the
  orientation it arrived in. A parse builds a product as a binary chain, so the prefix products of
  up to six sites traded by their own argmin and the rest froze as they came. On srbf's 125,127
  model predictions every answer that changed on a second call in `f64` and `real` (26 with the
  search, 24 without) was such a product.
* OPPOSITE TWINS. Two sums, each the other's negation, stayed apart wherever they met in one bag
  (a flip of one onto the other's base was refused) and collected wherever the parse traded one
  of them alone in a smaller product first.
"""
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


def chain(factors, nest='left'):
    """A binary `*` chain over prefix-token factors, left- or right-nested."""
    if nest == 'right':
        out = []
        for f in factors[:-1]:
            out += ['*'] + f
        return out + factors[-1]
    return ['*'] * (len(factors) - 1) + [t for f in factors for t in f]


def diff(a, b):
    return ['-', a, b]


# (1-x1)(x2-1)(1-x3)(x4-1)(1-x5)(x6-1)(1-x7): seven mixed-sign sums, alternating orientations.
SEVEN = [diff('1', 'x1'), diff('x2', '1'), diff('1', 'x3'), diff('x4', '1'), diff('1', 'x5'),
         diff('x6', '1'), diff('1', 'x7')]
# the same sums the other way round: flipping all seven negates the product, so a leading `neg`
# spells the same value
SEVEN_FLIPPED = [diff('x1', '1'), diff('1', 'x2'), diff('x3', '1'), diff('1', 'x4'), diff('x5', '1'),
                 diff('1', 'x6'), diff('x7', '1')]


@pytest.mark.parametrize('mode', MODES)
def test_seven_site_product_is_idempotent(eng, mode):
    answer = list(eng.simplify(chain(SEVEN), mode=mode))
    assert list(eng.simplify(answer, mode=mode)) == answer
    assert eng.complexity(answer, mode=mode) <= eng.complexity(chain(SEVEN), mode=mode)


@pytest.mark.parametrize('mode', MODES)
def test_seven_site_product_does_not_depend_on_the_route(eng, mode):
    spellings = [
        chain(SEVEN, 'left'),
        chain(SEVEN, 'right'),
        chain(SEVEN[::-1], 'left'),
        ['neg'] + chain(SEVEN_FLIPPED, 'left'),
        ['neg'] + chain(SEVEN_FLIPPED[::-1], 'right'),
    ]
    answers = {' '.join(eng.simplify(s, mode=mode, effort=0)) for s in spellings}
    assert len(answers) == 1, answers


@pytest.mark.parametrize('mode', ('f64', 'real'))
def test_wide_orbit_with_literal_coefficients_is_idempotent(eng, mode):
    # nine sites with integer coefficients and a rational coefficient on the product
    sums = [['-', '*', str(p), f'x{i}', str(q)] for i, (p, q) in
            enumerate([(2, 3), (3, 5), (5, 7), (7, 11), (11, 13), (13, 17), (17, 19), (19, 23), (23, 29)],
                      start=1)]
    for nest in ('left', 'right'):
        src = ['*', '/', '3', '7'] + chain(sums, nest)
        answer = list(eng.simplify(src, mode=mode, effort=0))
        assert list(eng.simplify(answer, mode=mode, effort=0)) == answer
        assert eng.complexity(answer, mode=mode) <= eng.complexity(src, mode=mode)


# (x1 - 3) * (3 - x1) / (x2^2 * x3^2): the first answer kept the pair and re-read cheaper, and a
# second call returned the collected form (the shape of srbf ground truth 4174, in every mode).
TWINS = ['*', '*', '-', 'x1', '3', '-', '3', 'x1', '*', 'inv', 'pow', 'x2', '2', 'inv', 'pow', 'x3', '2']


@pytest.mark.parametrize('mode', MODES)
def test_opposite_sums_collect_and_stay_put(eng, mode):
    answer = list(eng.simplify(TWINS, mode=mode))
    assert list(eng.simplify(answer, mode=mode)) == answer
    assert answer.count('-') == 1, answer  # one sum, squared
    # (the measure reads the input through the same constructors, so the pair prices collected)
    assert eng.complexity(answer, mode=mode) <= eng.complexity(TWINS, mode=mode)


@pytest.mark.parametrize('mode', MODES)
def test_opposite_sums_do_not_depend_on_the_route(eng, mode):
    spellings = [
        ['*', '-', 'x1', '1', '-', '1', 'x1'],
        ['*', '-', '1', 'x1', '-', 'x1', '1'],
        ['neg', 'pow', '-', 'x1', '1', '2'],
        ['neg', 'pow', '-', '1', 'x1', '2'],
    ]
    answers = {' '.join(eng.simplify(s, mode=mode, effort=0)) for s in spellings}
    assert len(answers) == 1, answers


@pytest.mark.parametrize('mode', MODES)
def test_a_sum_over_its_negation_reads_the_same_either_way(eng, mode):
    # the denominator's sum is no carrier (a negative power), so the numerator's moves onto it
    a = list(eng.simplify(['/', '-', 'x1', '3', '-', '3', 'x1'], mode=mode, effort=0))
    b = list(eng.simplify(['/', '-', '3', 'x1', '-', 'x1', '3'], mode=mode, effort=0))
    assert a == b
    assert list(eng.simplify(a, mode=mode, effort=0)) == a


# A UNIT coefficient decides a key's orientation too (srbf prediction 117892, variable renamed).
# A term `k * x1^-1.5 * (-a - b)` keeps the negated sum while |k| != 1 (it ties with `-k` and the
# positive sum, and the tie goes to the positive coefficient). Joined at coefficient 1, the key was
# returned as it was, although with a free sign `-(a + b)` is cheaper: the sum printed
# `-(a + b) / x1^1.5` and re-read with the sign on its coefficient, and a second call changed the
# answer.
PRED_117892 = ('exp * rootn x1 2 * 0.9156774815674904 - x1 * x1 / - / x1 + * 415.87771193151247 x1 + '
               '* -725.5868903908405 x1 - * 1442.938773405699 atan pow x1 3 x1 -33.3888943640316 '
               'pow x1 3').split()


def test_a_unit_coefficient_join_orients_the_key(eng):
    answer = list(eng.simplify(PRED_117892, mode='permissive'))
    assert list(eng.simplify(answer, mode='permissive')) == answer


# OPPOSITE SUMS ARE DECIDED PER CLASS. The first twin pass walked the factors once in arrival order
# and could flip a pair onto each other while a member that cannot move (a negative power) wanted
# the other side, so one flat bag collected differently by the order of its members, and the
# parse's own grouping lost collections 0.14.7 made.
_S = ['<add>', '3', '<sub>', 'x1', '</add>']    # 3 - x1
_N = ['<add>', 'x1', '<sub>', '3', '</add>']    # x1 - 3
BAGS = {
    'S, N, 1/S': [_S, _N, ['inv'] + _S],
    'S, N, S': [_S, _N, _S],
    'N^2, 1/S, S': [['pow'] + _N + ['2'], ['inv'] + _S, _S],
    'S^3, N, 1/S': [['pow'] + _S + ['3'], _N, ['inv'] + _S],
    'N^2, 1/S': [['pow'] + _N + ['2'], ['inv'] + _S],
    'S, N, 1/S, 1/N': [_S, _N, ['inv'] + _S, ['inv'] + _N],
}


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('bag', sorted(BAGS))
def test_one_bag_collects_the_same_in_every_order(eng, mode, bag):
    import itertools
    answers = set()
    for order in itertools.permutations(BAGS[bag]):
        tokens = ['<mul>'] + [t for m in order for t in m] + ['</mul>']
        answers.add(' '.join(eng.simplify(tokens, mode=mode, effort=0)))
    assert len(answers) == 1, answers


# The parse builds `(3 - x1)(x1 - 3)` first; the pair must still collect with the `1/(3 - x1)` it
# meets next (0.14.7: `x1 - 3`, and `(x1 - 3)/x2`).
@pytest.mark.parametrize('effort', (0, None))
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('src, parent', [
    ('* * - 3 x1 - x1 3 inv - 3 x1', '- x1 3'),
    ('/ * - 3 x1 - x1 3 * x2 - 3 x1', '/ - x1 3 x2'),
])
def test_a_collected_pair_still_meets_its_inverse(eng, mode, effort, src, parent):
    answer = list(eng.simplify(src.split(), mode=mode, effort=effort))
    assert eng.complexity(answer, mode=mode) <= eng.complexity(parent.split(), mode=mode), answer
    assert list(eng.simplify(answer, mode=mode, effort=effort)) == answer
