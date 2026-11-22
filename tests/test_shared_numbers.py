"""A sum groups a shared number, in every mode: after the flatten, a sum's terms are grouped by
the exact value of their coefficient up to sign, and each group of two or more is written
``c*(group)``, the sign in the coefficient, where that costs no more than its terms distributed.
The key is the exact value, so the grouping is deterministic; with the whole sum as the group it
is the content every mode extracts. Each group is the whole-sum factoring of its terms, which
float64 only rounds otherwise; strict takes a sign out of a group only where the group's sum holds
none (``tests/test_laws.py``'s sign rows).
"""
import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path, legacy_from_config


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return legacy_from_config(acj_config_path())


@pytest.mark.parametrize('expression, answer', [
    ('3.71649*(x1 + x2) + x3', 'x3 + 3.71649*(x1 + x2)'),
    ('3.71649*x1 + 3.71649*x2 + x3', 'x3 + 3.71649*(x1 + x2)'),
    ('3.71649*(x1 - x2) + x3', 'x3 + 3.71649*(x1 - x2)'),
    ('3.71649*(x1 + x2) + 5.11927*(x3 + x4)', '3.71649*(x1 + x2) + 5.11927*(x3 + x4)'),
    ('sin(3.71649*(x1 + x2) + 1.23457)', 'sin(3.71649*(x1 + x2) + 1.23457)'),
    # exact numbers too, one rule without a special case
    ('2*x1 + 2*x2 + x3', 'x3 + 2*(x1 + x2)'),
    ('x1/3 + x2/3 + x3', 'x3 + (x1 + x2)/3'),
    ('1.5*x1 + 1.5*x2 + 2.5*x3 + 2.5*x4 + 2.5', '3*(x1 + x2)/2 + 5*(x3 + x4 + 1)/2'),
    # one orientation per class, the sign in the coefficient
    ('-3.71649*x1 - 3.71649*x2 + x3', 'x3 - 3.71649*(x1 + x2)'),
    ('-3.71649*x1 + 3.71649*x2 + x3', 'x3 + 3.71649*(x2 - x1)'),
    # the key is the exact value: 4 is not 2, and a magnitude of one is no number to share
    ('2*x1 + 2*x2 + 4*x3', '4*x3 + 2*(x1 + x2)'),
    ('-x1 - x2 + 2*x3 + 2*x4', '2*(x3 + x4) - x1 - x2'),
    # a term holding a fitted constant is not grouped
    ('3.71649*x1 + 3.71649*x2 + <constant>', '3.71649*(x1 + x2) + <constant>'),
    # the whole sum as the group: the content every mode extracts
    ('3.71649*x1 + 3.71649*x2', '3.71649*(x1 + x2)'),
    ('2*x1 + 2*x2 + 2', '2*(x1 + x2 + 1)'),
])
@pytest.mark.parametrize('mode', ['strict', 'permissive'])
def test_each_mode_groups(engine: SimpliPyEngine, expression: str, answer: str, mode: str) -> None:
    assert engine.simplify(expression, mode=mode) == answer
    assert engine.simplify(answer, mode=mode) == answer


@pytest.mark.parametrize('spelling', [
    '3.71649*x1 + 3.71649*x2 + x3',
    '3.71649*x1 + (3.71649*x2 + x3)',
    '(3.71649*x1 + x3) + 3.71649*x2',
    'x3 + 3.71649*(x2 + x1)',
    '3.71649*x2 - (-3.71649*x1 - x3)',
])
@pytest.mark.parametrize('mode', ['strict', 'permissive'])
def test_the_grouping_is_decided_on_the_whole_sum(engine: SimpliPyEngine, spelling: str, mode: str) -> None:
    assert engine.simplify(spelling, mode=mode) == 'x3 + 3.71649*(x1 + x2)'


@pytest.mark.parametrize('mode', ['strict', 'permissive'])
def test_one_constant_per_shared_number(engine: SimpliPyEngine, mode: str) -> None:
    tokens = engine.simplify(['+', '+', '*', '3.71649', 'x1', '*', '3.71649', 'x2', 'x3'], mode=mode)
    assert tokens.count('3.71649') == 1
    masked = engine.mask(tokens, 'fittable')
    assert masked.count('<constant>') == 1


@pytest.mark.parametrize('expression', [
    # the quarter in x1/(4*x13) and 1/4 costs more grouped: it stays distributed
    'x13/2 + x1/(4*x13) + 1/4',
    '1/(x1 + x13/2 + x1/(4*x13) + 1/4)',
])
@pytest.mark.parametrize('mode', ['strict', 'permissive'])
def test_a_group_is_written_only_where_it_costs_no_more(engine: SimpliPyEngine, expression: str, mode: str) -> None:
    answer = engine.simplify(expression, mode=mode)
    assert '(x1/x13 + 1)/4' not in answer
    assert engine.complexity(answer, mode=mode) <= engine.complexity(expression, mode=mode)


def test_a_permissive_answer_is_no_dearer_on_stricts_yardstick(engine: SimpliPyEngine) -> None:
    # beside a group the sum's two orientations cost the same in permissive; where they tie, the
    # spelling strict prices lower is taken, and both modes read the group alike
    tokens = '+ - - * x16 x14 x16 x6 - neg x6 + acosh x9 - x16 inv - cosh x9 / x15 x14'.split()
    answer = engine.simplify(tokens, mode='permissive', effort=0)
    strict = engine.simplify(tokens, mode='strict', effort=0)
    assert engine.to_infix(answer) == engine.to_infix(strict) == \
        '1/(cosh(x9) - x15 / x14) + x14 * x16 - acosh(x9) - 2 * (x16 + x6)'
    price = engine._core.ac_complexity
    assert price(list(answer), 'default', rule_mode='strict') <= price(list(strict), 'default', rule_mode='strict')


@pytest.mark.parametrize('mode', ['strict', 'permissive'])
def test_prettify_shows_the_groups(engine: SimpliPyEngine, mode: str) -> None:
    answer = engine.simplify('3.71649*x1 + 3.71649*x2 + x3', mode=mode)
    assert engine.prettify(answer, mode=mode) == 'x3 + 3.71649*(x1 + x2)'
    assert engine.prettify(answer, mode=mode, keep_grouping=True) == 'x3 + 3.71649*(x1 + x2)'
    assert engine.prettify(answer) == 'x3 + 3.71649*(x1 + x2)'
