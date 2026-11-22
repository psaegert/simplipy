"""Permissive groups a shared number: after the flatten, a sum's terms are grouped by the exact
value of their coefficient up to sign, and each group of two or more is written ``c*(group)``,
the sign in the coefficient. The key is the exact value, so the grouping is deterministic; with
the whole sum as the group it is the content every mode extracts. Strict keeps the terms
distributed: in float64 ``c*a + c*b`` and ``c*(a + b)`` round differently.
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
def test_permissive_groups(engine: SimpliPyEngine, expression: str, answer: str) -> None:
    assert engine.simplify(expression, mode='permissive') == answer
    assert engine.simplify(answer, mode='permissive') == answer


@pytest.mark.parametrize('expression, answer', [
    ('3.71649*(x1 + x2) + x3', '3.71649*x1 + 3.71649*x2 + x3'),
    ('3.71649*(x1 + x2) + 5.11927*(x3 + x4)', '3.71649*x1 + 3.71649*x2 + 5.11927*x3 + 5.11927*x4'),
    ('2*x1 + 2*x2 + x3', '2*x1 + 2*x2 + x3'),
    ('3.71649*x1 + 3.71649*x2', '3.71649*(x1 + x2)'),
])
def test_strict_keeps_the_terms_distributed(engine: SimpliPyEngine, expression: str, answer: str) -> None:
    assert engine.simplify(expression, mode='strict') == answer


@pytest.mark.parametrize('spelling', [
    '3.71649*x1 + 3.71649*x2 + x3',
    '3.71649*x1 + (3.71649*x2 + x3)',
    '(3.71649*x1 + x3) + 3.71649*x2',
    'x3 + 3.71649*(x2 + x1)',
    '3.71649*x2 - (-3.71649*x1 - x3)',
])
def test_the_grouping_is_decided_on_the_whole_sum(engine: SimpliPyEngine, spelling: str) -> None:
    assert engine.simplify(spelling, mode='permissive') == 'x3 + 3.71649*(x1 + x2)'


def test_one_constant_per_shared_number(engine: SimpliPyEngine) -> None:
    tokens = engine.simplify(['+', '+', '*', '3.71649', 'x1', '*', '3.71649', 'x2', 'x3'], mode='permissive')
    assert tokens.count('3.71649') == 1
    masked = engine.mask(tokens, 'fittable')
    assert masked.count('<constant>') == 1


@pytest.mark.parametrize('expression', [
    # the quarter in x1/(4*x13) and 1/4 costs more grouped: it stays distributed
    'x13/2 + x1/(4*x13) + 1/4',
    '1/(x1 + x13/2 + x1/(4*x13) + 1/4)',
])
def test_a_group_is_written_only_where_it_costs_no_more(engine: SimpliPyEngine, expression: str) -> None:
    answer = engine.simplify(expression, mode='permissive')
    assert '(x1/x13 + 1)/4' not in answer
    assert engine.complexity(answer, mode='permissive') <= engine.complexity(expression, mode='permissive')


def test_a_tie_takes_the_spelling_strict_prices_lower(engine: SimpliPyEngine) -> None:
    # beside a group the sum's two orientations cost the same in permissive; strict writes the
    # pole's sign outside, a sign cheaper
    tokens = '+ - - * x16 x14 x16 x6 - neg x6 + acosh x9 - x16 inv - cosh x9 / x15 x14'.split()
    answer = engine.simplify(tokens, mode='permissive', effort=0)
    assert engine.to_infix(answer) == 'x14 * x16 - acosh(x9) - 1/(x15 / x14 - cosh(x9)) - 2 * (x16 + x6)'
    strict = engine.simplify(tokens, mode='strict', effort=0)
    price = engine._core.ac_complexity
    assert price(list(answer), 'default', rule_mode='strict') <= price(list(strict), 'default', rule_mode='strict')


def test_prettify_reads_in_the_answers_mode(engine: SimpliPyEngine) -> None:
    answer = engine.simplify('3.71649*x1 + 3.71649*x2 + x3', mode='permissive')
    assert engine.prettify(answer, mode='permissive') == 'x3 + 3.71649*(x1 + x2)'
    assert engine.prettify(answer, mode='permissive', keep_grouping=True) == 'x3 + 3.71649*(x1 + x2)'
    # strict, the default, reads the group distributed
    assert engine.prettify(answer) == '3.71649*x1 + 3.71649*x2 + x3'
