"""Division by zero is undefined however it is written, and powers and reciprocals agree.

The table's edge cells, one entry per operator: `1/0`, `inv(0)`, a negative power of zero, a
remainder by zero and a root whose index is no finite integer have no value, even where a limit
exists (`pow(0, -2)`: 1/x^2 tends to +inf at 0), so `x^-2` and `1/x^2` agree everywhere.
`pow(-inf, 0)` is 1 (a negative base admits only integer exponents, and near 0 the only one is 0),
`pow(+inf, 0)` has no value, and `mod(0, +-inf)` has none (the remainder tends to 0 along
(1/n, n) and to +inf along (-1/n, n)). The judge reads the table; permissive takes its value for
a fully numeric expression; f64 follows the deployed evaluator and keeps its values.
"""
import os

import pytest
import yaml
from mpmath import mp

from simplipy import SimpliPyEngine
from simplipy.verify import _contract
from simplipy.verify._contract import judge_rule

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPERATORS = yaml.safe_load(open(os.path.join(REPO, 'assets', 'engines', 'acj2-4', 'config.yaml')))[
    'operators']

#: Fully numeric expressions without a value.
UNDEFINED = [
    '/ 1 0', 'inv 0', 'pow 0 (-1)', 'pow 0 (-2)', 'pow 0 -0.5', 'pow 0 float("-inf")',
    'inv pow 0 2', '/ 1 pow 0 2', 'rootn 0 (-2)', 'rootn 0 (-3)', 'mod 1 0', 'mod 0 0',
    'rootn 2 float("inf")', 'rootn (-2) float("inf")', 'rootn 1 atanh (-1)', 'rootn 2 0.5',
    'mod 0 float("inf")', 'mod 0 float("-inf")', 'mod 0 atanh 1', 'pow float("inf") 0',
]
#: Fully numeric expressions with a value, as the judge spells it.
DEFINED = [
    ('pow float("-inf") 0', '1.0'), ('pow 0 0', '1.0'), ('mod 1 float("inf")', '1.0'),
    ('mod (-1) float("inf")', '+inf'), ('pow float("-inf") (-2)', '0.0'),
    ('rootn float("inf") (-2)', '0.0'),
]


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine(operators=OPERATORS, rules=[])


def table_value(expr: str) -> str:
    """The judge's value of a ground expression: 'nan', '+inf', '-inf' or a float's repr."""
    old = mp.dps
    try:
        mp.dps = 50
        v = _contract.c_eval(_contract.parse(expr.split()), {})
    finally:
        mp.dps = old
    if v != v:
        return 'nan'
    if v in (mp.inf, -mp.inf):
        return '+inf' if v > 0 else '-inf'
    return repr(float(v))


@pytest.mark.parametrize('expr', UNDEFINED)
def test_the_table_reads_no_value(expr) -> None:
    assert table_value(expr) == 'nan'


@pytest.mark.parametrize('expr, value', DEFINED)
def test_the_table_reads_the_limit(expr, value) -> None:
    assert table_value(expr) == value


@pytest.mark.parametrize('expr', UNDEFINED)
def test_permissive_takes_the_table_value(engine, expr) -> None:
    """A fully numeric expression takes its table value in permissive: no value stays none."""
    assert engine.simplify(expr.split(), mode='permissive') == ['float("nan")']


@pytest.mark.parametrize('expr, answer', [('pow float("-inf") 0', ['1']),
                                          ('mod 1 float("inf")', ['1']),
                                          ('mod (-1) float("inf")', ['float("inf")'])])
def test_permissive_takes_the_tables_limits(engine, expr, answer) -> None:
    assert engine.simplify(expr.split(), mode='permissive') == answer


@pytest.mark.parametrize('expr, answer', [
    ('/ 1 0', 'float("inf")'), ('pow 0 (-2)', 'float("inf")'), ('rootn 0 (-2)', 'float("inf")'),
    ('pow float("-inf") 0', '1'), ('pow float("inf") 0', '1'), ('mod 0 float("inf")', '0'),
    ('mod 0 float("-inf")', '-0.0'), ('mod 1 0', 'float("nan")')])
def test_f64_follows_the_evaluator(engine, expr, answer) -> None:
    """f64 is unchanged: it follows the realizations (numpy's 1/0.0 is inf, 0.0**-2 inf)."""
    assert ' '.join(engine.simplify(expr.split(), mode='f64')) == answer


@pytest.mark.parametrize('lhs, rhs', [
    ('pow ?0 (-2)', 'inv pow ?0 2'), ('pow _0 (-2)', 'inv pow _0 2'),
    ('pow ?0 (-1)', 'inv ?0'), ('rootn ?0 (-2)', 'inv rootn ?0 2'),
    ('rootn _0 (-3)', 'inv rootn _0 3')])
def test_powers_and_reciprocals_agree(lhs, rhs) -> None:
    """`x^-2` and `1/x^2` agree everywhere, at 0 too: neither has a value there."""
    got = judge_rule(lhs.split(), rhs.split())
    assert got['verdict'] == 'CERTIFIED', got
