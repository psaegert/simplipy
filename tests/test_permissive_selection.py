"""Permissive returns the cheapest of its candidates, each priced as it is returned.

Permissive runs three arms (its two fold disciplines and the f64 chain), and its literal fold moves
long exact literals to their floats and runs them again. Its answer is the cheapest of every
candidate those runs produce -- every state a search passed through and the input as read
included -- priced by re-reading what it prints in permissive's own measure. So, by construction,
the answer never prices above the input, never above the answer with the search off, and never
above the answer under a smaller effort. The inputs below broke these before.
"""
import pytest

from simplipy import SimpliPyEngine

# Training draws of flash-ansr's T8.1 catalog that permissive returned costlier than they came in.
PRICIER_THAN_INPUT = [
    '- * + * + x16 * pow 0.36421583172854655 1.5 - x16 x9 x8 log x2 * x17 x7 x6',
    '/ neg / / x14 - + x17 x17 x5 x5 + * x9 + - x9 + - rootn x15 -2 - atanh x13 acos x7 x4 x15 x1',
    '/ -0.3573144872201067 * acos * rootn 6.6344325385298895 5 * * pow x7 -3 / 10.387354569120916 - '
    'rootn x7 4 x7 * -2.3570761773421216 -15.27353204443006 x10',
]

# An srbf model prediction whose uncapped permissive answer was costlier than its answer at effort 4.
CAPPED_WAS_CHEAPER = (
    '+ + + * -0.61248131682658702 x_1 * * - 0.25748982793456728 * 1.386026691137763 x_1 - * '
    '-8.466887046162156e-11 x_0 0.6230267049169305 - * 0.5790175810396894 x_1 0.3411929660226874 * '
    '+ * 0.04684186038323518 x_0 0.02342092971136106 - * 0.008818190503282922 pow - * 13.42750931810752 '
    'x_0 6.713754626181707 3 5.337106214048569 0.13276475085596478')


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load('acj-5-4-llm', install=True)


def price(engine: SimpliPyEngine, expr: list[str]) -> int:
    return engine.complexity(list(expr), mode='permissive')


@pytest.mark.parametrize('expr', PRICIER_THAN_INPUT)
def test_the_answer_never_prices_above_the_input(engine, expr) -> None:
    t = expr.split()
    for effort in (0, None):
        assert price(engine, engine.simplify(t, mode='permissive', effort=effort)) <= price(engine, t)


def test_more_budget_never_ends_costlier(engine) -> None:
    t = CAPPED_WAS_CHEAPER.split()
    prices = [price(engine, engine.simplify(t, mode='permissive', effort=k)) for k in (0, 1, 2, 4, 8, 16, None)]
    assert prices == sorted(prices, reverse=True), prices
    # the uncapped answer is as cheap as the cheapest capped one, and cheaper than with the search off
    assert prices[-1] == min(prices) < prices[0]


@pytest.mark.parametrize('expr', PRICIER_THAN_INPUT + [CAPPED_WAS_CHEAPER])
def test_the_answer_is_a_fixpoint(engine, expr) -> None:
    once = engine.simplify(expr.split(), mode='permissive')
    assert engine.simplify(once, mode='permissive') == once
