"""simplify(work=...): an opt-in deterministic work budget for the search, no budget by default.

A unit is one step of the AC matcher or one canonical-constructor call, so a budget cuts the
same walk at the same place on every machine. ``work=None`` (the default in every mode) runs the
search until a round finds nothing; a budget ends it early and never changes which algorithm runs.
"""
import pytest

import simplipy
from simplipy import SimpliPyEngine


@pytest.fixture(scope='module')
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load('acj-5-4-llm', install=True)


def ladder(depth: int) -> list[str]:
    """x1*(x1 + (N + 1)/x1), nested ``depth`` deep: simplifies to a short form."""
    s = 'x1'
    for _ in range(depth):
        s = f'* x1 + x1 / + {s} 1 x1'
    return s.split()


def price(engine: SimpliPyEngine, expr: list[str], mode: str) -> int:
    return engine.complexity(list(expr), mode=mode)


def test_the_default_budget_is_exported() -> None:
    assert simplipy.DEFAULT_WORK is None


@pytest.mark.parametrize('work, error', [(True, TypeError), (-1, ValueError), ('all', ValueError), (1.5, TypeError)])
def test_work_is_validated(engine, work, error) -> None:
    with pytest.raises(error):
        engine.simplify(['+', 'x1', 'x1'], work=work)


@pytest.mark.parametrize('mode', ['f64', 'real', 'permissive'])
def test_every_mode_runs_unbounded_by_default(engine, mode) -> None:
    t = ladder(10)
    assert engine.simplify(t, mode=mode) == engine.simplify(t, mode=mode, work=None)


def test_a_budget_binds_when_asked_and_stays_sound(engine) -> None:
    # In f64 the budget is opt-in; 5,000 units cut this search short of the settled form,
    # and the cut answer is still never costlier than the input.
    t = ladder(10)
    settled = engine.simplify(t, mode='f64', work=None)
    cut = engine.simplify(t, mode='f64', work=5_000)
    assert cut != settled
    assert price(engine, settled, 'f64') < price(engine, cut, 'f64') <= price(engine, t, 'f64')


def test_permissive_defaults_to_no_budget(engine) -> None:
    for t in (ladder(6), ladder(10), '+ * 2.5 x1 * 1.5 x1'.split()):
        default = engine.simplify(t, mode='permissive')
        assert default == engine.simplify(t, mode='permissive', work=None)
        assert default == engine.simplify(t, mode='permissive', work='default')
        assert default == engine.simplify(t, mode='permissive')  # deterministic


@pytest.mark.parametrize('work', [1_000, 10_000])
def test_permissive_keeps_its_bounds_under_a_budget(engine, work) -> None:
    # never above the input, never above the search-off answer, never above a smaller effort
    for t in (ladder(8), '/ - 0.5 x1 / exp x2 float("inf")'.split()):
        prices = [price(engine, engine.simplify(t, mode='permissive', effort=k, work=work), 'permissive')
                  for k in (0, 1, 4, None)]
        assert prices == sorted(prices, reverse=True), prices
        assert prices[0] <= price(engine, t, 'permissive')
