"""The public ``effort=`` API: the search budget on ``simplify()``.

The falsifier row is SOOSE row-156: ``x2*(x2 + (x1+1)/x2)`` is a mu-hill -- the
valley ``1 + x1 + x2**2`` is strictly below it in the serve ordering, but every
step toward it ascends at its node (distribute first, recollect after), so the
strict-descent chain can never reach it. With a budget the engine must land in
the valley; with ``effort=0`` the call must keep the chain's own answer.
"""
import json

import pytest
import yaml

from simplipy import Mode, SimpliPyEngine, SimplifyStatus

# The pow-expand hill: the chain keeps the square, the search expands it and recollects.
HILL = '(x1 + 1)**2 - x1**2'
BUDGET = 64


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    # A bare acj-vocabulary engine with NO rules: the pow-expand valley is reachable
    # through the search's expansion and the constructors' recollection alone, so the
    # falsifier isolates the exploration phase from any mined ruleset.
    from conftest import acj_config_path, require_or_skip
    require_or_skip(acj_config_path(), 'needs the acj operator vocabulary')
    ops = yaml.safe_load(open(acj_config_path()))['operators']
    d = tmp_path_factory.mktemp('effort')
    (d / 'rules.json').write_text(json.dumps([]))
    (d / 'config.yaml').write_text(yaml.safe_dump({'operators': ops, 'rules': 'rules.json'}))
    with pytest.warns(UserWarning):  # the deliberate rules-less warning
        return SimpliPyEngine.from_config(str(d / 'config.yaml'))


class TestTheBudgetCrossesTheHill:
    def test_effort_zero_keeps_the_chain_answer(self, engine) -> None:
        out = engine.simplify(HILL, effort=0)
        assert engine.complexity(out) == engine.complexity(HILL), \
            'effort=0 must not enter the exploration phase'

    @pytest.mark.parametrize('mode', [Mode.f64, Mode.permissive])
    def test_the_valley_is_reached_and_is_idempotent(self, engine, mode) -> None:
        lo = engine.simplify(HILL, mode=mode, effort=0)
        hi = engine.simplify(HILL, mode=mode, effort=BUDGET)
        assert engine.complexity(hi) < engine.complexity(lo)
        assert engine.simplify(hi, mode=mode, effort=BUDGET) == hi

    def test_token_form_takes_the_budget_too(self, engine) -> None:
        tokens = engine.to_prefix(HILL)
        hi = engine.simplify(tokens, effort=BUDGET)
        assert isinstance(hi, list)
        assert engine.complexity(hi) < engine.complexity(tokens)

    def test_the_retired_mode_is_refused_whatever_the_effort(self, engine) -> None:
        with pytest.raises(ValueError, match='the real mode is retired'):
            engine.simplify(HILL, mode='real', effort=BUDGET)


class TestEffortValidation:
    def test_the_default_is_the_module_constant(self, engine) -> None:
        from simplipy.engine import DEFAULT_EFFORT
        assert engine.simplify(HILL) == engine.simplify(HILL, effort=DEFAULT_EFFORT)

    def test_effort_zero_is_byte_identical_to_the_plain_entry(self, engine) -> None:
        # The pre-effort core entry is the reference: the new dispatch at budget 0
        # must reproduce it byte-for-byte, tokens and rendering alike.
        tokens = engine.to_prefix(HILL)
        tagged = [str(t) for t in engine.to_tagged(tokens)]
        via_plain = engine._core.ac_simplify_explore(tagged, 48, "explicit", 0, rule_mode='default')
        assert via_plain == engine._core.ac_simplify_in_mode(tagged, 48, "explicit", 0, rule_mode="default")
        # The FFI's own default is the public one, so a direct `_core` caller that omits the
        # budget gets what `simplify` gives, not the chain alone.
        assert engine._core.ac_simplify_in_mode(tagged, 48, "explicit", rule_mode="default") \
            == engine._core.ac_simplify_in_mode(tagged, 48, "explicit", None, rule_mode="default")
        # The RULED default (once 4): None -- the default
        # call explores until a round finds nothing, and explicitly asking for the
        # chain alone differs on a mu-hill.
        from simplipy import DEFAULT_EFFORT
        assert DEFAULT_EFFORT is None
        assert engine.simplify(HILL) == engine.simplify(HILL, effort=None)
        assert engine.simplify(HILL) != engine.simplify(HILL, effort=0)

    @pytest.mark.parametrize('bad', [-1, -64])
    def test_negative_budgets_raise(self, engine, bad) -> None:
        with pytest.raises(ValueError, match='effort'):
            engine.simplify(HILL, effort=bad)

    @pytest.mark.parametrize('bad', [1.5, '8', True, False])
    def test_non_int_budgets_raise(self, engine, bad) -> None:
        with pytest.raises(TypeError, match='effort'):
            engine.simplify(HILL, effort=bad)

    def test_a_cap_beyond_the_index_range_is_no_cap(self, engine) -> None:
        # pyo3 cannot carry it as a usize; any such cap is the uncapped search.
        assert engine.simplify(HILL, effort=2 ** 70) == engine.simplify(HILL, effort=None)


class TestTheCallSaysWhenACapStoppedIt:
    """A call that a resource cap stops returns the state it reached and says so
    (``return_status=True``): the search's budget with candidates left, or ``max_passes`` before
    a pass confirmed the fixpoint."""

    TWO = 'x0*x1 + x0*x2 + x3*x4 + x3*x5'

    def test_a_call_that_ends_where_its_walk_ends_is_not_interrupted(self, engine) -> None:
        for effort in (0, None):
            answer, status = engine.simplify(HILL, effort=effort, return_status=True)
            assert answer == engine.simplify(HILL, effort=effort)
            assert status == SimplifyStatus(interrupted=False, caps=())

    def test_the_budget_stops_the_search_between_two_improvements(self, engine) -> None:
        answer, status = engine.simplify(self.TWO, effort=1, return_status=True)
        assert status == SimplifyStatus(interrupted=True, caps=('effort',))
        # a second call continues, and the uncapped search ends where nothing is left
        assert engine.simplify(answer, effort=1) != answer
        assert engine.simplify(self.TWO, return_status=True)[1].interrupted is False

    def test_the_status_comes_in_every_dialect(self, engine) -> None:
        tokens = engine.to_prefix(self.TWO)
        answer, status = engine.simplify(tokens, effort=1, return_status=True)
        assert answer == engine.simplify(tokens, effort=1)
        assert status.caps == ('effort',)

    def test_the_passes_stop_the_chain_before_its_fixpoint_is_confirmed(self) -> None:
        # One pass of a rule's step confirms nothing: the second pass would.
        from conftest import acj_config_path
        ops = yaml.safe_load(open(acj_config_path()))['operators']
        rules = [[['+', 'pow', 'sin', '_0', '2', 'pow', 'cos', '_0', '2'], ['1']]]
        e = SimpliPyEngine(operators=ops, rules=rules, rules_permissive=rules)
        expr = '+ x1 + pow sin x0 2 pow cos x0 2'.split()
        answer, status = e.simplify(expr, max_passes=1, effort=0, return_status=True)
        assert answer == ['+', 'x1', '1']
        assert status == SimplifyStatus(interrupted=True, caps=('max_passes',))
        assert e.simplify(expr, effort=0, return_status=True)[1].interrupted is False


# An srbf model prediction (f64): exact folds let the search multiply the 17-digit
# coefficients out, which takes more than 4 candidate descents. Capped at 4, the first call
# stopped between two improvements and a second call continued (473.8 -> 275.8 bits).
PARTIAL = ('* - * 1.8426336222334249e-5 x_0 66.651398870432352 - + + * 0.0017852549531278935 x_0 '
           '* - * -3.7410441585838554e-6 x_0 29.336341980886485 - * 0.003721137246172343 x_0 '
           '1.4144809534136968 pow + * 0.00020869130391839415 x_0 0.46214648267002759 2 '
           '41.682127161183045').split()


def _copies(k):
    # PARTIAL over k distinct variables, summed: every copy needs its own descents, so no
    # fixed cap fits every size.
    out = list(PARTIAL)
    for j in range(1, k):
        out = ['+'] + out + [t.replace('x_0', f'x_{j}') for t in PARTIAL]
    return out


@pytest.fixture(scope='module')
def shipped():
    # The 0.14 hub artifact: mined before 0.15, so read through the legacy opt-in.
    from conftest import legacy_from_config
    from simplipy.asset_manager import get_path
    try:
        return legacy_from_config(get_path('acj-5-4-llm'))
    except Exception:
        import os
        if os.environ.get('SIMPLIPY_TEST_REQUIRE_ASSETS'):
            raise
        pytest.skip('acj-5-4-llm not resolvable here')


class TestTheSearchRunsUntilItSettles:
    def test_the_default_answer_is_its_own_answer(self, shipped) -> None:
        once = shipped.simplify(PARTIAL)
        assert shipped.simplify(once) == once

    def test_a_cap_can_stop_between_two_improvements(self, shipped) -> None:
        # What the old default did: the cap is still available, and still a cap (two descents
        # stop short here; three reach the uncapped answer).
        once = shipped.simplify(PARTIAL, effort=2)
        twice = shipped.simplify(once, effort=2)
        assert twice != once
        assert shipped.complexity(twice) < shipped.complexity(once)
        assert shipped.complexity(shipped.simplify(PARTIAL)) <= shipped.complexity(twice)

    @pytest.mark.parametrize('k', [4, 5])
    def test_larger_expressions_need_more_than_any_small_cap(self, shipped, k) -> None:
        # What a search needs grows with the expression: the first copy takes five descents and
        # each further one three (its numbers group with the first copy's), so 2 copies reach
        # the uncapped answer within 6, 4 need 12 and 5 need 15, and a cap of 8 falls short here.
        t = _copies(k)
        once = shipped.simplify(t)
        assert shipped.simplify(once) == once
        capped = shipped.simplify(t, effort=8)
        assert shipped.complexity(once) < shipped.complexity(capped)
        assert shipped.simplify(capped, effort=8) != capped

    def test_the_cheapest_candidate_is_tried_first(self, shipped) -> None:
        # Eight products whose expansion does not pay, then one that does: the candidates are
        # tried cheapest rebuilt state first, so the first descent is the ninth product's.
        terms = [f'(x{4 * i + 1} + x{4 * i + 2})*(x{4 * i + 3} + x{4 * i + 4})' for i in range(8)]
        t = shipped.infix_to_prefix(' + '.join(terms + ['(y + 1)*(y - 1)']))
        once = shipped.simplify(t)
        assert shipped.simplify(t, effort=0) != once
        assert shipped.simplify(t, effort=1) == once
