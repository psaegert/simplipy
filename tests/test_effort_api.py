"""The public ``effort=`` API (ledger D39 B7): the search budget on ``simplify()``.

The falsifier row is SOOSE row-156: ``x2*(x2 + (x1+1)/x2)`` is a mu-hill -- the
valley ``1 + x1 + x2**2`` is strictly below it in the serve ordering, but every
step toward it ascends at its node (distribute first, recollect after), so the
strict-descent chain can never reach it. With a budget the engine must land in
the valley; with ``effort=0`` the call must keep the chain's own answer.
"""
import json

import pytest
import yaml

from simplipy import Mode, SimpliPyEngine

HILL = 'x2*(x2 + (x1+1)/x2)'
BUDGET = 64


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    # A bare acj-vocabulary engine with NO rules: the row-156 valley is reachable
    # through constructor-level cancellation alone, so the falsifier isolates the
    # exploration phase from any mined ruleset.
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

    def test_real_mode_fails_closed_regardless_of_effort(self, engine) -> None:
        # The B1 scaffolding never covered `real`; the public wire must not
        # weaken the fail-closed guard on an artifact without a real set.
        with pytest.raises(ValueError, match='real'):
            engine.simplify(HILL, mode=Mode.real, effort=BUDGET)


class TestRealModeExplores:
    def test_the_shipped_artifact_crosses_the_hill_in_real_mode(self) -> None:
        # Closes the D39 B1 coverage gap: exploration under `real`, on an
        # artifact that serves a real set.
        try:
            engine = SimpliPyEngine.load('acj-4')
        except Exception:
            import os
            if os.environ.get('SIMPLIPY_TEST_REQUIRE_ASSETS'):
                raise
            pytest.skip('acj-4 not resolvable here')
        hi = engine.simplify(HILL, mode=Mode.real, effort=BUDGET)
        assert engine.complexity(hi) < engine.complexity(engine.simplify(HILL, mode=Mode.real, effort=0))


class TestEffortValidation:
    def test_the_default_is_the_module_constant(self, engine) -> None:
        from simplipy.engine import DEFAULT_EFFORT
        assert engine.simplify(HILL) == engine.simplify(HILL, effort=DEFAULT_EFFORT)

    def test_effort_zero_is_byte_identical_to_the_plain_entry(self, engine) -> None:
        # The pre-effort core entry is the reference: the new dispatch at budget 0
        # must reproduce it byte-for-byte, tokens and rendering alike.
        tokens = engine.to_prefix(HILL)
        via_plain = engine._core.ac_simplify(
            [str(t) for t in engine.to_tagged(tokens)], 48, False, "explicit")
        assert via_plain == engine._core.ac_simplify_in_mode(
            [str(t) for t in engine.to_tagged(tokens)], 48, "default", "explicit", 0)
        # The RULED default (owner 2026-10-06; 4 from 2026-08-24): None -- the default
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


# An srbf model prediction (f64): exact folds let the search multiply the 17-digit
# coefficients out, which takes more than 4 candidate descents. Capped at 4, the first call
# stopped between two improvements and a second call continued (649.3 -> 521.4 bits).
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
    try:
        return SimpliPyEngine.load('acj-5-4-llm')
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
        # What the old default did: the cap is still available, and still a cap.
        once = shipped.simplify(PARTIAL, effort=4)
        twice = shipped.simplify(once, effort=4)
        assert twice != once
        assert shipped.complexity(twice) < shipped.complexity(once)
        assert shipped.complexity(shipped.simplify(PARTIAL)) <= shipped.complexity(twice)

    @pytest.mark.parametrize('k', [3, 4])
    def test_larger_expressions_need_more_than_any_small_cap(self, shipped, k) -> None:
        # What a search needs grows with the expression: 2 copies reach the uncapped answer
        # within 14 candidates, 3 need 19 and 4 need 23, so a cap of 16 falls short here.
        t = _copies(k)
        once = shipped.simplify(t)
        assert shipped.simplify(once) == once
        capped = shipped.simplify(t, effort=16)
        assert shipped.complexity(once) < shipped.complexity(capped)
        assert shipped.simplify(capped, effort=16) != capped

    def test_the_finish_does_not_retry_what_the_prefix_refused(self, shipped) -> None:
        # Eight products whose expansion does not pay, then one that does: the breadth-first
        # prefix refuses the first eight, and the finish's first round starts at the ninth.
        terms = [f'(x{4 * i + 1} + x{4 * i + 2})*(x{4 * i + 3} + x{4 * i + 4})' for i in range(8)]
        t = shipped.infix_to_prefix(' + '.join(terms + ['(y + 1)*(y - 1)']))
        once = shipped.simplify(t)
        assert shipped.simplify(t, effort=8) != once
        assert shipped.simplify(t, effort=9) == once
