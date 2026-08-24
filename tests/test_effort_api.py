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

    @pytest.mark.parametrize('mode', [Mode.f64, Mode.corpus])
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
        via_effort = engine.simplify(tokens, effort=0)
        via_plain = engine._core.ac_simplify(
            [str(t) for t in engine.to_tagged(tokens)], 48, False, "explicit")
        assert via_effort == engine.simplify(tokens)  # default is 0 today
        assert engine.simplify(HILL, effort=0) == engine.simplify(HILL)
        assert via_plain == engine._core.ac_simplify_in_mode(
            [str(t) for t in engine.to_tagged(tokens)], 48, "default", "explicit", 0)

    @pytest.mark.parametrize('bad', [-1, -64])
    def test_negative_budgets_raise(self, engine, bad) -> None:
        with pytest.raises(ValueError, match='effort'):
            engine.simplify(HILL, effort=bad)

    @pytest.mark.parametrize('bad', [1.5, '8', True, False])
    def test_non_int_budgets_raise(self, engine, bad) -> None:
        with pytest.raises(TypeError, match='effort'):
            engine.simplify(HILL, effort=bad)
