"""D39 B1 -- the row-156 falsifier and the post-fixpoint exploration scaffolding.

Ledger D39: the deterministic chain runs
UNCHANGED to its fixpoint (termination theorem intact), then an explicit,
budgeted exploration phase tries expansion moves through the same certified
machinery and accepts only endpoints STRICTLY below in the engine's own
reduction ordering. This suite is the B1 acceptance: it is measure-agnostic --
no measure number appears anywhere below; every comparison goes through
``ac_ordered_below``, the one serve-time ordering (rust/ac/rules.rs).

Row-156 (SOOSE, `the research harness`; the ledger's
named falsifier row): ``x2*(x2 + (x1+1)/x2)`` is a mu-hill. The valley
``1 + x1 + x2**2`` is strictly ordered below it, but every step toward it
ASCENDS at its node (distribute first, recollect after), so greedy strict
descent can never reach it -- the exact class the D39 exploration phase
exists for. The falsifier: with an exploration budget the engine must land
in the valley; with budget 0 the new entry point must be byte-identical to
today's chain (the effort=0 semantics the ledger fixes).

Scaffolding surface pinned here (B1; the public ``effort=`` API is B7):
NOTE: this suite drives the core through the retired two-valued ``wildcard_all``
bool, so it covers the default and corpus semantics and NOT ``real``. That is the
D39 scaffolding's own surface (ledger D39 B7: the public ``effort=`` API is
deliberately not wired to it yet), so the gap is scoped rather than accidental --
but a ``real``-mode explore regression would not be caught here.

``Engine._core.ac_simplify_explore(tokens, max_passes, wildcard_all, form,
explore_budget)`` -- ``ac_simplify``'s signature plus the trailing budget,
budget 0 the default. Tests observe from OUTSIDE through that entry point;
no switch exists inside the production chain.

Red-first (working rule 1): every test in the falsifier classes below is RED
against HEAD (the entry point does not exist); the red run is recorded in the
landing commit message. The premise pins in ``TestRow156Hill`` are green at
HEAD and must STAY green after landing -- they assert the default chain does
not move.
"""

import pytest

from simplipy import SimpliPyEngine
from conftest import acj_config_path

# The ledger's named falsifier row and its SOOSE source spelling -- one hill.
ROW156 = "x2*(x2+(x1+1)/x2)"
ROW156_SOOSE = "x2*(x2-(-x1-1)/x2)"
ROW156_VALLEY = "1+x1+x2**2"

# The pow-expand sibling of the same mu-hill class (expand the integer power
# of a sum, recollect across the subtraction).
POWSPLIT = "(x1+1)**2 - x1**2"
POWSPLIT_VALLEY = "2*x1+1"

# Row-156 with the variable replaced by a FAT-domain subterm: log(x1) is not
# finite a.e. (NaN on the whole negative half-line), so in SOUND mode both the
# distribute step and the exponent-cancelling recollection lose their
# licences. The same certified machinery that gates the chain must gate the
# exploration moves: sound stays on the hill, lossy (blanket licences, the
# shipped wildcard_all semantics) descends to the valley.
GATED = "log(x1)*(log(x1)+(x2+1)/log(x1))"
GATED_VALLEY = "log(x1)**2+x2+1"

# Nothing-to-find battery for never-worse and byte-stability: fixpoints the
# exploration must return unchanged, plus the hills themselves.
BATTERY = [
    ROW156,
    ROW156_SOOSE,
    POWSPLIT,
    GATED,
    "x1+x2",
    "sin(x0)**2+cos(x0)**2",
    "2.5*x3+x3",
    "x0/x0",
    "1/0",
]

BUDGET = 32


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.from_config(acj_config_path())


def tok(engine: SimpliPyEngine, s: str) -> list[str]:
    return engine._core.parse(s, True, False)


def fix(engine: SimpliPyEngine, s: str, lossy: bool = False) -> list[str]:
    """Today's greedy fixpoint, tagged form -- the chain the theorems cover."""
    return engine._core.ac_simplify(tok(engine, s), 48, lossy, 'tagged')


def explore(engine: SimpliPyEngine, tokens: list[str], budget: int,
            lossy: bool = False, form: str = 'tagged') -> list[str]:
    """The B1 entry point under test (does not exist at HEAD -- red)."""
    return engine._core.ac_simplify_explore(tokens, 48, lossy, form, budget)


class TestRow156Hill:
    """Premise pins -- green at HEAD, and they must stay green: the DEFAULT
    chain does not move under D39 (opt-in exploration, effort=0 default)."""

    def test_greedy_chain_stops_at_the_hill(self, engine: SimpliPyEngine) -> None:
        hill = fix(engine, ROW156)
        assert fix(engine, ROW156_SOOSE) == hill, \
            "both spellings of row-156 must reach one canonical hill"
        valley = fix(engine, ROW156_VALLEY)
        assert hill != valley, "row-156 stopped being a hill: D39's premise moved"
        # The falsifiable claim, in the engine's own ordering: a strictly
        # lower endpoint EXISTS and the greedy chain did not reach it.
        assert engine._core.ac_ordered_below(valley, hill) is True

    def test_powsplit_hill(self, engine: SimpliPyEngine) -> None:
        hill = fix(engine, POWSPLIT)
        valley = fix(engine, POWSPLIT_VALLEY)
        assert hill != valley
        assert engine._core.ac_ordered_below(valley, hill) is True

    def test_gated_hill_in_both_modes(self, engine: SimpliPyEngine) -> None:
        for lossy in (False, True):
            hill = fix(engine, GATED, lossy)
            valley = fix(engine, GATED_VALLEY, lossy)
            assert hill != valley
            assert engine._core.ac_ordered_below(valley, hill) is True


class TestRow156Falsifier:
    """RED at HEAD: the exploration entry point does not exist yet."""

    def test_exploration_reaches_the_valley(self, engine: SimpliPyEngine) -> None:
        hill = fix(engine, ROW156)
        valley = fix(engine, ROW156_VALLEY)
        for spelling in (ROW156, ROW156_SOOSE):
            out = explore(engine, tok(engine, spelling), BUDGET)
            assert out == valley
            assert engine._core.ac_ordered_below(out, hill) is True

    def test_powsplit_reaches_the_valley(self, engine: SimpliPyEngine) -> None:
        out = explore(engine, tok(engine, POWSPLIT), BUDGET)
        assert out == fix(engine, POWSPLIT_VALLEY)
        assert engine._core.ac_ordered_below(out, fix(engine, POWSPLIT)) is True

    def test_certificates_gate_the_moves(self, engine: SimpliPyEngine) -> None:
        # SOUND: log(x1) carries no finite-a.e. licence -- the moves are
        # refused by the same certificates that gate the chain; the endpoint
        # is the untouched sound fixpoint.
        assert explore(engine, tok(engine, GATED), BUDGET) == fix(engine, GATED)
        # LOSSY: blanket licences (the shipped wildcard_all semantics) -- the
        # same move now descends to the valley.
        assert explore(engine, tok(engine, GATED), BUDGET, lossy=True) \
            == fix(engine, GATED_VALLEY, True)

    def test_endpoint_is_idempotent(self, engine: SimpliPyEngine) -> None:
        # A reached valley re-explores to nothing (ledger D39, idempotence).
        valley = explore(engine, tok(engine, ROW156), BUDGET)
        assert explore(engine, valley, BUDGET) == valley
        # And the valley is a fixpoint of the plain chain (the endpoint is a
        # chain state, not a new kind of state).
        assert engine._core.ac_simplify(valley, 48, False, 'tagged') == valley

    def test_deterministic(self, engine: SimpliPyEngine) -> None:
        for s in BATTERY:
            t = tok(engine, s)
            assert explore(engine, t, BUDGET) == explore(engine, t, BUDGET)

    def test_never_worse(self, engine: SimpliPyEngine) -> None:
        # Fall back to the fixpoint (ledger D39): whatever the budget, the
        # endpoint is the greedy fixpoint or strictly below it -- asserted in
        # the serve ordering, never with measure numbers.
        for s in BATTERY:
            for lossy in (False, True):
                f = fix(engine, s, lossy)
                for budget in (1, BUDGET):
                    out = explore(engine, tok(engine, s), budget, lossy)
                    assert out == f or \
                        engine._core.ac_ordered_below(out, f) is True

    def test_budget_zero_is_byte_identical(self, engine: SimpliPyEngine) -> None:
        # The ledger's effort=0 semantics: unused exploration leaves every
        # behavior byte-identical to today's chain, mode by mode, form by form.
        for s in BATTERY:
            t = tok(engine, s)
            for lossy in (False, True):
                for form in ('tagged', 'explicit'):
                    assert explore(engine, t, 0, lossy, form) \
                        == engine._core.ac_simplify(t, 48, lossy, form)

    def test_budget_zero_is_the_default(self, engine: SimpliPyEngine) -> None:
        t = tok(engine, ROW156)
        assert engine._core.ac_simplify_explore(t) \
            == engine._core.ac_simplify(t, 48, False, 'tagged')

    def test_boundary_contracts_match_ac_simplify(self, engine: SimpliPyEngine) -> None:
        # Empty input: the documented `simplify([]) == []` contract (H-003).
        assert engine._core.ac_simplify_explore([], 48, False, 'tagged', BUDGET) == []
        # Malformed input: ValueError, exactly as ac_simplify (fail loudly,
        # never pass garbage through).
        with pytest.raises(ValueError):
            engine._core.ac_simplify_explore(['+', 'x0'], 48, False, 'tagged', BUDGET)
        with pytest.raises(ValueError):
            # unclosed tagged bag: the AC parser is the arbiter (H-003 sibling)
            engine._core.ac_simplify_explore(['<add>', 'x0'], 48, False,
                                             'tagged', BUDGET)
