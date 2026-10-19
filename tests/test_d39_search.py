"""D39 B1 -- the row-156 falsifier and the post-fixpoint exploration scaffolding.

Ledger D39: the deterministic chain runs
UNCHANGED to its fixpoint (termination theorem intact), then an explicit,
budgeted exploration phase tries expansion moves through the same certified
machinery and accepts only endpoints STRICTLY below in the engine's own
reduction ordering. This suite is the B1 acceptance: it is measure-agnostic --
no measure number appears anywhere below; every comparison goes through
``ac_ordered_below``, the one serve-time ordering (rust/ac/rules.rs).

Row-156 (SOOSE, recorded in the research ledger; the ledger's
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

``Engine._core.ac_simplify_explore(tokens, max_passes, form, explore_budget, rule_mode=('permissive' if wildcard_all else 'default'))`` -- ``ac_simplify``'s signature plus the trailing budget,
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
    return engine._core.ac_simplify_in_mode(tok(engine, s), 48, 'tagged', 0, rule_mode=('permissive' if lossy else 'default'))


def explore(engine: SimpliPyEngine, tokens: list[str], budget: int,
            lossy: bool = False, form: str = 'tagged') -> list[str]:
    """The B1 entry point under test (does not exist at HEAD -- red)."""
    return engine._core.ac_simplify_explore(tokens, 48, form, budget, rule_mode=('permissive' if lossy else 'default'))


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
        assert engine._core.ac_ordered_below_in(valley, hill, rule_mode='default') is True

    def test_powsplit_hill(self, engine: SimpliPyEngine) -> None:
        hill = fix(engine, POWSPLIT)
        valley = fix(engine, POWSPLIT_VALLEY)
        assert hill != valley
        assert engine._core.ac_ordered_below_in(valley, hill, rule_mode='default') is True

    def test_gated_hill_in_both_modes(self, engine: SimpliPyEngine) -> None:
        for lossy in (False, True):
            hill = fix(engine, GATED, lossy)
            valley = fix(engine, GATED_VALLEY, lossy)
            assert hill != valley
            assert engine._core.ac_ordered_below_in(valley, hill, rule_mode='default') is True


class TestRow156Falsifier:
    """RED at HEAD: the exploration entry point does not exist yet."""

    def test_exploration_reaches_the_valley(self, engine: SimpliPyEngine) -> None:
        hill = fix(engine, ROW156)
        valley = fix(engine, ROW156_VALLEY)
        for spelling in (ROW156, ROW156_SOOSE):
            out = explore(engine, tok(engine, spelling), BUDGET)
            assert out == valley
            assert engine._core.ac_ordered_below_in(out, hill, rule_mode='default') is True

    def test_powsplit_reaches_the_valley(self, engine: SimpliPyEngine) -> None:
        out = explore(engine, tok(engine, POWSPLIT), BUDGET)
        assert out == fix(engine, POWSPLIT_VALLEY)
        assert engine._core.ac_ordered_below_in(out, fix(engine, POWSPLIT), rule_mode='default') is True

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
        assert engine._core.ac_simplify_in_mode(valley, 48, 'tagged', 0, rule_mode='default') == valley

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
                        engine._core.ac_ordered_below_in(out, f, rule_mode='default') is True

    def test_budget_zero_is_byte_identical(self, engine: SimpliPyEngine) -> None:
        # The ledger's effort=0 semantics: unused exploration leaves every
        # behavior byte-identical to today's chain, mode by mode, form by form.
        for s in BATTERY:
            t = tok(engine, s)
            for lossy in (False, True):
                for form in ('tagged', 'explicit'):
                    assert explore(engine, t, 0, lossy, form) \
                        == engine._core.ac_simplify_in_mode(t, 48, form, 0, rule_mode=('permissive' if lossy else 'default'))

    def test_budget_zero_is_the_default(self, engine: SimpliPyEngine) -> None:
        t = tok(engine, ROW156)
        assert engine._core.ac_simplify_explore(t, rule_mode='default') \
            == engine._core.ac_simplify_in_mode(t, 48, 'tagged', 0, rule_mode='default')

    def test_boundary_contracts_match_ac_simplify(self, engine: SimpliPyEngine) -> None:
        # Empty input: the documented `simplify([]) == []` contract (H-003).
        assert engine._core.ac_simplify_explore([], 48, 'tagged', BUDGET, rule_mode='default') == []
        # Malformed input: ValueError, exactly as ac_simplify (fail loudly,
        # never pass garbage through).
        with pytest.raises(ValueError):
            engine._core.ac_simplify_explore(['+', 'x0'], 48, 'tagged', BUDGET, rule_mode='default')
        with pytest.raises(ValueError):
            # unclosed tagged bag: the AC parser is the arbiter (H-003 sibling)
            engine._core.ac_simplify_explore(['<add>', 'x0'], 48, 'tagged', BUDGET, rule_mode='default')


class TestFactorMove:
    """FACTOR, the reverse of distribute: a sum whose terms share a factor, as that factor
    times the quotient sum. Under a quotient the factored spelling pays only once the shared
    factor cancels across it, which the greedy chain never reaches: the factoring itself
    does not descend there."""

    HILL = "x1/(x1*x2 - x1**2)"
    VALLEY = "1/(x2 - x1)"

    def test_the_chain_stops_on_the_hill(self, engine: SimpliPyEngine) -> None:
        hill, valley = fix(engine, self.HILL), fix(engine, self.VALLEY)
        assert hill != valley
        assert engine._core.ac_ordered_below_in(valley, hill, rule_mode='default') is True

    def test_exploration_factors_and_cancels(self, engine: SimpliPyEngine) -> None:
        for lossy in (False, True):
            out = explore(engine, tok(engine, self.HILL), BUDGET, lossy=lossy)
            assert out == fix(engine, self.VALLEY, lossy), lossy

    def test_factoring_under_an_inverse_takes_its_licence(self, engine: SimpliPyEngine) -> None:
        # Factoring a sum under a negative power is distributing that power read backwards:
        # `1/(x1*x2 - x1*|x2|)` would become `1/(x1*(x2 - |x2|))`, where `x2 - |x2|` vanishes on
        # the whole of x2 > 0 and the inverse takes x1's sign there; `pow` holds such a product,
        # so the sum is not factored.
        src = ['inv', '-', '*', 'x1', 'x2', '*', 'x1', 'abs', 'x2']
        assert explore(engine, src, BUDGET) == fix(engine, engine.to_infix(src))

    def test_shared_constant_is_not_factored(self, engine: SimpliPyEngine) -> None:
        # Each `<constant>` is its own fitted constant: factoring a shared one out of two
        # terms would merge two constants into one, so the move does not propose it.
        src = ['+', '*', '<constant>', 'x1', '*', '<constant>', 'x2']
        out = explore(engine, src, BUDGET, lossy=True)
        assert out.count('<constant>') == 2, out


class TestTheSearchOwnsTheDistributiveLaw:
    """FACTOR takes what some terms of a sum share out of exactly those terms, and DISTRIBUTE
    moves one factor of a product into one of its sum factors; both are search moves under the
    whole-state acceptance, in every mode alike (no rule states the law)."""

    MODES = ('f64', 'real', 'permissive')

    @pytest.fixture(scope='class')
    def llm(self) -> SimpliPyEngine:
        return SimpliPyEngine.load('acj-5-4-llm', install=True)

    @pytest.mark.parametrize('src, want', [
        ('x1*x2 + x1*x3 + x4', 'x4 + x1*(x2 + x3)'),
        ('sin(x1)*x2 + sin(x1)*x3 + x4', 'x4 + sin(x1)*(x2 + x3)'),
        ('x2*x3 + x2 + x3 + 1', '(x2 + 1)*(x3 + 1)'),
        ('x1 + x1*x2 + x4', 'x4 + x1*(x2 + 1)'),
        # A sum factor spread among the other terms, in either orientation.
        ('x2 - x3 + (x2 - x3)*x4', '(x2 - x3)*(x4 + 1)'),
        ('x3 - x2 + (x2 - x3)*x4', '(x2 - x3)*(x4 - 1)'),
        # Two terms that share a base, beside a third holding it at another exponent.
        ('x8 + x3*x8 + 1/x8**3', '1/x8**3 + x8*(x3 + 1)'),
    ])
    def test_every_mode_factors_alike(self, llm: SimpliPyEngine, src: str, want: str) -> None:
        e = llm
        for mode in self.MODES:
            assert e.simplify(src, mode=mode) == e.simplify(want, mode=mode, effort=0), (src, mode)

    def test_one_factor_into_one_sum_factor(self, llm: SimpliPyEngine) -> None:
        # The reverse of FACTOR on some of a sum's terms: `rootn(x1, 2)` moves into the sum
        # (where permissive merges it with its square root), `exp(x2)` stays outside.
        src = 'exp(x2)*rootn(x1, 2)*(rootn(x1, 2) - x2)'
        want = 'exp(x2)*(x1 - x2*rootn(x1, 2))'
        assert llm.simplify(src, mode='permissive') == llm.simplify(want, mode='permissive', effort=0)

    def test_cheapest_candidate_first(self, llm: SimpliPyEngine) -> None:
        # The factoring is the cheapest rebuilt state among this state's candidates, so the
        # first descent takes it, whatever the expansions before it in pre-order.
        e = llm
        src = 'cos(x1*x2 + x1*x3) + sin((x4+1)*(x5+1)) + sin((x6+1)*(x7+1)) + sin((x8+1)*(x9+1))'
        out = e.simplify(src, effort=1)
        assert out == e.simplify(src) and 'x1*(x2 + x3)' in out, out

    def test_effort_zero_is_the_chain_alone(self, llm: SimpliPyEngine) -> None:
        e = llm
        for mode in self.MODES:
            assert e.simplify('x1*x2 + x1*x3 + x4', mode=mode, effort=0) == 'x4 + x1*x2 + x1*x3', mode
