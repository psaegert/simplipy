import numpy as np
import pytest

from simplipy import SimpliPyEngine
from simplipy.utils import violates_wildcard_multiplicity
from conftest import acj_config_path


# Minimal operator set for pruning tests
_MINIMAL_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "-": {"realization": "-", "alias": [], "inverse": "+", "arity": 2, "precedence": 1, "commutative": False},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
    "inv": {"realization": "simplipy.operators.inv", "alias": ["inverse"], "inverse": "inv", "arity": 1, "precedence": 4, "commutative": False},
    "sin": {"realization": "np.sin", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False},
}


class TestPruneCoveredRules:
    """Tests for SimpliPyEngine.prune_covered_rules()."""

    def test_composite_covered_rule_removed_needed_rule_kept(self) -> None:
        """A rule the base rule covers compositionally is removed; the base rule survives."""
        base = (("sin", "sin", "?0"), ("?0",))
        # sin(sin(sin(x))) -> sin(x) is derivable from `base` applied to the inner pair
        covered = (("sin", "sin", "sin", "?0"), ("sin", "?0"))
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[base, covered])

        n_pruned = engine.prune_covered_rules()

        assert n_pruned == 1
        assert engine.simplification_rules == [base]
        # The pruned set is live in the core: the covered rewrite still happens
        assert engine.simplify(["sin", "sin", "sin", "x0"]) == ["sin", "x0"]

    def test_composite_probe_keeps_wide_rule(self) -> None:
        """Leaf-only probing would wrongly remove a `_`-rule whose cover is `?`-sorted.

        The `?`-rule rewrites the wide rule's leaf-instantiated LHS variant, but
        cannot bind the composite `sin(x{i})` probe, so the wide rule is kept.
        (The rules are variable-preserving: a ground-collapsing pair would ride
        the engine's arithmetic -- the original `sin(...) -> 0` fixture leaned on
        the transcendental fold that fold unification removed, 2026-08-02, and a
        `neg` head is soundly STRIPPED by load canonicalization (`neg(A) -> 0`
        normalizes to `A -> 0`), which makes the wide rule genuinely wider than
        its cover and the prune of the cover correct.)
        """
        leaf_cover = (("+", "?0", "1"), ("?0",))
        wide = (("sin", "+", "_0", "1"), ("sin", "_0"))
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[leaf_cover, wide])

        # Leaf-only coverage holds: without `wide`, the leaf variant reaches the
        # wide rule's own answer ...
        probe = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[leaf_cover])
        assert probe.simplify(["sin", "+", "x0", "1"]) == ["sin", "x0"]
        # ... but the composite variant does not
        assert probe.simplify(["sin", "+", "sin", "x0", "1"]) != ["sin", "sin", "x0"]

        n_pruned = engine.prune_covered_rules()

        assert n_pruned == 0
        assert engine.simplification_rules == [leaf_cover, wide]
        assert engine.simplify(["sin", "+", "sin", "x0", "1"]) == ["sin", "sin", "x0"]

    def test_ground_rule_covered_by_exact_arithmetic(self) -> None:
        """A rule the AC core's exact extended-real arithmetic computes natively is covered.

        ``<constant> / -inf`` is 0 for EVERY finite refit of the constant (total, the
        forall direction of the contract), and the AC-judged coverage probe sees exactly
        that -- the old engine's numeric folder had to keep ``<constant>`` literal and
        so kept this rule alive; the serving engine does not need it.
        """
        ground = (("/", "<constant>", 'float("-inf")'), ("0",))
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[ground])

        # The AC engine computes the ground collapse natively, rule-free.
        no_rules = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[])
        assert no_rules.simplify(["/", "<constant>", 'float("-inf")']) == ["0"]

        n_pruned = engine.prune_covered_rules()

        assert n_pruned == 1
        assert engine.simplification_rules == []

    def test_pruning_is_idempotent(self) -> None:
        """A second prune removes nothing and leaves the rule list unchanged."""
        rules = [
            (("sin", "sin", "?0"), ("?0",)),
            (("sin", "sin", "sin", "?0"), ("sin", "?0")),      # covered
            (("sin", "sin", "sin", "sin", "?0"), ("?0",)),     # covered
        ]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        n_first = engine.prune_covered_rules()
        rules_after_first = list(engine.simplification_rules)

        n_second = engine.prune_covered_rules()

        assert n_first == 2
        assert n_second == 0
        assert engine.simplification_rules == rules_after_first

    def test_deterministic_across_hash_seeds(self) -> None:
        """Two subprocess runs under different PYTHONHASHSEED values agree exactly.

        First-match-wins makes rule order behavioral, so the prune must never
        let set iteration order leak into the probe-engine build.
        """
        import json
        import os
        import subprocess
        import sys

        script = (
            "import json, sys\n"
            "from simplipy import SimpliPyEngine\n"
            f"ops = json.loads({json.dumps(json.dumps(_MINIMAL_OPERATORS))})\n"
            "rules = [\n"
            "    (('sin', 'sin', '?0'), ('?0',)),\n"
            "    (('sin', 'sin', 'sin', '?0'), ('sin', '?0')),\n"
            "    (('sin', 'sin', 'sin', 'sin', '?0'), ('?0',)),\n"
            "    (('+', '?0', '1'), ('0',)),\n"
            "    (('sin', '+', '_0', '1'), ('0',)),\n"
            "    (('/', '<constant>', 'float(\"-inf\")'), ('0',)),\n"
            "]\n"
            "engine = SimpliPyEngine(operators=ops, rules=rules)\n"
            "engine.prune_covered_rules()\n"
            "print(json.dumps(engine.simplification_rules))\n"
        )
        outputs = []
        for hash_seed in ('0', '424242'):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed,
                       PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
            result = subprocess.run([sys.executable, '-c', script], env=env,
                                    capture_output=True, text=True, check=True)
            outputs.append(result.stdout)
        assert outputs[0] == outputs[1]
        # Sanity: the two covered sin-chain rules AND the exact-arithmetic ground rule
        # (<constant>/-inf, native under the AC judge) are pruned; the rest survive.
        assert len(json.loads(outputs[0])) == 3


class TestIsValid:
    """Tests for SimpliPyEngine.is_valid()."""

    def _engine(self) -> SimpliPyEngine:
        return SimpliPyEngine(operators=_MINIMAL_OPERATORS)

    def test_valid_binary_expression(self) -> None:
        assert self._engine().is_valid(["+", "x", "y"]) is True

    def test_valid_unary_expression(self) -> None:
        assert self._engine().is_valid(["neg", "x"]) is True

    def test_valid_nested_expression(self) -> None:
        assert self._engine().is_valid(["+", "*", "x", "y", "z"]) is True

    def test_valid_single_variable(self) -> None:
        assert self._engine().is_valid(["x"]) is True

    def test_invalid_variable_at_root(self) -> None:
        """A multi-token expression starting with a variable is invalid."""
        assert self._engine().is_valid(["x", "+", "y"]) is False

    def test_invalid_too_few_operands(self) -> None:
        assert self._engine().is_valid(["+", "x"]) is False

    def test_invalid_leftover_on_stack(self) -> None:
        assert self._engine().is_valid(["+", "x", "y", "z"]) is False

    def test_valid_numeric_constant(self) -> None:
        assert self._engine().is_valid(["+", "3.14", "x"]) is True


class TestMode:
    """The soundness `Mode` axis: SOUND (default) keeps a non-finite-a.e. subtree; LOSSY
    relaxes the constant-fold's finiteness gate (and the rule matcher's `!`-certificate)."""

    def test_mode_is_exported_and_ordinal(self) -> None:
        from simplipy import Mode
        assert Mode.SOUND < Mode.LOSSY
        assert [m.name for m in Mode] == ["SOUND", "LOSSY"]

    def test_sound_keeps_constant_over_zero(self) -> None:
        """`<constant>/0` is +-inf/nan for every constant, so SOUND must NOT fold it."""
        from simplipy import Mode
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[])
        # The AC engine computes inv(0) = +inf EXACTLY (one-zero contract), so the pole
        # is explicit: inf * <constant>. What SOUND must never do is collapse to a bare
        # <constant> -- the value class is not finite. (The LOSSY parity corner for this
        # spelling is a flagged flash-ansr-migration item.)
        out = engine.simplify(["/", "<constant>", "0"], mode=Mode.SOUND)
        assert out == ["<mul>", 'float("inf")', "<constant>", "</mul>"]
        assert engine.simplify(["/", "<constant>", "0"]) == out  # the default is SOUND

    def test_lossy_constant_over_zero_keeps_the_explicit_pole(self) -> None:
        """The AC core folds inv(0) to +inf exactly BEFORE any widening can see the
        quotient, so LOSSY also keeps the explicit pole spelling (the documented
        training-path parity corner, revisited at the flash-ansr migration)."""
        from simplipy import Mode
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[])
        assert engine.simplify(["/", "<constant>", "0"], mode=Mode.LOSSY) \
            == ["<mul>", 'float("inf")', "<constant>", "</mul>"]

    def test_finite_ae_fold_is_mode_independent(self) -> None:
        """A finite-a.e. subtree (`1/C`) folds in BOTH modes -- the pole is measure-zero."""
        from simplipy import Mode
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[])
        assert engine.simplify(["inv", "<constant>"], mode=Mode.SOUND) == ["<constant>"]
        assert engine.simplify(["inv", "<constant>"], mode=Mode.LOSSY) == ["<constant>"]

    def test_lossy_relaxes_cancellation_group_axioms(self) -> None:
        """The THIRD edge: SOUND cancellation respects the group axioms (`inf/inf`, `inf-inf`
        stay the sound `nan`); LOSSY relaxes them (structural cancel) -- the same relaxation LOSSY
        applies to the rule matcher's `!`-cert and the constant-fold's finiteness gate, so all
        three edges behave consistently under `Mode.LOSSY`."""
        from simplipy import Mode
        engine = SimpliPyEngine.from_config(
            acj_config_path())
        # (inf/inf)*x0: SOUND keeps the sound nan; LOSSY relaxes the $-certificate and the
        # structural cancel fires.
        c = ["*", "/", 'float("inf")', 'float("inf")', "x0"]
        assert list(engine.simplify(list(c), mode=Mode.SOUND)) == ['float("nan")']
        assert list(engine.simplify(list(c), mode=Mode.LOSSY)) == ["x0"]
        # (inf-inf)+x0: the AC CONSTRUCTORS compute inf + (-inf) = nan exactly (total
        # extended-real arithmetic, mode-independent) before any cancel could see it, so
        # BOTH modes return the true value -- there is no unsound step for LOSSY to relax.
        c = ["+", "-", 'float("inf")', 'float("inf")', "x0"]
        assert list(engine.simplify(list(c), mode=Mode.SOUND)) == ['float("nan")']
        assert list(engine.simplify(list(c), mode=Mode.LOSSY)) == ['float("nan")']


class TestOperatorConversions:
    """Tests for operators_to_realizations and realizations_to_operators."""

    def _engine(self) -> SimpliPyEngine:
        return SimpliPyEngine(operators=_MINIMAL_OPERATORS)

    def test_roundtrip(self) -> None:
        """operators_to_realizations → realizations_to_operators is identity."""
        engine = self._engine()
        expr = ["sin", "x"]
        realized = engine.operators_to_realizations(expr)
        recovered = engine.realizations_to_operators(realized)
        assert recovered == expr

    def test_operators_to_realizations_maps_correctly(self) -> None:
        engine = self._engine()
        result = engine.operators_to_realizations(["sin", "x"])
        assert result == ["np.sin", "x"]

    def test_unknown_token_passed_through(self) -> None:
        engine = self._engine()
        result = engine.operators_to_realizations(["sin", "my_var"])
        assert "my_var" in result

    from conftest import acj_config_path, construct_legacy_table
    engine = construct_legacy_table()
    expr = " + ".join(["x"] * 14)

    simplified = engine.simplify(expr, node_budget=2)
    simplified_prefix = engine.parse(simplified)

    assert "mult7" not in simplified_prefix
    assert "div7" not in simplified_prefix


def test_repeated_multiplication_avoids_unsupported_powers() -> None:
    from conftest import construct_legacy_table
    engine = construct_legacy_table()
    expr = "x / (" + " * ".join(["x"] * 15) + ")"

    simplified = engine.simplify(expr, node_budget=2)
    simplified_prefix = engine.parse(simplified)

    assert "pow7" not in simplified_prefix
    assert "mult7" not in simplified_prefix
    assert "div7" not in simplified_prefix


def test_simplify_accepts_numpy_array_tokens() -> None:
    engine = SimpliPyEngine.from_config(
        acj_config_path())
    expr = np.array(engine.parse("x1 + x2"), dtype=object)

    simplified = engine.simplify(expr, node_budget=2)

    assert isinstance(simplified, np.ndarray)
    assert simplified.dtype == expr.dtype
    assert list(simplified) == ["<add>", "x1", "x2", "</add>"]


def test_simplify_rejects_invalid_numpy_array_inputs() -> None:
    engine = SimpliPyEngine.from_config(
        acj_config_path())
    expr_2d = np.array([["+", "x1", "x2"]], dtype=object)
    expr_numeric = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        engine.simplify(expr_2d)

    with pytest.raises(ValueError):
        engine.simplify(expr_numeric)


class TestViolatesWildcardMultiplicity:
    """Tests for the wildcard multiplicity termination guard."""

    def test_valid_rule_no_wildcards(self) -> None:
        assert violates_wildcard_multiplicity(["+", "x", "0"], ["x"]) is False

    def test_valid_rule_equal_multiplicity(self) -> None:
        assert violates_wildcard_multiplicity(["+", "_0", "0"], ["_0"]) is False

    def test_valid_rule_decreasing_multiplicity(self) -> None:
        assert violates_wildcard_multiplicity(["+", "_0", "_0"], ["_0"]) is False

    def test_valid_rule_multiple_wildcards(self) -> None:
        assert violates_wildcard_multiplicity(["*", "_0", "_1"], ["*", "_1", "_0"]) is False

    def test_violating_rule_duplicated_wildcard(self) -> None:
        # _0 appears once on LHS but twice on RHS
        assert violates_wildcard_multiplicity(["f", "g", "_0"], ["_0", "_0"]) is True

    def test_violating_rule_new_wildcard_on_rhs(self) -> None:
        # _1 does not appear on LHS at all
        assert violates_wildcard_multiplicity(["+", "_0", "0"], ["_0", "_1"]) is True

    def test_valid_rule_with_tuples(self) -> None:
        assert violates_wildcard_multiplicity(("*", "_0", "_1"), ("*", "_1", "_0")) is False

    def test_violating_with_mixed_wildcards(self) -> None:
        # _0 is fine (1->1), but _1 goes from 1->2
        assert violates_wildcard_multiplicity(["f", "_0", "_1"], ["_0", "_1", "_1"]) is True


# Smaller operator set for find_rules tests (no sin — keeps search space tiny)
_FIND_RULES_OPERATORS = {
    "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
    "-": {"realization": "-", "alias": [], "inverse": "+", "arity": 2, "precedence": 1, "commutative": False},
    "neg": {"realization": "simplipy.operators.neg", "alias": [], "inverse": "neg", "arity": 1, "precedence": 2.5, "commutative": False},
    "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
    "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
    "inv": {"realization": "simplipy.operators.inv", "alias": ["inverse"], "inverse": "inv", "arity": 1, "precedence": 4, "commutative": False},
}

# The same vocabulary plus `abs` -- the one op here the AC constructors do NOT own.
# Over pure arithmetic the AC-judged mine correctly mints NOTHING (the engine owns
# every length<=3 identity natively), which turned the rule-property tests below
# into empty-set tautologies. `abs` restores a real, deterministic mining substrate:
# the fast test config mints exactly
#     abs abs ?0 -> abs ?0     and     abs neg ?0 -> abs ?0
# in well under a second, so the property tests quantify over a NON-EMPTY rule set.
# The vocabulary the minting helper below uses. `abs` alone used to mint (parity and
# abs-idempotence rules); as of the 2026-08-07 parity arm the constructors own that whole
# family and an abs-only mine yields ZERO rules -- the anti-vacuity tripwire caught it.
# `exp`/`log` are added because a transcendental inverse pair is something no constructor
# evaluates, so this mine keeps minting no matter how far the canon grows.
_FIND_RULES_OPERATORS_ABS = {
    **_FIND_RULES_OPERATORS,
    "abs": {"realization": "simplipy.operators.abs", "alias": [], "inverse": None, "arity": 1, "precedence": 3, "commutative": False},
    "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
    "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
}


class TestFindRules:
    """Tests for SimpliPyEngine.find_rules() (native-only since 0.5.0)."""

    def _core_engine(self, tmp_path, rules=None, operators=None) -> SimpliPyEngine:
        """Build an engine from tmp config+rules files so `from_config` attaches the core."""
        import json

        import yaml

        tmp_path.mkdir(parents=True, exist_ok=True)
        rules_path = tmp_path / "rules.json"
        rules_path.write_text(json.dumps(rules or []))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(
            {"operators": operators or _FIND_RULES_OPERATORS, "rules": "rules.json"}))
        engine = SimpliPyEngine.from_config(str(config_path))
        assert engine._core is not None, "compiled core failed to attach"
        return engine

    def _run_find_rules(self, tmp_path, operators=None, **kwargs) -> SimpliPyEngine:
        """Helper: run find_rules with a small, fast configuration."""
        defaults = dict(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=128,
            constants_fit_challenges=2,
            constants_fit_retries=1,
            # MINT-level pin (promote_sorts defaults ON since 2026-08-01): promotion
            # adds the ladder's seeded identities and reshapes sorts; the default-on
            # path is pinned separately by test_default_mine_ships_promoted_seeds.
            promote_sorts=False,
        )
        defaults.update(kwargs)
        engine = self._core_engine(tmp_path, operators=operators)
        engine.find_rules(**defaults)
        return engine

    def _run_minting_find_rules(self, tmp_path, **kwargs) -> SimpliPyEngine:
        """A mine that provably MINTS: abs vocabulary, seeded, with an anti-vacuity
        tripwire. Rule-property tests must use this helper so a future change that
        silently empties the mine FAILS them instead of passing them vacuously."""
        kwargs.setdefault("seed", 7)
        engine = self._run_find_rules(tmp_path, operators=_FIND_RULES_OPERATORS_ABS, **kwargs)
        assert engine.simplification_rules, (
            "anti-vacuity tripwire: the abs-vocabulary mine minted no rules; every "
            "rule-property assertion downstream of this helper would hold vacuously")
        return engine

    def test_construction_always_attaches_core(self) -> None:
        """Rust-only engine: EVERY construction path (including in-memory) attaches the
        compiled core, so mining (and everything else) is always available."""
        engine = SimpliPyEngine(operators=_FIND_RULES_OPERATORS)
        assert engine._core is not None

    def test_default_mine_ships_promoted_seeds(self, tmp_path) -> None:
        """promote_sorts defaults ON (owner ruling 2026-08-01: features do not ship
        disabled): a DEFAULT mine emits the deployment-strength artifact. Over pure
        arithmetic the mine itself mints nothing (identities are engine-native, see
        the mint-level twin below), so the default artifact is exactly the ladder's
        SEEDED canonical identities -- the additive-cancel family at `!` and the
        multiplicative twins at `$`, each certified through judge_bang, never by
        fiat (their carriers are pre-canceled or canonicalized away at enumeration,
        so no minable rule carrier exists; simplipy.promotion._ladder)."""
        import inspect
        default = inspect.signature(
            SimpliPyEngine.find_rules).parameters['promote_sorts'].default
        assert default is True
        engine = self._core_engine(tmp_path)
        engine.find_rules(max_source_pattern_length=3, dummy_variables=2,
                          extra_internal_terms=["0", "1"], X=128,
                          constants_fit_challenges=2, constants_fit_retries=1)
        seeds = {(('-', '!0', '!0'), ('0',)),
                 (('+', '!0', 'neg', '!0'), ('0',)),
                 (('+', 'neg', '!0', '!0'), ('0',)),
                 (('*', '0', '!0'), ('0',)),
                 (('/', '$0', '$0'), ('1',)),
                 (('*', '$0', 'inv', '$0'), ('1',)),
                 (('*', 'inv', '$0', '$0'), ('1',)),
                 (('/', '0', '$0'), ('0',))}
        got = {(tuple(lhs), tuple(rhs)) for lhs, rhs in engine.simplification_rules}
        assert got == seeds, got

    def test_arithmetic_identities_are_engine_native_not_mined(self, tmp_path) -> None:
        """The AC-judged mine refuses to mint what the serving engine computes natively.

        Over a pure-arithmetic vocabulary every length<=3 identity (`x+0`, `x*1`,
        `x-0`, ...) is owned by the AC constructors' exact arithmetic, so the mine
        correctly emits NOTHING -- and the identities hold rule-free."""
        engine = self._run_find_rules(tmp_path)
        assert engine.simplification_rules == []
        for expr in (["+", "x0", "0"], ["*", "x0", "1"], ["-", "x0", "0"]):
            assert engine.simplify(expr) == ["x0"]

    def test_deterministic_across_runs(self, tmp_path) -> None:
        """Same seed => identical NON-EMPTY rule set (regression: the mine must be
        seeded -- unseeded evaluation matrices made rulesets irreproducible).

        Runs on the abs vocabulary: on pure arithmetic this compared [] == [],
        which holds no matter how broken seeding gets. The stronger cross-process,
        cross-PYTHONHASHSEED discriminator lives in
        test_mining_soundness.py::test_determinism_across_processes."""
        rules_a = self._run_minting_find_rules(tmp_path / "a").simplification_rules
        rules_b = self._run_minting_find_rules(tmp_path / "b").simplification_rules
        assert rules_a == rules_b

    def test_all_rules_satisfy_wildcard_multiplicity(self, tmp_path) -> None:
        """Every discovered rule must satisfy non-increasing wildcard multiplicity.

        Uses the minting helper: on pure arithmetic this loop iterated an empty
        rule set and asserted nothing."""
        engine = self._run_minting_find_rules(tmp_path)
        for lhs, rhs in engine.simplification_rules:
            assert not violates_wildcard_multiplicity(lhs, rhs), (
                f"Rule violates wildcard multiplicity: {lhs} -> {rhs}"
            )

    def test_reset_rules_clears_existing(self, tmp_path) -> None:
        """reset_rules=True starts from an empty rule set."""
        engine = self._core_engine(tmp_path, rules=[(["+", "x0", "0"], ["x0"])])
        assert len(engine.simplification_rules) == 1
        engine.find_rules(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=128,
            constants_fit_challenges=2,
            constants_fit_retries=1,
            reset_rules=True,
            promote_sorts=False,
        )
        # The seeded rule is GONE (reset) and the AC-judged re-mine of this pure-
        # arithmetic universe correctly mints nothing (the engine owns it natively).
        assert (["+", "x0", "0"], ["x0"]) not in [
            (list(lhs), list(rhs)) for lhs, rhs in engine.simplification_rules]
        assert engine.simplification_rules == []
        assert engine.simplify(["+", "x0", "0"]) == ["x0"]

    def test_prune_reduces_rule_count(self, tmp_path) -> None:
        """prune='covered' on a NON-EMPTY mined set: never grows, keeps a subset,
        and every kept rule still fires on the serving engine.

        On pure arithmetic this reduced to `0 <= 0`; the prune never touched a
        single actual rule. On the abs vocabulary the mine mints both abs rules
        and the prune exercises its coverage judgment for real. Coverage is
        judged in the serve-time reduction ordering (state-coverage, the F5
        ruling): `abs neg ?0 -> abs ?0` is a strict descent there (a complexity
        TIE -- signs are free -- broken by the canonical order), and nothing
        else performs the rewrite, so the prune keeps BOTH rules. The old
        complexity-score judge read the tie as covered and dropped it, losing
        the rewrite from the serving engine."""
        engine_no_prune = self._run_minting_find_rules(tmp_path / "a", prune=False)
        engine_pruned = self._run_minting_find_rules(tmp_path / "b", prune="covered")
        full = engine_no_prune.simplification_rules
        kept = engine_pruned.simplification_rules
        # Same seed => same mine; the prune may only ever REMOVE rules from it.
        assert set(kept) <= set(full)
        assert kept, "prune='covered' emptied the mined rule set"
        # Neither abs rule is covered by the other: both survive, both fire.
        assert set(kept) == set(full)
        assert list(engine_pruned.simplify(["abs", "abs", "x0"])) == ["abs", "x0"]
        assert list(engine_pruned.simplify(["abs", "neg", "x0"])) == ["abs", "x0"]


class TestFindRulesWithCore:
    """find_rules on an engine with the compiled Rust core attached.

    Regression tests for the core mine: the fork-based Python pool mutates Python-side
    rule state while `simplify` runs on the immutable Rust core, so before the native
    delegation an engine from `from_config`/`load` mined 0 rules.
    """

    def _core_engine(self, tmp_path, operators=None) -> SimpliPyEngine:
        """Build an engine from tmp config+rules files so `from_config` attaches the core."""
        import json

        import yaml

        rules_path = tmp_path / "rules.json"
        rules_path.write_text(json.dumps([]))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(
            {"operators": operators or _FIND_RULES_OPERATORS, "rules": "rules.json"}))
        engine = SimpliPyEngine.from_config(str(config_path))
        assert engine._core is not None, "compiled core failed to attach; the native mine path is untested"
        return engine

    def test_native_mine_runs_and_arithmetic_stays_native(self, tmp_path) -> None:
        """The native mine path runs end to end on a core-attached engine.

        Under the AC judge a pure-arithmetic universe correctly mints ZERO rules
        (the pre-native-path bug this guarded against returned 0 rules while the
        identities also FAILED to hold; now 0 rules is the correct outcome and the
        identities hold natively -- both facts asserted)."""
        engine = self._core_engine(tmp_path)
        engine.find_rules(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=128,
            constants_fit_challenges=2,
            constants_fit_retries=1,
            promote_sorts=False,
        )
        assert engine.simplification_rules == []
        assert engine.simplify(["+", "x0", "0"]) == ["x0"]
        assert engine.simplify(["*", "x0", "1"]) == ["x0"]

    def test_native_mine_updates_the_live_core(self, tmp_path) -> None:
        """After the mine, `simplify` (which runs on the compiled core) must apply
        the FRESHLY MINTED rules -- the regression target is a mine whose output
        never reaches the immutable serving core (dead rules, silently).

        Uses the abs vocabulary so the mine actually mints; on pure arithmetic
        (mints nothing since the AC cutover) the old assertion only re-checked
        native behavior and the core-update claim was vacuous."""
        engine = self._core_engine(tmp_path, operators=_FIND_RULES_OPERATORS_ABS)
        engine.find_rules(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=128,
            constants_fit_challenges=2,
            constants_fit_retries=1,
            seed=7,
        )
        assert engine.simplification_rules, (
            "anti-vacuity tripwire: the abs-vocabulary mine minted no rules")
        # Both minted rules must fire on the LIVE core, not just sit in Python state.
        assert list(engine.simplify(["abs", "neg", "x0"])) == ["abs", "x0"]
        assert list(engine.simplify(["abs", "abs", "x0"])) == ["abs", "x0"]
        # And native arithmetic still holds rule-free.
        assert list(engine.simplify(["+", "x0", "0"])) == ["x0"]

    def test_x_as_array_is_accepted(self, tmp_path) -> None:
        """Passing X as an ndarray (documented) must work (it raised NameError before)."""
        engine = self._core_engine(tmp_path)
        engine.find_rules(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=np.random.default_rng(0).normal(0, 5, size=(128, 2)),
            constants_fit_challenges=2,
            constants_fit_retries=1,
            promote_sorts=False,
        )
        # Regression target was a NameError on ndarray X; completion is the assertion
        # (the AC-judged mine of this arithmetic universe correctly mints nothing).
        assert engine.simplification_rules == []


class TestConstantFolding:
    """Tests for numeric constant folding in apply_rules_top_down."""

    def _engine(self, rules=None) -> SimpliPyEngine:
        return SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)

    def test_binary_addition_folding(self) -> None:
        """1.23 + 4.56 should evaluate to a numeric result close to 5.79."""
        engine = self._engine()
        result = engine.simplify(["+", "1.23", "4.56"])
        assert len(result) == 1
        assert abs(float(result[0]) - 5.79) < 1e-10

    def test_binary_subtraction_folding(self) -> None:
        """5 - 3 should evaluate to 2."""
        engine = self._engine()
        result = engine.simplify(["-", "5", "3"])
        assert result == ["2"]

    def test_integer_result_formatting(self) -> None:
        """Integer-valued results should not have a decimal point."""
        engine = self._engine()
        result = engine.simplify(["+", "1", "2"])
        assert result == ["3"]

    def test_unary_folding(self) -> None:
        """neg(3) should evaluate to -3."""
        engine = self._engine()
        result = engine.simplify(["neg", "3"])
        assert result == ["-3"]

    def test_nested_constant_folding(self) -> None:
        """2 * 3 + 4 should evaluate to 10 via nested folding."""
        engine = self._engine()
        result = engine.simplify(["+", "*", "2", "3", "4"])
        assert result == ["10"]

    def test_division_by_zero_produces_inf(self) -> None:
        """1 / 0 should produce float("inf") token."""
        engine = self._engine()
        result = engine.simplify(["/", "1", "0"])
        assert result == ['float("inf")']

    def test_mixed_constant_and_placeholder(self) -> None:
        """<constant> + numeric should fold to <constant>."""
        engine = self._engine()
        result = engine.simplify(["+", "1.23", "<constant>"])
        assert result == ["<constant>"]

    def test_constant_placeholder_still_folds(self) -> None:
        """<constant> + <constant> should still fold to <constant>."""
        engine = self._engine()
        result = engine.simplify(["+", "<constant>", "<constant>"])
        assert result == ["<constant>"]

    def test_folding_enables_further_rules(self) -> None:
        """1 - 1 = 0, then x + 0 should simplify to x via rule."""
        engine = self._engine(rules=[(["+", "_0", "0"], ["_0"])])
        result = engine.simplify(["+", "x", "-", "1", "1"])
        assert result == ["x"]

    def test_simplify_infix_numeric_constants(self) -> None:
        """End-to-end: infix '1.23 + 4.56' folds to one literal; the DOWNSTREAM masking
        toolkit (masking is policy, not engine behavior -- owner ruling A2) abstracts it."""
        from simplipy import masking
        engine = self._engine()
        folded = engine.simplify("1.23 + 4.56")
        assert folded == "5.79"
        masked = masking.mask(engine.parse(folded), engine, masking.mask_all)
        assert masked == ["<constant>"]

    def test_constant_folding_observable(self) -> None:
        """Folding is observable through the simplify result itself (the pure-Python
        SimplificationStatistics instrumentation was removed in the Rust-only cutover)."""
        engine = self._engine()
        assert engine.simplify(["+", "1", "2"]) == ["3"]


class TestResolveConstantRules:
    """Tests for resolve_constant_rules()."""

    def _engine(self, rules) -> SimpliPyEngine:
        return SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)

    def test_resolves_numeric_rule(self) -> None:
        """A rule like sin(1) -> <constant> should be resolved to the actual value."""
        engine = self._engine(rules=[
            (["sin", "1"], ["<constant>"]),
        ])
        n = engine.resolve_constant_rules()
        assert n == 1
        # The rule should now have the actual sin(1) value
        rhs = engine.simplification_rules[0][1]
        assert '<constant>' not in rhs
        assert abs(float(rhs[0]) - 0.8414709848078965) < 1e-10

    def test_leaves_named_constant_rules_unchanged(self) -> None:
        """Rules with np.e or np.pi in the LHS should NOT be resolved."""
        engine = self._engine(rules=[
            (["sin", "np.pi"], ["<constant>"]),
        ])
        n = engine.resolve_constant_rules()
        assert n == 0
        assert tuple(engine.simplification_rules[0][1]) == ('<constant>',)

    def test_leaves_wildcard_rules_unchanged(self) -> None:
        """Pattern rules with wildcards should NOT be resolved."""
        engine = self._engine(rules=[
            (["-", "_0", "_0"], ["<constant>"]),
        ])
        n = engine.resolve_constant_rules()
        assert n == 0
        assert tuple(engine.simplification_rules[0][1]) == ('<constant>',)

    def test_leaves_non_constant_replacement_unchanged(self) -> None:
        """Rules whose replacement is not <constant> should NOT be touched."""
        engine = self._engine(rules=[
            (["+", "1", "0"], ["1"]),
        ])
        n = engine.resolve_constant_rules()
        assert n == 0
        assert tuple(engine.simplification_rules[0][1]) == ('1',)

    def test_compile_rules_called(self) -> None:
        """After resolution, compile_rules should update lookup tables."""
        engine = self._engine(rules=[
            (["sin", "1"], ["<constant>"]),
        ])
        engine.resolve_constant_rules()
        # The resolved rule holds the concrete value in the rule LIST (the plumbing
        # this test pins: compile_rules ran, tables updated) ...
        rules = {tuple(lhs): tuple(rhs) for lhs, rhs in engine.simplification_rules}
        assert ("sin", "1") in rules
        assert rules[("sin", "1")] != ("<constant>",)
        # ... but STAGE 2 translation refuses to SERVE it: `sin 1 -> 0.841...` is a
        # rounding rewrite (a ~105-unit literal above the 10-unit symbolic state,
        # non-descending under mu), so the engine's output stays symbolic. Resolved
        # values remain available to callers reading the rule list; simplify never
        # rounds.
        assert engine.simplify(["sin", "1"]) == ["sin", "1"]

    def test_multiple_rules_mixed(self) -> None:
        """Only the eligible rules should be resolved; others stay intact."""
        engine = self._engine(rules=[
            (["sin", "1"], ["<constant>"]),          # all-numeric -> resolve
            (["-", "_0", "_0"], ["<constant>"]),      # wildcard -> skip
            (["+", "1", "0"], ["1"]),                 # not <constant> rhs -> skip
            (["neg", "1"], ["<constant>"]),            # all-numeric -> resolve
        ])
        n = engine.resolve_constant_rules()
        assert n == 2
        # sin(1) resolved
        assert '<constant>' not in engine.simplification_rules[0][1]
        # _0 - _0 untouched
        assert tuple(engine.simplification_rules[1][1]) == ('<constant>',)
        # + 1 0 untouched
        assert tuple(engine.simplification_rules[2][1]) == ('1',)
        # neg(1) resolved to -1
        assert tuple(engine.simplification_rules[3][1]) == ('-1',)


class TestBangSort:
    """The `!` third sort: `!N` binds a variable leaf freely, or a
    SUBTREE only when the interval engine certifies it defined-and-finite a.e. (`finite_ae`).
    This is what recovers the E3 family (`exp(x) - exp(x) -> 0`) without resurrecting the
    unsound `_`-subtree binding (`log(x) - log(x)` is nan on half the line and must not fire).
    """

    def _engine(self, tmp_path):
        import json

        import yaml

        ops = {
            "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
            "-": {"realization": "-", "alias": [], "inverse": "+", "arity": 2, "precedence": 1, "commutative": False},
            "exp": {"realization": "np.exp", "alias": [], "inverse": "log", "arity": 1, "precedence": 3, "commutative": False},
            "log": {"realization": "np.log", "alias": [], "inverse": "exp", "arity": 1, "precedence": 3, "commutative": False},
            "sinh": {"realization": "np.sinh", "alias": [], "inverse": "asinh", "arity": 1, "precedence": 3, "commutative": False},
            "inv": {"realization": "simplipy.operators.inv", "alias": [], "inverse": "inv", "arity": 1, "precedence": 4, "commutative": False},
            "pow": {"realization": "simplipy.operators.pow", "alias": [], "inverse": "pow", "arity": 2, "precedence": 3, "commutative": False},
        }
        (tmp_path / "rules.json").write_text(json.dumps([[["-", "!0", "!0"], ["0"]]]))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"engine_generation": 2, "operators": ops, "rules": "rules.json"}))
        engine = SimpliPyEngine.from_config(str(tmp_path / "config.yaml"))
        assert engine._core is not None
        return engine

    def test_certified_subtrees_bind(self, tmp_path) -> None:
        engine = self._engine(tmp_path)
        # variable leaf: always binds (a leaf is trivially finite a.e.)
        assert '<constant>' in engine.simplify(["-", "x0", "x0"]) or \
               list(engine.simplify(["-", "x0", "x0"])) == ["0"]
        # exp/sinh: certified finite a.e. -> the rule fires (E3 recovered)
        for f in ("exp", "sinh"):
            out = list(engine.simplify(["-", f, "x0", f, "x0"]))
            assert out in (["0"], ["<constant>"]), (f, out)

    def test_uncertified_subtrees_do_not_bind(self, tmp_path) -> None:
        engine = self._engine(tmp_path)
        # log: nan on half the line -- rewriting to 0 would invent a function. The refusal
        # keeps both terms (native tagged spelling of the same refusal).
        assert list(engine.simplify(["-", "log", "x0", "log", "x0"])) \
            == ["<add>", "log", "x0", "<sub>", "log", "x0", "</add>"]
        # pow(x, inf): a.e. in {0, inf} -- inf - inf = nan on positive measure
        out = list(engine.simplify(["-", "pow", "x0", 'float("inf")', "pow", "x0", 'float("inf")']))
        assert out == ["<add>", "pow", "x0", 'float("inf")', "<sub>", "pow", "x0", 'float("inf")', "</add>"], out

    def test_pole_bearing_subtrees_bind_via_the_structural_path(self, tmp_path) -> None:
        # inv: finite a.e. with a measure-zero pole. Formerly the certificate's stated-scope
        # exclusion (subdivision cannot resolve a pole cell); the structural null-measure
        # certificate (`interval::nonfinite_null`) closes it: inv(x) - inv(x) -> 0 is sound
        # a.e. (the exceptional set {x = 0} is null).
        engine = self._engine(tmp_path)
        assert list(engine.simplify(["-", "inv", "x0", "inv", "x0"])) == ["0"]


class TestMultSort:
    """The `$` fourth sort: `$N` binds a variable leaf freely, or a SUBTREE only when the
    interval engine certifies it defined, finite AND nonzero a.e. (`finite_nonzero_ae` --
    the multiplicative group's regular domain). This is what makes the composite
    self-cancellation family (`f/f -> 1`, `f * inv(f) -> 1`, `0/f -> 0`) expressible at all
    (SELFCANCEL_MULTIPLICATIVE: `judge_bang` demotes `A/A -> 1` at the atom 0, so no
    pre-existing sort could carry it on composites)."""

    SEEDS = [
        [["/", "$0", "$0"], ["1"]],
        [["*", "$0", "inv", "$0"], ["1"]],
        [["*", "inv", "$0", "$0"], ["1"]],
        [["/", "0", "$0"], ["0"]],
    ]

    def _engine(self, tmp_path):
        import json

        import yaml

        ops = {
            "+": {"realization": "+", "alias": [], "inverse": "-", "arity": 2, "precedence": 1, "commutative": True},
            "-": {"realization": "-", "alias": [], "inverse": "+", "arity": 2, "precedence": 1, "commutative": False},
            "*": {"realization": "*", "alias": [], "inverse": "/", "arity": 2, "precedence": 2, "commutative": True},
            "/": {"realization": "simplipy.operators.div", "alias": [], "inverse": "*", "arity": 2, "precedence": 2, "commutative": False},
            "inv": {"realization": "simplipy.operators.inv", "alias": [], "inverse": "inv", "arity": 1, "precedence": 4, "commutative": False},
            "cos": {"realization": "simplipy.operators.cos", "alias": [], "inverse": "acos", "arity": 1, "precedence": 2, "commutative": False},
            "sin": {"realization": "simplipy.operators.sin", "alias": [], "inverse": "asin", "arity": 1, "precedence": 2, "commutative": False},
            "asin": {"realization": "simplipy.operators.asin", "alias": [], "inverse": "sin", "arity": 1, "precedence": 2, "commutative": False},
            "cosh": {"realization": "simplipy.operators.cosh", "alias": [], "inverse": "acosh", "arity": 1, "precedence": 2, "commutative": False},
        }
        (tmp_path / "rules.json").write_text(json.dumps(self.SEEDS))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"engine_generation": 2, "operators": ops, "rules": "rules.json"}))
        engine = SimpliPyEngine.from_config(str(tmp_path / "config.yaml"))
        assert engine._core is not None
        return engine

    def test_certified_composites_cancel(self, tmp_path) -> None:
        engine = self._engine(tmp_path)
        # variable leaf: binds freely (a variable is nonzero a.e.)
        assert list(engine.simplify(["/", "x0", "x0"])) == ["1"]
        # certified composites: bounded-away (cosh), analytic-witness (sin, x - cos x)
        for f in (["cosh", "x0"], ["sin", "x0"], ["-", "x0", "cos", "x0"]):
            assert list(engine.simplify(["/"] + f + f)) == ["1"], f
            assert list(engine.simplify(["*"] + f + ["inv"] + f)) == ["1"], f
        # 0 / certified-nonzero composite -> 0
        assert list(engine.simplify(["/", "0", "-", "x0", "cos", "x0"])) == ["0"]

    def test_uncertified_composites_do_not_cancel(self, tmp_path) -> None:
        engine = self._engine(tmp_path)
        # region-nan operand: finite_nonzero_ae refuses (not finite a.e.); the refusal
        # keeps the quotient (native tagged spelling).
        assert list(engine.simplify(["/", "asin", "x0", "asin", "x0"])) \
            == ["<mul>", "asin", "x0", "<div>", "asin", "x0", "</mul>"]
        # literal zero operand: 0/0 must never become 1 or 0 -- the all-valued exact fold
        # legitimately evaluates it to the nan literal (0/0 IS nan; that is not a $-binding)
        out = list(engine.simplify(["/", "0", "0"]))
        assert out == ['float("nan")'], out
        # identically-zero composite (plateau family): x0 - x0 folds to 0 upstream or stays;
        # either way no $-binding may produce 1
        out = list(engine.simplify(["/", "-", "x0", "x0", "-", "x0", "x0"]))
        assert out != ["1"], out
        # <constant>-bearing: the $-cancellation to 1 is refused (rebind guard + the
        # certificate's constant exclusion). Reduction to <constant> IS allowed: leaf
        # cancellation of x0 plus the independent-constant collapse C1/C2 -> <constant>
        # is sound skeleton semantics, not a $-binding.
        expr = ["/", "*", "<constant>", "x0", "*", "<constant>", "x0"]
        out = list(engine.simplify(list(expr)))
        assert out != ["1"], out

    def test_lossy_skips_the_certificate(self, tmp_path) -> None:
        from simplipy.engine import Mode
        engine = self._engine(tmp_path)
        assert list(engine.simplify(["/", "asin", "x0", "asin", "x0"], mode=Mode.LOSSY)) == ["1"]

    def test_judge_bang_mult_bar(self, tmp_path) -> None:
        # The promotion-side twin: the plain `!` bar demotes A/A -> 1 (0/0 = nan at the atom
        # 0); the multiplicative bar promotes it on the finite-nonzero lattice. The judges
        # REQUIRE the f64 oracle to be configured first (`configure(engine)`) -- calling them
        # bare now raises instead of looping (the _literal_spans termination guard).
        import numpy as np
        import pytest as _pytest
        from simplipy.promotion._f64_eval import configure
        from simplipy.promotion._ladder import judge_bang, judge_bang_mult
        from simplipy.promotion._pointwise import prefold
        rng = np.random.default_rng(0)
        from simplipy.promotion._f64_eval import ARITY
        if not ARITY:  # order-independent: another test may have configured the module state
            with _pytest.raises(RuntimeError, match='oracle not configured'):
                prefold(['/', '1', '1'])
        configure(self._engine(tmp_path))
        v_bang, _ = judge_bang(('/', '!0', '!0'), ('1',), rng)
        assert v_bang != 'PROMOTE'
        v_mult, _ = judge_bang_mult(('/', '$0', '$0'), ('1',), rng)
        assert v_mult == 'PROMOTE'
        v_zero, _ = judge_bang_mult(('/', '0', '$0'), ('0',), rng)
        assert v_zero == 'PROMOTE'
