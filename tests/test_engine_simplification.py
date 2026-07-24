import numpy as np
import pytest

from simplipy import SimpliPyEngine
from simplipy.utils import violates_wildcard_multiplicity


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


class TestPruneRedundantRules:
    """Tests for SimpliPyEngine.prune_redundant_rules()."""

    def test_explicit_rule_subsumed_by_wildcard_is_pruned(self) -> None:
        """An explicit rule covered by a wildcard-pattern rule is removed."""
        rules = [
            # Wildcard: sin(0) -> 0 would NOT cover this; need a general pattern
            # Wildcard: +(_0, 0) -> _0
            (["+", "_0", "0"], ["_0"]),
            # Explicit: +(x, 0) -> x  — subsumed by the wildcard above
            (["+", "x", "0"], ["x"]),
        ]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        n_pruned = engine.prune_redundant_rules()

        assert n_pruned == 1
        # Only the wildcard rule should remain
        assert len(engine.simplification_rules) == 1
        assert any("_0" in r[0] for r in engine.simplification_rules)

    def test_non_subsumed_explicit_rule_is_retained(self) -> None:
        """An explicit rule not covered by any wildcard rule survives pruning."""
        rules = [
            # Wildcard: *(_0, 0) -> 0
            (["*", "_0", "0"], ["0"]),
            # Explicit: +(x, 0) -> x — NOT subsumed (different operator)
            (["+", "x", "0"], ["x"]),
        ]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        n_pruned = engine.prune_redundant_rules()

        assert n_pruned == 0
        assert len(engine.simplification_rules) == 2

    def test_constant_folding_subsumes_all_constant_rule(self) -> None:
        """A rule where all operands are <constant> is pruned by constant folding."""
        rules = [
            # Explicit: sin(<constant>) -> <constant>
            # Constant folding in apply_rules_top_down handles this automatically
            (["sin", "<constant>"], ["<constant>"]),
        ]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        n_pruned = engine.prune_redundant_rules()

        assert n_pruned == 1
        assert len(engine.simplification_rules) == 0

    def test_returns_correct_count(self) -> None:
        """The return value equals the number of pruned rules."""
        rules = [
            (["+", "_0", "0"], ["_0"]),
            (["+", "x", "0"], ["x"]),       # subsumed
            (["+", "y", "0"], ["y"]),       # subsumed
            (["*", "x", "1"], ["x"]),       # NOT subsumed (no wildcard for *)
        ]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        n_pruned = engine.prune_redundant_rules()

        assert n_pruned == 2
        assert len(engine.simplification_rules) == 2

    def test_engine_still_simplifies_correctly_after_pruning(self) -> None:
        """Simplification results are preserved after pruning."""
        rules = [
            (["+", "_0", "0"], ["_0"]),
            (["+", "x", "0"], ["x"]),  # redundant
        ]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)

        # Before pruning
        result_before = engine.simplify(["+", "x", "0"])

        engine.prune_redundant_rules()

        # After pruning — same result
        result_after = engine.simplify(["+", "x", "0"])
        assert result_before == result_after

    def test_no_rules_is_noop(self) -> None:
        """Pruning an engine with no rules does nothing."""
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[])
        n_pruned = engine.prune_redundant_rules()

        assert n_pruned == 0
        assert len(engine.simplification_rules) == 0

    def test_only_pattern_rules_is_noop(self) -> None:
        """Pruning an engine with only wildcard rules does nothing."""
        rules = [
            (["+", "_0", "0"], ["_0"]),
            (["*", "_0", "1"], ["_0"]),
        ]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        n_pruned = engine.prune_redundant_rules()

        assert n_pruned == 0
        assert len(engine.simplification_rules) == 2

    def test_pruning_is_idempotent(self) -> None:
        """Running prune_redundant_rules a second time prunes no further rules."""
        rules = [
            (["+", "_0", "0"], ["_0"]),
            (["+", "x", "0"], ["x"]),       # subsumed
            (["+", "y", "0"], ["y"]),       # subsumed
            (["*", "x", "1"], ["x"]),       # NOT subsumed (no wildcard for *)
        ]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        engine.prune_redundant_rules()
        rules_after_first = list(engine.simplification_rules)

        n_second = engine.prune_redundant_rules()

        assert n_second == 0
        assert engine.simplification_rules == rules_after_first

    def test_remaining_explicit_rules_are_necessary(self) -> None:
        """After pruning, no remaining explicit rule is itself redundant.

        This verifies the key soundness property of serial pruning: since
        each rule is tested with previously-pruned rules already removed,
        every surviving explicit rule is individually necessary.
        """
        import re
        is_wildcard = re.compile(r'^_\d+$')

        rules = [
            (["+", "_0", "0"], ["_0"]),
            (["+", "x", "0"], ["x"]),       # subsumed
            (["+", "y", "0"], ["y"]),       # subsumed
            (["*", "x", "1"], ["x"]),       # NOT subsumed
            (["*", "_0", "0"], ["0"]),
            (["*", "x", "0"], ["0"]),       # subsumed
        ]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        engine.prune_redundant_rules()

        # Every remaining explicit rule must be necessary: remove it from the live rule
        # set (compile_rules syncs the core) and check the transformation is no longer
        # derivable, then restore.
        all_rules = list(engine.simplification_rules)
        for lhs, rhs in all_rules:
            if any(is_wildcard.match(t) for t in lhs):
                continue  # Skip pattern rules
            engine.simplification_rules = [r for r in all_rules if tuple(r[0]) != tuple(lhs)]
            engine.compile_rules()
            try:
                result = engine.simplify(list(lhs), mask_elementary_literals=False)
                assert tuple(result) != tuple(rhs), (
                    f"Rule {lhs} -> {rhs} is still redundant after pruning"
                )
            finally:
                engine.simplification_rules = list(all_rules)
                engine.compile_rules()


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
        """
        leaf_cover = (("+", "?0", "1"), ("0",))
        wide = (("sin", "+", "_0", "1"), ("0",))
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[leaf_cover, wide])

        # Leaf-only coverage holds: without `wide`, the leaf variant collapses
        probe = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[leaf_cover])
        assert len(probe.simplify(["sin", "+", "x0", "1"])) <= 1
        # ... but the composite variant does not
        assert len(probe.simplify(["sin", "+", "sin", "x0", "1"])) > 1

        n_pruned = engine.prune_covered_rules()

        assert n_pruned == 0
        assert engine.simplification_rules == [leaf_cover, wide]

    def test_constant_stays_literal(self) -> None:
        """An all-constant-foldable LHS must not count as covered via numeric folding.

        With ``<constant>`` substituted by a numeral, native constant folding
        collapses the LHS and the rule would be removed; the probe keeps
        ``<constant>`` literal, so the rule survives.
        """
        ground = (("/", "<constant>", 'float("-inf")'), ("0",))
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[ground])

        # Numeral instantiation IS natively folded (the trap the literal probe avoids)
        no_rules = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[])
        assert len(no_rules.simplify(["/", "7", 'float("-inf")'])) == 1
        # The literal probe is not
        assert no_rules.simplify(["/", "<constant>", 'float("-inf")']) == ["/", "<constant>", 'float("-inf")']

        n_pruned = engine.prune_covered_rules()

        assert n_pruned == 0
        assert engine.simplification_rules == [ground]

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
        # Sanity: the run pruned the two covered sin-chain rules and kept the rest
        assert len(json.loads(outputs[0])) == 4


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


class TestSortOperands:
    """Tests for the core operand-sort sub-unit (`_core.sort_only`; the pure-Python
    `sort_operands` was removed in the Rust-only cutover)."""

    def _engine(self) -> SimpliPyEngine:
        return SimpliPyEngine(operators=_MINIMAL_OPERATORS)

    def test_commutative_reorder(self) -> None:
        """Commutative operator sorts operands into canonical order."""
        engine = self._engine()
        result = engine._core.sort_only(["+", "b", "a"])
        # After sorting, variables should be in canonical order
        assert result == engine._core.sort_only(["+", "a", "b"])

    def test_non_commutative_unchanged(self) -> None:
        """Non-commutative operator preserves operand order."""
        engine = self._engine()
        assert engine._core.sort_only(["-", "b", "a"]) == ["-", "b", "a"]

    def test_idempotent(self) -> None:
        """Sorting an already-sorted expression is idempotent."""
        engine = self._engine()
        first = engine._core.sort_only(["+", "b", "a"])
        second = engine._core.sort_only(first)
        assert first == second

    def test_nested_commutative(self) -> None:
        """Nested commutative operators are sorted recursively."""
        engine = self._engine()
        result = engine._core.sort_only(["*", "z", "a"])
        assert result == engine._core.sort_only(["*", "a", "z"])


class TestCancelTerms:
    """Tests for the core term-cancellation sub-unit (`_core.cancel_only`, a faithful
    `cancel_terms(*collect_multiplicities(tokens))`; the pure-Python pipeline was removed
    in the Rust-only cutover)."""

    def _engine(self) -> SimpliPyEngine:
        return SimpliPyEngine(operators=_MINIMAL_OPERATORS)

    def test_cancel_x_minus_x(self) -> None:
        """x - x should cancel — both operands become neutral element 0."""
        engine = self._engine()
        expr = ["-", "x", "x"]
        result = engine._core.cancel_only(expr)
        assert result == ["-", "0", "0"]

    def test_cancel_x_div_x(self) -> None:
        """x / x should cancel — both operands become neutral element 1."""
        engine = self._engine()
        expr = ["/", "x", "x"]
        result = engine._core.cancel_only(expr)
        assert result == ["/", "1", "1"]

    def test_no_cancellation(self) -> None:
        """Expression with nothing to cancel is returned unchanged."""
        engine = self._engine()
        expr = ["+", "x", "y"]
        result = engine._core.cancel_only(expr)
        assert result == ["+", "x", "y"]


class TestApplySimplificationRules:
    """Tests for the core rule-application sub-unit (`_core.apply_rules`; the pure-Python
    `apply_simplifcation_rules` was removed in the Rust-only cutover)."""

    def test_applies_matching_rule(self) -> None:
        rules = [(["+", "_0", "0"], ["_0"])]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        result = engine._core.apply_rules(["+", "x", "0"])
        assert result == ["x"]

    def test_all_constants_returns_constant(self) -> None:
        """Expression of only operators and <constant> tokens reduces to <constant>."""
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS)
        result = engine._core.apply_rules(["+", "<constant>", "<constant>"])
        assert result == ["<constant>"]

    def test_no_matching_rule_unchanged(self) -> None:
        """Expression that matches no rule is returned unchanged."""
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[])
        result = engine._core.apply_rules(["+", "x", "y"])
        assert result == ["+", "x", "y"]


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

    engine = SimpliPyEngine.load("dev_7-3", install=True)
    expr = " + ".join(["x"] * 14)

    simplified = engine.simplify(expr, max_iter=1)
    simplified_prefix = engine.parse(simplified)

    assert "mult7" not in simplified_prefix
    assert "div7" not in simplified_prefix


def test_repeated_multiplication_avoids_unsupported_powers() -> None:
    engine = SimpliPyEngine.load("dev_7-3", install=True)
    expr = "x / (" + " * ".join(["x"] * 15) + ")"

    simplified = engine.simplify(expr, max_iter=1)
    simplified_prefix = engine.parse(simplified)

    assert "pow7" not in simplified_prefix
    assert "mult7" not in simplified_prefix
    assert "div7" not in simplified_prefix


def test_simplify_accepts_numpy_array_tokens() -> None:
    engine = SimpliPyEngine.load("dev_7-3", install=True)
    prefix_tokens = engine.parse("x1 + x2")
    expr = np.array(prefix_tokens, dtype=object)

    simplified = engine.simplify(expr, max_iter=1, apply_simplification_rules=False)

    assert isinstance(simplified, np.ndarray)
    assert simplified.dtype == expr.dtype
    assert np.array_equal(simplified, expr)


def test_simplify_rejects_invalid_numpy_array_inputs() -> None:
    engine = SimpliPyEngine.load("dev_7-3", install=True)
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


class TestFindRules:
    """Tests for SimpliPyEngine.find_rules() (native-only since 0.5.0)."""

    def _core_engine(self, tmp_path, rules=None) -> SimpliPyEngine:
        """Build an engine from tmp config+rules files so `from_config` attaches the core."""
        import json

        import yaml

        tmp_path.mkdir(parents=True, exist_ok=True)
        rules_path = tmp_path / "rules.json"
        rules_path.write_text(json.dumps(rules or []))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"operators": _FIND_RULES_OPERATORS, "rules": "rules.json"}))
        engine = SimpliPyEngine.from_config(str(config_path))
        assert engine._core is not None, "compiled core failed to attach"
        return engine

    def _run_find_rules(self, tmp_path, **kwargs) -> SimpliPyEngine:
        """Helper: run find_rules with a small, fast configuration."""
        defaults = dict(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=128,
            constants_fit_challenges=2,
            constants_fit_retries=1,
        )
        defaults.update(kwargs)
        engine = self._core_engine(tmp_path)
        engine.find_rules(**defaults)
        return engine

    def test_construction_always_attaches_core(self) -> None:
        """Rust-only engine: EVERY construction path (including in-memory) attaches the
        compiled core, so mining (and everything else) is always available."""
        engine = SimpliPyEngine(operators=_FIND_RULES_OPERATORS)
        assert engine._core is not None

    def test_discovers_basic_identities(self, tmp_path) -> None:
        """find_rules discovers well-known arithmetic identities (wildcard form)."""
        engine = self._run_find_rules(tmp_path)
        rules_lhs = {tuple(r[0]) for r in engine.simplification_rules}

        # These should always be discovered at length <= 3. The mine emits ?-sorted
        # rules: the miner's measure tolerance certifies the variable-leaf sort only;
        # `_`-subtree status needs the pointwise promotion stage.
        assert ("+", "?0", "0") in rules_lhs
        assert ("*", "?0", "1") in rules_lhs
        # x - x is handled natively by cancel_terms + constant folding,
        # so check for x - 0 instead.
        assert ("-", "?0", "0") in rules_lhs

    def test_deterministic_across_runs(self, tmp_path) -> None:
        """Same seed => identical rule set (regression: the mine must be seeded --
        unseeded evaluation matrices made rulesets irreproducible)."""
        rules_a = self._run_find_rules(tmp_path / "a", seed=7).simplification_rules
        rules_b = self._run_find_rules(tmp_path / "b", seed=7).simplification_rules
        assert rules_a == rules_b

    def test_all_rules_satisfy_wildcard_multiplicity(self, tmp_path) -> None:
        """Every discovered rule must satisfy non-increasing wildcard multiplicity."""
        engine = self._run_find_rules(tmp_path)
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
        )
        # Should have discovered fresh rules, not just the one we seeded
        assert len(engine.simplification_rules) > 1

    def test_prune_reduces_rule_count(self, tmp_path) -> None:
        """prune=True removes redundant explicit rules."""
        engine_no_prune = self._run_find_rules(tmp_path / "a", prune=False)
        engine_pruned = self._run_find_rules(tmp_path / "b", prune=True)
        # Pruning should remove at least some redundant explicit rules
        assert len(engine_pruned.simplification_rules) <= len(engine_no_prune.simplification_rules)


class TestFindRulesWithCore:
    """find_rules on an engine with the compiled Rust core attached.

    Regression tests for the core mine: the fork-based Python pool mutates Python-side
    rule state while `simplify` runs on the immutable Rust core, so before the native
    delegation an engine from `from_config`/`load` mined 0 rules.
    """

    def _core_engine(self, tmp_path) -> SimpliPyEngine:
        """Build an engine from tmp config+rules files so `from_config` attaches the core."""
        import json

        import yaml

        rules_path = tmp_path / "rules.json"
        rules_path.write_text(json.dumps([]))
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({"operators": _FIND_RULES_OPERATORS, "rules": "rules.json"}))
        engine = SimpliPyEngine.from_config(str(config_path))
        assert engine._core is not None, "compiled core failed to attach; the native mine path is untested"
        return engine

    def test_native_mine_discovers_basic_identities(self, tmp_path) -> None:
        """A core-attached engine must mine rules (it returned 0 before the native path)."""
        engine = self._core_engine(tmp_path)
        engine.find_rules(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=128,
            constants_fit_challenges=2,
            constants_fit_retries=1,
        )
        assert len(engine.simplification_rules) > 0
        # The native path always dedups/canonicalizes: dummy variables become `?j`
        # wildcards (variable-leaf sort -- the sort the mine's tolerance certifies).
        rules_lhs = {tuple(r[0]) for r in engine.simplification_rules}
        assert ("+", "?0", "0") in rules_lhs
        assert ("*", "?0", "1") in rules_lhs

    def test_native_mine_updates_the_live_core(self, tmp_path) -> None:
        """After the mine, `simplify` (which runs on the core) must apply the new rules."""
        engine = self._core_engine(tmp_path)
        engine.find_rules(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=128,
            constants_fit_challenges=2,
            constants_fit_retries=1,
        )
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
        )
        assert len(engine.simplification_rules) > 0


class TestConstantFolding:
    """Tests for numeric constant folding in apply_rules_top_down."""

    def _engine(self, rules=None) -> SimpliPyEngine:
        return SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)

    def test_binary_addition_folding(self) -> None:
        """1.23 + 4.56 should evaluate to a numeric result close to 5.79."""
        engine = self._engine()
        result = engine.simplify(["+", "1.23", "4.56"], mask_elementary_literals=False)
        assert len(result) == 1
        assert abs(float(result[0]) - 5.79) < 1e-10

    def test_binary_subtraction_folding(self) -> None:
        """5 - 3 should evaluate to 2."""
        engine = self._engine()
        result = engine.simplify(["-", "5", "3"], mask_elementary_literals=False)
        assert result == ["2"]

    def test_integer_result_formatting(self) -> None:
        """Integer-valued results should not have a decimal point."""
        engine = self._engine()
        result = engine.simplify(["+", "1", "2"], mask_elementary_literals=False)
        assert result == ["3"]

    def test_unary_folding(self) -> None:
        """neg(3) should evaluate to -3."""
        engine = self._engine()
        result = engine.simplify(["neg", "3"], mask_elementary_literals=False)
        assert result == ["-3"]

    def test_nested_constant_folding(self) -> None:
        """2 * 3 + 4 should evaluate to 10 via nested folding."""
        engine = self._engine()
        result = engine.simplify(["+", "*", "2", "3", "4"], mask_elementary_literals=False)
        assert result == ["10"]

    def test_division_by_zero_produces_inf(self) -> None:
        """1 / 0 should produce float("inf") token."""
        engine = self._engine()
        result = engine.simplify(["/", "1", "0"], mask_elementary_literals=False)
        assert result == ['float("inf")']

    def test_mixed_constant_and_placeholder(self) -> None:
        """<constant> + numeric should fold to <constant>."""
        engine = self._engine()
        result = engine.simplify(["+", "1.23", "<constant>"], mask_elementary_literals=False)
        assert result == ["<constant>"]

    def test_constant_placeholder_still_folds(self) -> None:
        """<constant> + <constant> should still fold to <constant>."""
        engine = self._engine()
        result = engine.simplify(["+", "<constant>", "<constant>"], mask_elementary_literals=False)
        assert result == ["<constant>"]

    def test_folding_enables_further_rules(self) -> None:
        """1 - 1 = 0, then x + 0 should simplify to x via rule."""
        engine = self._engine(rules=[(["+", "_0", "0"], ["_0"])])
        result = engine.simplify(["+", "x", "-", "1", "1"], mask_elementary_literals=False)
        assert result == ["x"]

    def test_simplify_infix_numeric_constants(self) -> None:
        """End-to-end: infix '1.23 + 4.56' should become '<constant>'."""
        engine = self._engine()
        result = engine.simplify("1.23 + 4.56")
        assert result == "<constant>"

    def test_constant_folding_observable(self) -> None:
        """Folding is observable through the simplify result itself (the pure-Python
        SimplificationStatistics instrumentation was removed in the Rust-only cutover)."""
        engine = self._engine()
        assert engine.simplify(["+", "1", "2"], mask_elementary_literals=False) == ["3"]


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
        # The resolved rule is live: the rule list holds the concrete value and the
        # (core-synced) engine applies it.
        rules = {tuple(lhs): tuple(rhs) for lhs, rhs in engine.simplification_rules}
        assert ("sin", "1") in rules
        assert rules[("sin", "1")] != ("<constant>",)
        result = engine.simplify(["sin", "1"], mask_elementary_literals=False)
        assert tuple(result) == rules[("sin", "1")]

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
            yaml.safe_dump({"operators": ops, "rules": "rules.json"}))
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
        # log: nan on half the line -- rewriting to 0 would invent a function
        assert list(engine.simplify(["-", "log", "x0", "log", "x0"])) == ["-", "log", "x0", "log", "x0"]
        # pow(x, inf): a.e. in {0, inf} -- inf - inf = nan on positive measure
        expr = ["-", "pow", "x0", 'float("inf")', "pow", "x0", 'float("inf")']
        assert list(engine.simplify(list(expr))) == expr
        # inv: finite a.e. but pole-bearing -- OUT of the certificate's stated scope (needs
        # inf_null); fail-closed means no binding, never unsoundness
        assert list(engine.simplify(["-", "inv", "x0", "inv", "x0"])) == ["-", "inv", "x0", "inv", "x0"]


class TestSearchAndAggressiveMode:
    """The cancel/rules SEARCH (`Engine::simplify_search`) and the `wildcard_all` apply-time
    aggressive mode -- the two public behaviours added on top of the plain fixpoint."""

    def _engine(self):
        import simplipy
        return SimpliPyEngine.from_config(simplipy.get_path('4-3', install=True))

    def test_search_never_grows_an_expression(self) -> None:
        """The search minimises over VISITED states and the input is state zero, so no result
        can ever be longer than its input -- for any node budget."""
        engine = self._engine()
        cases = [
            ["/", "x0", "inv", "x0"],
            ["*", "/", "x1", "x1", "inv", "x1"],
            ["+", "x0", "neg", "+", "x0", "x1"],
            ["/", "inv", "x4", "x4"],
            ["-", "x5", "+", "x5", "x5"],
        ]
        for expr in cases:
            assert len(engine.simplify(list(expr))) <= len(expr), expr

    def test_search_finds_the_shadowed_cancellation(self) -> None:
        """Regression for the candidate-shadowing tail: `inv(x)/x` must not be left at the
        greedy hyper-merge form, which is LONGER than simply not cancelling."""
        engine = self._engine()
        assert len(engine.simplify(["/", "inv", "x4", "x4"])) <= 4

    def test_simplify_is_idempotent(self) -> None:
        """A second pass must be a no-op: the search returns a state the next call cannot beat."""
        engine = self._engine()
        for expr in (["/", "x0", "inv", "x0"], ["*", "/", "x1", "x1", "inv", "x1"],
                     ["-", "x5", "+", "x5", "x5"], ["/", "inv", "x4", "x4"]):
            once = list(engine.simplify(list(expr)))
            assert list(engine.simplify(list(once))) == once, expr

    def test_wildcard_all_is_off_by_default_and_only_widens(self) -> None:
        """`wildcard_all` is the AGGRESSIVE apply-time mode: default OFF, and when ON it only
        ever binds MORE (never fewer) placeholders, so it can only shorten."""
        engine = self._engine()
        exprs = [
            ["/", "x0", "inv", "x0"],
            ["*", "x1", "/", "x1", "x1"],
            ["+", "sin", "x0", "neg", "sin", "x0"],
            ["-", "log", "x0", "log", "x0"],
        ]
        for expr in exprs:
            default = list(engine.simplify(list(expr)))
            explicit_off = list(engine.simplify(list(expr), wildcard_all=False))
            aggressive = list(engine.simplify(list(expr), wildcard_all=True))
            assert default == explicit_off, expr
            assert len(aggressive) <= len(default), (expr, default, aggressive)

    def test_cancel_only_applies_one_cancellation(self) -> None:
        """`cancel_only` exposes the cancellation unit alone: ONE cancellation under the default
        selection. It is not "what simplify does" -- the tree search branches over every
        candidate instead of privileging one -- so this pins the unit's own contract only."""
        engine = self._engine()
        expr = ["*", "/", "x1", "x1", "inv", "x1"]
        assert list(engine._core.cancel_only(list(expr))) == \
            ["*", "/", "inv", "x1", "1", "inv", "1"]
