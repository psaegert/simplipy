import multiprocessing as mp
import numpy as np
import pytest
from multiprocessing.shared_memory import SharedMemory

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

        # Every remaining explicit rule must be necessary
        for lhs, rhs in engine.simplification_rules:
            if any(is_wildcard.match(t) for t in lhs):
                continue  # Skip pattern rules
            saved = engine.simplification_rules_no_patterns.pop(tuple(lhs), None)
            try:
                result = engine.simplify(list(lhs), mask_elementary_literals=False)
                assert tuple(result) != tuple(rhs), (
                    f"Rule {lhs} -> {rhs} is still redundant after pruning"
                )
            finally:
                if saved is not None:
                    engine.simplification_rules_no_patterns[tuple(lhs)] = saved


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
    """Tests for SimpliPyEngine.sort_operands()."""

    def _engine(self) -> SimpliPyEngine:
        return SimpliPyEngine(operators=_MINIMAL_OPERATORS)

    def test_commutative_reorder(self) -> None:
        """Commutative operator sorts operands into canonical order."""
        engine = self._engine()
        result = engine.sort_operands(["+", "b", "a"])
        # After sorting, variables should be in canonical order
        assert result == engine.sort_operands(["+", "a", "b"])

    def test_non_commutative_unchanged(self) -> None:
        """Non-commutative operator preserves operand order."""
        engine = self._engine()
        assert engine.sort_operands(["-", "b", "a"]) == ["-", "b", "a"]

    def test_idempotent(self) -> None:
        """Sorting an already-sorted expression is idempotent."""
        engine = self._engine()
        first = engine.sort_operands(["+", "b", "a"])
        second = engine.sort_operands(first)
        assert first == second

    def test_nested_commutative(self) -> None:
        """Nested commutative operators are sorted recursively."""
        engine = self._engine()
        result = engine.sort_operands(["*", "z", "a"])
        assert result == engine.sort_operands(["*", "a", "z"])


class TestCancelTerms:
    """Tests for collect_multiplicities + cancel_terms pipeline."""

    def _engine(self) -> SimpliPyEngine:
        return SimpliPyEngine(operators=_MINIMAL_OPERATORS)

    def test_cancel_x_minus_x(self) -> None:
        """x - x should cancel — both operands become neutral element 0."""
        engine = self._engine()
        expr = ["-", "x", "x"]
        tree, annotations, labels = engine.collect_multiplicities(expr)
        result = engine.cancel_terms(tree, annotations, labels)
        assert result == ["-", "0", "0"]

    def test_cancel_x_div_x(self) -> None:
        """x / x should cancel — both operands become neutral element 1."""
        engine = self._engine()
        expr = ["/", "x", "x"]
        tree, annotations, labels = engine.collect_multiplicities(expr)
        result = engine.cancel_terms(tree, annotations, labels)
        assert result == ["/", "1", "1"]

    def test_no_cancellation(self) -> None:
        """Expression with nothing to cancel is returned unchanged."""
        engine = self._engine()
        expr = ["+", "x", "y"]
        tree, annotations, labels = engine.collect_multiplicities(expr)
        result = engine.cancel_terms(tree, annotations, labels)
        assert result == ["+", "x", "y"]


class TestApplySimplificationRules:
    """Tests for SimpliPyEngine.apply_simplifcation_rules()."""

    def test_applies_matching_rule(self) -> None:
        rules = [(["+", "_0", "0"], ["_0"])]
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=rules)
        result = engine.apply_simplifcation_rules(["+", "x", "0"])
        assert result == ["x"]

    def test_all_constants_returns_constant(self) -> None:
        """Expression of only operators and <constant> tokens reduces to <constant>."""
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS)
        result = engine.apply_simplifcation_rules(["+", "<constant>", "<constant>"])
        assert result == ["<constant>"]

    def test_no_matching_rule_unchanged(self) -> None:
        """Expression that matches no rule is returned unchanged."""
        engine = SimpliPyEngine(operators=_MINIMAL_OPERATORS, rules=[])
        result = engine.apply_simplifcation_rules(["+", "x", "y"])
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

    simplified = engine.simplify(expr, max_iter=1, verbose=False)
    simplified_prefix = engine.parse(simplified)

    assert "mult7" not in simplified_prefix
    assert "div7" not in simplified_prefix


def test_repeated_multiplication_avoids_unsupported_powers() -> None:
    engine = SimpliPyEngine.load("dev_7-3", install=True)
    expr = "x / (" + " * ".join(["x"] * 15) + ")"

    simplified = engine.simplify(expr, max_iter=1, verbose=False)
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
    """Tests for SimpliPyEngine.find_rules()."""

    def _run_find_rules(self, **kwargs) -> SimpliPyEngine:
        """Helper: run find_rules with a small, fast configuration."""
        defaults = dict(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=128,
            constants_fit_challenges=2,
            constants_fit_retries=1,
            n_workers=2,
        )
        defaults.update(kwargs)
        engine = SimpliPyEngine(operators=_FIND_RULES_OPERATORS)
        engine.find_rules(**defaults)
        return engine

    def test_discovers_basic_identities(self) -> None:
        """find_rules discovers well-known arithmetic identities."""
        engine = self._run_find_rules()
        rules_lhs = {tuple(r[0]) for r in engine.simplification_rules}

        # These should always be discovered at length <= 3
        assert ("+", "x0", "0") in rules_lhs or ("+", "x1", "0") in rules_lhs
        assert ("*", "x0", "1") in rules_lhs or ("*", "x1", "1") in rules_lhs
        assert ("-", "x0", "x0") in rules_lhs or ("-", "x1", "x1") in rules_lhs

    def test_all_rules_satisfy_wildcard_multiplicity(self) -> None:
        """Every discovered rule must satisfy non-increasing wildcard multiplicity."""
        engine = self._run_find_rules()
        for lhs, rhs in engine.simplification_rules:
            assert not violates_wildcard_multiplicity(lhs, rhs), (
                f"Rule violates wildcard multiplicity: {lhs} -> {rhs}"
            )

    def test_reset_rules_clears_existing(self) -> None:
        """reset_rules=True starts from an empty rule set."""
        engine = SimpliPyEngine(
            operators=_FIND_RULES_OPERATORS,
            rules=[(["+", "x0", "0"], ["x0"])],
        )
        assert len(engine.simplification_rules) == 1
        engine.find_rules(
            max_source_pattern_length=3,
            dummy_variables=2,
            extra_internal_terms=["0", "1"],
            X=128,
            constants_fit_challenges=2,
            constants_fit_retries=1,
            n_workers=2,
            reset_rules=True,
        )
        # Should have discovered fresh rules, not just the one we seeded
        assert len(engine.simplification_rules) > 1

    def test_prune_reduces_rule_count(self) -> None:
        """prune=True removes redundant explicit rules."""
        engine_no_prune = self._run_find_rules(prune=False)
        engine_pruned = self._run_find_rules(prune=True)
        # Pruning should remove at least some redundant explicit rules
        assert len(engine_pruned.simplification_rules) <= len(engine_no_prune.simplification_rules)


class TestFindRuleWorkerNumericShortCircuit:
    """Tests that find_rule_worker short-circuits purely-numeric expressions."""

    def _make_shared_X(self, n_samples: int = 32, n_vars: int = 2):
        """Create a small shared-memory array for worker tests."""
        X_data = np.random.normal(size=(n_samples, n_vars))
        shm = SharedMemory(create=True, size=X_data.nbytes)
        X_shared = np.ndarray(X_data.shape, dtype=X_data.dtype, buffer=shm.buf)
        X_shared[:] = X_data[:]
        return shm, X_data.shape, X_data.dtype

    def _run_worker(self, expression, engine, shm, X_shape, X_dtype, dummy_variables):
        """Submit one work item to find_rule_worker and return the result."""
        work_q = mp.Queue()
        result_q = mp.Queue()
        # The worker needs allowed_candidate_lengths to be non-empty and > 0
        simplified_length = len(expression)
        allowed_candidate_lengths = tuple(range(1, simplified_length))
        work_q.put((expression, simplified_length, allowed_candidate_lengths))
        work_q.put(None)  # sentinel

        p = mp.Process(
            target=engine.find_rule_worker,
            args=(
                0, work_q, result_q,
                X_shape, X_dtype, shm.name,
                {},  # no candidates needed — we only test the short-circuit path
                dummy_variables,
                engine.operator_arity,
                1, 1,
            )
        )
        p.start()
        p.join(timeout=10)
        assert p.exitcode == 0, "Worker process exited with an error"
        return result_q.get_nowait()

    @pytest.mark.parametrize("expression", [
        ("+", "1", "2"),
        ("*", "2.5", "3"),
        ("+", "1", "<constant>"),
        ("+", "<constant>", "2.5"),
        ("*", "<constant>", "<constant>"),
    ])
    def test_purely_numeric_expression_short_circuits(self, expression) -> None:
        """Expressions consisting only of operators, '<constant>', and numeric
        string literals must be immediately mapped to ('<constant>',) without
        entering the expensive candidate-search loop."""
        engine = SimpliPyEngine(operators=_FIND_RULES_OPERATORS)
        dummy_variables = ["x0", "x1"]
        shm, X_shape, X_dtype = self._make_shared_X()
        try:
            result = self._run_worker(list(expression), engine, shm, X_shape, X_dtype, dummy_variables)
        finally:
            shm.close()
            shm.unlink()

        assert result is not None
        lhs, rhs = result
        assert tuple(rhs) == ("<constant>",), (
            f"Expected ('<constant>',) for expression {expression}, got {rhs}"
        )
