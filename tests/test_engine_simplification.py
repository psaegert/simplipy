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
