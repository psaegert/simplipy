import numpy as np
import pytest

from simplipy import SimpliPyEngine


def test_repeated_addition_does_not_emit_unsupported_multipliers() -> None:
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
