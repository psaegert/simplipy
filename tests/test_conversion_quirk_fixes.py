"""Tests pinning the conversion-quirk fixes (branch fix/conversion-quirks).

See docs/CONVERSION_QUIRKS.md for the analysis + measured impact. Each fix corrects a behavior that the
deployed tag c84741f / engine-id dev_7-3 gets wrong; these tests assert the CORRECTED behavior and that
the surrounding (correct) behavior is unchanged (no over-fix).
"""
import pytest

from simplipy import SimpliPyEngine


@pytest.fixture(scope="module")
def engine():
    return SimpliPyEngine.load("dev_7-3", install=True)


# -- #1: fractional power no longer silently dropped (the serious one) ----------------------------
def test_quirk1_fractional_power_preserved(engine):
    assert engine.convert_expression(["pow2", "pow1_3", "x1"]) == ["pow2", "pow1_3", "x1"]
    assert engine.convert_expression(["pow4", "pow1_5", "x1"]) == ["pow4", "pow1_5", "x1"]
    assert engine.convert_expression(["pow2", "pow1_2", "x1"]) == ["pow2", "pow1_2", "x1"]


def test_quirk1_no_overfix_real_chains_still_combine(engine):
    # Genuine same-family chains must still combine (the fix only stops cross-family absorption).
    assert engine.convert_expression(["**", "x1", "6"]) == ["pow2", "pow3", "x1"]
    assert engine.convert_expression(["pow2", "pow2", "x1"]) == ["pow4", "x1"]
    assert engine.convert_expression(["pow2", "pow2", "pow2", "x1"]) == ["pow4", "pow2", "x1"]
    assert engine.convert_expression(["pow1_2", "pow1_2", "x1"]) == ["pow1_4", "x1"]


# -- #2: x**0 -> 1 (not the invalid 'pow0' token) -------------------------------------------------
def test_quirk2_pow_zero_is_one(engine):
    assert engine.convert_expression(["**", "x1", "0"]) == ["1"]
    assert engine.convert_expression(["**", "x1", "0.0"]) == ["1"]
    assert engine.convert_expression(["**", "x1", "/", "0", "3"]) == ["1"]
    # no 'pow0' token anywhere
    assert "pow0" not in engine.convert_expression(["**", "x1", "0"])


# -- #3: negating a numeric literal toggles ONE minus (no '--5') ----------------------------------
def test_quirk3_neg_literal_toggles_one_minus(engine):
    assert engine.convert_expression(["neg", "5"]) == ["-5"]
    assert engine.convert_expression(["neg", "-5"]) == ["5"]
    assert engine.convert_expression(["+", "neg", "2", "x1"]) == ["+", "-2", "x1"]


# -- #4: '^' parses unary minus like '**' ---------------------------------------------------------
def test_quirk4_caret_matches_starstar_for_unary_minus(engine):
    assert engine.infix_to_prefix("x1 ^ - x2") == engine.infix_to_prefix("x1 ** - x2")
    assert engine.infix_to_prefix("x1 ^ - x2") == ["neg", "**", "x1", "x2"]


# -- #5: operator associativity, fixed as a COORDINATED parse+render change (left-assoc parse +
#    right_allows_flatten disabled), so prefix<->infix round-trip identity is preserved. -----------
def test_quirk5_left_associativity(engine):
    # the kinetic-energy form: (1/2)*m*v**2, NOT 1/(2*m*v**2).
    assert engine.infix_to_prefix("1/2 * m * v ** 2") == ["*", "*", "/", "1", "2", "m", "**", "v", "2"]
    assert engine.infix_to_prefix("a - b - c") == ["-", "-", "a", "b", "c"]
    assert engine.infix_to_prefix("a / b / c") == ["/", "/", "a", "b", "c"]
    assert engine.infix_to_prefix("a ** b ** c") == ["**", "a", "**", "b", "c"]  # right-assoc kept
    assert engine.infix_to_prefix("a - (b - c)") == ["-", "a", "-", "b", "c"]  # parens respected


def test_quirk5_render_parenthesizes_right_chains(engine):
    # the render half: equal-precedence right operands keep parens (no flatten), so left-assoc parse
    # recovers the structure.
    assert engine.prefix_to_infix(["+", "a", "+", "b", "c"]) == "a + (b + c)"
    assert engine.prefix_to_infix(["+", "+", "a", "b", "c"]) == "a + b + c"  # left-nest flattens
    # round-trip identity holds for a right-nested associative chain.
    pre = ["*", "a", "*", "b", "c"]
    assert engine.parse(engine.prefix_to_infix(pre, power="**"), convert_expression=False) == pre


# -- #6: a raw unconfigured powN token no longer crashes pass-1 -----------------------------------
def test_quirk6_raw_powN_no_crash(engine):
    assert engine.convert_expression(["pow7", "x1"]) == ["pow7", "x1"]


# -- sanity: ordinary expressions are unchanged ---------------------------------------------------
def test_no_regression_on_ordinary_expressions(engine):
    assert engine.parse("x1 + x2") == ["+", "x1", "x2"]
    assert engine.parse("x1 ^ 2") == ["pow2", "x1"]
    assert engine.parse("x1 * x2 + x3") == ["+", "*", "x1", "x2", "x3"]
    assert engine.parse("sin(x1) / x2", mask_numbers=True) == ["/", "sin", "x1", "x2"]


def test_non_smooth_exponents_keep_binary_pow(engine):
    """FIX (phantom powN): exponents not decomposable into pow2..pow{max_power} must stay
    binary `pow`, never phantom tokens like `pow7` (which have no realization and corrupt
    arity downstream). Smooth exponents keep their exact previous form."""
    # non-smooth: binary pow survives and REALIZES
    assert engine.parse("v1 ** 7", mask_numbers=False) == ["pow", "v1", "7"]
    assert engine.parse("v1 ** 14", mask_numbers=False) == ["pow", "v1", "14"]
    assert engine.parse("v1 ** (7/3)", mask_numbers=False) == ["pow", "v1", "/", "7", "3"]
    # smooth: unchanged decompositions
    assert engine.parse("v1 ** 6 + v1", mask_numbers=False) == ["+", "pow2", "pow3", "v1", "v1"]
    assert engine.parse("v1 ** 8", mask_numbers=False) == ["pow4", "pow2", "v1"]
    assert engine.parse("v1 ** (4/3)", mask_numbers=False) == ["pow1_3", "pow4", "v1"]


def test_high_degree_polynomial_chain_realizes(engine):
    """End-to-end regression for the original failure (Nonic / Livermore-9/-22): a chain
    containing x**7 must parse, realize, codify, and evaluate."""
    import numpy as np
    from simplipy.utils import codify

    chain = " + ".join([f"v1 ** {k}" for k in range(9, 1, -1)] + ["v1"])
    prefix = engine.parse(chain, mask_numbers=False)
    realized = engine.operators_to_realizations(prefix)
    fn = engine.code_to_lambda(codify(engine.prefix_to_infix(realized, realization=True), ["v1"]))
    x = np.linspace(-1.0, 1.0, 7)
    assert np.allclose(fn(x), sum(x ** k for k in range(2, 10)) + x)
