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


# -- #6: a raw unconfigured powN token no longer crashes pass-1 -----------------------------------
def test_quirk6_raw_powN_no_crash(engine):
    assert engine.convert_expression(["pow7", "x1"]) == ["pow7", "x1"]


# -- sanity: ordinary expressions are unchanged ---------------------------------------------------
def test_no_regression_on_ordinary_expressions(engine):
    assert engine.parse("x1 + x2") == ["+", "x1", "x2"]
    assert engine.parse("x1 ^ 2") == ["pow2", "x1"]
    assert engine.parse("x1 * x2 + x3") == ["+", "*", "x1", "x2", "x3"]
    assert engine.parse("sin(x1) / x2", mask_numbers=True) == ["/", "sin", "x1", "x2"]
