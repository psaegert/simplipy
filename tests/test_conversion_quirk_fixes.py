"""Tests pinning the conversion-quirk fixes (branch fix/conversion-quirks).

See docs/CONVERSION_QUIRKS.md for the analysis + measured impact. Each fix corrects a behavior that the
deployed tag c84741f / engine-id dev_7-3 gets wrong; these tests assert the CORRECTED behavior and that
the surrounding (correct) behavior is unchanged (no over-fix).
"""
import pytest

from simplipy import SimpliPyEngine  # noqa: F401  (re-exported for the suite's historical imports)


@pytest.fixture(scope="module")
def engine():
    # Raw legacy TABLE from the in-repo fixture (conversion input-language tests
    # need no rules and no asset); the public artifact-load path refuses
    # generation 1 (see tests/conftest.py).
    from conftest import construct_legacy_table
    return construct_legacy_table()


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


# -- H-047 (2026-08-05): the **-exponent fold gate is SPELLING-EXACT --------------------------
# The port (and the legacy original) gated folds on limit_denominator(1e6) of the f64
# VALUE: exponents within ~5e-7 of 1 collapsed the pow entirely (x**(1+ulp) -> x, with NO
# vocabulary gate protecting the emit-nothing case), near-0 folded to 1, near -1 to inv,
# and near-misses of <=5/<=5 fractions snapped to the wrong powN/pow1_N function under a
# powN-declaring vocabulary. The exact decimal denotation now decides; every fold on an
# EXACT spelling survives.
def test_h047_near_one_exponent_keeps_pow(engine):
    assert engine.parse("v1 ** (1.0000000000000002)", mask_numbers=False) == ["pow", "v1", "1.0000000000000002"]
    assert engine.parse("v1 ** (0.9999995)", mask_numbers=False) == ["pow", "v1", "0.9999995"]
    assert engine.parse("v1 ** (1e-9)", mask_numbers=False) == ["pow", "v1", "1e-9"]
    assert engine.parse("v1 ** (-1.0000000000000002)", mask_numbers=False) == ["pow", "v1", "-1.0000000000000002"]


def test_h047_exact_spellings_still_fold(engine):
    assert engine.parse("v1 ** (1.0)", mask_numbers=False) == ["v1"]
    assert engine.parse("v1 ** (0.0)", mask_numbers=False) == ["1"]
    assert engine.parse("v1 ** (-1.0)", mask_numbers=False) == ["inv", "v1"]
    assert engine.parse("v1 ** (0.5)", mask_numbers=False) == ["pow1_2", "v1"]


def test_h047_near_fraction_no_longer_snaps(engine):
    # pre-fix: limit_denominator approximated to 1/2 and emitted pow1_2 = sqrt (value change)
    assert engine.parse("v1 ** (0.5000000000000001)", mask_numbers=False) == ["pow", "v1", "0.5000000000000001"]


# -- H-049 (2026-08-05): beyond-i128 exponent spellings KEEP binary pow, never raise ----------
# The legacy CPython int() was arbitrary-precision and never failed on a digit string; the
# i128 parse error was port-minted and raised on the engine's OWN rendered infix.
def test_h049_beyond_i128_integer_exponent_keeps_pow(engine):
    huge = "170141183460469231731687303715884105728"  # 2^127
    assert engine.parse(f"v1 ** {huge}", mask_numbers=False) == ["pow", "v1", huge]
    assert engine.parse(f"v1 ** ({huge}/3)", mask_numbers=False) == ["pow", "v1", "/", huge, "3"]


def test_h049_float_division_exponent_keeps_pow(engine):
    # the legacy dead branch RAISED here (int('2.0') failure-parity); the engine's own
    # divisor-side infix can render this shape, and its own output must always re-parse
    assert engine.parse("v1 ** (2.0/4.0)", mask_numbers=False) == ["pow", "v1", "/", "2.0", "4.0"]
