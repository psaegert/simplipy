"""Tests pinning the conversion-quirk fixes (branch fix/conversion-quirks).

See docs/CONVERSION_QUIRKS.md for the analysis + measured impact. Each fix corrects a behavior that the
deployed tag c84741f / engine-id dev_7-3 gets wrong; these tests assert the CORRECTED behavior and that
the surrounding (correct) behavior is unchanged (no over-fix).
"""
import pytest

from simplipy import Mode, SimpliPyEngine  # noqa: F401  (re-exported for the suite's historical imports)

from conftest import acj_config_path


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
    assert engine.read_infix(engine.prefix_to_infix(pre, power="**"), convert_expression=False) == pre


# -- #6: a raw unconfigured powN token no longer crashes pass-1 -----------------------------------
def test_quirk6_raw_powN_no_crash(engine):
    assert engine.convert_expression(["pow7", "x1"]) == ["pow7", "x1"]


# -- sanity: ordinary expressions are unchanged ---------------------------------------------------
def test_no_regression_on_ordinary_expressions(engine):
    assert engine.read_infix("x1 + x2") == ["+", "x1", "x2"]
    assert engine.read_infix("x1 ^ 2") == ["pow2", "x1"]
    assert engine.read_infix("x1 * x2 + x3") == ["+", "*", "x1", "x2", "x3"]
    assert engine.read_infix("sin(x1) / x2", mask_numbers=True) == ["/", "sin", "x1", "x2"]


def test_non_smooth_exponents_keep_binary_pow(engine):
    """FIX (phantom powN): exponents not decomposable into pow2..pow{max_power} must stay
    binary `pow`, never phantom tokens like `pow7` (which have no realization and corrupt
    arity downstream). Smooth exponents keep their exact previous form."""
    # non-smooth: binary pow survives and REALIZES
    assert engine.read_infix("v1 ** 7", mask_numbers=False) == ["pow", "v1", "7"]
    assert engine.read_infix("v1 ** 14", mask_numbers=False) == ["pow", "v1", "14"]
    assert engine.read_infix("v1 ** (7/3)", mask_numbers=False) == ["pow", "v1", "/", "7", "3"]
    # smooth: unchanged decompositions
    assert engine.read_infix("v1 ** 6 + v1", mask_numbers=False) == ["+", "pow2", "pow3", "v1", "v1"]
    assert engine.read_infix("v1 ** 8", mask_numbers=False) == ["pow4", "pow2", "v1"]
    assert engine.read_infix("v1 ** (4/3)", mask_numbers=False) == ["pow1_3", "pow4", "v1"]


# -- H-047 (2026-08-05): the **-exponent fold gate is SPELLING-EXACT --------------------------
# The port (and the legacy original) gated folds on limit_denominator(1e6) of the f64
# VALUE: exponents within ~5e-7 of 1 collapsed the pow entirely (x**(1+ulp) -> x, with NO
# vocabulary gate protecting the emit-nothing case), near-0 folded to 1, near -1 to inv,
# and near-misses of <=5/<=5 fractions snapped to the wrong powN/pow1_N function under a
# powN-declaring vocabulary. The exact decimal denotation now decides; every fold on an
# EXACT spelling survives.
def test_h047_near_one_exponent_keeps_pow(engine):
    assert engine.read_infix("v1 ** (1.0000000000000002)", mask_numbers=False) == ["pow", "v1", "1.0000000000000002"]
    assert engine.read_infix("v1 ** (0.9999995)", mask_numbers=False) == ["pow", "v1", "0.9999995"]
    assert engine.read_infix("v1 ** (1e-9)", mask_numbers=False) == ["pow", "v1", "1e-9"]
    assert engine.read_infix("v1 ** (-1.0000000000000002)", mask_numbers=False) == ["pow", "v1", "-1.0000000000000002"]


def test_h047_exact_spellings_still_fold(engine):
    assert engine.read_infix("v1 ** (1.0)", mask_numbers=False) == ["v1"]
    assert engine.read_infix("v1 ** (0.0)", mask_numbers=False) == ["1"]
    assert engine.read_infix("v1 ** (-1.0)", mask_numbers=False) == ["inv", "v1"]
    assert engine.read_infix("v1 ** (0.5)", mask_numbers=False) == ["pow1_2", "v1"]


def test_h047_near_fraction_no_longer_snaps(engine):
    # pre-fix: limit_denominator approximated to 1/2 and emitted pow1_2 = sqrt (value change)
    assert engine.read_infix("v1 ** (0.5000000000000001)", mask_numbers=False) == ["pow", "v1", "0.5000000000000001"]


# -- H-049 (2026-08-05): beyond-i128 exponent spellings KEEP binary pow, never raise ----------
# The legacy CPython int() was arbitrary-precision and never failed on a digit string; the
# i128 parse error was port-minted and raised on the engine's OWN rendered infix.
def test_h049_beyond_i128_integer_exponent_keeps_pow(engine):
    huge = "170141183460469231731687303715884105728"  # 2^127
    assert engine.read_infix(f"v1 ** {huge}", mask_numbers=False) == ["pow", "v1", huge]
    assert engine.read_infix(f"v1 ** ({huge}/3)", mask_numbers=False) == ["pow", "v1", "/", huge, "3"]


def test_h049_float_division_exponent_keeps_pow(engine):
    # the legacy dead branch RAISED here (int('2.0') failure-parity); the engine's own
    # divisor-side infix can render this shape, and its own output must always re-parse
    assert engine.read_infix("v1 ** (2.0/4.0)", mask_numbers=False) == ["pow", "v1", "/", "2.0", "4.0"]


# -- X10 (2026-08-15): a negative-literal pow base parenthesizes under power='**' -------------
# `['pow','-2','x0']` rendered as '-2 ** x0', which re-parses -- in this engine AND in
# SymPy 1.14 -- as -(2 ** x0): '**' outbinds the unary minus, so the render changed the
# value. Measured: an exact bijection, 59/5494 canonical outputs broke, every one a
# `pow <negative literal>` base. The composite spelling was already correct.
def test_x10_negative_literal_base_parenthesizes_under_starstar(engine):
    out = engine.prefix_to_infix(['pow', '-2', 'x0'], power='**')
    assert out == '(-2) ** x0', out
    # the round trip must land back on the same expression, not its negation
    assert engine.read_infix(out, mask_numbers=False) == ['pow', '-2', 'x0']


def test_x10_composite_and_positive_bases_unchanged(engine):
    assert engine.prefix_to_infix(['pow', 'neg', '2', 'x0'], power='**') == '(-2) ** x0'
    assert engine.prefix_to_infix(['pow', '2', 'x0'], power='**') == '2 ** x0'


def test_x10_site2_unary_pow_negative_literal_parenthesizes(engine):
    # render site 2: the unary powN family (legacy vocabulary) had the identical defect
    assert engine.prefix_to_infix(['pow2', '-2'], power='**') == '(-2)**2'


class TestEvaluateConstants:
    """The FRONT DOOR for numeric folding: explicit, opt-in, and priced by nothing.

    The 2026-08-02 fold unification deleted serve's last f64-evaluation arm, because
    "an f64 evaluation landing exactly on a cheap literal while truly differing at
    1e-17" is a rounding, not a simplification. That removed a real capability. This
    method gives it back with a name and a door, so it can never contaminate a
    canonical form.
    """

    @pytest.fixture(scope='class')
    def eng(self):
        return SimpliPyEngine.from_config(acj_config_path())

    def test_it_evaluates_what_simplify_refuses(self, eng):
        assert eng.simplify(['tan', '1']) == ['tan', '1']
        assert eng.evaluate_constants(['tan', '1']) == ['1.5574077246549023']

    def test_simplify_folds_only_where_mu_says_cheaper(self, eng):
        """The claim this used to make -- "no mode folds a ground transcendental" -- was
        retired by mode-aware folding, and its own `real` arm was wiped rather than
        checked (`set_mode_rules('real', [])`, against a docstring saying the artifact's
        real set was used).

        What holds now: `f64` and `corpus` fold when mu prices the literal cheaper, and
        `real` never folds a transcendental at all -- `tan(1)` is irrational and no
        literal spells it. `tan 1` survives everywhere because its float costs 58,877
        against the expression's 11,000; `asin 1e-08` folds where folding is allowed."""
        from conftest import require_triple_or_skip
        require_triple_or_skip()
        assert eng._core.mode_rules_len('real') is not None, 'the artifact must ship one'
        for mode in (Mode.f64, Mode.real, Mode.corpus):
            assert eng.simplify(['tan', '1'], mode=mode) == ['tan', '1'], mode
        assert eng.simplify(['asin', '1e-08'], mode=Mode.f64) == ['0.00000001']
        assert eng.simplify(['asin', '1e-08'], mode=Mode.corpus) == ['0.00000001']
        assert eng.simplify(['asin', '1e-08'], mode=Mode.real) == ['asin', '0.00000001']

    def test_a_fitted_constant_is_not_a_value(self, eng):
        """`<constant>` is a degree of freedom, so a subtree carrying one is not
        foldable -- but a sibling that IS foldable still folds."""
        assert eng.evaluate_constants(['*', 'tan', '1', '<constant>']) \
            == ['*', '1.5574077246549023', '<constant>']

    def test_a_non_finite_value_is_never_injected(self, eng):
        """Substituting inf/nan would put a non-finite token into an expression that had
        none -- the defect class that put a guard into flash-ansr. Refusing is safe and
        the caller can still evaluate it themselves."""
        for tokens in (['/', '1', '0'], ['log', '0']):
            assert eng.evaluate_constants(list(tokens)) == tokens

    def test_folding_is_maximal_and_leaves_variables_alone(self, eng):
        assert eng.evaluate_constants(['+', 'tan', '1', 'x0']) \
            == ['+', '1.5574077246549023', 'x0']
        assert eng.evaluate_constants(['+', 'x0', 'x1']) == ['+', 'x0', 'x1']

    def test_it_is_dialect_preserving_like_simplify(self, eng):
        assert eng.evaluate_constants('tan(1) + x0') == '1.5574077246549023 + x0'
        assert eng.evaluate_constants(eng.to_tagged(['+', 'tan', '1', 'x0'])) \
            == ['<add>', '1.5574077246549023', 'x0', '</add>']

    def test_it_is_more_accurate_than_the_rounding_rule_it_replaces(self, eng):
        """`e**sinh(-5)` is 5.94e-33, and a mined rule says 0 -- a 100% relative error
        that the judge's tolerance floor used to hide. The front door tells the truth."""
        assert eng.evaluate_constants(['pow', 'np.e', 'sinh', '(-5)']) \
            == ['5.942307292381135e-33']
